print("xai_vit_deit.py started", flush=True)

import os
import sys
import math
import argparse
import traceback
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np

from pathlib import Path
from PIL import Image
from torchvision import transforms

from config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    IMAGE_SIZE,
    NUM_CLASSES
)

print("Imports finished", flush=True)


CHECKPOINTS = {
    "vit": os.path.join(CHECKPOINT_DIR, "vit_best.pth"),
    "deit": os.path.join(CHECKPOINT_DIR, "deit_best.pth"),
    "swin": os.path.join(CHECKPOINT_DIR, "swin_best.pth")
}

XAI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "heatmaps")


# ============================================================
# Basic setup
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_image_path(image_path):
    """
    If the given path exists, use it as-is. Otherwise search the project
    (data_split and data_raw) for a file with the same name, in case of
    nested folders or a slightly wrong path, and print what was found.
    """
    if os.path.exists(image_path):
        print("Image found directly at:", image_path, flush=True)
        return image_path

    print(f"Image not found at given path: {image_path}", flush=True)
    print("Searching project for a file with this name...", flush=True)

    target_name = os.path.basename(image_path)
    project_root = os.path.dirname(os.path.abspath(__file__))

    matches = []
    for search_dir in ["data_split", "data_raw"]:
        full_search_dir = os.path.join(project_root, search_dir)
        if not os.path.exists(full_search_dir):
            continue

        for match in Path(full_search_dir).rglob(target_name):
            matches.append(str(match))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find any file named '{target_name}' under "
            f"data_split/ or data_raw/. Check the filename and folder "
            f"structure (run: find data_split -iname \"{target_name}\")."
        )

    print(f"Found {len(matches)} match(es):", flush=True)
    for m in matches:
        print(" -", m, flush=True)

    chosen = matches[0]
    print("Using:", chosen, flush=True)

    return chosen


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Loading checkpoint:", checkpoint_path, flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Checkpoint loaded successfully:", checkpoint_path, flush=True)

    return model, checkpoint


def overlay_cam_on_image(image_pil, cam, alpha=0.5):
    image_np = np.array(image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = heatmap * alpha + image_np * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)


def normalize_cam(cam):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


# ============================================================
# Algorithm 3 — Gradient Attention Rollout (ViT-B/16, DeiT-Base)
#
# timm's default Attention.forward uses the fused scaled-dot-product-
# attention kernel, which never materializes the (heads x N x N)
# attention matrix, so it cannot be captured or backpropagated through
# directly. Each Attention block's forward method is monkey-patched at
# runtime to force the explicit (non-fused) softmax computation and
# retain the attention tensor's gradient, exactly as described in
# Section 4.3.3 of the report.
# ============================================================

class GradientAttentionRollout:
    """
    Patches every Attention block in a timm ViT/DeiT model so the raw
    softmax attention matrix at each layer is captured and its gradient
    is retained after backward(). Implements steps 1-2 of Algorithm 3.
    """

    def __init__(self, model):
        self.model = model
        self.attention_matrices = []
        self._original_forwards = []
        self._patch_attention_layers()

    def _patch_attention_layers(self):
        for block in self.model.blocks:
            attn_module = block.attn
            self._original_forwards.append((attn_module, attn_module.forward))
            attn_module.forward = self._make_patched_forward(attn_module)

    def _make_patched_forward(self, attn_module):
        rollout_self = self

        def patched_forward(x, attn_mask=None, is_causal=False, **kwargs):
            if attn_mask is not None or is_causal:
                raise NotImplementedError(
                    "GradientAttentionRollout patch does not support "
                    "attn_mask or is_causal (not needed for standard "
                    "image classification forward passes)."
                )

            B, N, C = x.shape
            num_heads = attn_module.num_heads
            head_dim = getattr(attn_module, "head_dim", C // num_heads)
            scale = getattr(attn_module, "scale", head_dim ** -0.5)

            qkv = attn_module.qkv(x).reshape(
                B, N, 3, num_heads, head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            q_norm = getattr(attn_module, "q_norm", nn.Identity())
            k_norm = getattr(attn_module, "k_norm", nn.Identity())
            q = q_norm(q)
            k = k_norm(k)

            # Explicit (non-fused) attention so the matrix can be captured
            attn_raw = (q * scale) @ k.transpose(-2, -1)
            attn_raw = attn_raw.softmax(dim=-1)
            attn_raw.retain_grad()
            rollout_self.attention_matrices.append(attn_raw)

            attn_drop = getattr(attn_module, "attn_drop", nn.Identity())
            attn_for_value = attn_drop(attn_raw)

            out = attn_for_value @ v
            out = out.transpose(1, 2).reshape(B, N, C)
            out = attn_module.proj(out)

            proj_drop = getattr(attn_module, "proj_drop", nn.Identity())
            out = proj_drop(out)

            return out

        return patched_forward

    def restore(self):
        for attn_module, original_forward in self._original_forwards:
            attn_module.forward = original_forward


def compute_gradient_attention_rollout(model, input_tensor):
    """
    Implements Algorithm 3, steps 1-7:
    forward pass -> backprop predicted class -> gradient-weight each
    layer's attention -> add identity + renormalize -> multiply layers
    sequentially -> extract CLS row -> reshape to a spatial grid.
    """
    extractor = GradientAttentionRollout(model)

    try:
        model.zero_grad()
        output = model(input_tensor)

        pred_class = torch.argmax(output, dim=1).item()
        score = output[0, pred_class]
        score.backward()

        if len(extractor.attention_matrices) == 0:
            raise RuntimeError(
                "No attention matrices were captured. The Attention module "
                "patch may not match this timm version's implementation."
            )

        rollout = None

        for attn in extractor.attention_matrices:
            grad = attn.grad

            if grad is None:
                raise RuntimeError(
                    "Gradient was not captured for an attention layer. "
                    "retain_grad() may not have taken effect before backward()."
                )

            # Step 3: element-wise multiply by gradient, keep only
            # positive (class-supporting) contributions
            relevance = torch.relu(attn * grad)

            # Step 4: average across attention heads -> (N, N)
            relevance = relevance.mean(dim=1)[0]

            # Step 5: add identity (residual connection), re-normalize rows
            n_tokens = relevance.shape[-1]
            identity = torch.eye(n_tokens, device=relevance.device)
            relevance = relevance + identity
            relevance = relevance / relevance.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Step 6: multiply layers sequentially, first layer to last
            if rollout is None:
                rollout = relevance
            else:
                rollout = relevance @ rollout

        # Step 7: extract CLS row, reshape patch tokens to a square grid
        num_prefix = getattr(model, "num_prefix_tokens", 1)
        cls_row = rollout[0, num_prefix:]

        n_patches = cls_row.shape[0]
        grid_size = int(round(math.sqrt(n_patches)))

        if grid_size * grid_size != n_patches:
            raise ValueError(
                f"Cannot reshape {n_patches} patch tokens into a square grid."
            )

        cam = cls_row.reshape(grid_size, grid_size)
        cam = cam.detach().cpu().numpy()
        cam = normalize_cam(cam)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        probs = output.softmax(dim=1).detach().cpu().numpy()[0]

        return cam, pred_class, probs

    finally:
        extractor.restore()


# ============================================================
# Algorithm 4 — Grad-CAM Adaptation (Swin-Tiny)
#
# Swin-Tiny has no CLS token and uses windowed rather than global
# self-attention, so Algorithm 3 does not apply. Forward/backward hooks
# are registered on the final stage instead, per Section 3.3.2.
# ============================================================

class ActivationsAndGradients:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def swin_tokens_to_grid(tokens):
    """
    Reshapes the final Swin stage's output into a (H, W, C) spatial grid.
    Handles both (B, N, C) token sequences and (B, H, W, C) tensors,
    since this varies across timm versions.
    """
    if tokens.dim() == 4:
        return tokens[0]

    if tokens.dim() == 3:
        seq = tokens[0]
        n = seq.shape[0]
        h = w = int(round(math.sqrt(n)))

        if h * w != n:
            raise ValueError(
                f"Cannot reshape {n} tokens into a square grid for swin "
                f"(got shape {tuple(tokens.shape)})"
            )

        return seq.reshape(h, w, -1)

    raise ValueError(f"Unexpected activation shape: {tuple(tokens.shape)}")


def compute_swin_gradcam(model, input_tensor):
    """
    Implements Algorithm 4, steps 1-5.
    """
    target_layer = model.layers[-1].blocks[-1]
    hook = ActivationsAndGradients(target_layer)

    try:
        model.zero_grad()
        output = model(input_tensor)

        pred_class = torch.argmax(output, dim=1).item()
        score = output[0, pred_class]
        score.backward()

        activations = hook.activations
        gradients = hook.gradients

        act_grid = swin_tokens_to_grid(activations)   # (h, w, C)
        grad_grid = swin_tokens_to_grid(gradients)     # (h, w, C)

        # Step 3: global-average-pool gradient across spatial dims -> per-channel weight
        weights = grad_grid.mean(dim=(0, 1))

        # Step 4: weighted sum of channels, ReLU
        cam = torch.relu((act_grid * weights).sum(dim=-1))

        cam = cam.cpu().numpy()
        cam = normalize_cam(cam)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        probs = output.softmax(dim=1).detach().cpu().numpy()[0]

        return cam, pred_class, probs

    finally:
        hook.remove()


# ============================================================
# Orchestration
# ============================================================

def explain_image(image_path, save_prefix=None):
    print("\n=== Starting explain_image ===", flush=True)

    device = get_device()
    print("Using device:", device, flush=True)

    image_path = resolve_image_path(image_path)

    os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)
    print("Output folder ready:", XAI_OUTPUT_DIR, flush=True)

    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    print("Image loaded, size:", image.size, flush=True)

    if save_prefix is None:
        save_prefix = os.path.splitext(os.path.basename(image_path))[0]

    val_f1_scores = {}
    cams = {}
    predictions = {}
    class_to_idx = None

    for model_key, checkpoint_path in CHECKPOINTS.items():
        print(f"\n--- Processing model: {model_key} ---", flush=True)

        model, checkpoint = load_model(checkpoint_path, device)
        val_f1_scores[model_key] = checkpoint["best_val_f1"]

        if class_to_idx is None:
            class_to_idx = checkpoint["class_to_idx"]

        input_tensor = transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        if model_key in ("vit", "deit"):
            print(f"Running Gradient Attention Rollout (Algorithm 3) for {model_key}...", flush=True)
            cam, pred_idx, probs = compute_gradient_attention_rollout(model, input_tensor)
        else:
            print(f"Running Grad-CAM adaptation (Algorithm 4) for {model_key}...", flush=True)
            cam, pred_idx, probs = compute_swin_gradcam(model, input_tensor)

        print(f"Heatmap computed for {model_key}", flush=True)

        cams[model_key] = cam
        predictions[model_key] = {"pred_idx": pred_idx, "probs": probs}

        overlay = overlay_cam_on_image(image, cam)

        save_path = os.path.join(
            XAI_OUTPUT_DIR,
            f"{save_prefix}_{model_key}_explain.png"
        )
        Image.fromarray(overlay).save(save_path)
        print(f"Saved: {save_path}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    idx_to_class = {index: name for name, index in class_to_idx.items()}

    # Same F1-based weights as the classification ensemble (Equation 3.1)
    total_f1 = sum(val_f1_scores.values())
    if total_f1 == 0:
        weights = {k: 1 / 3 for k in CHECKPOINTS}
    else:
        weights = {k: v / total_f1 for k, v in val_f1_scores.items()}

    print("\nEnsemble weights (same as classification path):", flush=True)
    for key, weight in weights.items():
        print(f"{key}: {weight:.4f}", flush=True)

    # Consolidated, class-specific explanation (Section 3.3.2, final paragraph)
    consolidated_cam = sum(weights[k] * cams[k] for k in cams)
    consolidated_cam = normalize_cam(consolidated_cam)

    consolidated_overlay = overlay_cam_on_image(image, consolidated_cam)

    consolidated_path = os.path.join(XAI_OUTPUT_DIR, f"{save_prefix}_consolidated_explain.png")
    Image.fromarray(consolidated_overlay).save(consolidated_path)
    print(f"\nSaved consolidated explanation: {consolidated_path}", flush=True)

    print("\n================================", flush=True)
    print("Per-model predictions", flush=True)
    print("================================", flush=True)
    for model_key, result in predictions.items():
        pred_class = idx_to_class[result["pred_idx"]]
        prob = result["probs"][result["pred_idx"]]
        print(f"{model_key.upper()}: {pred_class} (prob={prob:.4f})", flush=True)

    print("\nDone.", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to ECG image"
    )

    args = parser.parse_args()
    print("Parsed args:", args, flush=True)

    try:
        explain_image(args.image)
    except Exception:
        print("\n!!! An error occurred. Full traceback below !!!\n", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()