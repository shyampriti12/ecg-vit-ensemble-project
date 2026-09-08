import os
import sys
import io
import base64
import traceback

# config.py and xai_vit_deit.py live one directory above /web, so add
# PROJECT_ROOT to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template

from config import CHECKPOINT_DIR, NUM_CLASSES
from xai_vit_deit import (
    get_transform,
    load_model,
    compute_gradient_attention_rollout,
    compute_swin_gradcam,
    overlay_cam_on_image,
    normalize_cam,
)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

CHECKPOINTS = {
    "vit": os.path.join(CHECKPOINT_DIR, "vit_best.pth"),
    "deit": os.path.join(CHECKPOINT_DIR, "deit_best.pth"),
    "swin": os.path.join(CHECKPOINT_DIR, "swin_best.pth"),
}

MODEL_LABELS = {
    "vit": "ViT-B/16",
    "deit": "DeiT-Base",
    "swin": "Swin-Tiny",
}

MODEL_METHODS = {
    "vit": "Gradient Attention Rollout",
    "deit": "Gradient Attention Rollout",
    "swin": "Grad-CAM",
}

# Models are loaded once at startup and kept warm in memory here.
_state = {
    "device": None,
    "models": {},
    "weights": {},
    "idx_to_class": None,
    "transform": None,
    "loaded": False,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_all_models():
    device = get_device()
    print("Using device:", device)

    transform = get_transform()

    models = {}
    val_f1_scores = {}
    idx_to_class = None

    for model_key, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading {model_key} checkpoint...")
        model, checkpoint = load_model(checkpoint_path, device)

        models[model_key] = model
        val_f1_scores[model_key] = checkpoint["best_val_f1"]

        if idx_to_class is None:
            idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

        print(f"{model_key} loaded | val F1 = {val_f1_scores[model_key]:.4f}")

    # Equation 3.1: normalized, validation-F1-weighted ensemble weights
    total_f1 = sum(val_f1_scores.values())
    if total_f1 == 0:
        print("Warning: all validation F1 scores are zero. Using equal weights.")
        weights = {k: 1 / len(models) for k in models}
    else:
        weights = {k: v / total_f1 for k, v in val_f1_scores.items()}

    print("Ensemble weights:", weights)

    _state.update({
        "device": device,
        "models": models,
        "weights": weights,
        "idx_to_class": idx_to_class,
        "transform": transform,
        "loaded": True,
    })

    print("All models loaded. Ready to serve predictions.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_to_base64(np_image_uint8):
    buffer = io.BytesIO()
    Image.fromarray(np_image_uint8).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------------
# Explanation generation (Algorithms 3 & 4, Section 3.3.2)
# --------------------------------------------------------------------------

def generate_model_explanation(model_key, model, image_tensor, image_pil):
    """
    Runs Gradient Attention Rollout (ViT/DeiT) or the Grad-CAM adaptation
    (Swin) for one model and returns its prediction, cam, and heatmap
    overlay. `image_tensor` must have requires_grad_(True) set, since both
    methods backpropagate to get their cam.
    """
    if model_key == "swin":
        cam, pred_idx, probs = compute_swin_gradcam(model, image_tensor)
    else:
        cam, pred_idx, probs = compute_gradient_attention_rollout(model, image_tensor)

    overlay = overlay_cam_on_image(image_pil, cam, alpha=0.5)

    return {
        "predicted_class_idx": pred_idx,
        "probs": probs,
        "cam": cam,
        "confidence": round(float(probs[pred_idx]), 4),
        "heatmap": encode_image_to_base64(overlay),
    }


def generate_consolidated_explanation(model_cams, weights, image_pil):
    """
    Combines each model's cam into a single weighted, class-consistent
    heatmap (Section 3.3.2, final paragraph), using the same validation-F1
    weights as the classification ensemble.
    """
    consolidated_cam = sum(weights[key] * cam for key, cam in model_cams.items())
    consolidated_cam = normalize_cam(consolidated_cam)

    overlay = overlay_cam_on_image(image_pil, consolidated_cam, alpha=0.5)

    return encode_image_to_base64(overlay)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": _state["loaded"]}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not _state["loaded"]:
        return jsonify({"error": "Models are still loading. Try again in a moment."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        device = _state["device"]
        transform = _state["transform"]
        idx_to_class = _state["idx_to_class"]
        weights = _state["weights"]

        model_explanations = {}
        final_probs = np.zeros(NUM_CLASSES, dtype=np.float32)

        for model_key, model in _state["models"].items():
            # Fresh tensor per model: both XAI methods backpropagate through
            # it, so each model needs its own graph.
            image_tensor = transform(image).unsqueeze(0).to(device)
            image_tensor.requires_grad_(True)

            explanation = generate_model_explanation(model_key, model, image_tensor, image)

            model_explanations[model_key] = explanation
            final_probs += weights[model_key] * explanation["probs"]

        final_pred_idx = int(final_probs.argmax())

        individual_predictions = {
            model_key: {
                "label": MODEL_LABELS[model_key],
                "method": MODEL_METHODS[model_key],
                "predicted_class": idx_to_class[explanation["predicted_class_idx"]],
                "confidence": explanation["confidence"],
                "heatmap": explanation["heatmap"],
            }
            for model_key, explanation in model_explanations.items()
        }

        consolidated_heatmap = generate_consolidated_explanation(
            {key: explanation["cam"] for key, explanation in model_explanations.items()},
            weights,
            image
        )

        response = {
            "predicted_class": idx_to_class[final_pred_idx],
            "confidence": round(float(final_probs[final_pred_idx]), 4),
            "class_probabilities": {
                idx_to_class[i]: round(float(p), 4)
                for i, p in enumerate(final_probs)
            },
            "ensemble_weights": {
                MODEL_LABELS[k]: round(v, 4) for k, v in weights.items()
            },
            "individual_models": individual_predictions,
            "consolidated_heatmap": consolidated_heatmap,
        }

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    load_all_models()
    # use_reloader=False avoids loading all three checkpoints twice on startup
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)