import os
import sys
import io
import traceback

# config.py lives one directory above /web, so add PROJECT_ROOT to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import timm
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, render_template

from config import CHECKPOINT_DIR, IMAGE_SIZE, NUM_CLASSES

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

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    models = {}
    val_f1_scores = {}
    idx_to_class = None

    for model_key, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading {model_key} checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = timm.create_model(
            checkpoint["model_name"],
            pretrained=False,
            num_classes=NUM_CLASSES,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        models[model_key] = model
        val_f1_scores[model_key] = checkpoint["best_val_f1"]

        if idx_to_class is None:
            idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

        print(f"{model_key} loaded | val F1 = {val_f1_scores[model_key]:.4f}")

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

        image_tensor = transform(image).unsqueeze(0).to(device)

        final_probs = torch.zeros((1, NUM_CLASSES)).to(device)
        individual_predictions = {}

        with torch.no_grad():
            for model_key, model in _state["models"].items():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                final_probs += weights[model_key] * probs

                pred_idx = torch.argmax(probs, dim=1).item()
                individual_predictions[model_key] = {
                    "label": MODEL_LABELS[model_key],
                    "predicted_class": idx_to_class[pred_idx],
                    "confidence": round(float(probs[0][pred_idx]), 4),
                }

        final_probs_cpu = final_probs.cpu().numpy()[0]
        final_pred_idx = int(final_probs_cpu.argmax())

        response = {
            "predicted_class": idx_to_class[final_pred_idx],
            "confidence": round(float(final_probs_cpu[final_pred_idx]), 4),
            "class_probabilities": {
                idx_to_class[i]: round(float(p), 4)
                for i, p in enumerate(final_probs_cpu)
            },
            "ensemble_weights": {
                MODEL_LABELS[k]: round(v, 4) for k, v in weights.items()
            },
            "individual_models": individual_predictions,
        }

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    load_all_models()
    # use_reloader=False avoids loading all three checkpoints twice on startup
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)