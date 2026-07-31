import os
import argparse
import torch
import timm
from PIL import Image
from torchvision import transforms

from config import (
    CHECKPOINT_DIR,
    IMAGE_SIZE,
    NUM_CLASSES
)


CHECKPOINTS = {
    "vit": os.path.join(CHECKPOINT_DIR, "vit_best.pth"),
    "deit": os.path.join(CHECKPOINT_DIR, "deit_best.pth"),
    "swin": os.path.join(CHECKPOINT_DIR, "swin_best.pth")
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def predict_image(image_path):
    device = get_device()
    print("Using device:", device)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    models = {}
    val_f1_scores = {}
    class_to_idx = None

    print("\nLoading trained models...")

    for model_key, checkpoint_path in CHECKPOINTS.items():
        model, checkpoint = load_model(checkpoint_path, device)

        models[model_key] = model
        val_f1_scores[model_key] = checkpoint["best_val_f1"]

        if class_to_idx is None:
            class_to_idx = checkpoint["class_to_idx"]

        print(f"{model_key} loaded | validation F1 = {val_f1_scores[model_key]:.4f}")

    idx_to_class = {
        index: class_name
        for class_name, index in class_to_idx.items()
    }

    total_f1 = sum(val_f1_scores.values())

    if total_f1 == 0:
        print("\nWarning: All validation F1 scores are zero. Using equal weights.")
        weights = {
            "vit": 1 / 3,
            "deit": 1 / 3,
            "swin": 1 / 3
        }
    else:
        weights = {
            key: value / total_f1
            for key, value in val_f1_scores.items()
        }

    print("\nEnsemble weights:")
    for key, weight in weights.items():
        print(f"{key}: {weight:.4f}")

    final_probs = torch.zeros((1, NUM_CLASSES)).to(device)

    individual_predictions = {}

    with torch.no_grad():
        for model_key, model in models.items():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)

            final_probs += weights[model_key] * probs

            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = idx_to_class[pred_idx]

            individual_predictions[model_key] = {
                "class": pred_class,
                "probabilities": probs.cpu().numpy()[0]
            }

    final_pred_idx = torch.argmax(final_probs, dim=1).item()
    final_pred_class = idx_to_class[final_pred_idx]

    print("\n================================")
    print("Individual Model Predictions")
    print("================================")

    for model_key, result in individual_predictions.items():
        print(f"{model_key.upper()} prediction: {result['class']}")

    print("\n================================")
    print("Final Ensemble Prediction")
    print("================================")
    print("Image:", image_path)
    print("Predicted class:", final_pred_class)

    print("\nClass probabilities:")

    final_probs_cpu = final_probs.cpu().numpy()[0]

    for i in range(NUM_CLASSES):
        class_name = idx_to_class[i]
        probability = final_probs_cpu[i]
        print(f"{class_name}: {probability:.4f}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to ECG image"
    )

    args = parser.parse_args()

    predict_image(args.image)


if __name__ == "__main__":
    main()