


import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from config import (
    TEST_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
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


def get_test_loader():
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return test_dataset, test_loader


def load_model(checkpoint_path, device):
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


def get_model_probabilities(model, loader, device):
    all_probs = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    return all_probs, all_labels


def save_confusion_matrix(y_true, y_pred, class_names, title):
    os.makedirs(
        os.path.join(OUTPUT_DIR, "confusion_matrices"),
        exist_ok=True
    )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrices",
        f"{title.replace(' ', '_')}.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Confusion matrix saved:", save_path)


def save_roc_curve(y_true, probabilities, class_names, title):
    os.makedirs(
        os.path.join(OUTPUT_DIR, "roc_curves"),
        exist_ok=True
    )

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average: interpolate all per-class curves onto a common
    # FPR grid, then average the TPR values (standard sklearn approach)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i],
            color=color, lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})"
        )

    plt.plot(
        fpr["macro"], tpr["macro"],
        color="black", lw=2, linestyle="--",
        label=f"Macro-average (AUC = {roc_auc['macro']:.3f})"
    )

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "roc_curves",
        f"{title.replace(' ', '_')}.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("ROC curve saved:", save_path)

    return roc_auc["macro"]


def evaluate_model(probabilities, labels, class_names, title):
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n====================================")
    print(title)
    print("====================================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            zero_division=0
        )
    )

    try:
        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro"
        )
        print(f"Macro AUC-ROC: {auc:.4f}")
    except Exception as e:
        print("AUC-ROC could not be calculated:", e)

    save_confusion_matrix(
        labels,
        predictions,
        class_names,
        title
    )

    save_roc_curve(
        labels,
        probabilities,
        class_names,
        title
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    device = get_device()
    print("Using device:", device)

    test_dataset, test_loader = get_test_loader()

    print("Test images:", len(test_dataset))
    print("Class mapping:", test_dataset.class_to_idx)

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check data_split/test folder.")

    class_names = test_dataset.classes

    models = {}
    val_f1_scores = {}

    for model_key, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        print(f"\nLoading {model_key} checkpoint...")
        model, checkpoint = load_model(checkpoint_path, device)

        models[model_key] = model
        val_f1_scores[model_key] = checkpoint["best_val_f1"]

        print(
            f"{model_key} best validation F1: "
            f"{val_f1_scores[model_key]:.4f}"
        )

    total_f1 = sum(val_f1_scores.values())

    if total_f1 == 0:
        print("Warning: all validation F1 scores are zero.")
        print("Using equal weights.")
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

    all_model_probs = {}
    labels_reference = None

    for model_key, model in models.items():
        print(f"\nGetting probabilities for {model_key}...")

        probs, labels = get_model_probabilities(
            model,
            test_loader,
            device
        )

        all_model_probs[model_key] = probs

        if labels_reference is None:
            labels_reference = labels

    ensemble_probs = (
        weights["vit"] * all_model_probs["vit"] +
        weights["deit"] * all_model_probs["deit"] +
        weights["swin"] * all_model_probs["swin"]
    )

    results = {}

    results["ViT"] = evaluate_model(
        all_model_probs["vit"],
        labels_reference,
        class_names,
        "ViT Test Results"
    )

    results["DeiT"] = evaluate_model(
        all_model_probs["deit"],
        labels_reference,
        class_names,
        "DeiT Test Results"
    )

    results["Swin"] = evaluate_model(
        all_model_probs["swin"],
        labels_reference,
        class_names,
        "Swin Test Results"
    )

    results["Weighted Ensemble"] = evaluate_model(
        ensemble_probs,
        labels_reference,
        class_names,
        "Weighted Ensemble Test Results"
    )

    print("\nFinal Summary")
    print("=============")
    print("Model\t\tAccuracy\tPrecision\tRecall\t\tF1")

    for model_name, metric in results.items():
        print(
            f"{model_name:18s}"
            f"{metric['accuracy']:.4f}\t\t"
            f"{metric['precision']:.4f}\t\t"
            f"{metric['recall']:.4f}\t\t"
            f"{metric['f1']:.4f}"
        )


if __name__ == "__main__":
    main()