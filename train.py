print("train.py started", flush=True)

import os
import argparse
import copy
import torch
import timm
import numpy as np

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from config import (
    TRAIN_DIR,
    VAL_DIR,
    CHECKPOINT_DIR,
    MODEL_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
    PATIENCE
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Light augmentation only, because ECG shape is important
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            shear=3
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def build_model(model_key, device):
    model_name = MODEL_NAMES[model_key]

    print("Loading model:", model_name, flush=True)

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    return model


def calculate_class_weights(dataset):
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)

    print("Class counts:", class_counts, flush=True)

    if np.any(class_counts == 0):
        raise ValueError("One or more classes have zero images.")

    weights = len(targets) / (len(class_counts) * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32)

    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, acc, precision, recall, f1, cm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vit", "deit", "swin"],
        help="Choose model: vit, deit, or swin"
    )

    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = get_device()

    print("Using device:", device, flush=True)
    print("TRAIN_DIR:", TRAIN_DIR, flush=True)
    print("VAL_DIR:", VAL_DIR, flush=True)

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    print("Training images:", len(train_dataset), flush=True)
    print("Validation images:", len(val_dataset), flush=True)
    print("Class mapping:", train_dataset.class_to_idx, flush=True)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check data_split/train.")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Check data_split/val.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = build_model(args.model, device)

    class_weights = calculate_class_weights(train_dataset).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    best_f1 = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]", flush=True)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, cm = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}", flush=True)
        print(f"Train Accuracy: {train_acc:.4f}", flush=True)
        print(f"Train F1: {train_f1:.4f}", flush=True)

        print(f"Val Loss: {val_loss:.4f}", flush=True)
        print(f"Val Accuracy: {val_acc:.4f}", flush=True)
        print(f"Val Precision: {val_precision:.4f}", flush=True)
        print(f"Val Recall: {val_recall:.4f}", flush=True)
        print(f"Val F1: {val_f1:.4f}", flush=True)

        print("Confusion Matrix:", flush=True)
        print(cm, flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0

            save_path = os.path.join(
                CHECKPOINT_DIR,
                f"{args.model}_best.pth"
            )

            torch.save({
                "model_key": args.model,
                "model_name": MODEL_NAMES[args.model],
                "model_state_dict": best_model_weights,
                "best_val_f1": best_f1,
                "class_to_idx": train_dataset.class_to_idx
            }, save_path)

            print("Best model saved:", save_path, flush=True)

        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{PATIENCE}",
                flush=True
            )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.", flush=True)
            break

    print(f"\nTraining completed for {args.model}", flush=True)
    print(f"Best validation F1-score: {best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()