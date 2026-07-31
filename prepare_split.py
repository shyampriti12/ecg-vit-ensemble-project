import os
import shutil
import random
from pathlib import Path

from config import RAW_DATA_DIR, SPLIT_DATA_DIR, CLASSES

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

IMAGE_EXTENSIONS = [
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP"
]


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def find_images_recursively(class_dir):
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(Path(class_dir).rglob(ext))

    return list(images)


def split_dataset():
    print("Starting dataset split...")
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("SPLIT_DATA_DIR:", SPLIT_DATA_DIR)

    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"RAW_DATA_DIR not found: {RAW_DATA_DIR}")

    print("\nFolders found inside data_raw:")
    for item in os.listdir(RAW_DATA_DIR):
        print("-", item)

    if os.path.exists(SPLIT_DATA_DIR):
        print("\nRemoving old data_split folder...")
        shutil.rmtree(SPLIT_DATA_DIR)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            create_dir(os.path.join(SPLIT_DATA_DIR, split, cls))

    total_images_all_classes = 0

    for cls in CLASSES:
        class_dir = os.path.join(RAW_DATA_DIR, cls)

        print("\nChecking class folder:", class_dir)

        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        images = find_images_recursively(class_dir)
        random.shuffle(images)

        total = len(images)
        total_images_all_classes += total

        if total == 0:
            print(f"WARNING: No images found for class: {cls}")
            continue

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        split_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split, files in split_map.items():
            for file in files:
                dest = os.path.join(SPLIT_DATA_DIR, split, cls, file.name)

                # Prevent overwrite if duplicate filenames exist
                if os.path.exists(dest):
                    dest = os.path.join(
                        SPLIT_DATA_DIR,
                        split,
                        cls,
                        f"{file.stem}_{random.randint(100000, 999999)}{file.suffix}"
                    )

                shutil.copy2(file, dest)

        print(
            f"{cls}: total={total}, "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

    print("\nTotal images found:", total_images_all_classes)

    if total_images_all_classes == 0:
        print("\nERROR: No images found.")
        print("Your class folders exist, but no image files were found inside them.")
        print("Check whether your images are .jpg, .png, .jpeg, .bmp, .tif, .tiff, or .webp.")
    else:
        print("\nDataset split completed successfully.")


if __name__ == "__main__":
    split_dataset()