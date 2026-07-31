import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data_raw")
SPLIT_DATA_DIR = os.path.join(PROJECT_ROOT, "data_split")

TRAIN_DIR = os.path.join(SPLIT_DATA_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DATA_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DATA_DIR, "test")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Classes
CLASSES = [
    "Normal",
    "Myocardial Infarction",
    "Abnormal Heartbeat",
    "History of MI"
]

NUM_CLASSES = len(CLASSES)

# Training settings
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

# Model names
MODEL_NAMES = {
    "vit": "vit_base_patch16_224",
    "deit": "deit_base_patch16_224",
    "swin": "swin_tiny_patch4_window7_224"
}
