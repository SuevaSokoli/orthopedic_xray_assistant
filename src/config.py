import os

# ── Dataset Paths ──────────────────────────────────────────
MURA_ROOT = r"C:\thesis_data\MURA"
MURA_TRAIN = os.path.join(MURA_ROOT, "train")
MURA_VALID = os.path.join(MURA_ROOT, "valid")

KNEE_OA_ROOT = r"C:\thesis_data\knee_oa"      # to be downloaded later
FRACATLAS_ROOT = r"C:\thesis_data\fracatlas"   # to be downloaded later

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\evaso\OneDrive\Documents\thesis\code\orthopedic_xray_assistant"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRADCAM_DIR = os.path.join(PROJECT_ROOT, "outputs", "gradcam")

# ── Body Parts (MURA classes) ──────────────────────────────
BODY_PARTS = [
    "XR_ELBOW",
    "XR_FINGER",
    "XR_FOREARM",
    "XR_HAND",
    "XR_HUMERUS",
    "XR_SHOULDER",
    "XR_WRIST"
]

# ── Conditions ─────────────────────────────────────────────
CONDITIONS = ["normal", "abnormal"]

# ── Model Parameters ───────────────────────────────────────
IMAGE_SIZE = 224          # ResNet50 expects 224x224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_WORKERS = 0           # 0 is safest on Windows

# ── Random Seed (for reproducibility) ─────────────────────
SEED = 42