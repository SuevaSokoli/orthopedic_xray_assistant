import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import BODY_PARTS, IMAGE_SIZE

# ── Label mappings ─────────────────────────────────────────
BODY_PART_TO_IDX = {part: idx for idx, part in enumerate(BODY_PARTS)}
CONDITION_TO_IDX = {"normal": 0, "abnormal": 1}
IDX_TO_BODY_PART = {idx: part for part, idx in BODY_PART_TO_IDX.items()}
IDX_TO_CONDITION = {0: "normal", 1: "abnormal"}

# ── Image transforms ───────────────────────────────────────
def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

# ── Dataset class ──────────────────────────────────────────
class MURADataset(Dataset):
    """
    Fast MURA dataset loader using the image paths CSV files.
    Much faster than scanning folders — reads a single file
    instead of making thousands of network requests.
    """
    def __init__(self, root_dir, mode="train"):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = get_transforms(mode)
        self.samples = []
        self._load_from_csv()

    def _load_from_csv(self):
        """
        Loads image paths from the MURA CSV file.
        Each line looks like:
            MURA-v1.1/train/XR_ELBOW/patient00001/study1_positive/image1.png
        """
        # Determine which CSV file to use
        if self.mode == "train":
            csv_filename = "train_image_paths.csv"
        else:
            csv_filename = "valid_image_paths.csv"

        # The CSV is one level up from train/valid folders
        mura_root = os.path.dirname(self.root_dir)
        csv_path = os.path.join(mura_root, csv_filename)

        if not os.path.exists(csv_path):
            print(f"CSV not found at {csv_path}, falling back to folder scan...")
            self._load_from_folders()
            return

        print(f"Loading from CSV: {csv_path}")
        with open(csv_path, "r") as f:
            lines = f.read().strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract body part from path
            # Path format: MURA-v1.1/train/XR_ELBOW/patient.../study.../image.png
            parts = line.replace("\\", "/").split("/")

            body_part = None
            is_abnormal = None

            for part in parts:
                if part.startswith("XR_"):
                    body_part = part
                if "positive" in part.lower():
                    is_abnormal = True
                elif "negative" in part.lower():
                    is_abnormal = False

            if body_part is None or is_abnormal is None:
                continue
            if body_part not in BODY_PART_TO_IDX:
                continue

            # Build full image path
            # The CSV paths start with MURA-v1.1/train/...
            # We need to map to our actual root_dir
            img_filename = parts[-1]
            study_part = parts[-2]
            patient_part = parts[-3]
            body_part_part = parts[-4]

            img_path = os.path.join(
                self.root_dir,
                body_part_part,
                patient_part,
                study_part,
                img_filename
            )

            self.samples.append((
                img_path,
                BODY_PART_TO_IDX[body_part],
                1 if is_abnormal else 0
            ))

        print(f"Loaded {len(self.samples)} samples from CSV")

    def _load_from_folders(self):
        """Fallback: scan folders (slow on Google Drive)"""
        for body_part in BODY_PARTS:
            body_part_dir = os.path.join(self.root_dir, body_part)
            if not os.path.exists(body_part_dir):
                continue
            body_part_idx = BODY_PART_TO_IDX[body_part]
            for patient in os.listdir(body_part_dir):
                patient_dir = os.path.join(body_part_dir, patient)
                if not os.path.isdir(patient_dir):
                    continue
                for study in os.listdir(patient_dir):
                    study_dir = os.path.join(patient_dir, study)
                    if not os.path.isdir(study_dir):
                        continue
                    if "positive" in study.lower():
                        condition_idx = 1
                    elif "negative" in study.lower():
                        condition_idx = 0
                    else:
                        continue
                    for img_file in os.listdir(study_dir):
                        if img_file.lower().endswith(".png"):
                            img_path = os.path.join(study_dir, img_file)
                            self.samples.append((img_path, body_part_idx, condition_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, body_part_idx, condition_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return (
            image,
            torch.tensor(body_part_idx, dtype=torch.long),
            torch.tensor(condition_idx, dtype=torch.long)
        )