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

# ── Image transforms ───────────────────────────────────────
def get_transforms(mode="train"):
    """
    Returns image transformations for training or validation.
    Training includes augmentation (flips, rotation) to prevent overfitting.
    Validation only resizes and normalizes.
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
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
    Custom PyTorch Dataset for the MURA X-ray dataset.
    Each sample returns:
        - image tensor (3 x 224 x 224)
        - body part label (integer 0-6)
        - condition label (0=normal, 1=abnormal)
    """
    def __init__(self, root_dir, mode="train"):
        """
        Args:
            root_dir: path to MURA train or valid folder
            mode: "train" or "valid"
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = get_transforms(mode)
        self.samples = []  # list of (image_path, body_part_idx, condition_idx)
        self._load_samples()

    def _load_samples(self):
        """
        Walks through the MURA folder structure and collects all image paths.
        MURA structure:
            root/
              XR_ELBOW/
                patient00001/
                  study1_positive/   ← abnormal
                    image1.png
                  study1_negative/   ← normal
                    image1.png
        """
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
                    study_dir = os.path.join(patient_dir, patient, study)
                    if not os.path.isdir(study_dir):
                        # Try direct path
                        study_dir = os.path.join(patient_dir, study)
                    if not os.path.isdir(study_dir):
                        continue

                    # Determine condition from folder name
                    if "positive" in study.lower():
                        condition_idx = CONDITION_TO_IDX["abnormal"]
                    elif "negative" in study.lower():
                        condition_idx = CONDITION_TO_IDX["normal"]
                    else:
                        continue

                    # Collect all PNG images in this study folder
                    for img_file in os.listdir(study_dir):
                        if img_file.lower().endswith(".png"):
                            img_path = os.path.join(study_dir, img_file)
                            self.samples.append((
                                img_path,
                                body_part_idx,
                                condition_idx
                            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, body_part_idx, condition_idx = self.samples[idx]

        # Load image and convert to RGB (MURA images are grayscale)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return (
            image,
            torch.tensor(body_part_idx, dtype=torch.long),
            torch.tensor(condition_idx, dtype=torch.long)
        )