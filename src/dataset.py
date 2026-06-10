import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import BODY_PARTS, IMAGE_SIZE

BODY_PART_TO_IDX = {part: idx for idx, part in enumerate(BODY_PARTS)}
CONDITION_TO_IDX = {"normal": 0, "abnormal": 1}
IDX_TO_BODY_PART = {idx: part for part, idx in BODY_PART_TO_IDX.items()}
IDX_TO_CONDITION = {0: "normal", 1: "abnormal"}

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

class MURADataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = get_transforms(mode)
        self.samples = []
        self._load_from_folders()

    def _load_from_folders(self):
        print(f"Scanning folders in: {self.root_dir}")
        for body_part in BODY_PARTS:
            body_part_dir = os.path.join(self.root_dir, body_part)
            if not os.path.exists(body_part_dir):
                continue

            body_part_idx = BODY_PART_TO_IDX[body_part]

            for patient in os.listdir(body_part_dir):
                if patient.startswith("."):
                    continue
                patient_dir = os.path.join(body_part_dir, patient)
                if not os.path.isdir(patient_dir):
                    continue

                for study in os.listdir(patient_dir):
                    if study.startswith("."):
                        continue
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
                        if img_file.lower().endswith(".png") and not img_file.startswith("."):
                            img_path = os.path.join(study_dir, img_file)
                            self.samples.append((
                                img_path,
                                body_part_idx,
                                condition_idx
                            ))

        print(f"Found {len(self.samples)} samples")

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