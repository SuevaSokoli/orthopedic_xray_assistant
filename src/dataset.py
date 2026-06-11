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
        self._load_from_csv()

    def _load_from_csv(self):
        if self.mode == "train":
            csv_filename = "train_image_paths.csv"
        else:
            csv_filename = "valid_image_paths.csv"

        mura_root = os.path.dirname(self.root_dir)
        csv_path = os.path.join(mura_root, csv_filename)

        if not os.path.exists(csv_path):
            print(f"CSV not found at {csv_path}")
            return

        print(f"Loading from CSV: {csv_path}")
        with open(csv_path, "r") as f:
            lines = f.read().strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.replace("\\", "/").split("/")

            if len(parts) < 4:
                continue

            img_file = parts[-1]
            study = parts[-2]
            patient = parts[-3]
            body_part = parts[-4]

            if not body_part.startswith("XR_"):
                continue
            if body_part not in BODY_PART_TO_IDX:
                continue
            if not img_file.lower().endswith(".png"):
                continue
            if img_file.startswith("."):
                continue

            if "positive" in study.lower():
                condition_idx = 1
            elif "negative" in study.lower():
                condition_idx = 0
            else:
                continue

            img_path = os.path.join(
                self.root_dir,
                body_part,
                patient,
                study,
                img_file
            )

            self.samples.append((
                img_path,
                BODY_PART_TO_IDX[body_part],
                condition_idx
            ))

        print(f"Loaded {len(self.samples)} samples from CSV")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, body_part_idx, condition_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            # If image can't be opened, return a blank image
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        return (
            image,
            torch.tensor(body_part_idx, dtype=torch.long),
            torch.tensor(condition_idx, dtype=torch.long)
        )