import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import MURADataset, BODY_PART_TO_IDX
from src.config import MURA_TRAIN, MURA_VALID

print("Loading MURA training dataset...")
train_dataset = MURADataset(root_dir=MURA_TRAIN, mode="train")
print(f"✅ Training samples found: {len(train_dataset)}")

print("\nLoading MURA validation dataset...")
valid_dataset = MURADataset(root_dir=MURA_VALID, mode="valid")
print(f"✅ Validation samples found: {len(valid_dataset)}")

print("\nTesting a single sample...")
image, body_part, condition = train_dataset[0]
print(f"✅ Image shape: {image.shape}")
print(f"✅ Body part index: {body_part.item()}")
print(f"✅ Condition index: {condition.item()}")

print("\n🎉 Dataset loader is working correctly!")