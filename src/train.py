import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MURA_TRAIN, MURA_VALID, CHECKPOINTS_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, SEED, NUM_WORKERS
)
from src.dataset import MURADataset
from src.model import build_model, count_parameters

# ── Reproducibility ────────────────────────────────────────
torch.manual_seed(SEED)

# ── Device setup ───────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU found, using CPU")
    return device

# ── Training one epoch ─────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_body = 0
    correct_cond = 0
    total = 0

    for images, body_parts, conditions in tqdm(loader, desc="Training"):
        images = images.to(device)
        body_parts = body_parts.to(device)
        conditions = conditions.to(device)

        optimizer.zero_grad()

        # Forward pass
        body_logits, cond_logits = model(images)

        # Compute losses for both heads
        loss_body = criterion(body_logits, body_parts)
        loss_cond = criterion(cond_logits, conditions)
        loss = loss_body + loss_cond  # combined loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        correct_body += (body_logits.argmax(1) == body_parts).sum().item()
        correct_cond += (cond_logits.argmax(1) == conditions).sum().item()
        total += images.size(0)

    avg_loss = total_loss / len(loader)
    acc_body = correct_body / total * 100
    acc_cond = correct_cond / total * 100
    return avg_loss, acc_body, acc_cond

# ── Validation one epoch ───────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_body = 0
    correct_cond = 0
    total = 0

    with torch.no_grad():
        for images, body_parts, conditions in tqdm(loader, desc="Validating"):
            images = images.to(device)
            body_parts = body_parts.to(device)
            conditions = conditions.to(device)

            body_logits, cond_logits = model(images)

            loss_body = criterion(body_logits, body_parts)
            loss_cond = criterion(cond_logits, conditions)
            loss = loss_body + loss_cond

            total_loss += loss.item()
            correct_body += (body_logits.argmax(1) == body_parts).sum().item()
            correct_cond += (cond_logits.argmax(1) == conditions).sum().item()
            total += images.size(0)

    avg_loss = total_loss / len(loader)
    acc_body = correct_body / total * 100
    acc_cond = correct_cond / total * 100
    return avg_loss, acc_body, acc_cond

# ── Main training loop ─────────────────────────────────────
def main():
    device = get_device()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MURADataset(root_dir=MURA_TRAIN, mode="train")
    valid_dataset = MURADataset(root_dir=MURA_VALID, mode="valid")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Build model
    print("\nBuilding model...")
    model = build_model(pretrained=True)
    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    # Training history
    history = {
        "train_loss": [], "train_acc_body": [], "train_acc_cond": [],
        "val_loss": [], "val_acc_body": [], "val_acc_cond": []
    }

    best_val_loss = float("inf")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("\n🚀 Starting training...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 50)

        # Train
        train_loss, train_acc_body, train_acc_cond = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc_body, val_acc_cond = validate(
            model, valid_loader, criterion, device
        )

        # Step scheduler
        scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc_body"].append(train_acc_body)
        history["train_acc_cond"].append(train_acc_cond)
        history["val_loss"].append(val_loss)
        history["val_acc_body"].append(val_acc_body)
        history["val_acc_cond"].append(val_acc_cond)

        # Print results
        print(f"Train Loss: {train_loss:.4f} | Body Acc: {train_acc_body:.2f}% | Cond Acc: {train_acc_cond:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Body Acc: {val_acc_body:.2f}% | Cond Acc: {val_acc_cond:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc_body": val_acc_body,
                "val_acc_cond": val_acc_cond,
            }, checkpoint_path)
            print(f"✅ Best model saved! (val_loss: {val_loss:.4f})")

        print()

    # Save training history
    history_path = os.path.join(CHECKPOINTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to {history_path}")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()