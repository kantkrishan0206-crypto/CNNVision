# src/train.py
#
# Trains CNN on data/processed using manifests and stats.
# Saves best-performing checkpoint to models/cnn_model.pth and logs metrics to models/metrics.csv.
#
# Usage:
#   python -m src.train --proc-root data/processed --models-root models --epochs 20 --batch-size 64 --lr 1e-3
#   python -m src.train --resume models/checkpoints/epoch_10.pth   # resume training
#
# Notes:
# - Expects data/processed/manifests/{train,val}.json and data/processed/stats.json.
# - Applies normalization from stats.json.
# - Supports mixed precision training (--amp).
# - Early stopping and checkpointing included.

import os
import csv
import json
import math
import random
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.model import create_model

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Dataset
# ---------------------------
class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, transform):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.transform = transform
        # Class name to index map (stable ordering)
        self.class_names = sorted({it["class_name"] for it in self.items})
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # Use label provided; ensure consistency if label mapping is needed
        label = int(it["label"])
        return img, label

def build_transforms(proc_root: Path, image_size: Tuple[int, int]) -> Tuple[transforms.Compose, transforms.Compose]:
    stats_path = proc_root / "stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = stats["normalization"]["mean"]
    std = stats["normalization"]["std"]

    train_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_tf, val_tf

# ---------------------------
# Metrics
# ---------------------------
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()

# ---------------------------
# Training loop
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs.detach(), labels.detach())
        n_batches += 1

    return {
        "loss": running_loss / max(n_batches, 1),
        "acc": running_acc / max(n_batches, 1)
    }

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        n_batches += 1

    return {
        "loss": running_loss / max(n_batches, 1),
        "acc": running_acc / max(n_batches, 1)
    }

# ---------------------------
# Checkpointing
# ---------------------------
def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_acc: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }, path)

def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt.get("best_val_acc", 0.0)

# ---------------------------
# Orchestration
# ---------------------------
def run_training(
    proc_root: str,
    models_root: str,
    num_classes: int,
    image_size: Tuple[int, int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    amp: bool,
    resume: str = None,
    save_every: int = 0,
    early_stopping_patience: int = 10
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    proc_root = Path(proc_root)
    models_root = Path(models_root)
    manifests_root = proc_root / "manifests"

    train_manifest = manifests_root / "train.json"
    val_manifest = manifests_root / "val.json"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("Processed manifests not found. Run preprocessing to create data/processed/manifests.")

    train_tf, val_tf = build_transforms(proc_root, image_size)
    train_ds = ManifestImageDataset(train_manifest, transform=train_tf)
    val_ds = ManifestImageDataset(val_manifest, transform=val_tf)

    num_workers = min(os.cpu_count() or 4, 8)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = create_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None

    # Resume support
    start_epoch = 1
    best_val_acc = 0.0
    checkpoints_dir = models_root / "checkpoints"
    final_model_path = models_root / "cnn_model.pth"
    metrics_path = models_root / "metrics.csv"
    models_root.mkdir(parents=True, exist_ok=True)

    if resume is not None:
        resume_path = Path(resume)
        if resume_path.exists():
            e, best_val_acc = load_checkpoint(resume_path, model, optimizer)
            start_epoch = e + 1
            print(f"[INFO] Resumed from {resume_path}, starting epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
        else:
            print(f"[WARN] Resume path {resume_path} not found. Starting fresh.")

    # Metrics CSV init
    init_metrics_csv(metrics_path)

    # Early stopping
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}] "
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

        append_metrics_csv(metrics_path, epoch, train_metrics, val_metrics)

        # Save best model
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), final_model_path)
            print(f"[INFO] New best val_acc={best_val_acc:.4f}. Saved {final_model_path}")

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Optional epoch checkpointing
        if save_every and (epoch % save_every == 0):
            save_checkpoint(checkpoints_dir / f"epoch_{epoch}.pth", model, optimizer, epoch, best_val_acc)

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"[INFO] Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break

    print(f"[INFO] Training completed. Best val_acc={best_val_acc:.4f}. Model saved to {final_model_path}")

def init_metrics_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

def append_metrics_csv(path: Path, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_metrics['loss']:.6f}",
            f"{train_metrics['acc']:.6f}",
            f"{val_metrics['loss']:.6f}",
            f"{val_metrics['acc']:.6f}"
        ])

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train CNN and save best checkpoint to models/cnn_model.pth.")
    p.add_argument("--proc-root", type=str, default="data/processed", help="Processed dataset root.")
    p.add_argument("--models-root", type=str, default="models", help="Directory to save models and metrics.")
    p.add_argument("--num-classes", type=int, default=10, help="Number of classes.")
    p.add_argument("--image-size", nargs=2, type=int, default=[64, 64], metavar=("W", "H"), help="Resize for transforms.")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--save-every", type=int, default=0, help="Save epoch checkpoints every N epochs (0 disables).")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    return p.parse_args()

def main():
    args = parse_args()
    run_training(
        proc_root=args.proc_root,
        models_root=args.models_root,
        num_classes=args.num_classes,
        image_size=(args.image_size[0], args.image_size[1]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        amp=bool(args.amp),
        resume=args.resume,
        save_every=int(args.save_every),
        early_stopping_patience=int(args.patience)
    )

if __name__ == "__main__":
    main()
