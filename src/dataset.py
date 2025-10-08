# src/dataset.py
from pathlib import Path
from typing import Tuple, Dict, List
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from .config import RAW_DIR, PROCESSED_DIR, LOGS_DIR

# CIFAR-10 normalization values (ImageNet-like, commonly used for CIFAR too)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

def download_cifar10(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _ = datasets.CIFAR10(root=str(root), train=True, download=True)
    _ = datasets.CIFAR10(root=str(root), train=False, download=True)
    print(f"[Dataset] CIFAR-10 downloaded to: {root.resolve()}")

def _build_transforms() -> Dict[str, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return {"train": train_tf, "eval": eval_tf}

def get_datasets(raw_dir: Path, processed_dir: Path, val_size: int = 5000, seed: int = 42):
    # Ensure dataset exists
    try:
        transforms_dict = _build_transforms()
        full_train = datasets.CIFAR10(root=str(raw_dir), train=True, download=True, transform=transforms_dict["train"])
        test_ds = datasets.CIFAR10(root=str(raw_dir), train=False, download=True, transform=transforms_dict["eval"])
    except Exception:
        download_cifar10(raw_dir)
        transforms_dict = _build_transforms()
        full_train = datasets.CIFAR10(root=str(raw_dir), train=True, download=False, transform=transforms_dict["train"])
        test_ds = datasets.CIFAR10(root=str(raw_dir), train=False, download=False, transform=transforms_dict["eval"])

    # Split train/val
    generator = torch.Generator().manual_seed(seed)
    train_size = max(1, len(full_train) - val_size)
    val_size = min(val_size, len(full_train) - 1)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    classes = full_train.classes
    return train_ds, val_ds, test_ds, classes

def get_dataloaders(
    raw_dir: Path,
    processed_dir: Path,
    batch_size: int,
    num_workers: int,
    val_size: int,
    seed: int,
    pin_memory: bool,
):
    train_ds, val_ds, test_ds, classes = get_datasets(raw_dir, processed_dir, val_size, seed)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }
    return loaders, classes

def save_stats(train_ds, val_ds, test_ds, classes: List[str], out_path: Path) -> None:
    stats = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "num_classes": len(classes),
        "classes": classes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Logs] Dataset stats saved: {out_path.resolve()}")
