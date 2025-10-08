# src/train.py
from pathlib import Path
from typing import Dict, Any
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .config import LOGS_DIR, MODELS_DIR
from .evaluate import evaluate_model

def _to_device(batch, device):
    inputs, labels = batch
    return inputs.to(device), labels.to(device)

def train_one_epoch(model, loader, device, criterion, optimizer) -> Dict[str, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {"loss": running_loss / total, "acc": 100.0 * correct / total}

def save_checkpoint(model, optimizer, epoch: int, best_acc: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
    }, out_path)
    print(f"[Checkpoint] Saved: {out_path.resolve()}")

def train_model(model, loaders, device, cfg: Dict[str, Any]):
    epochs = int(cfg.get("epochs", 10))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_acc = 0.0
    best_path = MODELS_DIR / "best_cnn.pth"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], device, criterion, optimizer)
        val_metrics = evaluate_model(model, loaders["val"], device, criterion)
        scheduler.step(val_metrics["acc"])
        print(f"[Scheduler] LR adjusted to {optimizer.param_groups[0]['lr']:.6f}")

        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['acc']:.2f}% || "
              f"Val Loss {val_metrics['loss']:.4f} | Val Acc {val_metrics['acc']:.2f}% | LR {optimizer.param_groups[0]['lr']:.5f}")

        # Checkpoint if improved
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(model, optimizer, epoch, best_acc, best_path)

        # Optional early stopping: stop if val_acc plateaus for many epochs
        patience = 5
        if len(history["val_acc"]) > patience:
            recent = history["val_acc"][-patience:]
            if max(recent) < best_acc - 0.01:  # no improvement margin
                print("[EarlyStop] Validation accuracy plateaued. Stopping.")
                break

    # Save final history
    hist_path = LOGS_DIR / "history_last.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Logs] Training history saved: {hist_path.resolve()}")

    return model, history