# src/evaluate.py
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(model, loader, device, criterion: nn.Module = None) -> Dict[str, float]:
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    criterion = criterion or nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {"loss": loss_total / total, "acc": 100.0 * correct / total}

def collect_predictions(model, loader, device) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred

def confusion_matrix_plot(y_true: List[int], y_pred: List[int], classes: List[str], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Eval] Confusion matrix saved: {out_path.resolve()}")
    plt.close()
