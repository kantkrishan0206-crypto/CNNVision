# src/predict.py
from pathlib import Path
from typing import Dict, List
import torch
from PIL import Image
from torchvision import transforms
from .model import build_model
from .config import MODELS_DIR
from .dataset import CIFAR_MEAN, CIFAR_STD

def _predict_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def predict_image(model, image_path: Path, device, classes: List[str]) -> Dict:
    tf = _predict_transform()
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    pred_idx = int(torch.argmax(torch.tensor(probs)).item())
    return {"pred_class": classes[pred_idx], "pred_idx": pred_idx, "probs": probs}
