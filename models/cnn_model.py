# src/model.py
#
# CNN backbone with BatchNorm, Dropout, and a configurable classifier head.
# Designed for small images (e.g., CIFAR-10 processed to 64x64).
# Extendable for larger inputs by adjusting strides/pooling.

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),        # 64x64 -> 32x32
            ConvBlock(32, 64),                 # 32x32 -> 16x16
            ConvBlock(64, 128),                # 16x16 -> 8x8
            ConvBlock(128, 256, pool=False),   # 8x8 -> 8x8
            nn.MaxPool2d(2)                    # 8x8 -> 4x4
        )
        # Compute flattened size: 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_model(num_classes: int = 10) -> nn.Module:
    return CNNClassifier(num_classes=num_classes)
