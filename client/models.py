"""
Heterogeneous Neural Network Architectures adapted for EuroSAT (64x64 RGB, 10 classes).
Optimized for CPU training and inference.
"""
from typing import Dict, Any
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    mobilenet_v3_small,
    shufflenet_v2_x0_5,
)


class CustomConvNet(nn.Module):
    """
    Lightweight 4-stage convolutional baseline for resource-constrained edge CPUs.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for 64x64 EuroSAT satellite images.
    Replaces initial 7x7 conv with 3x3 conv to avoid premature spatial downsampling.
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Preserve spatial dimension on 64x64 images
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenet_v3(num_classes: int = 10) -> nn.Module:
    """
    MobileNetV3-Small adapted for 64x64 EuroSAT satellite images.
    Utilizes depthwise separable convolutions and squeeze-and-excitation blocks.
    """
    model = mobilenet_v3_small(weights=None)
    # Adjust classifier head for num_classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_shufflenet_v2(num_classes: int = 10) -> nn.Module:
    """
    ShuffleNetV2 (0.5x) adapted for 64x64 EuroSAT satellite images.
    Utilizes channel split and channel shuffle operations with minimal CPU FLOPs.
    """
    model = shufflenet_v2_x0_5(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# Model Registry
MODEL_REGISTRY = {
    "resnet18": build_resnet18,
    "mobilenetv3": build_mobilenet_v3,
    "shufflenetv2": build_shufflenet_v2,
    "custom": CustomConvNet,
}


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Factory function to retrieve model instance by name.
    """
    key = model_name.lower().replace("-", "").replace("_", "")
    for reg_key, builder in MODEL_REGISTRY.items():
        if reg_key in key or key in reg_key:
            return builder(num_classes=num_classes)
    raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Computes parameter metrics and estimated memory footprint.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 32-bit floats
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(model_size_mb, 2),
    }
