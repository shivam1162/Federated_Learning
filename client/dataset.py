"""
Dataset loaders and image preprocessors for EuroSAT (64x64 RGB).
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from common.config import EUROSAT_CLASSES, IMAGE_SIZE, DEFAULT_BATCH_SIZE


# Standard ImageNet / EuroSAT normalization constants
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Data augmentation for local private training
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# Deterministic transform for validation and public reference distillation
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


class EuroSATDataset(Dataset):
    """
    Loads labeled EuroSAT images organized in subfolders by class name.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or EVAL_TRANSFORM
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(EUROSAT_CLASSES)}

        if self.root_dir.exists():
            for cls_name, cls_idx in self.class_to_idx.items():
                cls_dir = self.root_dir / cls_name
                if cls_dir.is_dir():
                    for img_file in cls_dir.glob("*.jpg"):
                        self.samples.append((img_file, cls_idx))
                    for img_file in cls_dir.glob("*.png"):
                        self.samples.append((img_file, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


class PublicReferenceDataset(Dataset):
    """
    Shared unlabelled reference dataset for logit distillation consensus.
    Maintains deterministic image order across all client devices.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or EVAL_TRANSFORM
        self.image_paths: List[Path] = []

        if self.root_dir.exists():
            # Support both flat directory and class-subfolder directory structures
            for ext in ("*.jpg", "*.png", "*/*.jpg", "*/*.png"):
                self.image_paths.extend(sorted(self.root_dir.glob(ext)))
            # Ensure deterministic sort across all laptops
            self.image_paths = sorted(list(set(self.image_paths)), key=lambda p: str(p.name))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(img_path.name)


def get_client_dataloaders(
    partition_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits client private partition into local train and validation sets.
    """
    full_ds = EuroSATDataset(partition_dir, transform=TRAIN_TRANSFORM)
    total_len = len(full_ds)
    if total_len == 0:
        raise ValueError(f"No EuroSAT images found in partition directory: {partition_dir}")

    val_len = max(1, int(total_len * val_split))
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len], generator=generator)

    # Use eval transform for validation
    val_ds.dataset.transform = EVAL_TRANSFORM

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_public_dataloader(
    public_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> DataLoader:
    """
    DataLoader for public reference dataset (deterministic, no shuffle).
    """
    public_ds = PublicReferenceDataset(public_dir, transform=EVAL_TRANSFORM)
    if len(public_ds) == 0:
        raise ValueError(f"No images found in public reference directory: {public_dir}")

    return DataLoader(public_ds, batch_size=batch_size, shuffle=False)
