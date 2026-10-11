"""
EuroSAT dataset downloader, Non-IID partitioner, and distribution packager.
"""
import os
import shutil
import zipfile
import requests
import urllib.request
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image

from common.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    EUROSAT_DIR,
    PARTITIONS_DIR,
    PUBLIC_REF_DIR,
    GLOBAL_VAL_DIR,
    EUROSAT_CLASSES,
    IMAGE_SIZE,
    DEFAULT_PUBLIC_SAMPLES,
    DEFAULT_VAL_SAMPLES,
)

OFFICIAL_EUROSAT_RGB_URL = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip"


def ensure_directories():
    """Creates required data directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_REF_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_VAL_DIR.mkdir(parents=True, exist_ok=True)


def download_eurosat(progress_callback=None, force: bool = False) -> Path:
    """
    Downloads and extracts the official EuroSAT RGB dataset if not already present.
    """
    ensure_directories()
    zip_path = RAW_DATA_DIR / "EuroSAT.zip"

    # Check if extracted folder already exists with full dataset
    if not force:
        if EUROSAT_DIR.exists():
            total = sum(len(list((EUROSAT_DIR / c).glob("*.jpg"))) for c in EUROSAT_CLASSES if (EUROSAT_DIR / c).is_dir())
            if total >= 10000:
                return EUROSAT_DIR

        direct_subdirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir() and d.name in EUROSAT_CLASSES]
        if len(direct_subdirs) == len(EUROSAT_CLASSES):
            total = sum(len(list((RAW_DATA_DIR / c).glob("*.jpg"))) for c in EUROSAT_CLASSES)
            if total >= 10000:
                return RAW_DATA_DIR

    if not zip_path.exists():
        if progress_callback:
            progress_callback("Downloading EuroSAT dataset (~90MB)...")
        print(f"Downloading EuroSAT from {OFFICIAL_EUROSAT_RGB_URL}...")
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        with requests.get(OFFICIAL_EUROSAT_RGB_URL, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            progress_callback(f"Downloading EuroSAT: {percent}% ({downloaded // (1024*1024)} MB)")

    if progress_callback:
        progress_callback("Extracting EuroSAT.zip archive...")
    print("Extracting EuroSAT.zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    # Return actual root containing class folders
    if EUROSAT_DIR.exists():
        return EUROSAT_DIR
    return RAW_DATA_DIR


def generate_synthetic_eurosat_if_missing(samples_per_class: int = 50):
    """
    Generates a lightweight synthetic EuroSAT dataset for instant offline testing and development.
    Creates 64x64 colored images with synthetic patterns for each class.
    """
    ensure_directories()
    target_root = RAW_DATA_DIR / "2750"
    target_root.mkdir(parents=True, exist_ok=True)

    # Specific color palettes for each satellite class
    class_colors = {
        "AnnualCrop": (190, 160, 110),
        "Forest": (34, 139, 34),
        "HerbaceousVegetation": (124, 252, 0),
        "Highway": (128, 128, 128),
        "Industrial": (169, 169, 169),
        "Pasture": (154, 205, 50),
        "PermanentCrop": (107, 142, 35),
        "Residential": (205, 92, 92),
        "River": (65, 105, 225),
        "SeaLake": (30, 144, 255),
    }

    for cls_name in EUROSAT_CLASSES:
        cls_dir = target_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        base_color = class_colors.get(cls_name, (100, 100, 100))

        # Check existing count
        existing = list(cls_dir.glob("*.jpg"))
        if len(existing) >= samples_per_class:
            continue

        for i in range(samples_per_class):
            # Create a 64x64 noisy image representative of the satellite class
            noise = np.random.randint(-30, 30, (IMAGE_SIZE, IMAGE_SIZE, 3))
            img_arr = np.clip(np.array(base_color, dtype=np.int32) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            img.save(cls_dir / f"{cls_name}_{i+1}.jpg")


def get_eurosat_root() -> Path:
    """Finds and returns the root path containing the 10 EuroSAT class directories."""
    possible_roots = [
        EUROSAT_DIR,
        RAW_DATA_DIR,
        RAW_DATA_DIR / "EuroSAT",
        DATA_DIR / "EuroSAT",
    ]
    for root in possible_roots:
        if root.exists() and all((root / cls_name).is_dir() for cls_name in EUROSAT_CLASSES):
            return root
    return None


def partition_dataset_non_iid(
    client_ids: List[str],
    public_samples: int = DEFAULT_PUBLIC_SAMPLES,
    val_samples: int = DEFAULT_VAL_SAMPLES,
    samples_per_client: int = 600,
    use_full_dataset: bool = False,
    progress_callback=None,
) -> Dict[str, Dict[str, int]]:
    """
    Performs Non-IID partitioning of EuroSAT across clients, sets aside public distillation reference,
    and stages downloadable zip archives.
    
    Returns partition statistics: {client_id: {class_name: count}}
    """
    ensure_directories()
    root_path = get_eurosat_root()

    if root_path is None:
        # Fall back to synthetic data if EuroSAT hasn't been downloaded yet
        generate_synthetic_eurosat_if_missing(samples_per_class=60)
        root_path = get_eurosat_root()

    if progress_callback:
        progress_callback(0.05, "Purging previous partition archives...")

    # 0. Force wipe any previous partitions and packages completely
    if PARTITIONS_DIR.exists():
        for item in list(PARTITIONS_DIR.glob("*")):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except Exception:
                pass
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)

    if PUBLIC_REF_DIR.exists():
        shutil.rmtree(PUBLIC_REF_DIR, ignore_errors=True)
    PUBLIC_REF_DIR.mkdir(parents=True, exist_ok=True)

    if GLOBAL_VAL_DIR.exists():
        shutil.rmtree(GLOBAL_VAL_DIR, ignore_errors=True)
    GLOBAL_VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all image files by class
    class_files: Dict[str, List[Path]] = {}
    for cls_name in EUROSAT_CLASSES:
        cls_dir = root_path / cls_name
        files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
        np.random.seed(42)
        np.random.shuffle(files)
        class_files[cls_name] = files

    # 1. Set aside Public Reference Dataset (Balanced across all 10 classes)
    if progress_callback:
        progress_callback(0.12, "Extracting public reference distillation set...")
    per_class_public = max(1, public_samples // len(EUROSAT_CLASSES))
    for cls_name in EUROSAT_CLASSES:
        pub_files = class_files[cls_name][:per_class_public]
        class_files[cls_name] = class_files[cls_name][per_class_public:]
        for f in pub_files:
            shutil.copy2(f, PUBLIC_REF_DIR / f.name)

    # 2. Set aside Global Validation Dataset (Balanced across all 10 classes)
    if progress_callback:
        progress_callback(0.20, "Extracting global validation evaluation set...")
    per_class_val = max(1, val_samples // len(EUROSAT_CLASSES))
    for cls_name in EUROSAT_CLASSES:
        val_cls_dir = GLOBAL_VAL_DIR / cls_name
        val_cls_dir.mkdir(parents=True, exist_ok=True)
        v_files = class_files[cls_name][:per_class_val]
        class_files[cls_name] = class_files[cls_name][per_class_val:]
        for f in v_files:
            shutil.copy2(f, val_cls_dir / f.name)

    # Calculate samples per client if full dataset requested
    total_remaining = sum(len(files) for files in class_files.values())
    if use_full_dataset and len(client_ids) > 0:
        samples_per_client = total_remaining // len(client_ids)

    # Non-IID Domain Specialization Groups:
    class_groups = [
        ["AnnualCrop", "Pasture", "PermanentCrop"],
        ["Forest", "HerbaceousVegetation"],
        ["Highway", "Industrial", "Residential"],
        ["River", "SeaLake"],
    ]

    stats: Dict[str, Dict[str, int]] = {}
    n_clients = len(client_ids)

    for i, c_id in enumerate(client_ids):
        if progress_callback:
            pct_alloc = 0.25 + (i / max(n_clients, 1)) * 0.35
            progress_callback(pct_alloc, f"Allocating Non-IID data for {c_id} ({i+1}/{n_clients})...")

        stats[c_id] = {cls: 0 for cls in EUROSAT_CLASSES}
        client_dir = PARTITIONS_DIR / c_id
        client_data_dir = client_dir / "private_data"
        client_data_dir.mkdir(parents=True, exist_ok=True)

        primary_classes = class_groups[i % len(class_groups)]
        secondary_classes = [c for c in EUROSAT_CLASSES if c not in primary_classes]

        # Allocate 80% of client capacity to primary classes, 20% to others
        num_primary = int(samples_per_client * 0.80)
        num_secondary = samples_per_client - num_primary

        per_primary = max(1, num_primary // len(primary_classes))
        per_secondary = max(1, num_secondary // max(len(secondary_classes), 1))

        # Copy primary class images
        for cls in primary_classes:
            target_cls_dir = client_data_dir / cls
            target_cls_dir.mkdir(parents=True, exist_ok=True)
            avail = class_files[cls]
            take_count = min(per_primary, len(avail))
            selected = avail[:take_count]
            class_files[cls] = avail[take_count:]

            for f in selected:
                shutil.copy2(f, target_cls_dir / f.name)
            stats[c_id][cls] += take_count

        # Copy secondary class images
        for cls in secondary_classes:
            target_cls_dir = client_data_dir / cls
            target_cls_dir.mkdir(parents=True, exist_ok=True)
            avail = class_files[cls]
            take_count = min(per_secondary, len(avail))
            selected = avail[:take_count]
            class_files[cls] = avail[take_count:]

            for f in selected:
                shutil.copy2(f, target_cls_dir / f.name)
            stats[c_id][cls] += take_count

        # Also copy public reference data into client's staging bundle
        client_pub_dir = client_dir / "public_reference"
        if client_pub_dir.exists():
            shutil.rmtree(client_pub_dir)
        shutil.copytree(PUBLIC_REF_DIR, client_pub_dir)

        # 4. Create downloadable ZIP archive for client
        if progress_callback:
            pct_zip = 0.60 + (i / max(n_clients, 1)) * 0.38
            progress_callback(pct_zip, f"Compressing {c_id}.zip package ({i+1}/{n_clients})...")

        zip_output_path = PARTITIONS_DIR / f"{c_id}.zip"
        with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zip_f:
            for root, _, files in os.walk(client_dir):
                for file in files:
                    full_p = Path(root) / file
                    arcname = full_p.relative_to(client_dir)
                    zip_f.write(full_p, arcname)

    if progress_callback:
        progress_callback(1.0, f"Partitions ready for {n_clients} clients.")

    return stats

