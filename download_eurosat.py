"""
Script to download and verify the real EuroSAT RGB dataset from official repository.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from master.partitioner import download_eurosat, get_eurosat_root
from common.config import EUROSAT_CLASSES


def main():
    print("=" * 60)
    print("      EuroSAT Dataset Downloader & Validator")
    print("=" * 60)

    existing_root = get_eurosat_root()
    if existing_root:
        total_imgs = sum(len(list((existing_root / cls).glob("*.jpg"))) for cls in EUROSAT_CLASSES)
        if total_imgs >= 10000:
            print(f"Official full EuroSAT dataset already present at: {existing_root} ({total_imgs:,} total images)")
            return
        else:
            print(f"Currently found {total_imgs} sample images. Upgrading to full official EuroSAT dataset (~27,000 images, ~90MB)...")

    print("Downloading official EuroSAT RGB dataset (~90MB)...")
    root = download_eurosat(progress_callback=print, force=True)
    print(f"\nDownload & extraction completed to: {root}")


if __name__ == "__main__":
    main()
