"""
Common configuration, constants, and hyperparameters for FedMD.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EUROSAT_DIR = RAW_DATA_DIR / "2750"  # EuroSAT RGB folder is typically named '2750'
PARTITIONS_DIR = DATA_DIR / "partitions"
PUBLIC_REF_DIR = DATA_DIR / "public_reference"
GLOBAL_VAL_DIR = DATA_DIR / "val"

# EuroSAT Classes (10 classes)
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(EUROSAT_CLASSES)
IMAGE_SIZE = 64  # EuroSAT native resolution is 64x64

# Default Network Settings
DEFAULT_MASTER_PORT = 8000
DEFAULT_STREAMLIT_PORT = 8501

# FedMD Distillation & Training Hyperparameters
DEFAULT_ROUNDS = 5
DEFAULT_LOCAL_EPOCHS = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001
DEFAULT_TEMPERATURE = 2.0  # Temperature tau for softening logits in KL divergence
DEFAULT_ALPHA = 0.5        # Loss balance: (1 - alpha) * CE + alpha * KD

# Number of samples for public reference distillation set
DEFAULT_PUBLIC_SAMPLES = 400
# Number of samples for global validation set
DEFAULT_VAL_SAMPLES = 500
