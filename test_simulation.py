"""
End-to-End Local Pipeline Verification Test for FedMD.
Tests data loaders, forward pass, logit computation, and distillation logic.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import DEFAULT_TEMPERATURE
from client.models import get_model
from client.dataset import get_client_dataloaders, get_public_dataloader
from client.trainer import LocalTrainer


def test_pipeline():
    print("1. Testing Dataset Loaders...")
    train_loader, val_loader = get_client_dataloaders("data/partitions/client_1/private_data", batch_size=16)
    public_loader = get_public_dataloader("data/partitions/client_1/public_reference", batch_size=16)
    print(f"   Train samples: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Public: {len(public_loader.dataset)}")

    print("\n2. Initializing Heterogeneous Model (MobileNetV3)...")
    model = get_model("mobilenetv3")
    trainer = LocalTrainer(model)

    print("\n3. Testing Local Supervised Epoch (Cross-Entropy)...")
    loss, acc = trainer.train_private_epoch(train_loader)
    print(f"   Epoch 1 -> Loss: {loss:.4f}, Acc: {acc:.2f}%")

    print("\n4. Testing Public Logit Generation...")
    logits = trainer.compute_public_logits(public_loader)
    print(f"   Computed {len(logits)} public logit vectors. Sample shape: {len(logits[0])}")

    print("\n5. Testing FedMD Distillation with Mock Consensus Logits...")
    mock_consensus = logits  # Use self-logits as consensus for test
    kd_loss = trainer.distill_from_consensus(public_loader, mock_consensus, temperature=DEFAULT_TEMPERATURE, epochs=1)
    print(f"   Distillation Complete! KD Loss: {kd_loss:.4f}")

    print("\n=== ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    test_pipeline()
