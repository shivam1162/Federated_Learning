"""
Local Trainer implementing Cross-Entropy private training and FedMD KL-divergence distillation.
"""
from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.config import DEFAULT_TEMPERATURE, DEFAULT_LR


class LocalTrainer:
    """
    Manages local model training on private data and distillation against public consensus logits.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        lr: float = DEFAULT_LR
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.ce_criterion = nn.CrossEntropyLoss()

    def train_private_epoch(
        self,
        train_loader: DataLoader,
        batch_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> Tuple[float, float]:
        """
        Executes one epoch of supervised training on private Non-IID data.
        Returns: (average_loss, accuracy_percentage)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.ce_criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_callback and (batch_idx % 2 == 0 or batch_idx == num_batches - 1):
                cur_acc = (correct / max(total, 1)) * 100.0
                batch_callback(batch_idx + 1, num_batches, batch_loss, cur_acc)

        avg_loss = total_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100.0
        return avg_loss, acc

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates model on private validation data.
        Returns: (average_loss, accuracy_percentage)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.ce_criterion(outputs, targets)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100.0
        return avg_loss, acc

    def compute_public_logits(self, public_loader: DataLoader) -> List[List[float]]:
        """
        Computes forward pass on public reference set to generate unnormalized logits.
        Returns: List of float lists, shape [N_public, NUM_CLASSES]
        """
        self.model.eval()
        all_logits = []

        with torch.no_grad():
            for images, _ in public_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                all_logits.append(outputs.cpu().numpy())

        if len(all_logits) == 0:
            return []

        concatenated = np.vstack(all_logits)
        return concatenated.tolist()

    def distill_from_consensus(
        self,
        public_loader: DataLoader,
        consensus_logits: List[List[float]],
        temperature: float = DEFAULT_TEMPERATURE,
        epochs: int = 1,
        batch_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> float:
        """
        Fine-tunes model on public reference data to mimic ensemble consensus logits
        using Kullback-Leibler (KL) Divergence loss.
        """
        self.model.train()
        consensus_tensor = torch.tensor(consensus_logits, dtype=torch.float32)

        total_kd_loss = 0.0
        step_count = 0
        num_batches = len(public_loader)

        for _ in range(epochs):
            offset = 0
            for batch_idx, (images, _) in enumerate(public_loader):
                batch_size = images.size(0)
                images = images.to(self.device)
                target_logits = consensus_tensor[offset : offset + batch_size].to(self.device)
                offset += batch_size

                self.optimizer.zero_grad()
                student_logits = self.model(images)

                # Softened probability distributions
                log_prob_student = F.log_softmax(student_logits / temperature, dim=1)
                prob_teacher = F.softmax(target_logits / temperature, dim=1)

                # KL divergence loss scaled by temperature squared
                kd_loss = F.kl_div(log_prob_student, prob_teacher, reduction="batchmean") * (temperature ** 2)
                kd_loss.backward()
                self.optimizer.step()

                batch_kd = kd_loss.item()
                total_kd_loss += batch_kd
                step_count += 1

                if batch_callback and (batch_idx % 2 == 0 or batch_idx == num_batches - 1):
                    batch_callback(batch_idx + 1, num_batches, batch_kd)

        return total_kd_loss / max(step_count, 1)

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
