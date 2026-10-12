"""
Client Worker Daemon for Heterogeneous Federated Model Distillation.
Executes local private training, generates public reference logits, and distills knowledge from Master consensus.
"""
import os
import sys
import time
import zipfile
import argparse
import platform
import shutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DEFAULT_MASTER_PORT, EUROSAT_CLASSES
from client.models import get_model, get_model_summary
from client.dataset import get_client_dataloaders, get_public_dataloader
from client.trainer import LocalTrainer


class FedMDClient:
    def __init__(
        self,
        master_ip: str,
        client_id: str,
        model_name: str,
        device_name: str,
        master_port: int = DEFAULT_MASTER_PORT,
    ):
        self.master_url = f"http://{master_ip}:{master_port}"
        self.client_id = client_id
        self.model_name = model_name
        self.device_name = device_name
        self.base_dir = Path("data") / client_id
        self.private_data_dir = self.base_dir / "private_data"
        self.public_ref_dir = self.base_dir / "public_reference"
        self.weights_dir = Path("weights") / client_id
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Detect CPU specifications
        self.cpu_info = f"{platform.processor()} ({os.cpu_count()} cores)"

        # Initialize local heterogeneous model
        print(f"[{self.client_id}] Initializing architecture '{self.model_name}' on CPU...")
        self.device = torch.device("cpu")
        self.model = get_model(self.model_name, num_classes=len(EUROSAT_CLASSES))
        self.summary = get_model_summary(self.model)
        print(f"[{self.client_id}] Model parameters: {self.summary['total_params']:,} ({self.summary['size_mb']} MB)")

        self.trainer = LocalTrainer(self.model, device=self.device)
        self.last_completed_round = 0
        self.train_loader = None
        self.val_loader = None
        self.public_loader = None
        self.current_distribution_id = 0
        self.current_training_session_id = 0
        self.has_logged_finished = False

        # Thread-safe telemetry state for background heartbeat daemon
        self._telemetry_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.telemetry: Dict[str, Any] = {
            "sub_stage": "Connected",
            "current_epoch": 0,
            "total_epochs": 0,
            "batch_progress": 0.0,
            "step_loss": None,
            "step_acc": None,
        }

    def set_telemetry(
        self,
        sub_stage: str,
        current_epoch: int = 0,
        total_epochs: int = 0,
        batch_progress: float = 0.0,
        step_loss: Optional[float] = None,
        step_acc: Optional[float] = None,
    ):
        """Thread-safe update of local client telemetry."""
        with self._telemetry_lock:
            self.telemetry["sub_stage"] = sub_stage
            self.telemetry["current_epoch"] = current_epoch
            self.telemetry["total_epochs"] = total_epochs
            self.telemetry["batch_progress"] = round(batch_progress, 3)
            if step_loss is not None:
                self.telemetry["step_loss"] = round(float(step_loss), 4)
            if step_acc is not None:
                self.telemetry["step_acc"] = round(float(step_acc), 2)

    def start_heartbeat_daemon(self):
        """Spawns dedicated background thread for unblocked heartbeats and live telemetry."""
        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"heartbeat-{self.client_id}"
            )
            self._heartbeat_thread.start()
            print(f"[{self.client_id}] Dedicated background telemetry & heartbeat daemon started.")

    def _heartbeat_loop(self):
        """
        Runs continuously in background, sending HTTP heartbeats + live telemetry
        every 1.5 seconds without being blocked by PyTorch training on CPU.
        """
        while not self._stop_event.is_set():
            try:
                with self._telemetry_lock:
                    payload = dict(self.telemetry)
                requests.post(
                    f"{self.master_url}/api/heartbeat/{self.client_id}",
                    json=payload,
                    timeout=3
                )
            except Exception:
                pass
            time.sleep(1.5)

    def reset_model_and_weights(self, session_id: int):
        """
        Purges older saved checkpoints from disk and reinitializes the neural network
        model to fresh initial weights for a new training session.
        """
        print(f"\n[{self.client_id}] ==================================================")
        print(f"[{self.client_id}] NEW TRAINING SESSION (ID: {session_id}) DETECTED")
        print(f"[{self.client_id}] PURGING OLDER WEIGHTS & RE-INITIALIZING MODEL")
        print(f"[{self.client_id}] ==================================================")

        # 1. Purge checkpoint directory
        purged_count = 0
        if self.weights_dir.exists():
            for pt_file in self.weights_dir.glob("*.pt"):
                try:
                    pt_file.unlink(missing_ok=True)
                    purged_count += 1
                except Exception as e:
                    print(f"[{self.client_id}] Warning deleting checkpoint {pt_file.name}: {e}")
        print(f"[{self.client_id}] Purged {purged_count} previous checkpoint file(s) from {self.weights_dir}")

        # 2. Re-initialize model architecture to clean baseline state
        self.model = get_model(self.model_name, num_classes=len(EUROSAT_CLASSES))
        self.trainer = LocalTrainer(self.model, device=self.device)
        self.last_completed_round = 0
        self.has_logged_finished = False
        print(f"[{self.client_id}] Model architecture '{self.model_name}' re-initialized to initial baseline state.")

    def register(self) -> bool:
        """Sends registration handshake to Master coordinator."""
        payload = {
            "client_id": self.client_id,
            "device_name": self.device_name,
            "model_name": self.model_name,
            "cpu_info": self.cpu_info,
        }
        try:
            res = requests.post(f"{self.master_url}/api/register", json=payload, timeout=5)
            if res.status_code == 200:
                print(f"[{self.client_id}] Successfully registered with Master at {self.master_url}")
                return True
            print(f"[{self.client_id}] Registration failed: {res.text}")
        except Exception as e:
            print(f"[{self.client_id}] Cannot connect to Master at {self.master_url}: {e}")
        return False

    def handle_distribution(self, distribution_id: int) -> bool:
        """
        Executes explicit dataset distribution lifecycle:
        1. Reports 'downloading' status to Master.
        2. Force-deletes any existing local dataset directory.
        3. Downloads assigned partition ZIP archive from Master.
        4. Extracts archive into data/{client_id}/.
        5. Re-initializes PyTorch data loaders.
        6. Reports 'ready' status with verified sample counts back to Master.
        """
        print(f"\n[{self.client_id}] ==================================================")
        print(f"[{self.client_id}] DISTRIBUTION SIGNAL RECEIVED (Batch ID: {distribution_id})")
        print(f"[{self.client_id}] ==================================================")

        self.set_telemetry(sub_stage="Purging Local Dataset & Downloading Package", batch_progress=0.1)

        # 1. Report downloading status
        try:
            requests.post(
                f"{self.master_url}/api/report_distribution_status",
                json={
                    "client_id": self.client_id,
                    "distribution_id": distribution_id,
                    "status": "downloading",
                    "message": "Purging old dataset and downloading new package...",
                },
                timeout=5,
            )
        except Exception as e:
            print(f"[{self.client_id}] Could not report downloading status: {e}")

        # 2. Force delete existing local dataset directory
        if self.base_dir.exists():
            print(f"[{self.client_id}] Purging existing local dataset at {self.base_dir}...")
            try:
                shutil.rmtree(self.base_dir, ignore_errors=True)
            except Exception as e:
                print(f"[{self.client_id}] Warning while purging old data: {e}")

        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 3. Download fresh partition archive from Master
        print(f"[{self.client_id}] Downloading partition package from Master: {self.master_url}/api/download_partition/{self.client_id}")
        zip_path = Path(f"{self.client_id}_temp.zip")
        try:
            self.set_telemetry(sub_stage="Downloading Partition Package via LAN", batch_progress=0.3)
            res = requests.get(f"{self.master_url}/api/download_partition/{self.client_id}", stream=True, timeout=90)
            if res.status_code != 200:
                err_msg = f"Download failed ({res.status_code}): {res.text}"
                print(f"[{self.client_id}] Error: {err_msg}")
                self._report_distribution_error(distribution_id, err_msg)
                return False

            with open(zip_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)

            # 4. Extract archive
            self.set_telemetry(sub_stage="Extracting ZIP Archive", batch_progress=0.7)
            print(f"[{self.client_id}] Extracting partition archive into {self.base_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.base_dir)

            if zip_path.exists():
                zip_path.unlink()

            # 5. Build and verify data loaders
            self.set_telemetry(sub_stage="Initializing PyTorch DataLoaders", batch_progress=0.9)
            print(f"[{self.client_id}] Initializing PyTorch DataLoaders...")
            self.train_loader, self.val_loader = get_client_dataloaders(str(self.private_data_dir), batch_size=32)
            self.public_loader = get_public_dataloader(str(self.public_ref_dir), batch_size=32)

            n_train = len(self.train_loader.dataset)
            n_val = len(self.val_loader.dataset)
            n_pub = len(self.public_loader.dataset)

            print(f"[{self.client_id}] Dataset verified: {n_train} train, {n_val} val, {n_pub} public reference images.")
            self.current_distribution_id = distribution_id

            # 6. Report ready to Master
            requests.post(
                f"{self.master_url}/api/report_distribution_status",
                json={
                    "client_id": self.client_id,
                    "distribution_id": distribution_id,
                    "status": "ready",
                    "message": "Dataset successfully installed and verified.",
                    "train_samples": n_train,
                    "val_samples": n_val,
                    "public_samples": n_pub,
                },
                timeout=5,
            )
            self.set_telemetry(sub_stage="Dataset Partition Ready", batch_progress=1.0)
            print(f"[{self.client_id}] Distribution confirmed [READY] to Master.")
            return True

        except Exception as e:
            err_msg = f"Distribution failed: {e}"
            print(f"[{self.client_id}] {err_msg}")
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            self.set_telemetry(sub_stage=f"Error: {err_msg}")
            self._report_distribution_error(distribution_id, err_msg)
            return False

    def _report_distribution_error(self, distribution_id: int, message: str):
        try:
            requests.post(
                f"{self.master_url}/api/report_distribution_status",
                json={
                    "client_id": self.client_id,
                    "distribution_id": distribution_id,
                    "status": "error",
                    "message": message,
                },
                timeout=5,
            )
        except Exception:
            pass

    def run_training_cycle(self):
        """
        Main worker daemon loop: listens for Master round states, executes local CE training,
        uploads reference logits, and distills from ensemble consensus.
        """
        print(f"[{self.client_id}] Worker daemon active. Waiting for dataset distribution and training rounds...")
        self.set_telemetry(sub_stage="Connected & Idle")

        while True:
            time.sleep(1.0)

            try:
                res = requests.get(f"{self.master_url}/api/round_state", timeout=3)
                if res.status_code != 200:
                    continue
                state = res.json()
            except Exception:
                continue

            # Check for Dataset Distribution Trigger from Master
            dist_id = state.get("distribution_id", 0)
            if dist_id > self.current_distribution_id:
                self.handle_distribution(dist_id)
                continue

            # Check for Training Trigger
            if not state.get("training_active"):
                continue

            # Check for New Training Session Trigger (Purges older weights & resets model)
            session_id = state.get("training_session_id", 0)
            if session_id > self.current_training_session_id:
                self.reset_model_and_weights(session_id)
                self.current_training_session_id = session_id

            # Ensure data loaders are ready before training
            if self.train_loader is None or self.public_loader is None:
                if self.private_data_dir.exists() and self.public_ref_dir.exists():
                    try:
                        print(f"[{self.client_id}] Initializing PyTorch DataLoaders from local directory...")
                        self.train_loader, self.val_loader = get_client_dataloaders(str(self.private_data_dir), batch_size=32)
                        self.public_loader = get_public_dataloader(str(self.public_ref_dir), batch_size=32)
                    except Exception as e:
                        print(f"[{self.client_id}] Error initializing DataLoaders: {e}")
                        continue
                else:
                    self.set_telemetry(sub_stage="Awaiting Dataset Distribution")
                    print(f"[{self.client_id}] Training requested but dataset not distributed yet. Awaiting distribution...")
                    time.sleep(1.0)
                    continue

            current_round = state.get("current_round", 1)
            stage = state.get("stage", "idle")
            local_epochs = state.get("local_epochs", 2)
            temperature = state.get("temperature", 2.0)

            # Stage 1: Private local training + Public logit submission
            if current_round > self.last_completed_round and stage == "waiting_for_logits":
                print(f"\n[{self.client_id}] === STARTING ROUND {current_round}/{state.get('total_rounds')} ===")

                # 1. Train on private Non-IID data with live batch callback
                print(f"[{self.client_id}] Training {local_epochs} local epochs on private Non-IID data...")
                for ep in range(local_epochs):
                    def on_batch(b_idx, total_b, b_loss, b_acc):
                        pct = b_idx / max(total_b, 1)
                        self.set_telemetry(
                            sub_stage=f"Round {current_round} | Epoch {ep+1}/{local_epochs} ({int(pct*100)}%)",
                            current_epoch=ep+1,
                            total_epochs=local_epochs,
                            batch_progress=pct,
                            step_loss=b_loss,
                            step_acc=b_acc
                        )

                    loss, acc = self.trainer.train_private_epoch(self.train_loader, batch_callback=on_batch)
                    print(f"[{self.client_id}] [Round {current_round} | Epoch {ep+1}/{local_epochs}] Private Loss: {loss:.4f} | Acc: {acc:.2f}%")

                # 2. Evaluate on private val
                self.set_telemetry(sub_stage=f"Round {current_round} | Evaluating Private Validation")
                val_loss, val_acc = self.trainer.evaluate(self.val_loader)
                print(f"[{self.client_id}] Validation Acc: {val_acc:.2f}% | Loss: {val_loss:.4f}")

                # 3. Generate public logits on shared reference set
                self.set_telemetry(sub_stage=f"Round {current_round} | Computing Public Logits")
                print(f"[{self.client_id}] Generating logits on {len(self.public_loader.dataset)} public reference images...")
                logits = self.trainer.compute_public_logits(self.public_loader)

                # 4. Upload logits to Master
                self.set_telemetry(sub_stage=f"Round {current_round} | Uploading Public Logits")
                print(f"[{self.client_id}] Uploading public logits to Master...")
                submit_payload = {
                    "client_id": self.client_id,
                    "round_num": current_round,
                    "logits": logits,
                    "local_loss": float(val_loss),
                    "local_acc": float(val_acc),
                }
                res_sub = requests.post(f"{self.master_url}/api/submit_logits", json=submit_payload, timeout=10)
                self.set_telemetry(sub_stage=f"Round {current_round} | Awaiting Ensemble Consensus")
                print(f"[{self.client_id}] Logits submitted successfully. Waiting for ensemble consensus...")

            # Stage 2: Distill from ensemble consensus
            elif current_round > self.last_completed_round and stage == "consensus_ready":
                self.set_telemetry(sub_stage=f"Round {current_round} | Fetching Consensus Logits")
                print(f"[{self.client_id}] Fetching consensus logits for Round {current_round} from Master...")
                res_cons = requests.get(f"{self.master_url}/api/get_consensus/{current_round}", timeout=5)
                if res_cons.status_code == 200 and res_cons.json().get("ready"):
                    consensus_logits = res_cons.json()["consensus_logits"]
                    print(f"[{self.client_id}] Applying FedMD Knowledge Distillation (KL Divergence, tau={temperature})...")

                    def on_kd_batch(b_idx, total_b, kd_l):
                        pct = b_idx / max(total_b, 1)
                        self.set_telemetry(
                            sub_stage=f"Round {current_round} | FedMD Distillation ({int(pct*100)}%)",
                            batch_progress=pct,
                            step_loss=kd_l
                        )

                    kd_loss = self.trainer.distill_from_consensus(
                        public_loader=self.public_loader,
                        consensus_logits=consensus_logits,
                        temperature=temperature,
                        epochs=1,
                        batch_callback=on_kd_batch
                    )
                    print(f"[{self.client_id}] Knowledge Distillation complete! KD Loss: {kd_loss:.4f}")

                    # Save weights
                    ckpt_path = self.weights_dir / f"model_round_{current_round}.pt"
                    self.trainer.save_checkpoint(str(ckpt_path))
                    latest_ckpt = self.weights_dir / "latest.pt"
                    self.trainer.save_checkpoint(str(latest_ckpt))

                    # Upload checkpoint to Master for centralized evaluation in Tab 4
                    try:
                        with open(latest_ckpt, "rb") as f_up:
                            requests.post(
                                f"{self.master_url}/api/upload_weights/{self.client_id}",
                                files={"file": (f"{self.client_id}_latest.pt", f_up, "application/octet-stream")},
                                timeout=20
                            )
                    except Exception as e:
                        print(f"[{self.client_id}] Note: Could not upload checkpoint to Master: {e}")

                    # Notify Master
                    requests.post(f"{self.master_url}/api/complete_round/{self.client_id}?round_num={current_round}", timeout=5)
                    self.last_completed_round = current_round
                    self.set_telemetry(sub_stage=f"Round {current_round} Complete (Awaiting Next Round)")
                    print(f"[{self.client_id}] Round {current_round} completed.")

            elif stage == "finished" and self.last_completed_round > 0:
                if not self.has_logged_finished:
                    self.set_telemetry(sub_stage="Training Complete (Ready for Inference)")
                    print(f"[{self.client_id}] Federated learning completed successfully! Ready for inference or next training run.")
                    self.has_logged_finished = True
                continue


def main():
    parser = argparse.ArgumentParser(description="FedMD Heterogeneous Edge Client")
    parser.add_argument("--master_ip", type=str, default="127.0.0.1", help="Master Coordinator IP address")
    parser.add_argument("--master_port", type=int, default=DEFAULT_MASTER_PORT, help="Master port")
    parser.add_argument("--client_id", type=str, default="client_1", help="Client unique ID (e.g. client_1, client_2)")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenetv3", "shufflenetv2", "custom"], help="Model architecture")
    parser.add_argument("--name", type=str, default=None, help="Device friendly name (e.g. Dell-XPS, ThinkPad)")
    args = parser.parse_args()

    device_name = args.name or f"{platform.node()} ({args.model})"
    client = FedMDClient(
        master_ip=args.master_ip,
        client_id=args.client_id,
        model_name=args.model,
        device_name=device_name,
        master_port=args.master_port,
    )

    if not client.register():
        print("Registration failed. Please make sure the Master server is running.")
        sys.exit(1)

    # Launch dedicated background heartbeat & live telemetry thread
    client.start_heartbeat_daemon()

    try:
        client.run_training_cycle()
    except KeyboardInterrupt:
        print(f"\n[{args.client_id}] Client stopped by user.")


if __name__ == "__main__":
    main()
