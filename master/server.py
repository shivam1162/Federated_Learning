"""
FastAPI Coordination Server for Heterogeneous FedMD.
Orchestrates client registration, Non-IID dataset distribution, and logit consensus aggregation.
"""
import time
import threading
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common.config import (
    PARTITIONS_DIR,
    DEFAULT_ROUNDS,
    DEFAULT_LOCAL_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_ALPHA,
)
from common.protocols import (
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    ClientStatus,
    ClusterStatusResponse,
    StartTrainingRequest,
    LogitSubmissionRequest,
    ConsensusResponse,
    RoundMetrics,
    DistributionStatusReport,
    HeartbeatPayload,
)
from master.partitioner import partition_dataset_non_iid, download_eurosat

app = FastAPI(
    title="FedMD Heterogeneous Coordination Server",
    description="Federated Model Distillation orchestrator for heterogeneous edge devices",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MasterState:
    def __init__(self):
        self.clients: Dict[str, ClientStatus] = {}
        self.current_round: int = 0
        self.total_rounds: int = DEFAULT_ROUNDS
        self.training_active: bool = False
        self.stage: str = "idle"  # idle, waiting_for_logits, consensus_ready, round_complete, finished
        
        # Hyperparameters
        self.local_epochs: int = DEFAULT_LOCAL_EPOCHS
        self.batch_size: int = DEFAULT_BATCH_SIZE
        self.temperature: float = DEFAULT_TEMPERATURE
        self.alpha: float = DEFAULT_ALPHA

        # Round tracking: round_num -> {client_id: logits_matrix}
        self.round_logits: Dict[int, Dict[str, List[List[float]]]] = {}
        # round_num -> consensus_matrix
        self.consensus_cache: Dict[int, List[List[float]]] = {}
        # round_num -> RoundMetrics
        self.history: List[RoundMetrics] = []
        # Partition statistics
        self.partition_stats: Dict[str, Dict[str, int]] = {}
        # Dataset distribution tracking
        self.distribution_id: int = 0
        self.distribution_active: bool = False
        self.distribution_status: Dict[str, Dict[str, Any]] = {}
        # Training session tracking (increments on each Start Federated Distillation trigger)
        self.training_session_id: int = 0

state = MasterState()


@app.get("/api/status", response_model=ClusterStatusResponse)
def get_cluster_status():
    """Returns cluster topology, active client list, and current round state."""
    # Check for inactive clients (heartbeat > 30s)
    now = time.time()
    active_count = sum(1 for c in state.clients.values() if (now - c.last_seen) < 45.0)

    return ClusterStatusResponse(
        master_status="Online",
        active_clients_count=active_count,
        current_round=state.current_round,
        total_rounds=state.total_rounds,
        training_active=state.training_active,
        clients=state.clients,
    )


@app.post("/api/register", response_model=ClientRegistrationResponse)
def register_client(req: ClientRegistrationRequest, request: Request):
    """
    Registers a new heterogeneous edge client into the coordination pool.
    """
    client_ip = request.client.host if request.client else "unknown"
    state.clients[req.client_id] = ClientStatus(
        client_id=req.client_id,
        device_name=req.device_name,
        model_name=req.model_name,
        cpu_info=req.cpu_info,
        ip_address=client_ip,
        last_seen=time.time(),
        current_round=state.current_round,
        is_training=state.training_active,
    )

    # If a distribution is already active or staged, initialize entry for this client
    if state.distribution_id > 0 and req.client_id not in state.distribution_status:
        state.distribution_status[req.client_id] = {
            "status": "pending",
            "message": "Newly registered. Awaiting dataset download...",
            "train_samples": 0,
            "val_samples": 0,
            "public_samples": 0,
            "timestamp": time.time(),
        }

    return ClientRegistrationResponse(
        status="success",
        message=f"Client '{req.client_id}' registered with model '{req.model_name}'",
        client_id=req.client_id,
        assigned_partition=f"{req.client_id}.zip",
    )


@app.post("/api/heartbeat/{client_id}")
def heartbeat(client_id: str, payload: Optional[HeartbeatPayload] = None):
    """Updates client presence timestamp and records live telemetry state."""
    if client_id in state.clients:
        client = state.clients[client_id]
        client.last_seen = time.time()
        if payload:
            if payload.sub_stage is not None:
                client.sub_stage = payload.sub_stage
            if payload.current_epoch is not None:
                client.current_epoch = payload.current_epoch
            if payload.total_epochs is not None:
                client.total_epochs = payload.total_epochs
            if payload.batch_progress is not None:
                client.batch_progress = payload.batch_progress
            if payload.step_loss is not None:
                client.step_loss = payload.step_loss
            if payload.step_acc is not None:
                client.step_acc = payload.step_acc
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Client not registered")


class PreparePartitionsRequest(BaseModel):
    num_clients: Optional[int] = None
    client_ids: Optional[List[str]] = None
    use_full_dataset: bool = False
    samples_per_client: int = 600


# Background partition task state
partition_task_state = {
    "active": False,
    "percent": 0.0,
    "message": "Idle",
    "status": "idle",
    "stats": {},
}


def _run_partition_worker(client_ids: List[str], samples_per_client: int, use_full_dataset: bool, mode_text: str):
    def update_progress(pct: float, msg: str):
        partition_task_state["percent"] = pct
        partition_task_state["message"] = msg

    try:
        partition_task_state["active"] = True
        partition_task_state["status"] = "running"
        stats = partition_dataset_non_iid(
            client_ids=client_ids,
            samples_per_client=samples_per_client,
            use_full_dataset=use_full_dataset,
            progress_callback=update_progress,
        )
        state.partition_stats = stats
        partition_task_state["status"] = "success"
        partition_task_state["percent"] = 1.0
        partition_task_state["stats"] = stats
        partition_task_state["message"] = f"Partitions generated [{mode_text}] for {len(client_ids)} clients: {', '.join(client_ids)}"
    except Exception as e:
        partition_task_state["status"] = "error"
        partition_task_state["message"] = str(e)
    finally:
        partition_task_state["active"] = False


@app.post("/api/prepare_partitions_async")
def prepare_partitions_async(req: Optional[PreparePartitionsRequest] = None):
    """
    Initiates dataset partitioning asynchronously with live progress tracking.
    """
    if partition_task_state.get("active"):
        return {"status": "busy", "message": "Partitioning task already in progress."}

    registered = list(state.clients.keys())
    use_full = req.use_full_dataset if req else False
    samples = req.samples_per_client if req else 600

    if req and req.client_ids and len(req.client_ids) > 0:
        client_ids = req.client_ids
    elif req and req.num_clients and req.num_clients > 0:
        client_ids = [f"client_{i+1}" for i in range(req.num_clients)]
    elif registered:
        client_ids = registered
    else:
        client_ids = ["client_1", "client_2"]

    mode_text = "Full Dataset (27,000 images)" if use_full else f"Sampled Mode ({samples} samples/node)"
    partition_task_state["percent"] = 0.02
    partition_task_state["message"] = "Initializing partition worker..."
    partition_task_state["status"] = "running"
    partition_task_state["active"] = True

    thread = threading.Thread(
        target=_run_partition_worker,
        args=(client_ids, samples, use_full, mode_text),
        daemon=True
    )
    thread.start()

    return {"status": "started", "message": "Partitioning job launched."}


@app.get("/api/partition_progress")
def get_partition_progress():
    """Returns current real-time partition progress and status."""
    return partition_task_state


@app.post("/api/prepare_partitions")
def prepare_partitions(req: Optional[PreparePartitionsRequest] = None):
    """
    Synchronous fallback: slices EuroSAT into Non-IID partitions.
    """
    registered = list(state.clients.keys())
    use_full = req.use_full_dataset if req else False
    samples = req.samples_per_client if req else 600

    if req and req.client_ids and len(req.client_ids) > 0:
        client_ids = req.client_ids
    elif req and req.num_clients and req.num_clients > 0:
        client_ids = [f"client_{i+1}" for i in range(req.num_clients)]
    elif registered:
        client_ids = registered
    else:
        client_ids = ["client_1", "client_2"]

    stats = partition_dataset_non_iid(
        client_ids=client_ids,
        samples_per_client=samples,
        use_full_dataset=use_full,
    )
    state.partition_stats = stats
    mode_text = "Full Dataset (27,000 images)" if use_full else f"Sampled Mode ({samples} samples/node)"
    return {
        "status": "success",
        "message": f"Partitions generated [{mode_text}] for {len(client_ids)} clients: {', '.join(client_ids)}",
        "stats": stats,
    }


@app.get("/api/download_partition/{client_id}")
def download_partition(client_id: str):
    """
    Serves the pre-packaged zip archive containing private data and public reference set.
    """
    zip_path = PARTITIONS_DIR / f"{client_id}.zip"
    if not zip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Partition archive for '{client_id}' not found on Coordinator. Please click 'Generate Non-IID Partitions' first."
        )

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{client_id}.zip"
    )


@app.post("/api/trigger_distribution")
def trigger_distribution():
    """
    Signals all registered worker clients to delete their existing local data,
    download the staged partition package, and reinitialize their data loaders.
    """
    # 1. Verify partition packages exist on Coordinator
    if not PARTITIONS_DIR.exists() or not list(PARTITIONS_DIR.glob("*.zip")):
        raise HTTPException(
            status_code=400,
            detail="No partition packages found in data/partitions/. Please click 'Generate Non-IID Partitions' first."
        )

    # 2. Verify registered clients
    if not state.clients:
        raise HTTPException(
            status_code=400,
            detail="No worker nodes registered. Launch worker clients first before distributing."
        )

    state.distribution_id += 1
    state.distribution_active = True
    state.distribution_status = {}

    for cid in state.clients:
        state.distribution_status[cid] = {
            "status": "pending",
            "message": "Awaiting worker response...",
            "train_samples": 0,
            "val_samples": 0,
            "public_samples": 0,
            "timestamp": time.time(),
        }

    return {
        "status": "started",
        "distribution_id": state.distribution_id,
        "target_clients": list(state.clients.keys()),
    }


@app.get("/api/distribution_state")
def get_distribution_state():
    """Returns the live dataset distribution status across all active clients."""
    registered_cids = list(state.clients.keys())
    if not registered_cids or state.distribution_id == 0:
        return {
            "distribution_id": state.distribution_id,
            "distribution_active": state.distribution_active,
            "all_ready": False,
            "clients": state.distribution_status,
        }

    all_ready = (
        len(state.distribution_status) >= len(registered_cids)
        and all(
            state.distribution_status.get(cid, {}).get("status") == "ready"
            for cid in registered_cids
        )
    )

    if all_ready and state.distribution_active:
        state.distribution_active = False

    return {
        "distribution_id": state.distribution_id,
        "distribution_active": state.distribution_active,
        "all_ready": all_ready,
        "clients": state.distribution_status,
    }


@app.post("/api/report_distribution_status")
def report_distribution_status(req: DistributionStatusReport):
    """Receives distribution progress or completion reports from worker clients."""
    state.distribution_status[req.client_id] = {
        "status": req.status,
        "message": req.message or "",
        "train_samples": req.train_samples or 0,
        "val_samples": req.val_samples or 0,
        "public_samples": req.public_samples or 0,
        "timestamp": time.time(),
    }

    if req.client_id in state.clients:
        state.clients[req.client_id].last_seen = time.time()

    return {"status": "recorded"}


@app.post("/api/start_training")
def start_training(req: StartTrainingRequest):
    """
    Initiates federated distillation across all registered clients.
    """
    if len(state.clients) == 0:
        raise HTTPException(status_code=400, detail="Cannot start: No clients registered yet.")

    state.training_session_id += 1
    state.total_rounds = req.total_rounds
    state.local_epochs = req.local_epochs
    state.batch_size = req.batch_size
    state.temperature = req.temperature
    state.alpha = req.alpha
    state.current_round = 1
    state.training_active = True
    state.stage = "waiting_for_logits"
    state.round_logits = {}
    state.consensus_cache = {}
    state.history = []

    for c in state.clients.values():
        c.is_training = True
        c.current_round = 1

    return {
        "status": "started",
        "session_id": state.training_session_id,
        "round": 1,
        "total_rounds": state.total_rounds,
    }


@app.get("/api/round_state")
def get_round_state():
    """Returns training parameters and synchronization signals for worker clients."""
    return {
        "training_active": state.training_active,
        "training_session_id": state.training_session_id,
        "current_round": state.current_round,
        "total_rounds": state.total_rounds,
        "stage": state.stage,
        "distribution_id": state.distribution_id,
        "local_epochs": state.local_epochs,
        "batch_size": state.batch_size,
        "temperature": state.temperature,
        "alpha": state.alpha,
    }


@app.post("/api/submit_logits")
def submit_logits(req: LogitSubmissionRequest):
    """
    Collects public-set logits from a client for the current round.
    Computes ensemble consensus when all active registered clients have submitted.
    """
    r_num = req.round_num
    if r_num not in state.round_logits:
        state.round_logits[r_num] = {}

    state.round_logits[r_num][req.client_id] = req.logits

    # Update client telemetry
    if req.client_id in state.clients:
        state.clients[req.client_id].metrics = {
            "loss": req.local_loss,
            "acc": req.local_acc,
        }
        state.clients[req.client_id].current_round = r_num

    registered_client_count = max(len(state.clients), 1)
    submitted_count = len(state.round_logits[r_num])

    # Check if all clients submitted
    if submitted_count >= registered_client_count:
        # 1. Compute Ensemble Consensus Logits: Z_bar = (1/K) * Sum(Z_k)
        all_matrices = [np.array(logits) for logits in state.round_logits[r_num].values()]
        stacked = np.stack(all_matrices, axis=0)  # Shape: [K, N_public, NUM_CLASSES]
        consensus = np.mean(stacked, axis=0)       # Shape: [N_public, NUM_CLASSES]
        state.consensus_cache[r_num] = consensus.tolist()
        state.stage = "consensus_ready"

        # 2. Compute Agreement Score (Cosine Similarity between clients)
        agreement = 1.0
        if len(all_matrices) >= 2:
            norm1 = np.linalg.norm(all_matrices[0].flatten())
            norm2 = np.linalg.norm(all_matrices[1].flatten())
            if norm1 > 0 and norm2 > 0:
                agreement = float(np.dot(all_matrices[0].flatten(), all_matrices[1].flatten()) / (norm1 * norm2))

        # 3. Record Round Telemetry
        metrics_entry = RoundMetrics(
            round_num=r_num,
            client_metrics={
                cid: state.clients[cid].metrics for cid in state.clients if cid in state.clients
            },
            agreement_score=round(max(0.0, agreement), 4),
        )
        state.history.append(metrics_entry)

    return {
        "status": "received",
        "submitted": submitted_count,
        "required": registered_client_count,
        "consensus_ready": (r_num in state.consensus_cache),
    }


@app.get("/api/get_consensus/{round_num}", response_model=ConsensusResponse)
def get_consensus(round_num: int):
    """
    Returns the computed consensus logits for round_num if ready.
    """
    if round_num in state.consensus_cache:
        return ConsensusResponse(
            ready=True,
            round_num=round_num,
            consensus_logits=state.consensus_cache[round_num],
            message="Consensus ready",
        )
    return ConsensusResponse(
        ready=False,
        round_num=round_num,
        consensus_logits=None,
        message="Consensus still pending from other workers",
    )


@app.post("/api/complete_round/{client_id}")
def complete_round(client_id: str, round_num: int):
    """
    Client reports completion of distillation fine-tuning for round_num.
    When all clients finish, increments current_round.
    """
    if client_id in state.clients:
        state.clients[client_id].current_round = round_num

    # Check if all clients reached this round
    all_done = all(c.current_round >= round_num for c in state.clients.values())
    if all_done:
        if round_num >= state.total_rounds:
            state.training_active = False
            state.stage = "finished"
        else:
            state.current_round = round_num + 1
            state.stage = "waiting_for_logits"

    return {"current_round": state.current_round, "stage": state.stage}


@app.get("/api/history")
def get_training_history():
    """Returns historical round metrics for plotting in Streamlit."""
    return {
        "rounds": [m.round_num for m in state.history],
        "history": [m.model_dump() for m in state.history],
        "stats": state.partition_stats,
    }
