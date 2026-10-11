"""
Pydantic schemas and data protocols for REST API communication between Master and Workers.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClientRegistrationRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier (e.g., client_1, client_2)")
    device_name: str = Field(..., description="Human readable device name or hostname")
    model_name: str = Field(..., description="Chosen architecture (resnet18, mobilenetv3, shufflenetv2, custom)")
    cpu_info: Optional[str] = Field(default=None, description="Processor specification / thread count")


class ClientRegistrationResponse(BaseModel):
    status: str
    message: str
    client_id: str
    assigned_partition: str


class ClientStatus(BaseModel):
    client_id: str
    device_name: str
    model_name: str
    cpu_info: Optional[str] = None
    ip_address: Optional[str] = None
    last_seen: float
    current_round: int = 0
    is_training: bool = False
    metrics: Dict[str, float] = Field(default_factory=dict)


class ClusterStatusResponse(BaseModel):
    master_status: str
    active_clients_count: int
    current_round: int
    total_rounds: int
    training_active: bool
    clients: Dict[str, ClientStatus]


class StartTrainingRequest(BaseModel):
    total_rounds: int = 5
    local_epochs: int = 2
    batch_size: int = 32
    temperature: float = 2.0
    alpha: float = 0.5


class LogitSubmissionRequest(BaseModel):
    client_id: str
    round_num: int
    # List of float logits: shape [N_public, NUM_CLASSES]
    logits: List[List[float]]
    local_loss: float
    local_acc: float


class ConsensusResponse(BaseModel):
    ready: bool
    round_num: int
    consensus_logits: Optional[List[List[float]]] = None
    message: str = "Consensus pending"


class RoundMetrics(BaseModel):
    round_num: int
    client_metrics: Dict[str, Dict[str, float]]
    agreement_score: float = 0.0
    val_accuracy: Dict[str, float] = Field(default_factory=dict)
    ensemble_val_accuracy: float = 0.0
