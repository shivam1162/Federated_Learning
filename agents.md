# Heterogeneous Model Federated Learning (FedMD) Architecture & Specification

## 1. Executive Summary
This project implements a decentralized, privacy-preserving **Heterogeneous Model Federated Learning** system on physical hardware across a local Wi-Fi network. 

Because participating devices utilize fundamentally different neural network architectures (e.g., ResNet vs. MobileNet vs. ShuffleNet), standard coordinate-wise parameter averaging (FedAvg) is mathematically impossible. Instead, this system leverages **Federated Model Distillation (FedMD)** to establish a **Common Knowledge Base** via logit consensus over an unlabelled public reference dataset.

The system is controlled and visualized via an interactive **Streamlit Web Portal** hosted on a dedicated **Coordinator (Master)** laptop.

---

## 2. Hardware Topology & Network Design

```
                                 MASTER LAPTOP (Coordinator)
                      ┌───────────────────────────────────────────────┐
                      │  • Streamlit Web Portal (Port 8501)           │
                      │  • FastAPI Orchestrator (Port 8000)           │
                      │  • Dataset Partitioner & Distribution Engine  │
                      │  • Logit Consensus Aggregator Engine          │
                      │  • (Zero model training on Master CPU)        │
                      └───────┬───────────────────────────────┬───────┘
                              │                               │
                    Local LAN │ (Wi-Fi)             Local LAN │ (Wi-Fi)
                              ▼                               ▼
               WORKER LAPTOP 1 (Client 1)       WORKER LAPTOP 2 (Client 2)
            ┌──────────────────────────────┐ ┌──────────────────────────────┐
            │ • Architecture: ResNet-18    │ │ • Architecture: MobileNetV3  │
            │ • Private Non-IID Partition  │ │ • Private Non-IID Partition  │
            │ • Local CPU Training Loop    │ │ • Local CPU Training Loop    │
            │ • FedMD Distillation Engine  │ │ • FedMD Distillation Engine  │
            └──────────────────────────────┘ └──────────────────────────────┘
                                              ▲
                                              │ (Scalable to Client 3..N)
                                     FUTURE WORKER LAPTOPS
```

### Roles & Responsibilities:
1. **Master Laptop (Dedicated Coordinator)**:
   * **Host**: Runs on the current machine.
   * **Compute Load**: Lightweight (orchestration, consensus averaging, network I/O, UI rendering).
   * **Responsibilities**:
     * Tracks connected worker laptops dynamically via `/register`.
     * Prepares EuroSAT partitions and serves them via HTTP when triggered.
     * Collects public-set logits from all active workers each round.
     * Computes the ensemble consensus logits $\bar{Z}_{\text{public}}$ and serves them back.
     * Renders real-time telemetry, model comparisons, and inference demos on Streamlit.

2. **Worker Laptops (Edge Training Clients)**:
   * **Hardware**: CPU-only laptops connected to the same Wi-Fi.
   * **Compute Load**: Model training & inference on private local data + public reference set.
   * **Responsibilities**:
     * Registers with Master upon launch with device info and chosen model architecture.
     * Downloads its assigned dataset partition on command.
     * Executes local training epochs on private Non-IID data.
     * Computes logits on the public reference set and uploads them to Master.
     * Updates local model weights using Knowledge Distillation (KL Divergence) against the consensus.

---

## 3. Dataset & Partitioning Strategy

### Dataset: EuroSAT (RGB)
* **Domain**: Sentinel-2 Satellite Earth Observation.
* **Image Specifications**: 64 × 64 pixels, 3 channels (RGB).
* **Classes (10)**:
  1. `AnnualCrop`
  2. `Forest`
  3. `HerbaceousVegetation`
  4. `Highway`
  5. `Industrial`
  6. `Pasture`
  7. `PermanentCrop`
  8. `Residential`
  9. `River`
  10. `SeaLake`

### Data Slices:
1. **Public Reference Distillation Set ($D_{\text{public}}$)**:
   * A shared unlabelled set (~300–500 images) sampled across all classes.
   * Stored on Master and distributed to all clients for knowledge transfer.
2. **Private Non-IID Client Partitions ($D_{k}^{\text{private}}$)**:
   * Extreme Non-IID skewing by class groups to demonstrate cross-domain knowledge transfer:
     * **Client 1**: Heavy bias towards Agricultural / Vegetation classes (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Pasture`).
     * **Client 2**: Heavy bias towards Urban / Infrastructure classes (`Highway`, `Industrial`, `Residential`).
     * **Future Clients**: Bias towards Water bodies (`River`, `SeaLake`) or mixed distributions.
3. **Global Validation Set ($D_{\text{val}}$)**:
   * A balanced test set retained on Master to evaluate individual model and ensemble accuracy across rounds.

### One-Click Distribution Flow:
1. User clicks **"Distribute Dataset"** on the Streamlit dashboard.
2. Master slices the dataset into Non-IID packages for all registered clients.
3. Master notifies clients (or clients poll/listen) to download their partition archive via:
   `GET http://<MASTER_IP>:8000/api/download_partition/{client_id}`
4. Clients unpack the archive locally into their `data/` directory.

---

## 4. Heterogeneous Model Architecture Suite

All models are adapted for 64×64 input resolution and optimized for CPU inference and training:

| Model ID | Target Architecture | Key Architectural Traits | Target Device |
| :--- | :--- | :--- | :--- |
| **Model 1** | `ResNet-18 (Compact)` | Residual skip connections; standard 3x3 conv blocks. | Worker 1 |
| **Model 2** | `MobileNetV3-Small` | Depthwise separable convolutions; inverted residuals; squeeze-and-excitation. | Worker 2 |
| **Model 3** | `ShuffleNetV2` | Channel split & channel shuffle; extremely low FLOP count on CPU. | Future Worker 3 |
| **Model 4** | `CustomConvNet` | Lightweight 4-layer CNN baseline for low-spec edge nodes. | Future Worker 4 |

---

## 5. Federated Model Distillation (FedMD) Formulation

Standard FedAvg cannot aggregate weights $W_1$ and $W_2$ because their layer shapes and dimensions do not align. FedMD exchanges prediction outputs (logits) instead.

### Algorithmic Steps Per Communication Round $t$:

```
Round t Cycle:
  [Worker k] Local Training on Private Data -> L_CE
       │
       ▼
  [Worker k] Public Forward Pass on D_public -> Output Logits Z_k
       │
       ▼ (HTTP POST to Master)
  [Master] Logit Aggregation: Z_bar = (1/K) * Sum(Z_k)
       │
       ▼ (HTTP GET from Master)
  [Worker k] Distillation on D_public: L_total = (1 - alpha)*L_CE + alpha*L_KD(Z_k, Z_bar)
```

1. **Step 1: Local Private Training**
   Each worker $k$ trains on its private Non-IID dataset $D_k$ using Cross-Entropy loss:
   $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^C y_c \log p_c(x)$$

2. **Step 2: Public Logit Generation**
   Worker $k$ evaluates the public reference set $D_{\text{public}}$ and records unnormalized logits:
   $$Z_k = f_k(X_{\text{public}}; \theta_k) \in \mathbb{R}^{N_{\text{public}} \times C}$$
   Worker $k$ uploads $Z_k$ to the Master.

3. **Step 3: Master Consensus Aggregation**
   Master collects logits from all $K$ currently registered active workers and computes consensus logits:
   $$\bar{Z}_{\text{public}} = \frac{1}{K} \sum_{k=1}^K Z_k$$

4. **Step 4: Knowledge Distillation Fine-Tuning**
   Workers fetch $\bar{Z}_{\text{public}}$ from Master and update their parameters by minimizing the Kullback-Leibler (KL) Divergence loss on the public dataset:
   $$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \text{KL}\left(\text{Softmax}\left(\frac{\bar{Z}_{\text{public}}}{\tau}\right) \parallel \text{Softmax}\left(\frac{Z_k}{\tau}\right)\right)$$
   where $\tau$ is the distillation temperature (default $\tau = 2.0$).

---

## 6. Dynamic Scaling Mechanism (Scalable to $K$ Clients)

The system does **not** hardcode client count:
* **Dynamic Registration**:
  * Endpoint: `POST /api/register`
  * Payload: `{ "client_id": "...", "device_name": "...", "model_name": "...", "cpu_info": "..." }`
  * Master adds the client to its active pool.
* **Dynamic Consensus**:
  * Master determines consensus once all $K$ registered clients have submitted logits for round $t$.
  * Configurable timeouts handle dropped nodes or slow stragglers.
* **Modular Client Script**:
  * A single worker entrypoint:
    ```bash
    python client/client.py --master_ip 192.168.1.15 --client_id client_2 --model mobilenetv3 --name "Lenovo-Yoga"
    ```

---

## 7. Streamlit Web Portal Specifications

The web dashboard (`master/dashboard.py`) contains four core tabs/views:

1. **Cluster Topology & Node Monitor**:
   * Master status, local IP, active port.
   * Dynamic cards for each connected worker: Client ID, hardware specs, ping/latency, active model architecture, parameter count.
2. **Dataset Management & One-Click Distribution**:
   * EuroSAT dataset status (Downloaded / Ready).
   * Visual chart of Non-IID class distribution across clients.
   * **[Distribute Dataset to Clients]** button with real-time transfer progress.
3. **Training Control & Live Telemetry**:
   * Hyperparameter controls: Number of FL rounds, local epochs, batch size, temperature $\tau$.
   * **[Start Federated Distillation]** button.
   * Real-time charts:
     * Local accuracy & loss progression per client.
     * Consensus agreement score (cosine similarity of logits across models).
     * Global validation accuracy over rounds.
4. **Interactive Satellite Inference Playground**:
   * Drag-and-drop or select sample EuroSAT images.
   * Side-by-side breakdown:
     * Model 1 (ResNet-18) prediction & confidence bar.
     * Model 2 (MobileNetV3) prediction & confidence bar.
     * Model 3...N predictions.
     * **Consensus Knowledge Ensemble** final classification.

---

## 8. Directory Structure

```
federated_learning/
├── agents.md                    # System architecture & specification (this file)
├── notes.txt                    # Project notes & source dataset links
├── requirements.txt             # Master & Client Python dependencies
│
├── common/                      # Shared across Master and Workers
│   ├── __init__.py
│   ├── config.py                # Hyperparameters, paths, class names, network ports
│   └── protocols.py             # Pydantic models for REST API requests & responses
│
├── master/                      # Runs on Master Coordinator Laptop
│   ├── __init__.py
│   ├── server.py                # FastAPI coordination server (registration, sync, logits, data staging)
│   ├── partitioner.py           # EuroSAT Non-IID dataset slicer and packager
│   └── dashboard.py             # Streamlit visual control center & inference demo
│
├── client/                      # Runs on Worker Laptops
│   ├── __init__.py
│   ├── client.py                # Worker client daemon (polls master, manages training cycles)
│   ├── models.py                # PyTorch heterogeneous architectures (ResNet18, MobileNetV3, ShuffleNet)
│   ├── dataset.py               # Local dataset loaders & preprocessors
│   └── trainer.py               # Local training loop + FedMD KL distillation logic
│
└── data/                        # Local data directory
    ├── raw/                     # Original EuroSAT dataset
    ├── partitions/              # Packaged client zips (client_1.zip, client_2.zip, etc.)
    └── public_reference/        # Shared unlabelled public distillation set
```

---

## 9. Implementation Roadmap

1. **Phase 1: Environment & Dataset Preparation**
   * Create `requirements.txt` with PyTorch (CPU), FastAPI, Uvicorn, Streamlit, Pillow, etc.
   * Implement automated EuroSAT dataset downloader and verification script.
2. **Phase 2: Common Library & Heterogeneous Models**
   * Define shared schemas (`protocols.py`) and config constants (`config.py`).
   * Implement heterogeneous model definitions (`models.py`) adapted for 64×64 images.
3. **Phase 3: Master Coordination Server & Partitioner**
   * Build Non-IID EuroSAT partitioner with public reference subset generation (`partitioner.py`).
   * Build FastAPI server with endpoints: registration, partition download, logit submission, consensus broadcast.
4. **Phase 4: Client Training Daemon**
   * Build client connection lifecycle: register, download partition, train local epochs, generate public logits, apply distillation.
5. **Phase 5: Streamlit Web Portal & Inference Playground**
   * Develop real-time dashboard with cluster topology, dataset distribution trigger, round monitoring, and live image classifier demo.
6. **Phase 6: Verification & Multi-Machine LAN Deployment**
   * Test end-to-end simulation locally (multiple client processes on localhost).
   * Deploy across physical laptops over the local Wi-Fi network.

---

## 10. Operational Guide: How to Run the System

### Prerequisites:
1. Ensure all laptops (Master Coordinator + Worker Laptops) are connected to the **same Wi-Fi / LAN network**.
2. Python 3.10+ installed on all machines with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Windows Firewall check: On the Master Coordinator, ensure inbound connections on port `8000` (FastAPI) and `8501` (Streamlit) are permitted on private networks.

---

### Step 1: Launch Master Coordinator (This Laptop)
On the Master laptop, run the all-in-one launcher:
```powershell
python run_master.py
```
* **What happens**:
  * Scans and prints your local Wi-Fi LAN IP (e.g., `192.168.1.15`).
  * Boots the FastAPI coordination hub on `http://0.0.0.0:8000`.
  * Launches the Streamlit Web Portal on `http://localhost:8501`.
  * Outputs the exact connection commands for worker laptops.

---

### Step 2: Launch Worker Laptop 1 (Client 1 - ResNet-18)
On the first worker laptop, open a terminal in the project directory and run:
```bash
python client/client.py --master_ip <MASTER_IP> --client_id client_1 --model resnet18 --name "Worker-1-ResNet"
```
*(Replace `<MASTER_IP>` with the IP printed by `run_master.py`, e.g., `192.168.1.15`).*

---

### Step 3: Launch Worker Laptop 2 (Client 2 - MobileNetV3)
On the second worker laptop, open a terminal in the project directory and run:
```bash
python client/client.py --master_ip <MASTER_IP> --client_id client_2 --model mobilenetv3 --name "Worker-2-MobileNet"
```

---

### Step 4: Web Portal Operations
1. Open your browser on the Master laptop to **`http://localhost:8501`**.
2. **Tab 1 (Cluster Topology)**: Confirm both worker laptops appear with green `ACTIVE` status cards and their respective model architectures.
3. **Tab 2 (Dataset Management)**:
   * Click **"Generate & Partition Dataset"** to slice EuroSAT into Non-IID partitions and prepare the unlabelled public reference set.
   * Click **"Distribute Dataset to Clients"** (clients will automatically fetch their assigned `.zip` archives over LAN).
4. **Tab 3 (Training Telemetry)**:
   * Select the number of FL rounds (e.g., 5) and local CPU epochs (e.g., 2).
   * Click **"Start Federated Distillation"**.
   * Watch live progress: local CE training $\rightarrow$ public logit submission $\rightarrow$ consensus computation $\rightarrow$ KL distillation.
5. **Tab 4 (Satellite Inference Playground)**:
   * Select or upload a test satellite image.
   * Observe how individual model predictions compare against the **Common Knowledge Ensemble**.

---

## 11. Step-by-Step Guide: Adding a New Client (Scaling to Client 3, 4, ... N)

The system is built on a **dynamic auto-discovery architecture**. You do **not** need to restart the Master server or modify any code to add new devices.

Follow these steps whenever a new worker laptop joins:

### Step 1: Connect to Wi-Fi
Connect the new laptop to the exact same Wi-Fi network as the Master Coordinator.

### Step 2: Copy Code & Install Dependencies
1. Copy or clone the `federated_learning/` project folder to the new laptop.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Select an Architecture for the New Device
Choose an architecture suited to the new device's CPU capacity from the registry:
* `shufflenetv2` (Ultra-lightweight, 352k parameters — great for budget CPUs)
* `custom` (Compact 4-layer CNN)
* `mobilenetv3` (MobileNetV3-Small)
* `resnet18` (ResNet-18)

### Step 4: Launch the Client Daemon
Run the client script on the new laptop with a **unique `--client_id`**:
```bash
python client/client.py --master_ip <MASTER_IP> --client_id client_3 --model shufflenetv2 --name "Worker-3-ShuffleNet"
```

### Step 5: What Happens Automatically Behind the Scenes
1. **Dynamic Registration Handshake**:
   * The new client sends `POST /api/register` to Master with its ID, device hostname, CPU specifications, and model name.
2. **Instant UI Discovery**:
   * A new hardware card for `client_3` immediately renders on the Streamlit **Cluster Topology** tab.
3. **Partition Allocation**:
   * When dataset distribution is triggered, Master slices an additional Non-IID class partition specifically for `client_3`.
   * `client_3` downloads `http://<MASTER_IP>:8000/api/download_partition/client_3` and unpacks it locally.
4. **Dynamic Consensus Expansion**:
   * During the next federated round, Master automatically expands the consensus aggregation pool from $K=2$ to $K=3$:
     $$\bar{Z}_{\text{public}} = \frac{1}{3} \left( Z_{\text{client\_1}} + Z_{\text{client\_2}} + Z_{\text{client\_3}} \right)$$
   * Consensus logits are computed only when all 3 clients have uploaded their logits.
5. **Ensemble Participation**:
   * The new client's model automatically contributes to the real-time inference playground predictions and agreement score metrics.

