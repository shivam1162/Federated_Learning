# Heterogeneous Model Federated Learning (FedMD) Architecture & Specification

## 1. Executive Summary
This project implements a decentralized, privacy-preserving **Heterogeneous Model Federated Learning** system on physical hardware across a local Wi-Fi network. 

Because participating edge devices utilize fundamentally different neural network architectures (e.g., ResNet vs. MobileNet vs. ShuffleNet), standard coordinate-wise parameter averaging (FedAvg) is mathematically impossible. Instead, this system leverages **Federated Model Distillation (FedMD)** to establish a **Common Knowledge Base** via logit consensus over an unlabelled public reference dataset.

The system is controlled and visualized via an enterprise **Streamlit Web Portal** hosted on a dedicated **Coordinator (Master)** laptop.

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
   * **Host**: Runs on the central coordinator machine.
   * **Compute Load**: Lightweight (orchestration, consensus averaging, network I/O, UI rendering). Zero neural network training load on CPU.
   * **Responsibilities**:
     * Tracks connected worker laptops dynamically via `/api/register`.
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
* **Total Images Available**: 27,000 images (~2,000 to 3,000 images per class) stored in `data/raw/2750/`.

### 4-Domain Non-IID Class Specialization Matrix
To rigorously demonstrate cross-domain knowledge transfer, EuroSAT classes are grouped into 4 distinct domains:

| Domain ID | Specialization Name | Primary High-Density Classes | Target Node |
| :--- | :--- | :--- | :--- |
| **Domain 0** | **Agriculture & Farmland** | `AnnualCrop`, `Pasture`, `PermanentCrop` | Client 1 |
| **Domain 1** | **Natural Forestry & Woods** | `Forest`, `HerbaceousVegetation` | Client 2 |
| **Domain 2** | **Urban & Built Environment** | `Highway`, `Industrial`, `Residential` | Client 3 |
| **Domain 3** | **Water Bodies & Aquatic Terrain** | `River`, `SeaLake` | Client 4 |

*Each client receives an 80% density skew toward its primary domain, with the remaining 20% distributed across other classes.*

### Data Slices:
1. **Public Reference Distillation Set ($D_{\text{public}}$)**:
   * A shared unlabelled set (~400 images) sampled evenly across all 10 classes (40/class).
   * Bundled into every client package for knowledge transfer.
2. **Private Non-IID Client Partitions ($D_{k}^{\text{private}}$)**:
   * Sliced per client based on the domain specialization matrix.
3. **Global Validation Set ($D_{\text{val}}$)**:
   * A balanced test set (500 images) retained on Master to evaluate individual model and ensemble accuracy across rounds.

### Dual Dataset Modes:
* **Fast Demo Mode (Recommended for CPU Demos)**:
  * Uses ~600–800 images per client (adjustable via slider from 300 to 2,000).
  * Fast generation (~2 seconds) and rapid CPU training (~20–30 seconds per round).
* **Full Dataset Mode**:
  * Divides all ~26,100 remaining images equally across the $K$ clients (~13,000/node for 2 clients, ~8,700/node for 3 clients).
  * Generates ~35 MB `.zip` packages per client.

### Dynamic Generation Flow with Live Progress:
1. **Automatic Pre-Clean**: The partitioner forces a clean wipe of `data/partitions/` before every generation so no stale packages linger.
2. **Live Progress Tracking**: Streamlit tracks progress in real time via `/api/partition_progress` and renders a live `st.progress` bar (purging $\rightarrow$ public reference $\rightarrow$ validation set $\rightarrow$ class allocation $\rightarrow$ zip compression).
3. **One-Click Download**: Worker clients pull their assigned partition via `GET /api/download_partition/{client_id}` and unpack it locally into `data/{client_id}/`.

---

## 4. Heterogeneous Model Architecture Suite

All models are adapted for 64×64 input resolution and optimized for CPU inference and training:

| Model ID | Target Architecture | Parameters | Size (MB) | Architectural Traits | Target Device |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | `ResNet-18 (Compact)` | 11,173,962 | ~42.6 MB | Residual skip connections; 3x3 conv stem adapted for 64x64. | Worker 1 |
| **Model 2** | `MobileNetV3-Small` | 1,528,106 | ~5.8 MB | Depthwise separable convolutions; inverted residuals; squeeze-and-excitation. | Worker 2 |
| **Model 3** | `ShuffleNetV2` | 352,042 | ~1.3 MB | Channel split & channel shuffle; ultra-low CPU FLOP count. | Worker 3 |
| **Model 4** | `CustomConvNet` | ~250,000 | ~1.0 MB | Lightweight 4-stage convolutional baseline for budget edge nodes. | Worker 4 |

---

## 5. Federated Model Distillation (FedMD) Formulation

Standard FedAvg cannot aggregate weights $W_1$ and $W_2$ because layer shapes and dimensions do not align. FedMD exchanges prediction outputs (logits) instead.

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
  [Worker k] Distillation on D_public: L_KD = tau^2 * KL(Softmax(Z_bar/tau) || Softmax(Z_k/tau))
```

1. **Step 1: Local Private Training**
   Each worker $k$ trains on its private Non-IID dataset $D_k$ using Cross-Entropy loss:
   $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^C y_c \log p_c(x)$$

2. **Step 2: Public Logit Generation**
   Worker $k$ evaluates the public reference set $D_{\text{public}}$ and records unnormalized logits:
   $$Z_k = f_k(X_{\text{public}}; \theta_k) \in \mathbb{R}^{N_{\text{public}} \times C}$$
   Worker $k$ uploads $Z_k$ to the Master via `POST /api/submit_logits`.

3. **Step 3: Master Consensus Aggregation**
   Master collects logits from all $K$ currently active workers and computes consensus logits:
   $$\bar{Z}_{\text{public}} = \frac{1}{K} \sum_{k=1}^K Z_k$$
   Master also calculates the **Consensus Agreement Score** (cosine similarity of logits across models).

4. **Step 4: Knowledge Distillation Fine-Tuning**
   Workers fetch $\bar{Z}_{\text{public}}$ via `GET /api/get_consensus/{round_num}` and update their weights by minimizing the Kullback-Leibler (KL) Divergence loss on the public dataset:
   $$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \text{KL}\left(\text{Softmax}\left(\frac{\bar{Z}_{\text{public}}}{\tau}\right) \parallel \text{Softmax}\left(\frac{Z_k}{\tau}\right)\right)$$
   where $\tau$ is the distillation temperature (default $\tau = 2.0$).

---

## 6. Dynamic Scaling Mechanism (Scalable to $K$ Clients)

The system does **not** hardcode client count:
* **Dynamic Registration**:
  * Endpoint: `POST /api/register`
  * Payload: `{ "client_id": "...", "device_name": "...", "model_name": "...", "cpu_info": "..." }`
  * Master adds the client to its active pool.
* **Dynamic Consensus Quorum**:
  * Master determines consensus once all $K$ currently registered clients have submitted logits for round $t$:
    $$\bar{Z}_{\text{public}} = \frac{1}{K} \sum_{k=1}^K Z_k$$
* **Modular Client Script**:
  * Run on any machine with:
    ```bash
    python client/client.py --master_ip <MASTER_IP> --client_id client_3 --model shufflenetv2 --name "Worker-3-ShuffleNet"
    ```

---

## 7. Streamlit Web Portal Specifications

The web dashboard (`master/dashboard.py`) follows an enterprise, professional design (zero emojis, clean slate/neutral color palette):

1. **Cluster Topology Tab**:
   * Master status, local IP, active port, hardware role.
   * Dynamic status cards for each registered edge laptop: Client ID, hardware specs, status badge (`[ACTIVE]` / `[DISCONNECTED]`), current round progress.
2. **Dataset Management & Distribution Tab**:
   * EuroSAT dataset status.
   * **Target Worker Count Selector**: Automatically detects connected workers, or allows manual pre-staging ($K = 2, 3, 4, \dots$).
   * **Dataset Allocation Mode Toggle**: Toggle between **Fast Demo Mode** (with slider) and **Full Dataset Mode** (~27,000 images).
   * **Live Progress Bar**: Step-by-step visual progress tracking during generation.
   * **Non-IID Distribution Chart**: Grouped bar chart displaying exact sample counts per class across all clients.
3. **Training Control & Live Telemetry Tab**:
   * Hyperparameter controls: Number of FL rounds, local epochs, batch size, temperature $\tau$.
   * **[Start Federated Distillation]** button.
   * Real-time Plotly charts:
     * Local validation accuracy progression per client.
     * Consensus agreement score (cosine similarity of logits across disparate models over rounds).
4. **Interactive Satellite Inference Playground Tab**:
   * Drag-and-drop custom 64x64 satellite tiles or filter benchmark samples across all 10 EuroSAT land cover classes.
   * Ground truth detection and verification badge (`[CORRECT MATCH]` vs `[MISMATCH]`).
   * **Live PyTorch Evaluation**: Executes real-time forward passes on CPU across all connected heterogeneous models using trained `.pt` checkpoints (or baseline initial weights).
   * **Consensus Analytics**: Highlights whether models reached unanimous agreement or if the ensemble reconciled domain-biased divergence.
   * **Visual Telemetry**:
     * Hero consensus card with confidence progress bar and Top-3 candidate ranking.
     * Expandable individual model breakdown cards with architecture parameter counts and weights status badges.
     * Comparative Plotly grouped bar chart showing the full 10-class probability spectrum across all disparate architectures and the Common Knowledge Ensemble.

---

## 8. Current Directory Structure

```
federated_learning/
├── .gitignore                   # Excludes raw images, weights, zip packages, and cache
├── agents.md                    # System architecture & specification (this file)
├── notes.txt                    # Project notes & source dataset links
├── requirements.txt             # Master & Client Python dependencies
├── run_master.py                # All-in-one Coordinator launcher (FastAPI + Streamlit)
├── download_eurosat.py          # EuroSAT RGB dataset downloader & validator
├── test_simulation.py           # End-to-end local pipeline verification test
│
├── common/                      # Shared across Master and Workers
│   ├── __init__.py
│   ├── config.py                # Hyperparameters, paths, class names, network ports
│   └── protocols.py             # Pydantic schemas for REST API requests & responses
│
├── master/                      # Runs on Master Coordinator Laptop
│   ├── __init__.py
│   ├── server.py                # FastAPI coordination server (registration, sync, logits, data staging)
│   ├── partitioner.py           # EuroSAT Non-IID dataset slicer and packager with progress callback
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
    ├── .gitkeep                 # Preserves data folder in git
    ├── raw/
    │   └── 2750/                # Full official EuroSAT dataset (27,000 satellite images)
    ├── partitions/              # Dynamically generated client packages (client_1.zip, client_2.zip, ...)
    ├── public_reference/        # Shared unlabelled public distillation set (~400 images)
    └── val/                     # Global validation evaluation set (~500 images)
```

---

## 9. Remote Git Repository

* **Remote Origin**: `git@github.com:shivam1162/Federated_Learning.git`
* **Default Branch**: `main`
* **Web URL**: [https://github.com/shivam1162/Federated_Learning](https://github.com/shivam1162/Federated_Learning)

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
  * Scans and prints your local Wi-Fi LAN IP (e.g., `192.168.0.244`).
  * Boots the FastAPI coordination hub on `http://0.0.0.0:8000` with hot-reloading.
  * Launches the Streamlit Web Portal on `http://localhost:8501`.
  * Outputs the exact connection commands for worker laptops.

---

### Step 2: Launch Worker Laptop 1 (Client 1 - ResNet-18)
On the first worker laptop, open a terminal in the project directory and run:
```bash
python client/client.py --master_ip <MASTER_IP> --client_id client_1 --model resnet18 --name "Worker-1-ResNet"
```
*(Replace `<MASTER_IP>` with the IP printed by `run_master.py`, e.g., `192.168.0.244`).*

---

### Step 3: Launch Worker Laptop 2 (Client 2 - MobileNetV3)
On the second worker laptop, open a terminal in the project directory and run:
```bash
python client/client.py --master_ip <MASTER_IP> --client_id client_2 --model mobilenetv3 --name "Worker-2-MobileNet"
```

---

### Step 4: Web Portal Operations
1. Open your browser on the Master laptop to **`http://localhost:8501`**.
2. **Tab 1 (Cluster Topology)**: Confirm connected worker laptops appear with `[ACTIVE]` status badges and their respective model architectures.
3. **Tab 2 (Dataset Management)**:
   * Select your **Dataset Allocation Mode** (Fast Demo Mode or Full Dataset Mode).
   * Click **"Generate Non-IID Partitions"**: Partitions are generated and staged strictly on Master (`data/partitions/`) with a real-time progress bar. Zero network distribution happens at this step.
   * Click **"Distribute Dataset Packages"**: Broadcasts a distribution command to all registered edge nodes. Connected workers purge their previous local dataset (`data/{client_id}/`), download the fresh `.zip` archive over Wi-Fi, extract it, initialize PyTorch DataLoaders, and report back `[READY]` with sample counts. Streamlit displays live progress per worker node.
4. **Tab 3 (Training Telemetry)**:
   * Select the number of FL rounds (e.g., 5) and local CPU epochs (e.g., 2).
   * Click **"Start Federated Distillation"**.
   * Watch live telemetry: local CE training $\rightarrow$ public logit submission $\rightarrow$ consensus computation $\rightarrow$ KL distillation.
5. **Tab 4 (Satellite Inference Playground)**:
   * Select or upload a test satellite image.
   * Observe how individual model predictions compare against the **Common Knowledge Ensemble**.

---

## 11. Step-by-Step Guide: Adding a New Client (Scaling to Client 3, 4, ... N)

The system is built on a **dynamic auto-discovery architecture**. You do **not** need to restart the Master server or modify any code to add new devices.

### Step 1: Connect to Wi-Fi
Connect the new laptop to the exact same Wi-Fi network as the Master Coordinator.

### Step 2: Copy Code & Install Dependencies
1. Clone the repository on the new laptop:
   ```bash
   git clone git@github.com:shivam1162/Federated_Learning.git
   cd Federated_Learning
   pip install -r requirements.txt
   ```

### Step 3: Select an Architecture for the New Device
Choose an architecture suited to the new device's CPU capacity from the registry:
* `shufflenetv2` (Ultra-lightweight, 352k parameters — ideal for budget CPUs)
* `custom` (Compact 4-layer CNN)
* `mobilenetv3` (MobileNetV3-Small)
* `resnet18` (ResNet-18)

### Step 4: Launch the Client Daemon
Run the client script on the new laptop with a **unique `--client_id`**:
```bash
python client/client.py --master_ip <MASTER_IP> --client_id client_3 --model shufflenetv2 --name "Worker-3-ShuffleNet"
```

### Step 5: What Happens Automatically Behind the Scenes
1. **Dynamic Registration Handshake**: The new client registers via `POST /api/register` with its specs.
2. **Instant UI Discovery**: A new hardware card for `client_3` renders immediately in the Streamlit **Cluster Topology** tab.
3. **Partition Allocation**: In Tab 2, set Target Worker Count to `3` and click **"Generate Non-IID Partitions"**. Master stages `client_3.zip` with Domain 2 specialization (`Highway`, `Industrial`, `Residential`).
4. **Dynamic Consensus Expansion**: Master expands the consensus aggregation pool to $K=3$:
   $$\bar{Z}_{\text{public}} = \frac{1}{3} \left( Z_{\text{client\_1}} + Z_{\text{client\_2}} + Z_{\text{client\_3}} \right)$$
5. **Ensemble Participation**: The new model participates in live inference predictions and telemetry curves.
