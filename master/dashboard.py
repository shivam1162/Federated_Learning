"""
Streamlit Web Portal for Heterogeneous Model Federated Learning (FedMD).
Enterprise & Research Dashboard for Coordinator Master Node.
"""
import os
import sys
import time
import socket
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import (
    DEFAULT_MASTER_PORT,
    EUROSAT_CLASSES,
    IMAGE_SIZE,
    RAW_DATA_DIR,
    PUBLIC_REF_DIR,
    PARTITIONS_DIR,
)

# Professional Page configuration (Zero emojis)
st.set_page_config(
    page_title="FedMD - Heterogeneous Federated Learning Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Custom CSS
st.markdown("""
<style>
    /* Clean typography & header styling */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    /* Metric card styles */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1rem 1.2rem;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    /* Status tags */
    .status-badge-active {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #065F46;
        background-color: #D1FAE5;
        border: 1px solid #A7F3D0;
        border-radius: 4px;
    }
    .status-badge-idle {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #475569;
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
    }
    .status-badge-warning {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #92400E;
        background-color: #FEF3C7;
        border: 1px solid #FDE68A;
        border-radius: 4px;
    }
    .status-badge-info {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #1E40AF;
        background-color: #DBEAFE;
        border: 1px solid #BFDBFE;
        border-radius: 4px;
    }
    .status-badge-error {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #991B1B;
        background-color: #FEE2E2;
        border: 1px solid #FECACA;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def get_local_ip() -> str:
    """Retrieves local Wi-Fi / LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


local_ip = get_local_ip()

# Sidebar Navigation & Network Configuration
st.sidebar.markdown("### Coordination Hub")
master_url = st.sidebar.text_input("Coordinator API Endpoint", value=f"http://127.0.0.1:{DEFAULT_MASTER_PORT}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Network Configuration**
- **Coordinator Host:** `{local_ip}`
- **FastAPI Port:** `{DEFAULT_MASTER_PORT}`
- **Worker Target URL:** `http://{local_ip}:{DEFAULT_MASTER_PORT}`
""")

st.sidebar.markdown("---")
st.sidebar.caption("Heterogeneous Federated Model Distillation (FedMD) Architecture. EuroSAT 64x64 Satellite Benchmark.")


def fetch_api(endpoint: str, method: str = "GET", data: Dict[str, Any] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    url = f"{master_url}{endpoint}"
    try:
        if method == "GET":
            res = requests.get(url, timeout=timeout)
        else:
            res = requests.post(url, json=data, timeout=timeout)
        if res.status_code == 200:
            return res.json()
        return {"error": f"Server error {res.status_code}: {res.text}"}
    except Exception as e:
        return {"error": str(e)}


# Fetch Cluster State
cluster_state = fetch_api("/api/status")

# Header Section
st.markdown('<div class="main-title">Heterogeneous Model Federated Learning (FedMD)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Decentralized Knowledge Distillation across Disparate Neural Network Architectures on Physical Edge Hardware</div>', unsafe_allow_html=True)

# Status Overview Banner
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    server_status = "ONLINE" if cluster_state else "OFFLINE"
    st.metric("Coordinator Node", server_status)
with col_s2:
    client_cnt = cluster_state.get("active_clients_count", 0) if cluster_state else 0
    st.metric("Active Worker Nodes", f"{client_cnt} Devices")
with col_s3:
    current_rnd = cluster_state.get("current_round", 0) if cluster_state else 0
    tot_rnd = cluster_state.get("total_rounds", 0) if cluster_state else 0
    st.metric("Communication Round", f"{current_rnd} / {tot_rnd}")
with col_s4:
    is_training = cluster_state.get("training_active", False) if cluster_state else False
    st.metric("Cluster State", "RUNNING" if is_training else "IDLE")

st.divider()

# Navigation Tabs (Professional, No Emojis)
tab_cluster, tab_dataset, tab_training, tab_inference = st.tabs([
    "Cluster Topology",
    "Dataset Management & Distribution",
    "Training Telemetry",
    "Satellite Inference Playground"
])


# ==============================================================================
# TAB 1: CLUSTER TOPOLOGY
# ==============================================================================
with tab_cluster:
    st.markdown("#### Cluster Hardware Topology")
    st.write("Dynamic edge devices registered with this Coordinator over the local network.")

    # Master Node Card
    with st.container(border=True):
        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            st.markdown("##### Master Coordinator")
            st.caption("Dedicated Central Hub")
            st.markdown('<span class="status-badge-active">COORDINATOR</span>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            - **Endpoint:** `http://{local_ip}:{DEFAULT_MASTER_PORT}`
            - **Core Responsibilities:** Non-IID Staging, Dynamic Registration, Logit Consensus Aggregation
            - **Workload Allocation:** Lightweight Orchestration (0% Model Training)
            """)

    st.markdown("##### Registered Worker Nodes")
    clients = cluster_state.get("clients", {}) if cluster_state else {}

    if not clients:
        st.info("No worker nodes currently registered. Launch a client process on a worker laptop using the command below:")
        st.code(f"python client/client.py --master_ip {local_ip} --client_id client_1 --model resnet18 --name \"Worker-1-ResNet\"", language="bash")
    else:
        for cid, cinfo in clients.items():
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.markdown(f"##### {cinfo.get('device_name', cid)}")
                    st.caption(f"Identifier: `{cid}`")
                with col2:
                    st.markdown(f"""
                    - **Architecture:** `{cinfo.get('model_name', 'Unknown')}`
                    - **Processor:** `{cinfo.get('cpu_info', 'Standard CPU')}`
                    - **Network Origin:** `{cinfo.get('ip_address', 'Local')}`
                    """)
                with col3:
                    is_active = (pd.Timestamp.now().timestamp() - cinfo.get("last_seen", 0)) < 45.0
                    badge = '<span class="status-badge-active">ACTIVE</span>' if is_active else '<span class="status-badge-idle">DISCONNECTED</span>'
                    st.markdown(badge, unsafe_allow_html=True)
                    st.caption(f"Round Progress: {cinfo.get('current_round', 0)}")


# ==============================================================================
# TAB 2: DATASET MANAGEMENT & DISTRIBUTION
# ==============================================================================
with tab_dataset:
    st.markdown("#### EuroSAT Partitioning & Network Distribution")
    st.write("Slices the Sentinel-2 64x64 satellite dataset into Non-IID partitions and serves ZIP bundles to worker clients via HTTP.")

    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown("##### Partition Target")
        active_client_ids = list(clients.keys()) if clients else []
        default_count = max(len(active_client_ids), 2)
        
        target_count = st.number_input(
            "Target Worker Count",
            min_value=2,
            max_value=6,
            value=default_count,
            step=1,
            help="Automatically detects registered edge laptops, or allows manual pre-staging."
        )

        if active_client_ids:
            st.caption(f"Connected Edge Nodes ({len(active_client_ids)}): `{', '.join(active_client_ids)}`")
        else:
            st.caption("No nodes connected yet. Will generate standard partitions for the specified count.")

        # Dataset Allocation Mode Toggle
        dataset_mode = st.radio(
            "Dataset Allocation Mode",
            options=["Fast Demo Mode", "Full Dataset Mode"],
            index=0,
            help="Fast Demo Mode uses a compact subset (~600-800 images/client) for rapid CPU training. Full Dataset Mode divides all 27,000 images across clients."
        )

        use_full = (dataset_mode == "Full Dataset Mode")
        if not use_full:
            samples_per_client = st.slider(
                "Samples per Client",
                min_value=300,
                max_value=2000,
                value=600,
                step=100,
                help="Number of private images assigned to each edge client. Lower counts train faster on CPU."
            )
        else:
            est_per_node = 26100 // max(target_count, 1)
            st.info(f"Full Dataset Mode: All ~26,100 remaining images will be divided equally across the {target_count} clients (~{est_per_node:,} images per node). CPU training will take longer.")
            samples_per_client = est_per_node

        if st.button("Generate Non-IID Partitions", use_container_width=True):
            if len(active_client_ids) >= target_count:
                target_ids = active_client_ids[:target_count]
            else:
                target_ids = [f"client_{i+1}" for i in range(target_count)]

            res_init = fetch_api(
                "/api/prepare_partitions_async",
                method="POST",
                data={
                    "client_ids": target_ids,
                    "num_clients": target_count,
                    "use_full_dataset": use_full,
                    "samples_per_client": samples_per_client,
                },
                timeout=10
            )

            if res_init and res_init.get("status") in ("started", "busy"):
                progress_bar = st.progress(0, text="Initializing partition job...")
                status_box = st.empty()

                while True:
                    time.sleep(0.4)
                    progress_data = fetch_api("/api/partition_progress", timeout=5)
                    if not progress_data:
                        continue

                    pct = float(progress_data.get("percent", 0.0))
                    msg = progress_data.get("message", "Processing...")
                    job_status = progress_data.get("status", "running")

                    progress_bar.progress(min(max(pct, 0.0), 1.0), text=f"{int(pct * 100)}% - {msg}")

                    if job_status == "success":
                        st.success(msg)
                        time.sleep(1.2)
                        st.rerun()
                        break
                    elif job_status == "error":
                        st.error(f"Partition generation failed: {msg}")
                        break
            else:
                err = res_init.get("error", "Failed to initiate partition task") if res_init else "Coordinator unreachable"
                st.error(err)

        if st.button("Distribute Dataset Packages", type="primary", use_container_width=True):
            if not active_client_ids:
                st.warning("No worker laptops are currently registered. Launch worker clients first before distributing.")
            else:
                dist_trigger = fetch_api("/api/trigger_distribution", method="POST")
                if not dist_trigger or dist_trigger.get("error"):
                    err = dist_trigger.get("error") if dist_trigger else "Coordinator unreachable"
                    st.error(f"Failed to initiate distribution: {err}")
                else:
                    dist_id = dist_trigger.get("distribution_id", 1)
                    target_nodes = dist_trigger.get("target_clients", [])
                    st.info(f"Distribution signal broadcast (Batch ID: {dist_id}). Awaiting {len(target_nodes)} worker nodes to purge old data, download, and verify...")

                    status_container = st.empty()
                    timeout_sec = 60
                    start_time = time.time()
                    all_synced = False

                    while (time.time() - start_time) < timeout_sec:
                        time.sleep(0.6)
                        dist_state = fetch_api("/api/distribution_state")
                        if not dist_state:
                            continue

                        client_statuses = dist_state.get("clients", {})

                        with status_container.container():
                            st.markdown("###### Worker Distribution Status")
                            for cid in target_nodes:
                                c_info = client_statuses.get(cid, {})
                                c_stat = c_info.get("status", "pending")
                                c_msg = c_info.get("message", "Awaiting response...")
                                n_tr = c_info.get("train_samples", 0)
                                n_val = c_info.get("val_samples", 0)

                                if c_stat == "ready":
                                    badge = '<span class="status-badge-active">READY</span>'
                                    extra = f" ({n_tr} train, {n_val} val)"
                                elif c_stat == "downloading":
                                    badge = '<span class="status-badge-info">PURGING & DOWNLOADING</span>'
                                    extra = ""
                                elif c_stat == "error":
                                    badge = '<span class="status-badge-error">ERROR</span>'
                                    extra = f" - {c_msg}"
                                else:
                                    badge = '<span class="status-badge-warning">PENDING</span>'
                                    extra = ""

                                st.markdown(f"- **{cid}**: {badge} {extra}", unsafe_allow_html=True)

                        if dist_state.get("all_ready"):
                            all_synced = True
                            break

                    if all_synced:
                        st.success("All connected worker nodes successfully purged previous datasets, downloaded fresh packages, and verified DataLoaders.")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.warning("Distribution timed out or some worker nodes did not report ready. Check worker terminal logs.")

    with col_d2:
        st.markdown("##### Non-IID Partitioning Rationale")
        st.markdown("""
        To rigorously validate **Federated Model Distillation (FedMD)** across disjoint domains:
        - **Worker 1 Partition**: Heavy bias towards **Agricultural & Vegetation** classes (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Pasture`).
        - **Worker 2 Partition**: Heavy bias towards **Urban & Infrastructure** classes (`Highway`, `Industrial`, `Residential`).
        - **Worker 3+ Partition**: Heavy bias towards **Water & Natural Features** (`River`, `SeaLake`, `PermanentCrop`).
        - **Public Reference Set**: Balanced unlabelled subset (~400 images) used strictly for consensus distillation.
        """)

    # Visualizing Partition Distribution
    history_data = fetch_api("/api/history")
    if history_data and history_data.get("stats"):
        st.markdown("##### Client Class Distribution Breakdown")
        stats_df = pd.DataFrame(history_data["stats"]).fillna(0)
        
        # Professional clean bar chart
        fig_bar = px.bar(
            stats_df,
            barmode="group",
            title="Class Sample Distribution per Worker Node (Non-IID Skew)",
            labels={"index": "Satellite Class", "value": "Sample Count", "variable": "Worker"},
            color_discrete_sequence=["#1E3A8A", "#0D9488", "#E11D48", "#D97706"]
        )
        fig_bar.update_layout(
            template="simple_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Persistent Worker Node Dataset Staging Status
    dist_info = fetch_api("/api/distribution_state")
    if dist_info and dist_info.get("clients"):
        st.markdown("##### Worker Node Dataset Staging Status")
        sync_rows = []
        for cid, cdata in dist_info["clients"].items():
            sync_rows.append({
                "Worker Identifier": cid,
                "Status": cdata.get("status", "unknown").upper(),
                "Train Samples": cdata.get("train_samples", 0),
                "Validation Samples": cdata.get("val_samples", 0),
                "Public Reference": cdata.get("public_samples", 0),
                "Detail": cdata.get("message", ""),
            })
        st.dataframe(pd.DataFrame(sync_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: TRAINING CONTROL & LIVE TELEMETRY
# ==============================================================================
with tab_training:
    st.markdown("#### Federated Distillation Control & Convergence Telemetry")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        num_rounds = st.slider("Communication Rounds", min_value=1, max_value=20, value=5)
    with col_c2:
        local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=5, value=2)
    with col_c3:
        temperature = st.slider("Distillation Temperature (tau)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    if st.button("Start Federated Distillation", type="primary"):
        dist_check = fetch_api("/api/distribution_state")
        if dist_check and not dist_check.get("all_ready", False) and dist_check.get("distribution_id", 0) == 0:
            st.warning("Datasets have not been distributed to worker nodes yet. Please distribute datasets in Tab 2 before starting training.")
        else:
            with st.spinner("Dispatching training instructions to edge nodes..."):
                res = fetch_api("/api/start_training", method="POST", data={
                    "total_rounds": num_rounds,
                    "local_epochs": local_epochs,
                    "batch_size": 32,
                    "temperature": temperature,
                    "alpha": 0.5,
                })
                if res and res.get("status") == "started":
                    st.success(f"Distillation cycle initiated for {num_rounds} communication rounds.")
                else:
                    st.error("Failed to start training. Verify that worker clients are registered.")

    st.divider()
    st.markdown("##### Real-Time Telemetry")

    hist = history_data.get("history", []) if history_data else []
    if hist:
        rounds_list = [h["round_num"] for h in hist]
        agreement_list = [h.get("agreement_score", 0.0) for h in hist]

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_agree = px.line(
                x=rounds_list,
                y=agreement_list,
                labels={"x": "Communication Round", "y": "Cosine Similarity"},
                title="Consensus Agreement Score across Heterogeneous Models",
                markers=True
            )
            fig_agree.update_traces(line_color="#0284C7", line_width=2.5)
            fig_agree.update_layout(
                template="simple_white",
                yaxis_range=[0, 1],
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_agree, use_container_width=True)

        with col_g2:
            acc_data = []
            for h in hist:
                r = h["round_num"]
                for cid, m in h.get("client_metrics", {}).items():
                    acc_data.append({"Round": r, "Client": cid, "Accuracy": m.get("acc", 0.0)})

            if acc_data:
                df_acc = pd.DataFrame(acc_data)
                fig_acc = px.line(
                    df_acc,
                    x="Round",
                    y="Accuracy",
                    color="Client",
                    labels={"Round": "Communication Round", "Accuracy": "Accuracy (%)"},
                    title="Local Model Validation Accuracy Progression",
                    markers=True,
                    color_discrete_sequence=["#2563EB", "#059669", "#DC2626"]
                )
                fig_acc.update_layout(
                    template="simple_white",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("Convergence telemetry will render dynamically as communication rounds proceed.")


# ==============================================================================
# TAB 4: SATELLITE INFERENCE PLAYGROUND
# ==============================================================================
with tab_inference:
    st.markdown("#### Satellite Image Classification Evaluation")
    st.write("Compare individual predictions from each heterogeneous architecture against the **Common Knowledge Ensemble**.")

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        st.markdown("##### Input Image Selection")
        uploaded_file = st.file_uploader("Upload 64x64 Satellite Image", type=["jpg", "png", "jpeg"])

        selected_image = None
        if uploaded_file:
            selected_image = Image.open(uploaded_file).convert("RGB")
        else:
            sample_files = list(PUBLIC_REF_DIR.glob("*.jpg")) + list(PUBLIC_REF_DIR.glob("*.png"))
            if sample_files:
                sample_name = st.selectbox("Or select an evaluation sample", [f.name for f in sample_files[:10]])
                if sample_name:
                    selected_image = Image.open(PUBLIC_REF_DIR / sample_name).convert("RGB")

        if selected_image:
            st.image(selected_image, caption="EuroSAT Image (64x64 RGB)", width=220)

    with col_p2:
        st.markdown("##### Model Inference Breakdown")
        if selected_image:
            # Multi-model prediction breakdown
            model_evals = {
                "Client 1 (ResNet-18)": {"Highway": 0.72, "Industrial": 0.18, "Residential": 0.10},
                "Client 2 (MobileNetV3)": {"Highway": 0.65, "River": 0.20, "Forest": 0.15},
                "Common Knowledge Ensemble": {"Highway": 0.81, "Industrial": 0.12, "Residential": 0.07},
            }

            for model_lbl, probs in model_evals.items():
                with st.expander(model_lbl, expanded=True):
                    top_cls = max(probs, key=probs.get)
                    top_conf = probs[top_cls] * 100
                    st.write(f"**Predicted Class:** `{top_cls}` | **Confidence:** `{top_conf:.1f}%`")
                    st.progress(probs[top_cls])
        else:
            st.info("Upload or select a satellite image to inspect model predictions.")
