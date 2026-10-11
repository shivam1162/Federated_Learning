"""
Master Coordinator All-in-One Runner.
Launches both the FastAPI Coordination Server (port 8000) and the Streamlit Web Dashboard (port 8501).
"""
import sys
import time
import socket
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import DEFAULT_MASTER_PORT, DEFAULT_STREAMLIT_PORT


def get_local_ip() -> str:
    """Finds the local LAN / Wi-Fi IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    local_ip = get_local_ip()
    print("=" * 70)
    print("      FedMD: HETEROGENEOUS FEDERATED LEARNING COORDINATOR")
    print("=" * 70)
    print(f"Master IP Address:           {local_ip}")
    print(f"FastAPI Coordination Port:   {DEFAULT_MASTER_PORT}")
    print(f"Streamlit Web Dashboard:     http://{local_ip}:{DEFAULT_STREAMLIT_PORT}")
    print("=" * 70)
    print("\nCommand for Worker Laptop 1:")
    print(f"  python client/client.py --master_ip {local_ip} --client_id client_1 --model resnet18 --name \"Worker-1-ResNet\"\n")
    print("Command for Worker Laptop 2:")
    print(f"  python client/client.py --master_ip {local_ip} --client_id client_2 --model mobilenetv3 --name \"Worker-2-MobileNet\"\n")
    print("=" * 70)
    print("Starting FastAPI and Streamlit services...\n")

    # Start FastAPI server
    fastapi_cmd = [
        sys.executable, "-m", "uvicorn",
        "master.server:app",
        "--host", "0.0.0.0",
        "--port", str(DEFAULT_MASTER_PORT),
        "--reload",
        "--log-level", "info",
    ]
    fastapi_proc = subprocess.Popen(fastapi_cmd)

    # Allow FastAPI a moment to bind
    time.sleep(1.5)

    # Start Streamlit dashboard
    streamlit_cmd = [
        sys.executable, "-m", "streamlit",
        "run", "master/dashboard.py",
        "--server.port", str(DEFAULT_STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ]
    streamlit_proc = subprocess.Popen(streamlit_cmd)

    try:
        while True:
            time.sleep(1)
            # Check if either crashed
            if fastapi_proc.poll() is not None:
                print("FastAPI server exited unexpectedly.")
                break
            if streamlit_proc.poll() is not None:
                print("Streamlit exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nShutting down FedMD Coordinator services...")
    finally:
        fastapi_proc.terminate()
        streamlit_proc.terminate()
        fastapi_proc.wait()
        streamlit_proc.wait()
        print("Master services terminated successfully.")


if __name__ == "__main__":
    main()
