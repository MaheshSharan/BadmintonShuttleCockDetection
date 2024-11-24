"""
Script to start training monitoring dashboards.
"""
import json
from pathlib import Path
from visualization.training_dashboard import TrainingDashboard
import threading
import subprocess

def start_tensorboard():
    """Start TensorBoard server."""
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(["tensorboard", "--logdir", str(log_dir), "--port", "6006"])
    print("TensorBoard started at http://localhost:6006")

def start_training_dashboard():
    """Start web-based training dashboard."""
    # Load dashboard config
    config_path = Path("config/visualization_config.json")
    if not config_path.exists():
        config = {
            "update_interval": 1.0,  # seconds
            "max_points": 1000,
            "metrics": ["loss", "accuracy", "learning_rate", "gpu_util"],
            "plot_layout": {
                "height": 800,
                "template": "plotly_dark"
            }
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    # Create log directory
    log_dir = Path("outputs/dashboard_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Start dashboard
    dashboard = TrainingDashboard(
        config=config,
        log_dir=str(log_dir),
        port=8050,
        dark_mode=True
    )
    dashboard.start()
    print("Training Dashboard started at http://localhost:8050")

if __name__ == "__main__":
    print("Starting monitoring dashboards...")
    
    # Start TensorBoard in a separate thread
    tensorboard_thread = threading.Thread(target=start_tensorboard)
    tensorboard_thread.start()
    
    # Start Training Dashboard
    start_training_dashboard()
