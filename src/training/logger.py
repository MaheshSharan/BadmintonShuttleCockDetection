"""
Training logging and monitoring utilities.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TrainingLogger:
    """Comprehensive training logger with metrics tracking and visualization."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
        use_tensorboard: bool = True,
        log_memory: bool = True,
        log_gradients: bool = True
    ):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use tensorboard
            log_memory: Whether to log memory usage
            log_gradients: Whether to log gradient statistics
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.log_memory = log_memory
        self.log_gradients = log_gradients
        
        # Create log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(self.experiment_dir / "tensorboard")
            
        # Setup logging
        self.setup_logging()
        
        # Initialize metrics storage
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.experiment_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ):
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            phase: Training phase ('train', 'val', 'test')
        """
        # Add step and timestamp
        metrics_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Store metrics
        self.metrics_history[phase].append(metrics_entry)
        
        # Log to tensorboard
        if self.use_tensorboard:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{name}", value, step)
                
        # Log to file
        logger.info(f"Step {step} {phase} metrics: {metrics}")
        
    def log_model_gradients(
        self,
        model: torch.nn.Module,
        step: int
    ):
        """
        Log model gradient statistics.
        
        Args:
            model: Model to log gradients for
            step: Current step
        """
        if not self.log_gradients:
            return
            
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats[f"{name}_grad_norm"] = param.grad.norm().item()
                
        if self.use_tensorboard:
            for name, value in grad_stats.items():
                self.writer.add_scalar(f"gradients/{name}", value, step)
                
    def log_memory_stats(
        self,
        memory_stats: Dict[str, int],
        step: int
    ):
        """
        Log memory usage statistics.
        
        Args:
            memory_stats: Dictionary of memory statistics
            step: Current step
        """
        if not self.log_memory:
            return
            
        if self.use_tensorboard:
            for name, value in memory_stats.items():
                self.writer.add_scalar(f"memory/{name}", value, step)
                
    def plot_metrics(
        self,
        metric_name: str,
        phases: Optional[list] = None,
        save: bool = True
    ):
        """
        Plot metrics over time.
        
        Args:
            metric_name: Name of metric to plot
            phases: List of phases to plot
            save: Whether to save the plot
        """
        phases = phases or ['train', 'val']
        plt.figure(figsize=(10, 6))
        
        for phase in phases:
            data = self.metrics_history[phase]
            if not data:
                continue
                
            df = pd.DataFrame(data)
            if metric_name not in df.columns:
                continue
                
            plt.plot(df['step'], df[metric_name], label=f"{phase}")
            
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over time')
        plt.legend()
        
        if save:
            plt.savefig(self.experiment_dir / f"{metric_name}_plot.png")
            plt.close()
        else:
            plt.show()
            
    def save_metrics_history(self):
        """Save metrics history to file."""
        save_path = self.experiment_dir / "metrics_history.json"
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def close(self):
        """Cleanup and close logger."""
        if self.use_tensorboard:
            self.writer.close()
        self.save_metrics_history()
