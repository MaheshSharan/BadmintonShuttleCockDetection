"""
Training metrics and logging utilities.
"""
import torch
from typing import Dict, Any
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracks and logs training metrics."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = 'logs'):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        self.epoch = 0
        
        # Initialize metric storage
        self.current_metrics = {}
        self.best_metrics = {}
        self.metric_history = []
        
        # Training stats
        self.epoch_start_time = None
        self.batch_start_time = None
        self.samples_seen = 0
        
        # Save config
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.current_metrics = {}
        self.samples_seen = 0
        
    def end_epoch(self):
        """Process the end of an epoch."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Compute epoch metrics
        epoch_metrics = {
            name: np.mean(values) if values else 0
            for name, values in self.current_metrics.items()
        }
        
        # Update best metrics
        for name, value in epoch_metrics.items():
            if name not in self.best_metrics or value < self.best_metrics[name]:
                self.best_metrics[name] = value
                
        # Log to TensorBoard
        for name, value in epoch_metrics.items():
            self.writer.add_scalar(f'epoch/{name}', value, self.epoch)
            
        # Log epoch summary
        logger.info(
            f'Epoch {self.epoch}: '
            f'time: {epoch_time:.2f}s, '
            f'samples: {self.samples_seen}, '
            + ', '.join(f'{k}: {v:.4f}' for k, v in epoch_metrics.items())
        )
        
        # Save metrics history
        self.metric_history.append({
            'epoch': self.epoch,
            'time': epoch_time,
            'samples': self.samples_seen,
            **epoch_metrics
        })
        
        # Save metrics to file
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump({
                'history': self.metric_history,
                'best': self.best_metrics
            }, f, indent=4)
            
        self.epoch += 1
        
    def update(self, metrics: Dict[str, torch.Tensor], batch_size: int):
        """Update metrics with new batch results."""
        # Update sample count
        self.samples_seen += batch_size
        
        # Update metrics
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            if name not in self.current_metrics:
                self.current_metrics[name] = []
            self.current_metrics[name].append(value)
            
        # Log to TensorBoard
        if self.step % self.config['logging']['log_frequency'] == 0:
            for name, value in metrics.items():
                self.writer.add_scalar(f'batch/{name}', value, self.step)
                
        self.step += 1
        
    def should_stop_early(self) -> bool:
        """Check if training should stop based on early stopping criteria."""
        if len(self.metric_history) < self.config['training']['early_stopping_patience']:
            return False
            
        patience = self.config['training']['early_stopping_patience']
        metric_name = self.config['checkpointing']['metric_name']
        metric_mode = self.config['checkpointing']['metric_mode']
        
        recent_metrics = [
            epoch[metric_name]
            for epoch in self.metric_history[-patience:]
        ]
        
        if metric_mode == 'min':
            best_metric = min(recent_metrics)
            return best_metric == recent_metrics[0]
        else:
            best_metric = max(recent_metrics)
            return best_metric == recent_metrics[0]
            
    def close(self):
        """Clean up resources."""
        self.writer.close()
        
    def log_resources(self):
        """Log resource usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            self.writer.add_scalar('resources/gpu_memory_allocated_mb', memory_allocated, self.step)
            self.writer.add_scalar('resources/gpu_memory_cached_mb', memory_cached, self.step)
            
    def log_model_gradients(self, model: torch.nn.Module):
        """Log model gradient statistics."""
        if not self.config['logging']['log_gradients']:
            return
            
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.writer.add_scalar(f'gradients/{name}_norm', grad_norm, self.step)
