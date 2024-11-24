"""
Performance monitoring module for tracking system resources and training metrics.
"""
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors system resources and training performance metrics."""
    
    def __init__(self, log_dir: str, update_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            log_dir: Directory to save logs
            update_interval: How often to update metrics (seconds)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval
        
        # Initialize metrics storage
        self.metrics = {
            'gpu_utilization': [],
            'gpu_memory': [],
            'cpu_utilization': [],
            'ram_usage': [],
            'batch_time': [],
            'throughput': []
        }
        
        # CUDA metrics
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 0
            
    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU utilization and memory usage."""
        if not self.cuda_available:
            return {'utilization': 0.0, 'memory': 0.0}
            
        stats = {
            'utilization': [],
            'memory': []
        }
        
        for i in range(self.num_gpus):
            # Get GPU utilization
            with torch.cuda.device(i):
                utilization = torch.cuda.utilization()
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                
            stats['utilization'].append(utilization)
            stats['memory'].append(memory_allocated)
            
        return {
            'utilization': np.mean(stats['utilization']),
            'memory': np.mean(stats['memory'])
        }
        
    def get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU utilization and RAM usage."""
        return {
            'cpu_util': psutil.cpu_percent(),
            'ram_usage': psutil.virtual_memory().percent
        }
        
    def update(self, batch_time: Optional[float] = None, batch_size: Optional[int] = None):
        """
        Update performance metrics.
        
        Args:
            batch_time: Time taken for last batch
            batch_size: Size of last batch
        """
        # Get GPU stats
        gpu_stats = self.get_gpu_stats()
        self.metrics['gpu_utilization'].append(gpu_stats['utilization'])
        self.metrics['gpu_memory'].append(gpu_stats['memory'])
        
        # Get CPU stats
        cpu_stats = self.get_cpu_stats()
        self.metrics['cpu_utilization'].append(cpu_stats['cpu_util'])
        self.metrics['ram_usage'].append(cpu_stats['ram_usage'])
        
        # Update batch metrics if provided
        if batch_time is not None:
            self.metrics['batch_time'].append(batch_time)
            if batch_size is not None:
                throughput = batch_size / batch_time
                self.metrics['throughput'].append(throughput)
                
        # Trim metrics to keep memory usage bounded
        max_points = 1000
        for key in self.metrics:
            if len(self.metrics[key]) > max_points:
                self.metrics[key] = self.metrics[key][-max_points:]
                
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get current metrics."""
        return self.metrics.copy()
