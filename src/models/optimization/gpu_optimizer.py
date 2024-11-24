"""
GPU Optimization Module
Implements GPU utilization and optimization strategies for efficient model training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import numpy as np
from torch.cuda import amp
import threading
from queue import Queue
from contextlib import contextmanager


class GPUOptimizer:
    """
    Handles GPU optimization strategies including multi-GPU training,
    mixed precision, and efficient memory management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_gpus: Optional[int] = None,
        mixed_precision: bool = True,
        memory_efficient: bool = True,
        prefetch_factor: int = 2,
        async_loading: bool = True
    ):
        """
        Initialize GPU optimizer with specified settings.
        
        Args:
            model: The neural network model
            num_gpus: Number of GPUs to use (None for auto-detection)
            mixed_precision: Whether to use mixed precision training
            memory_efficient: Whether to use memory-efficient optimizations
            prefetch_factor: Number of batches to prefetch
            async_loading: Whether to use asynchronous data loading
        """
        self.model = model
        self.mixed_precision = mixed_precision
        self.memory_efficient = memory_efficient
        self.prefetch_factor = prefetch_factor
        self.async_loading = async_loading
        
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.scaler = amp.GradScaler() if mixed_precision else None
        self.data_queue = Queue(maxsize=prefetch_factor) if async_loading else None
        self.prefetch_thread = None
        
        self._setup_gpu_optimization()
    
    def _setup_gpu_optimization(self):
        """Configure GPU optimization settings."""
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return
        
        # Set up multi-GPU if available
        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        
        # Move model to GPU
        self.model = self.model.cuda()
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        if self.memory_efficient:
            # Enable memory-efficient optimizations
            torch.cuda.set_per_process_memory_fraction(0.9)  # Reserve some memory
            torch.cuda.empty_cache()
    
    def optimize_input(
        self,
        batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Optimize input data for GPU processing.
        
        Args:
            batch: Input batch data
            
        Returns:
            Optimized batch data
        """
        if isinstance(batch, torch.Tensor):
            return self._optimize_tensor(batch)
        elif isinstance(batch, dict):
            return {k: self._optimize_tensor(v) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        return batch
    
    def _optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize a single tensor for GPU processing."""
        if not tensor.is_cuda and torch.cuda.is_available():
            tensor = tensor.cuda(non_blocking=True)
        
        if self.mixed_precision and tensor.dtype == torch.float32:
            tensor = tensor.half()
        
        return tensor
    
    @staticmethod
    def get_gpu_stats() -> Dict[str, float]:
        """Get current GPU utilization statistics."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'memory_allocated': torch.cuda.memory_allocated() / (1024 * 1024),
            'memory_cached': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_memory_allocated': torch.cuda.max_memory_allocated() / (1024 * 1024)
        }
    
    def start_prefetch(self, data_loader: torch.utils.data.DataLoader):
        """Start asynchronous data prefetching."""
        if not self.async_loading:
            return
        
        def prefetch_worker():
            try:
                for batch in data_loader:
                    optimized_batch = self.optimize_input(batch)
                    self.data_queue.put(optimized_batch)
            finally:
                self.data_queue.put(None)  # Signal end of data
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
    
    def get_next_batch(self) -> Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Get next prefetched batch."""
        if not self.async_loading:
            return None
        
        batch = self.data_queue.get()
        if batch is None:  # End of data
            self.prefetch_thread = None
        return batch
    
    def synchronize(self):
        """Synchronize GPU operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def cleanup(self):
        """Clean up GPU resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
            self.prefetch_thread = None
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        with amp.autocast(enabled=self.mixed_precision):
            yield
    
    def backward_pass(self, loss: torch.Tensor):
        """Perform backward pass with mixed precision if enabled."""
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Perform optimizer step with mixed precision if enabled."""
        if self.mixed_precision:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
