"""
Memory Optimization Module
Implements memory management and optimization strategies for efficient model training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import gc
import numpy as np
from contextlib import contextmanager
import functools
import psutil


class MemoryOptimizer:
    """
    Handles memory optimization strategies including gradient checkpointing,
    memory caching, and efficient tensor management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        enable_checkpoint: bool = True,
        cache_size_limit: int = 1024,  # MB
        pin_memory: bool = True,
        optimize_cuda_cache: bool = True
    ):
        """
        Initialize memory optimizer with specified settings.
        
        Args:
            model: The neural network model
            enable_checkpoint: Whether to enable gradient checkpointing
            cache_size_limit: Maximum cache size in MB
            pin_memory: Whether to pin memory for faster GPU transfer
            optimize_cuda_cache: Whether to optimize CUDA memory cache
        """
        self.model = model
        self.enable_checkpoint = enable_checkpoint
        self.cache_size_limit = cache_size_limit * 1024 * 1024  # Convert to bytes
        self.pin_memory = pin_memory
        self.optimize_cuda_cache = optimize_cuda_cache
        
        self.tensor_cache = {}
        self.cached_memory = 0
        
        self._setup_model_optimization()
    
    def _setup_model_optimization(self):
        """Configure model optimization settings."""
        if self.enable_checkpoint:
            self._enable_gradient_checkpointing()
        
        if self.optimize_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        def wrap_forward(module, forward):
            @functools.wraps(forward)
            def wrapped_forward(*args, **kwargs):
                def custom_forward(*inputs):
                    return forward(*inputs)
                return torch.utils.checkpoint.checkpoint(custom_forward, *args, **kwargs)
            return wrapped_forward

        # Apply checkpointing to backbone
        if hasattr(self.model, 'backbone'):
            original_forward = self.model.backbone.forward
            self.model.backbone.forward = wrap_forward(self.model.backbone, original_forward)

        # Apply checkpointing to neck
        if hasattr(self.model, 'neck'):
            original_forward = self.model.neck.forward
            self.model.neck.forward = wrap_forward(self.model.neck, original_forward)

        # Apply checkpointing to detection head
        if hasattr(self.model, 'detection_head'):
            original_forward = self.model.detection_head.forward
            self.model.detection_head.forward = wrap_forward(self.model.detection_head, original_forward)
    
    @contextmanager
    def optimize_memory_context(self):
        """Context manager for temporary memory optimization."""
        try:
            if self.optimize_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            yield
            
        finally:
            if self.optimize_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                final_memory = torch.cuda.memory_allocated()
                if final_memory > initial_memory * 1.5:  # Significant memory increase
                    self._cleanup_cache()
    
    def optimize_tensors(
        self,
        tensors: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Optimize tensor memory usage.
        
        Args:
            tensors: Input tensors to optimize
            
        Returns:
            Optimized tensors
        """
        if isinstance(tensors, torch.Tensor):
            return self._optimize_single_tensor(tensors)
        elif isinstance(tensors, dict):
            return {k: self._optimize_single_tensor(v) for k, v in tensors.items()}
        elif isinstance(tensors, list):
            return [self._optimize_single_tensor(t) for t in tensors]
        return tensors
    
    def _optimize_single_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize a single tensor's memory usage."""
        if not isinstance(tensor, torch.Tensor):
            return tensor
        
        # Check cache for similar tensors
        tensor_key = (tensor.shape, tensor.dtype, tensor.device)
        if tensor_key in self.tensor_cache:
            cached_tensor = self.tensor_cache[tensor_key]
            if cached_tensor.shape == tensor.shape:
                cached_tensor.copy_(tensor)
                return cached_tensor
        
        # Pin memory if needed
        if self.pin_memory and tensor.device.type == 'cpu':
            tensor = tensor.pin_memory()
        
        # Update cache
        tensor_size = tensor.element_size() * tensor.nelement()
        if self.cached_memory + tensor_size <= self.cache_size_limit:
            self.tensor_cache[tensor_key] = tensor
            self.cached_memory += tensor_size
        
        return tensor
    
    def _cleanup_cache(self):
        """Clean up tensor cache to free memory."""
        self.tensor_cache.clear()
        self.cached_memory = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            'cpu_memory': psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_cached': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / (1024 * 1024)
            })
        
        return stats
    
    def optimize_model_memory(self):
        """Apply memory optimization techniques to the model."""
        # Convert model to half precision where beneficial
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Keep batch norm in float32 for stability
                module.float()
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                # Convert compute-heavy layers to half precision
                module.half()
        
        # Enable gradient checkpointing for memory efficiency
        if self.enable_checkpoint:
            self._enable_gradient_checkpointing()
        
        # Clear unused memory
        self._cleanup_cache()
        
        return self.model
