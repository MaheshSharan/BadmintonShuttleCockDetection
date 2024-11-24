"""
Resource management for efficient training.
"""
import torch
import torch.cuda.amp as amp
from typing import Dict, Optional, Any
import logging
from contextlib import contextmanager
import gc

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages training resources and optimization strategies."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        memory_efficient: bool = True
    ):
        """
        Initialize resource manager.
        
        Args:
            model: Model to manage resources for
            optimizer: Optimizer to manage
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            mixed_precision: Whether to use mixed precision training
            memory_efficient: Whether to use memory efficient features
        """
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.memory_efficient = memory_efficient
        
        # Initialize mixed precision scaler
        self.scaler = amp.GradScaler() if mixed_precision else None
        self.current_step = 0
        
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.mixed_precision:
            with amp.autocast():
                yield
        else:
            yield
            
    def backward_pass(self, loss: torch.Tensor):
        """
        Perform backward pass with gradient accumulation.
        
        Args:
            loss: Loss to backpropagate
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.gradient_accumulation_steps
        
        if self.mixed_precision:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        self.current_step += 1
        
        # Only update on accumulation steps
        if self.current_step % self.gradient_accumulation_steps == 0:
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
    def memory_cleanup(self):
        """Perform memory cleanup."""
        if self.memory_efficient:
            gc.collect()
            torch.cuda.empty_cache()
            
    @contextmanager
    def train_step(self):
        """Context manager for a training step."""
        try:
            with self.autocast_context():
                yield
        finally:
            if self.memory_efficient:
                self.memory_cleanup()
                
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        if not torch.cuda.is_available():
            return {}
            
        return {
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
            'max_cached': torch.cuda.max_memory_reserved()
        }
