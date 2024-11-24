"""
Batch Processing Module
Implements efficient batch processing strategies for the shuttlecock detection model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch.cuda.amp import autocast, GradScaler
import numpy as np


class BatchProcessor:
    """
    Handles efficient batch processing with dynamic batching, gradient accumulation,
    and mixed precision training.
    """
    
    def __init__(
        self,
        base_batch_size: int = 32,
        accumulation_steps: int = 4,
        mixed_precision: bool = True,
        dynamic_batching: bool = True,
        memory_fraction: float = 0.9,
    ):
        """
        Initialize batch processor with optimization settings.
        
        Args:
            base_batch_size: Base batch size for training
            accumulation_steps: Number of gradient accumulation steps
            mixed_precision: Whether to use mixed precision training
            dynamic_batching: Whether to use dynamic batch sizing
            memory_fraction: Target GPU memory utilization fraction
        """
        self.base_batch_size = base_batch_size
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.dynamic_batching = dynamic_batching
        self.memory_fraction = memory_fraction
        
        self.scaler = GradScaler() if mixed_precision else None
        self.current_batch_size = base_batch_size
        self.current_step = 0
        
    def process_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process a single batch with optimizations.
        
        Args:
            model: The neural network model
            batch: Dictionary containing batch data
            optimizer: The optimizer
            training: Whether in training mode
            
        Returns:
            Tuple of (loss, metrics)
        """
        if training and self.dynamic_batching:
            self._adjust_batch_size(model)
        
        # Split batch if needed
        sub_batches = self._split_batch(batch)
        total_loss = 0
        accumulated_metrics = {}
        
        for sub_batch in sub_batches:
            with autocast(enabled=self.mixed_precision):
                # Forward pass
                outputs = model(sub_batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / len(sub_batches)
                
            if training:
                # Backward pass with gradient scaling
                if self.mixed_precision:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Gradient accumulation
                if (self.current_step + 1) % self.accumulation_steps == 0:
                    if self.mixed_precision:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss.item()
            self._accumulate_metrics(accumulated_metrics, outputs)
            
            self.current_step += 1
        
        # Average metrics
        metrics = self._average_metrics(accumulated_metrics, len(sub_batches))
        metrics['loss'] = total_loss / len(sub_batches)
        
        return total_loss / len(sub_batches), metrics
    
    def _split_batch(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split batch into sub-batches based on current batch size."""
        if not self.dynamic_batching:
            return [batch]
        
        total_size = next(iter(batch.values())).size(0)
        n_splits = (total_size + self.current_batch_size - 1) // self.current_batch_size
        
        sub_batches = []
        for i in range(n_splits):
            start_idx = i * self.current_batch_size
            end_idx = min((i + 1) * self.current_batch_size, total_size)
            
            sub_batch = {
                k: v[start_idx:end_idx] for k, v in batch.items()
            }
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    def _adjust_batch_size(self, model: nn.Module):
        """Dynamically adjust batch size based on GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        try:
            # Get current GPU memory usage
            torch.cuda.synchronize()
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            # Adjust batch size based on memory usage
            if current_memory > self.memory_fraction:
                self.current_batch_size = max(1, self.current_batch_size // 2)
            elif current_memory < self.memory_fraction * 0.7:
                self.current_batch_size = min(
                    self.base_batch_size,
                    self.current_batch_size * 2
                )
        except Exception as e:
            print(f"Warning: Failed to adjust batch size: {e}")
    
    def _accumulate_metrics(
        self,
        accumulated_metrics: Dict[str, float],
        outputs: Dict[str, torch.Tensor]
    ):
        """Accumulate metrics from batch outputs."""
        if not isinstance(outputs, dict):
            return
        
        for k, v in outputs.items():
            if k != 'loss' and isinstance(v, (torch.Tensor, float)):
                if k not in accumulated_metrics:
                    accumulated_metrics[k] = 0
                accumulated_metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
    
    def _average_metrics(
        self,
        accumulated_metrics: Dict[str, float],
        n_batches: int
    ) -> Dict[str, float]:
        """Average accumulated metrics over number of batches."""
        return {
            k: v / n_batches for k, v in accumulated_metrics.items()
        }
