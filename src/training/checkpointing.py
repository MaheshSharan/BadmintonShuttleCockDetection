"""
Model checkpointing and state management.
"""
import os
import torch
from pathlib import Path
from typing import Dict, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpointing and state restoration."""
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_checkpoints: int = 5,
        save_best_only: bool = True,
        metric_name: str = "val_loss",
        metric_mode: str = "min"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            scheduler: Optional learning rate scheduler
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only best checkpoints
            metric_name: Metric to track for best checkpoint
            metric_mode: "min" or "max" for metric optimization
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        self.best_metric = float('inf') if metric_mode == "min" else float('-inf')
        self.checkpoints = []
        
    def save(
        self,
        epoch: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            extra_state: Additional state to save
            
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            logger.warning(f"Metric {self.metric_name} not found in metrics")
            return None
            
        is_best = (
            (self.metric_mode == "min" and metric_value < self.best_metric) or
            (self.metric_mode == "max" and metric_value > self.best_metric)
        )
        
        if self.save_best_only and not is_best:
            return None
            
        if is_best:
            self.best_metric = metric_value
            
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if extra_state:
            checkpoint['extra_state'] = extra_state
            
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"checkpoint_epoch{epoch}_{timestamp}.pt"
        save_path = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        self.checkpoints.append(save_path)
        
        # Remove old checkpoints if needed
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_checkpoint.unlink()
            
        return save_path
        
    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint, loads latest if None
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Dictionary containing loaded state
        """
        if checkpoint_path is None:
            if not self.checkpoints:
                raise ValueError("No checkpoints found")
            checkpoint_path = self.checkpoints[-1]
            
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if requested
        if (load_scheduler and self.scheduler is not None and
            'scheduler_state_dict' in checkpoint):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        
        return checkpoint
