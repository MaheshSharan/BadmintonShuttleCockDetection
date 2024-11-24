"""
Unified trainer module for shuttlecock detection model.
"""
import torch
import torch.nn as nn
from torch.cuda import amp
from pathlib import Path
from typing import Dict, Union
from tqdm import tqdm
import logging
import time
import psutil
from .distributed import DistributedTrainer
from .resource_manager import ResourceManager
from .checkpointing import CheckpointManager

logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """
    Unified trainer for shuttlecock detection model.
    Handles training, validation, logging, and visualization.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        data_module: 'ShuttlecockDataModule',
        logger: 'TrainingLogger',
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            data_module: Data module
            logger: Training logger
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.data_module = data_module
        self.logger = logger
        self.device = device
        
        # Get data loaders
        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup learning rate scheduler
        num_training_steps = len(self.train_loader) * config['training']['num_epochs']
        num_warmup_steps = len(self.train_loader) * config['training']['warmup_epochs']
        self.scheduler = self._create_scheduler(num_training_steps, num_warmup_steps)
        
        # Setup distributed training
        if config['distributed']['world_size'] and torch.cuda.device_count() > 1:
            self.distributed = DistributedTrainer(
                model=self.model,
                world_size=config['distributed']['world_size'],
                backend=config['distributed']['backend'],
                find_unused_parameters=config['distributed']['find_unused_parameters']
            )
            if config['distributed']['sync_bn']:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        else:
            self.distributed = None
        
        # Setup resource management
        self.resource_manager = ResourceManager(
            model=self.model,
            optimizer=self.optimizer,
            gradient_accumulation_steps=config['optimization']['gradient_accumulation_steps'],
            max_grad_norm=config['optimization']['gradient_clipping'],
            mixed_precision=config['optimization']['mixed_precision']
        )
        
        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(
            save_dir=Path(config['checkpointing']['save_dir']),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            max_checkpoints=config['checkpointing']['max_checkpoints']
        )

    def train(self):
        """Train the model."""
        logger.info("=== Starting Training ===")
        logger.info(f"Total epochs: {self.config['training']['num_epochs']}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Training sequences: {len(self.train_loader.dataset)}")
        logger.info(f"Validation sequences: {len(self.val_loader.dataset)}")
        logger.info("========================")
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            epoch_start = time.time()
            
            # Training epoch
            self.model.train()
            train_loss = 0
            train_metrics = {}
            
            # Create progress bar for this epoch
            with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    try:
                        # Memory usage info
                        if batch_idx == 0:
                            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                            logger.info(f"Memory usage: {memory:.2f} MB")
                        
                        # Move batch to device
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        
                        # Forward pass
                        with amp.autocast(enabled=self.config['optimization']['mixed_precision']):
                            outputs = self.model(batch)
                            loss = self._compute_loss(outputs, batch)
                        
                        # Backward pass with gradient accumulation
                        loss = loss / self.config['optimization']['gradient_accumulation_steps']
                        self.resource_manager.backward_step(loss)
                        
                        if (batch_idx + 1) % self.config['optimization']['gradient_accumulation_steps'] == 0:
                            self.resource_manager.optimizer_step()
                            if self.scheduler is not None:
                                self.scheduler.step()
                        
                        # Update metrics
                        train_loss += loss.item()
                        metrics = self._compute_metrics(outputs, batch)
                        for k, v in metrics.items():
                            train_metrics[k] = train_metrics.get(k, 0) + v
                        
                        # Update progress bar
                        avg_loss = train_loss / (batch_idx + 1)
                        avg_metrics = {k: v / (batch_idx + 1) for k, v in train_metrics.items()}
                        
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            **{k: f'{v:.4f}' for k, v in avg_metrics.items()},
                            'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                        })
                        
                        # Log every N batches
                        if (batch_idx + 1) % self.config['logging']['log_frequency'] == 0:
                            logger.info(
                                f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                                f"Loss: {avg_loss:.4f} "
                                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                            )
                            
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        raise e
            
            # Validation
            val_loss, val_metrics = self.validate()
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(
                f"\nEpoch {epoch+1} Summary:"
                f"\n - Time: {epoch_time:.2f}s"
                f"\n - Train Loss: {avg_loss:.4f}"
                f"\n - Val Loss: {val_loss:.4f}"
                f"\n - Train Metrics: {avg_metrics}"
                f"\n - Val Metrics: {val_metrics}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    metrics={'val_loss': val_loss, **val_metrics}
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
    def _create_scheduler(self, num_training_steps: int, num_warmup_steps: int):
        """Create learning rate scheduler."""
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load checkpoint."""
        self.checkpoint_manager.load(checkpoint_path)
        
    def validate(self):
        """Run validation."""
        self.model.eval()
        val_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch['frames'])
                loss_dict = self.criterion(predictions, batch)
                
                val_metrics.append(loss_dict)
                
        # Compute mean metrics
        mean_metrics = {}
        for metric in val_metrics[0].keys():
            values = [m[metric].item() if isinstance(m[metric], torch.Tensor) else m[metric]
                     for m in val_metrics]
            mean_metrics[f'val_{metric}'] = np.mean(values)
        
        # Update metrics
        self.metrics.update(mean_metrics, 0)
        
        # Save checkpoint if best
        if self.checkpoint_manager.is_best(mean_metrics):
            self.checkpoint_manager.save(
                epoch=self.current_epoch,
                metric_value=mean_metrics[self.config['checkpointing']['metric_name']]
            )
        
        return mean_metrics
