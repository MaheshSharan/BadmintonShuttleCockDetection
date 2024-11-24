"""
Data module for shuttlecock detection.
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from .shuttlecock_dataset import ShuttlecockDataset
from .transforms import ShuttlecockTransform, collate_fn

logger = logging.getLogger(__name__)

class ShuttlecockDataModule:
    """Data module for shuttlecock detection."""
    
    def __init__(self, config: Dict):
        """
        Initialize data module.
        
        Args:
            config: Data configuration dictionary containing:
                - train_path: Path to training data
                - val_path: Path to validation data
                - batch_size: Batch size
                - num_workers: Number of workers for data loading
                - sequence_length: Number of frames per sequence
                - frame_size: Target frame size (height, width)
                - prefetch_factor: Number of batches to prefetch
        """
        self.train_path = Path(config['train_path'])
        self.val_path = Path(config['val_path'])
        self.batch_size = config.get('batch_size', 8)
        self.num_workers = config.get('num_workers', 4)
        self.sequence_length = config.get('sequence_length', 16)
        self.frame_size = tuple(config.get('frame_size', [720, 1280]))
        self.prefetch_factor = config.get('prefetch_factor', 2)
        
        logger.info(f"Initializing data module with:")
        logger.info(f"  Train path: {self.train_path}")
        logger.info(f"  Val path: {self.val_path}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Frame size: {self.frame_size}")
        
        # Create transforms
        self.train_transform = ShuttlecockTransform(train=True, size=self.frame_size)
        self.val_transform = ShuttlecockTransform(train=False, size=self.frame_size)
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.setup()
    
    def setup(self):
        """Set up train and validation datasets."""
        self.train_dataset = ShuttlecockDataset(
            root_dir=self.train_path,
            sequence_length=self.sequence_length,
            transform=self.train_transform,
            target_size=self.frame_size
        )
        
        self.val_dataset = ShuttlecockDataset(
            root_dir=self.val_path,
            sequence_length=self.sequence_length,
            transform=self.val_transform,
            target_size=self.frame_size
        )
        
        logger.info(f"Found {len(self.train_dataset)} training sequences")
        logger.info(f"Found {len(self.val_dataset)} validation sequences")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=collate_fn
        )
