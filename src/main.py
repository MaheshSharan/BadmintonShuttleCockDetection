"""
Main entry point for shuttlecock detection training.
"""
import json
from pathlib import Path
import torch
import logging

from src.models.shuttlecock_tracker import ShuttlecockTracker
from src.data.data_module import ShuttlecockDataModule
from src.training.unified_trainer import UnifiedTrainer
from src.training.logger import TrainingLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Load configurations
    model_config = load_config('config/model_config.json')
    training_config = load_config('config/training_config.json')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = ShuttlecockTracker(model_config)
    model = model.to(device)
    
    # Initialize data module
    data_module = ShuttlecockDataModule(training_config['data'])
    
    # Initialize logger
    training_logger = TrainingLogger(
        log_dir='logs',
        experiment_name='shuttlecock_detection',
        use_tensorboard=True
    )
    
    # Initialize trainer
    trainer = UnifiedTrainer(
        model=model,
        config=training_config,
        data_module=data_module,
        logger=training_logger,
        device=device
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()

if __name__ == '__main__':
    main()
