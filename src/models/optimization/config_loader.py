"""
Configuration Loader for Optimization Modules
Handles loading and validation of optimization configuration parameters.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class OptimizationConfigLoader:
    """Loads and validates optimization configuration parameters."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def get_batch_processor_config(self) -> Dict[str, Any]:
        """Get batch processor configuration."""
        return self.config.get('batch_processor', {})
    
    def get_memory_optimizer_config(self) -> Dict[str, Any]:
        """Get memory optimizer configuration."""
        return self.config.get('memory_optimizer', {})
    
    def get_gpu_optimizer_config(self) -> Dict[str, Any]:
        """Get GPU optimizer configuration."""
        return self.config.get('gpu_optimizer', {})
    
    def get_model_pruner_config(self) -> Dict[str, Any]:
        """Get model pruner configuration."""
        return self.config.get('model_pruner', {})
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return self.config.get('model_params', {})
    
    def get_physics_params(self) -> Dict[str, Any]:
        """Get physics-related parameters."""
        return self.config.get('physics_params', {})
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self.config.get('training', {})
    
    def get_validation_params(self) -> Dict[str, Any]:
        """Get validation parameters."""
        return self.config.get('validation', {})
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """Get tracking parameters."""
        return self.config.get('tracking', {})
    
    def get_prediction_params(self) -> Dict[str, Any]:
        """Get trajectory prediction parameters."""
        return self.config.get('prediction', {})
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """Get trajectory optimization parameters."""
        return self.config.get('optimization', {})


def create_optimizers(model: 'nn.Module', config_path: str) -> Dict[str, Any]:
    """
    Create all optimizers with configuration parameters.
    
    Args:
        model: The model to optimize
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing all optimizer instances
    """
    config_loader = OptimizationConfigLoader(config_path)
    
    from .batch_processor import BatchProcessor
    from .memory_optimizer import MemoryOptimizer
    from .gpu_optimizer import GPUOptimizer
    from .model_pruner import ModelPruner
    
    batch_config = config_loader.get_batch_processor_config()
    memory_config = config_loader.get_memory_optimizer_config()
    gpu_config = config_loader.get_gpu_optimizer_config()
    pruner_config = config_loader.get_model_pruner_config()
    
    return {
        'batch_processor': BatchProcessor(**batch_config),
        'memory_optimizer': MemoryOptimizer(model, **memory_config),
        'gpu_optimizer': GPUOptimizer(model, **gpu_config),
        'model_pruner': ModelPruner(model, **pruner_config)
    }
