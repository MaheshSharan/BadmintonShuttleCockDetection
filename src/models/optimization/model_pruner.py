"""
Model Pruning Module
Implements model pruning and quantization strategies for model optimization.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub


class ModelPruner:
    """
    Handles model pruning and quantization for model size and inference optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_method: str = 'l1_unstructured',
        target_sparsity: float = 0.5,
        quantize: bool = True,
        pruning_schedule: str = 'gradual',
        fine_tune_steps: int = 1000
    ):
        """
        Initialize model pruner with specified settings.
        
        Args:
            model: The neural network model
            pruning_method: Pruning method to use
            target_sparsity: Target sparsity ratio
            quantize: Whether to quantize the model
            pruning_schedule: Pruning schedule ('one_shot' or 'gradual')
            fine_tune_steps: Number of fine-tuning steps after pruning
        """
        self.model = model
        self.pruning_method = pruning_method
        self.target_sparsity = target_sparsity
        self.quantize = quantize
        self.pruning_schedule = pruning_schedule
        self.fine_tune_steps = fine_tune_steps
        
        self.pruning_methods = {
            'l1_unstructured': prune.l1_unstructured,
            'random_unstructured': prune.random_unstructured,
            'ln_structured': prune.ln_structured
        }
        
        self.current_step = 0
        self.pruned_layers = []
        
        if quantize:
            self._prepare_for_quantization()
    
    def _prepare_for_quantization(self):
        """Prepare model for quantization."""
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Wrap the model with quantization stubs
        class QuantWrappedModel(nn.Module):
            def __init__(self, model, quant, dequant):
                super().__init__()
                self.quant = quant
                self.model = model
                self.dequant = dequant
            
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        self.model = QuantWrappedModel(self.model, self.quant, self.dequant)
    
    def prune_model(
        self,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> nn.Module:
        """
        Apply pruning to the model.
        
        Args:
            dataloader: DataLoader for fine-tuning
            optimizer: Optimizer for fine-tuning
            
        Returns:
            Pruned model
        """
        if self.pruning_schedule == 'gradual':
            return self._gradual_pruning(dataloader, optimizer)
        else:
            return self._one_shot_pruning()
    
    def _one_shot_pruning(self) -> nn.Module:
        """Apply one-shot pruning to the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune_fn = self.pruning_methods[self.pruning_method]
                prune_fn(
                    module,
                    name='weight',
                    amount=self.target_sparsity
                )
                self.pruned_layers.append(name)
        
        return self.model
    
    def _gradual_pruning(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> nn.Module:
        """Apply gradual pruning with fine-tuning."""
        n_steps = self.fine_tune_steps
        sparsity_schedule = np.linspace(0, self.target_sparsity, n_steps)
        
        for step, sparsity in enumerate(sparsity_schedule):
            # Apply pruning
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if name not in self.pruned_layers:
                        prune_fn = self.pruning_methods[self.pruning_method]
                        prune_fn(
                            module,
                            name='weight',
                            amount=sparsity
                        )
                        self.pruned_layers.append(name)
            
            # Fine-tune if dataloader and optimizer provided
            if dataloader is not None and optimizer is not None:
                self._fine_tune_step(dataloader, optimizer)
            
            self.current_step = step
        
        return self.model
    
    def _fine_tune_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ):
        """Perform one step of fine-tuning."""
        self.model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            
            if isinstance(batch, (tuple, list)):
                x, y = batch
            else:
                x = batch
            
            outputs = self.model(x)
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = outputs
            
            loss.backward()
            optimizer.step()
    
    def quantize_model(
        self,
        calibration_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """
        Quantize the model for improved inference efficiency.
        
        Args:
            calibration_dataloader: DataLoader for calibrating quantization
            
        Returns:
            Quantized model
        """
        if not self.quantize:
            return self.model
        
        # Prepare for quantization
        self.model.eval()
        
        # Fuse modules where possible
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'relu']]
        )
        
        # Configure quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate if dataloader provided
        if calibration_dataloader is not None:
            self._calibrate_quantization(calibration_dataloader)
        
        # Convert to quantized model
        torch.quantization.convert(self.model, inplace=True)
        
        return self.model
    
    def _calibrate_quantization(self, dataloader: torch.utils.data.DataLoader):
        """Calibrate quantization using provided data."""
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                self.model(x)
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size statistics."""
        total_params = 0
        total_nonzero = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                total_nonzero += torch.count_nonzero(param).item()
        
        return {
            'total_parameters': total_params,
            'nonzero_parameters': total_nonzero,
            'sparsity': 1 - (total_nonzero / total_params),
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def remove_pruning(self):
        """Remove pruning and make weights permanent."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.pruned_layers:
                prune.remove(module, 'weight')
        
        self.pruned_layers = []
    
    def export_optimized_model(
        self,
        path: str,
        input_shape: tuple,
        export_format: str = 'torchscript'
    ):
        """
        Export the optimized model.
        
        Args:
            path: Path to save the model
            input_shape: Input shape for tracing
            export_format: Format to export ('torchscript' or 'onnx')
        """
        self.model.eval()
        dummy_input = torch.randn(input_shape)
        
        if export_format == 'torchscript':
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(path)
        elif export_format == 'onnx':
            torch.onnx.export(
                self.model,
                dummy_input,
                path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
