"""
Model Registry
This module provides a registry system for model components,
allowing easy registration and instantiation of different architectures.
"""

from typing import Dict, Type, Any, Callable
import torch.nn as nn

from .backbone.csp_darknet import CSPDarknet
from .neck.fpn import FPN
from .head.detection_head import DetectionHead
from .head.rpn import RPN


class Registry:
    """
    Registry class for managing model components.
    Supports registration and building of backbones, necks, and heads.
    """
    
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type[nn.Module]] = {}
    
    def register_module(self, name: str = None) -> Callable:
        """
        Register a module class.
        
        Args:
            name (str, optional): Module name. If not specified, use class name.
            
        Returns:
            Callable: Decorator function
        """
        def _register(cls):
            module_name = name if name else cls.__name__
            if module_name in self._module_dict:
                raise ValueError(f'Module {module_name} is already registered in {self._name}')
            self._module_dict[module_name] = cls
            return cls
        
        return _register
    
    def get(self, name: str) -> Type[nn.Module]:
        """Get registered module class by name."""
        if name not in self._module_dict:
            raise KeyError(f'Module {name} is not registered in {self._name}')
        return self._module_dict[name]
    
    def build(self, cfg: Dict[str, Any]) -> nn.Module:
        """
        Build module from config dict.
        
        Args:
            cfg (Dict[str, Any]): Config dict with module parameters
            
        Returns:
            nn.Module: Built module
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'Config must be a dict, got {type(cfg)}')
            
        module_class = self.get(cfg.pop('type', None))
        module = module_class(**cfg)
        return module


# Create registries for different components
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
RPN_HEADS = Registry('rpn')

# Register backbone
@BACKBONES.register_module()
class CSPDarknet(CSPDarknet):
    """CSPDarknet backbone."""
    pass

# Register FPN neck
@NECKS.register_module()
class FPN(FPN):
    """Feature Pyramid Network."""
    pass

# Register Detection Head
@HEADS.register_module()
class DetectionHead(DetectionHead):
    """Detection head for shuttlecock detection."""
    pass

# Register RPN
@RPN_HEADS.register_module()
class RPN(RPN):
    """Region Proposal Network."""
    pass


def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """Build backbone from config."""
    return BACKBONES.build(cfg)

def build_neck(cfg: Dict[str, Any]) -> nn.Module:
    """Build neck from config."""
    return NECKS.build(cfg)

def build_head(cfg: Dict[str, Any]) -> nn.Module:
    """Build head from config."""
    return HEADS.build(cfg)

def build_rpn(cfg: Dict[str, Any]) -> nn.Module:
    """Build RPN from config."""
    return RPN_HEADS.build(cfg)

def build_detector(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build complete detector from config.
    
    Args:
        cfg (Dict[str, Any]): Config dict with backbone, neck, and head configs
        
    Returns:
        nn.Module: Complete detector model
    """
    backbone = build_backbone(cfg['backbone'])
    neck = build_neck(cfg['neck'])
    rpn = build_rpn(cfg['rpn'])
    head = build_head(cfg['head'])
    
    return nn.ModuleDict({
        'backbone': backbone,
        'neck': neck,
        'rpn': rpn,
        'head': head
    })
