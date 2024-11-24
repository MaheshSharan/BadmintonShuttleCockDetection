"""
Feature Pyramid Network Implementation
This module implements the FPN architecture for multi-scale feature extraction,
with additional enhancements for shuttlecock detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Enhanced Feature Pyramid Network implementation.
    
    Features:
    - Multi-scale feature fusion
    - Adaptive feature enhancement
    - Configurable extra convolutions
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs='on_input',
        relu_before_extra_convs=False
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
            
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        # Output convolutions
        self.fpn_convs = nn.ModuleList()
        
        # Build lateral and output convolutions
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=True,
                bias=False
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=True,
                bias=False
            )
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
        # Extra FPN levels
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                             if self.add_extra_convs == 'on_input' else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=True,
                    bias=False
                )
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
            
        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # Extra levels
        if self.num_outs > len(outs):
            if self.add_extra_convs == 'on_input':
                extra_source = inputs[self.backbone_end_level - 1]
            else:
                extra_source = outs[-1]
                
            for i in range(used_backbone_levels, self.num_outs):
                if self.relu_before_extra_convs:
                    extra_source = F.relu(extra_source)
                outs.append(self.fpn_convs[i](extra_source))
                extra_source = outs[-1]
                
        return tuple(outs)


class ConvModule(nn.Module):
    """Basic convolution module with normalization and activation."""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        normalize=True,
        bias=True
    ):
        super().__init__()
        self.with_norm = normalize
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias and not normalize
        )
        
        if self.with_norm:
            self.bn = nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        return F.relu(x)


def build_fpn(config):
    """Build FPN from config dictionary."""
    return FPN(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_outs=config['num_outs'],
        start_level=config.get('start_level', 0),
        add_extra_convs=config.get('add_extra_convs', 'on_input'),
        relu_before_extra_convs=config.get('relu_before_extra_convs', False)
    )
