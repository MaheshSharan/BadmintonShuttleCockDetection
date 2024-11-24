"""
Custom Fusion Module Implementation
This module implements feature fusion strategies for shuttlecock detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class AdaptiveFusion(nn.Module):
    """
    Adaptive Feature Fusion module for combining multi-scale features.
    Uses attention mechanisms to weight different feature levels.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_levels: int = 5,
        use_spatial_attention: bool = True,
        use_channel_attention: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention
        
        # Feature adaptation layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1)
            for i in range(num_levels)
        ])
        
        # Channel attention modules
        if use_channel_attention:
            self.channel_attentions = nn.ModuleList([
                ChannelAttention(out_channels)
                for _ in range(num_levels)
            ])
            
        # Spatial attention modules
        if use_spatial_attention:
            self.spatial_attentions = nn.ModuleList([
                SpatialAttention()
                for _ in range(num_levels)
            ])
            
        # Feature refinement
        self.refine_conv = nn.Conv2d(
            out_channels * num_levels,
            out_channels,
            3,
            padding=1
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights of the fusion module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of fusion module.
        
        Args:
            inputs: List of multi-scale feature maps
            
        Returns:
            Fused feature map
        """
        assert len(inputs) == self.num_levels
        
        # Adapt features to common channel dimension
        feats = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Apply channel attention if enabled
        if self.use_channel_attention:
            feats = [
                att(feat)
                for feat, att in zip(feats, self.channel_attentions)
            ]
            
        # Apply spatial attention if enabled
        if self.use_spatial_attention:
            feats = [
                att(feat)
                for feat, att in zip(feats, self.spatial_attentions)
            ]
            
        # Resize all features to the same size
        target_size = feats[0].shape[2:]
        feats = [
            F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            if feat.shape[2:] != target_size else feat
            for feat in feats
        ]
        
        # Concatenate and refine
        x = torch.cat(feats, dim=1)
        x = self.refine_conv(x)
        
        return x


class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average pool features
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pool features
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


class MotionAwareFusion(nn.Module):
    """
    Motion-aware feature fusion for better tracking of fast-moving shuttlecocks.
    """
    
    def __init__(
        self,
        in_channels: int,
        motion_channels: int = 2,  # For optical flow
        use_temporal: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.motion_channels = motion_channels
        self.use_temporal = use_temporal
        
        # Motion feature extraction
        self.motion_conv = nn.Conv2d(
            motion_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(
            in_channels * 2 if use_temporal else in_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(
        self,
        x: torch.Tensor,
        motion_feat: Optional[torch.Tensor] = None,
        prev_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of motion-aware fusion.
        
        Args:
            x: Current feature map
            motion_feat: Motion features (e.g., optical flow)
            prev_feat: Previous frame features
            
        Returns:
            Motion-enhanced features
        """
        # Process motion features if available
        if motion_feat is not None:
            motion_feat = self.motion_conv(motion_feat)
            x = x + motion_feat
            
        # Incorporate temporal information if available and enabled
        if self.use_temporal and prev_feat is not None:
            x = torch.cat([x, prev_feat], dim=1)
            
        # Final fusion
        x = self.fusion_conv(x)
        
        return x
