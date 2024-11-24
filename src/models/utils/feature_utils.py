"""
Multi-scale Feature Handling Utilities
This module provides utilities for handling multi-scale features in the detection pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np


def resize_features(features, size, mode='bilinear', align_corners=False):
    """
    Resize features to target size.
    
    Args:
        features (torch.Tensor): Input features of shape (B, C, H, W)
        size (tuple): Target size (H, W)
        mode (str): Interpolation mode
        align_corners (bool): Align corners parameter for interpolation
        
    Returns:
        torch.Tensor: Resized features
    """
    if features.shape[2:] == size:
        return features
    return F.interpolate(
        features,
        size=size,
        mode=mode,
        align_corners=align_corners if mode != 'nearest' else None
    )


class FeaturePyramidAdapter:
    """Adapts features from different scales for unified processing."""
    
    def __init__(self, feature_strides=[4, 8, 16, 32, 64], target_stride=8):
        self.feature_strides = feature_strides
        self.target_stride = target_stride
        
    def adapt_features(self, features, input_size):
        """
        Adapt features from different scales to target stride.
        
        Args:
            features (list): List of feature tensors
            input_size (tuple): Original input size (H, W)
            
        Returns:
            list: Adapted features at target stride
        """
        adapted_features = []
        target_size = (
            input_size[0] // self.target_stride,
            input_size[1] // self.target_stride
        )
        
        for feat, stride in zip(features, self.feature_strides):
            if stride < self.target_stride:
                # Downsample features
                scale_factor = self.target_stride / stride
                feat = F.avg_pool2d(
                    feat,
                    kernel_size=int(scale_factor),
                    stride=int(scale_factor)
                )
            elif stride > self.target_stride:
                # Upsample features
                feat = resize_features(feat, target_size)
            adapted_features.append(feat)
            
        return adapted_features


class FeatureEnhancer:
    """Enhances features using various techniques."""
    
    def __init__(self, channels, reduction=16):
        self.channels = channels
        self.reduction = reduction
        
    def apply_channel_attention(self, x):
        """Apply channel attention to features."""
        batch, channels, height, width = x.size()
        
        # Global average pooling
        y = F.avg_pool2d(x, kernel_size=(height, width))
        
        # Channel excitation
        y = y.view(batch, channels)
        y = torch.sigmoid(y).view(batch, channels, 1, 1)
        
        return x * y.expand_as(x)
    
    def apply_spatial_attention(self, x):
        """Apply spatial attention to features."""
        batch, channels, height, width = x.size()
        
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        y = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(y)
        
        return x * attention


def generate_anchors(base_size=8, ratios=[0.5, 1.0, 2.0], scales=np.array([2, 4, 8, 16, 32])):
    """
    Generate anchor boxes for feature maps.
    
    Args:
        base_size (int): Base anchor size
        ratios (list): Aspect ratios for anchors
        scales (np.array): Scales for anchors
        
    Returns:
        np.array: Generated anchors of shape (N, 4) in (x1, y1, x2, y2) format
    """
    ratios = np.array(ratios)
    scales = np.array(scales)
    
    # Get all combinations of scales and ratios
    scales_grid, ratios_grid = np.meshgrid(scales, ratios)
    scales_grid = scales_grid.flatten()
    ratios_grid = ratios_grid.flatten()
    
    # Calculate anchor widths and heights
    heights = base_size * scales_grid
    widths = base_size * scales_grid * np.sqrt(ratios_grid)
    
    # Generate anchor boxes
    x1 = -widths / 2
    y1 = -heights / 2
    x2 = widths / 2
    y2 = heights / 2
    
    anchors = np.stack([x1, y1, x2, y2], axis=1)
    return anchors.astype(np.float32)


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1 (torch.Tensor): First set of boxes (N, 4)
        boxes2 (torch.Tensor): Second set of boxes (M, 4)
        
    Returns:
        torch.Tensor: IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    return inter / union
