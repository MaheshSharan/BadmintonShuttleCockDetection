"""
CSPDarknet Backbone Implementation
This module implements the CSPDarknet backbone architecture, which is an efficient feature extractor
that uses Cross Stage Partial Networks to reduce computational complexity while maintaining accuracy.
"""

import torch
import torch.nn as nn


class ConvBNMish(nn.Module):
    """Convolution-BatchNorm-Mish block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial Block"""
    def __init__(self, in_channels, out_channels, num_bottlenecks):
        super().__init__()
        mid_channels = out_channels // 2
        
        # Main branch
        self.main_conv = ConvBNMish(in_channels, mid_channels, 1)
        self.main_bottlenecks = nn.Sequential(*[
            Bottleneck(mid_channels, mid_channels)
            for _ in range(num_bottlenecks)
        ])
        
        # Shortcut branch
        self.shortcut_conv = ConvBNMish(in_channels, mid_channels, 1)
        
        # Final fusion
        self.final_conv = ConvBNMish(mid_channels * 2, out_channels, 1)

    def forward(self, x):
        main = self.main_bottlenecks(self.main_conv(x))
        shortcut = self.shortcut_conv(x)
        return self.final_conv(torch.cat([main, shortcut], dim=1))


class Bottleneck(nn.Module):
    """Standard bottleneck block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBNMish(in_channels, mid_channels, 1)
        self.conv2 = ConvBNMish(mid_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class CSPDarknet(nn.Module):
    """CSPDarknet backbone architecture"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Initial convolution
        self.conv1 = ConvBNMish(in_channels, base_channels, 3, padding=1)
        
        # Downsample and CSP stages
        self.stage1 = self._make_stage(base_channels, base_channels * 2, 1)      # 1/2
        self.stage2 = self._make_stage(base_channels * 2, base_channels * 4, 2)  # 1/4
        self.stage3 = self._make_stage(base_channels * 4, base_channels * 8, 8)  # 1/8
        self.stage4 = self._make_stage(base_channels * 8, base_channels * 16, 8) # 1/16
        self.stage5 = self._make_stage(base_channels * 16, base_channels * 32, 4) # 1/32

    def _make_stage(self, in_channels, out_channels, num_bottlenecks):
        """Create a stage with downsampling and CSP block"""
        return nn.Sequential(
            ConvBNMish(in_channels, out_channels, 3, stride=2, padding=1),  # Downsample
            CSPBlock(out_channels, out_channels, num_bottlenecks)
        )

    def forward(self, x):
        # Store intermediate features for FPN
        features = []
        
        x = self.conv1(x)
        x = self.stage1(x)
        features.append(x)  # P1 (1/2)
        
        x = self.stage2(x)
        features.append(x)  # P2 (1/4)
        
        x = self.stage3(x)
        features.append(x)  # P3 (1/8)
        
        x = self.stage4(x)
        features.append(x)  # P4 (1/16)
        
        x = self.stage5(x)
        features.append(x)  # P5 (1/32)
        
        return features


def build_csp_backbone(config):
    """Build CSPDarknet backbone from config"""
    return CSPDarknet(
        in_channels=config.get('in_channels', 3),
        base_channels=config.get('base_channels', 64)
    )
