"""
Region Proposal Network (RPN) Implementation
This module implements the RPN for generating shuttlecock proposals from feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from ..utils.feature_utils import generate_anchors, compute_iou_matrix


class RPN(nn.Module):
    """
    Region Proposal Network for shuttlecock detection.
    Generates proposals from FPN features using anchor-based detection.
    """
    
    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        anchor_scales: List[int] = [8, 16, 32],
        anchor_ratios: List[float] = [0.5, 1.0, 2.0],
        feat_strides: List[int] = [4, 8, 16, 32, 64],
        train_cfg: Dict = None,
        test_cfg: Dict = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.feat_strides = feat_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Number of anchors per location
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        
        # RPN components
        self.rpn_conv = nn.Conv2d(
            in_channels, feat_channels, 3, padding=1
        )
        self.rpn_cls = nn.Conv2d(
            feat_channels,
            self.num_anchors * 2,  # 2 classes: background and shuttlecock
            1
        )
        self.rpn_reg = nn.Conv2d(
            feat_channels,
            self.num_anchors * 4,  # 4 for bbox regression
            1
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights of RPN."""
        nn.init.normal_(self.rpn_conv.weight, std=0.01)
        nn.init.normal_(self.rpn_cls.weight, std=0.01)
        nn.init.normal_(self.rpn_reg.weight, std=0.01)
        
        nn.init.constant_(self.rpn_conv.bias, 0)
        nn.init.constant_(self.rpn_cls.bias, 0)
        nn.init.constant_(self.rpn_reg.bias, 0)
        
    def forward_single(self, x: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for single feature level.
        
        Args:
            x: Feature map from FPN
            stride: Feature stride for this level
            
        Returns:
            Tuple of (classification scores, bbox predictions)
        """
        x = F.relu(self.rpn_conv(x))
        
        # Shape: (N, num_anchors * 2, H, W)
        rpn_cls_score = self.rpn_cls(x)
        # Shape: (N, num_anchors * 4, H, W)
        rpn_bbox_pred = self.rpn_reg(x)
        
        return rpn_cls_score, rpn_bbox_pred
        
    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of RPN.
        
        Args:
            feats: List of feature maps from FPN
            
        Returns:
            Tuple of (list of classification scores, list of bbox predictions)
        """
        return multi_apply(self.forward_single, feats, self.feat_strides)
    
    def get_anchors(self, featmap_sizes: List[Tuple[int, int]], img_shape: Tuple[int, int]) -> List[torch.Tensor]:
        """
        Generate anchors for each feature map level.
        
        Args:
            featmap_sizes: List of (height, width) for each feature level
            img_shape: Original image shape (height, width)
            
        Returns:
            List of anchors for each level
        """
        multi_level_anchors = []
        
        for i, stride in enumerate(self.feat_strides):
            # Generate base anchors for this stride
            base_anchors = generate_anchors(
                base_size=stride,
                ratios=self.anchor_ratios,
                scales=self.anchor_scales
            )
            base_anchors = torch.from_numpy(base_anchors).float()
            
            # Generate grid anchors based on feature map size
            feat_h, feat_w = featmap_sizes[i]
            shifts_x = torch.arange(0, feat_w * stride, stride, dtype=torch.float32)
            shifts_y = torch.arange(0, feat_h * stride, stride, dtype=torch.float32)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                                shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
            
            # Add anchors for this feature level
            anchors = base_anchors[None, :, :] + shifts[:, None, :]
            anchors = anchors.reshape(-1, 4)
            
            # Clip anchors to image boundaries
            anchors[:, 0::2].clamp_(min=0, max=img_shape[1])
            anchors[:, 1::2].clamp_(min=0, max=img_shape[0])
            
            multi_level_anchors.append(anchors)
            
        return multi_level_anchors
    
    def loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        img_shape: Tuple[int, int],
        gt_labels: List[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RPN loss.
        
        Args:
            cls_scores: List of classification scores
            bbox_preds: List of bbox predictions
            gt_bboxes: Ground truth bounding boxes
            img_shape: Image shape
            gt_labels: Ground truth labels
            
        Returns:
            Dictionary of losses
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchors = self.get_anchors(featmap_sizes, img_shape)
        
        # Assign targets
        cls_targets = []
        reg_targets = []
        
        for i in range(len(anchors)):
            anchor = anchors[i]
            gt_bbox = gt_bboxes[i]
            
            # Compute IoU between anchors and gt_boxes
            iou_matrix = compute_iou_matrix(anchor, gt_bbox)
            
            # Assign positive and negative samples
            max_iou, argmax_iou = iou_matrix.max(dim=1)
            pos_mask = max_iou >= self.train_cfg.get('pos_iou_thr', 0.7)
            neg_mask = max_iou < self.train_cfg.get('neg_iou_thr', 0.3)
            
            # Prepare classification targets
            cls_target = torch.zeros_like(pos_mask, dtype=torch.long)
            cls_target[pos_mask] = 1  # Positive samples
            cls_target[~(pos_mask | neg_mask)] = -1  # Ignore these samples
            
            # Prepare regression targets
            reg_target = torch.zeros_like(anchor)
            if pos_mask.any():
                pos_bbox_targets = gt_bbox[argmax_iou[pos_mask]]
                reg_target[pos_mask] = self.bbox2delta(anchor[pos_mask], pos_bbox_targets)
            
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
        
        # Compute losses
        losses = {}
        
        # Classification loss
        cls_scores = [
            score.permute(0, 2, 3, 1).reshape(-1, 2)
            for score in cls_scores
        ]
        cls_targets = torch.cat(cls_targets)
        valid_mask = cls_targets >= 0
        losses['loss_rpn_cls'] = F.cross_entropy(
            cls_scores[0][valid_mask],
            cls_targets[valid_mask],
            reduction='mean'
        )
        
        # Regression loss (smooth L1)
        bbox_preds = [
            pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for pred in bbox_preds
        ]
        reg_targets = torch.cat(reg_targets)
        pos_mask = cls_targets > 0
        
        if pos_mask.any():
            losses['loss_rpn_reg'] = F.smooth_l1_loss(
                bbox_preds[0][pos_mask],
                reg_targets[pos_mask],
                reduction='mean',
                beta=1.0
            )
        else:
            losses['loss_rpn_reg'] = bbox_preds[0].sum() * 0
            
        return losses
    
    @staticmethod
    def bbox2delta(anchors: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Convert bbox targets to delta format."""
        px = (anchors[:, 0] + anchors[:, 2]) * 0.5
        py = (anchors[:, 1] + anchors[:, 3]) * 0.5
        pw = anchors[:, 2] - anchors[:, 0]
        ph = anchors[:, 3] - anchors[:, 1]
        
        gx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5
        gy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5
        gw = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gh = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)
        return deltas


def multi_apply(func, *args, **kwargs):
    """Apply function to multiple inputs."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
