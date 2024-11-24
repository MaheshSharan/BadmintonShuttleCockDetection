"""
Detection Head Implementation
This module implements the detection head for precise shuttlecock localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from ..utils.feature_utils import compute_iou_matrix


class DetectionHead(nn.Module):
    """
    Detection head for shuttlecock detection.
    Performs fine-grained detection using RPN proposals.
    """
    
    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        num_classes: int = 1,  # Only shuttlecock
        reg_class_agnostic: bool = False,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Layers for shared features
        self.shared_fcs = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification branch
        self.fc_cls = nn.Linear(feat_channels, num_classes + 1)  # +1 for background
        
        # Regression branch
        out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
        self.fc_reg = nn.Linear(feat_channels, out_dim_reg)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.shared_fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        nn.init.normal_(self.fc_cls.weight, std=0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        
        nn.init.normal_(self.fc_reg.weight, std=0.001)
        nn.init.constant_(self.fc_reg.bias, 0)
        
    def forward(
        self,
        x: torch.Tensor,
        rois: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of detection head.
        
        Args:
            x: Input features
            rois: Region proposals from RPN
            
        Returns:
            Tuple of (classification scores, bbox predictions)
        """
        # ROI pooling (simplified, you might want to use torchvision.ops.RoIPool)
        roi_feats = self.roi_pool(x, rois)
        
        # Flatten features
        roi_feats = roi_feats.view(roi_feats.size(0), -1)
        
        # Shared fully connected layers
        shared_feats = self.shared_fcs(roi_feats)
        
        # Get classification scores
        cls_score = self.fc_cls(shared_feats)
        
        # Get bbox predictions
        bbox_pred = self.fc_reg(shared_feats)
        
        return cls_score, bbox_pred
    
    def roi_pool(
        self,
        features: torch.Tensor,
        rois: torch.Tensor,
        output_size: Tuple[int, int] = (7, 7)
    ) -> torch.Tensor:
        """
        Perform ROI pooling on features.
        
        Args:
            features: Input feature maps
            rois: Region proposals
            output_size: Size of output features
            
        Returns:
            Pooled features for each ROI
        """
        # Implement ROI pooling (you might want to use torchvision.ops.RoIPool)
        # This is a simplified version
        roi_feats = []
        
        for roi in rois:
            x1, y1, x2, y2 = roi.int()
            roi_feat = features[:, :, y1:y2+1, x1:x2+1]
            roi_feat = F.adaptive_avg_pool2d(roi_feat, output_size)
            roi_feats.append(roi_feat)
            
        return torch.stack(roi_feats, dim=0)
    
    def loss(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        rois: torch.Tensor,
        labels: torch.Tensor,
        bbox_targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection head loss.
        
        Args:
            cls_score: Classification scores
            bbox_pred: BBox predictions
            rois: Input ROIs
            labels: Ground truth labels
            bbox_targets: Ground truth bbox targets
            reduction: Loss reduction method
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Classification loss
        cls_loss = F.cross_entropy(
            cls_score,
            labels,
            reduction=reduction
        )
        losses['loss_cls'] = cls_loss
        
        # Regression loss (smooth L1)
        pos_inds = labels > 0
        
        if pos_inds.any():
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                    pos_inds, labels[pos_inds] - 1
                ]
                
            reg_loss = F.smooth_l1_loss(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                reduction=reduction,
                beta=1.0
            )
            losses['loss_reg'] = reg_loss
        else:
            losses['loss_reg'] = bbox_pred.sum() * 0
            
        return losses
    
    def get_bboxes(
        self,
        rois: torch.Tensor,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        img_shape: Tuple[int, int],
        scale_factor: Optional[torch.Tensor] = None,
        rescale: bool = False,
        cfg: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get detection bboxes from network output.
        
        Args:
            rois: Input ROIs
            cls_score: Classification scores
            bbox_pred: BBox predictions
            img_shape: Original image shape
            scale_factor: Scale factor for rescaling bboxes
            rescale: Whether to rescale bboxes
            cfg: Test config
            
        Returns:
            Tuple of (bboxes, scores)
        """
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
            
        scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None
        
        if bbox_pred is not None:
            bboxes = self.bbox_pred2bbox(rois, bbox_pred)
            if rescale and scale_factor is not None:
                bboxes /= scale_factor
        else:
            bboxes = rois
            
        # Remove background class
        if scores is not None:
            scores = scores[:, 1:]
            
        # Apply score threshold
        if cfg is not None and cfg.get('score_thr', 0) > 0:
            score_thr = cfg['score_thr']
            valid_mask = scores.max(dim=1)[0] > score_thr
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]
            
        return bboxes, scores
    
    @staticmethod
    def bbox_pred2bbox(rois: torch.Tensor, bbox_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert bbox predictions to bboxes.
        
        Args:
            rois: Input ROIs
            bbox_pred: BBox predictions
            
        Returns:
            Predicted bboxes
        """
        means = torch.tensor([0., 0., 0., 0.]).to(bbox_pred.device)
        stds = torch.tensor([0.1, 0.1, 0.2, 0.2]).to(bbox_pred.device)
        
        denorm_deltas = bbox_pred * stds + means
        
        dx = denorm_deltas[:, 0::4]
        dy = denorm_deltas[:, 1::4]
        dw = denorm_deltas[:, 2::4]
        dh = denorm_deltas[:, 3::4]
        
        # Convert rois to (x_ctr, y_ctr, w, h)
        x1, y1, x2, y2 = rois.split(1, dim=1)
        px = ((x1 + x2) * 0.5)
        py = ((y1 + y2) * 0.5)
        pw = (x2 - x1)
        ph = (y2 - y1)
        
        # Apply deltas
        gx = px + pw * dx
        gy = py + ph * dy
        gw = pw * torch.exp(dw)
        gh = ph * torch.exp(dh)
        
        # Convert back to (x1, y1, x2, y2)
        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5
        
        return torch.cat([x1, y1, x2, y2], dim=1)
