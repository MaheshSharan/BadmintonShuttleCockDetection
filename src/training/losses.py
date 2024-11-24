"""
Comprehensive loss functions for shuttlecock detection and tracking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """Detection loss combining classification and regression."""
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, pred, target, mask=None):
        cls_pred, reg_pred = pred['classification'], pred['regression']
        cls_target, reg_target = target['classification'], target['regression']
        
        # Classification loss
        cls_loss = self.cls_loss(cls_pred, cls_target)
        if mask is not None:
            cls_loss = cls_loss * mask
            
        # Regression loss (only for positive samples)
        pos_mask = (cls_target > 0.5).float()
        reg_loss = self.reg_loss(reg_pred, reg_target) * pos_mask
        
        return {
            'cls_loss': cls_loss.mean(),
            'reg_loss': reg_loss.sum() / (pos_mask.sum() + 1e-6)
        }


class TrackingLoss(nn.Module):
    """Tracking loss for trajectory consistency."""
    def __init__(self):
        super().__init__()
        self.id_loss = nn.CrossEntropyLoss()
        self.motion_loss = nn.SmoothL1Loss()
        
    def forward(self, pred, target):
        id_pred, motion_pred = pred['track_id'], pred['motion']
        id_target, motion_target = target['track_id'], target['motion']
        
        id_loss = self.id_loss(id_pred, id_target)
        motion_loss = self.motion_loss(motion_pred, motion_target)
        
        return {
            'id_loss': id_loss,
            'motion_loss': motion_loss
        }


class PhysicsLoss(nn.Module):
    """Physics-based loss for trajectory validation."""
    def __init__(self, gravity=9.81, air_resistance=0.1):
        super().__init__()
        self.gravity = gravity
        self.air_resistance = air_resistance
        
    def forward(self, pred_trajectory, time_steps):
        # Compute acceleration
        velocity = torch.diff(pred_trajectory, dim=1) / time_steps
        acceleration = torch.diff(velocity, dim=1) / time_steps
        
        # Gravity component
        gravity_loss = F.mse_loss(
            acceleration[..., 1],  # vertical component
            -self.gravity * torch.ones_like(acceleration[..., 1])
        )
        
        # Air resistance
        speed = torch.norm(velocity, dim=-1)
        air_resistance_loss = F.mse_loss(
            acceleration,
            -self.air_resistance * speed.unsqueeze(-1) * velocity
        )
        
        return {
            'gravity_loss': gravity_loss,
            'air_resistance_loss': air_resistance_loss
        }


class UnifiedLoss(nn.Module):
    """Combined loss function for end-to-end training."""
    def __init__(self, config):
        super().__init__()
        self.detection_loss = DetectionLoss()
        self.tracking_loss = TrackingLoss()
        self.physics_loss = PhysicsLoss()
        
        # Loss weights from config
        self.weights = config['optimization']['loss_weights']
        
    def forward(self, predictions, targets):
        # Detection losses
        det_losses = self.detection_loss(
            predictions['detection'],
            targets['detection']
        )
        
        # Tracking losses
        track_losses = self.tracking_loss(
            predictions['tracking'],
            targets['tracking']
        )
        
        # Physics losses
        physics_losses = self.physics_loss(
            predictions['trajectory'],
            targets['time_steps']
        )
        
        # Combine all losses
        total_loss = (
            self.weights['detection'] * (det_losses['cls_loss'] + det_losses['reg_loss']) +
            self.weights['tracking'] * (track_losses['id_loss'] + track_losses['motion_loss']) +
            self.weights['trajectory'] * (physics_losses['gravity_loss'] + physics_losses['air_resistance_loss'])
        )
        
        return {
            'total_loss': total_loss,
            'detection_loss': det_losses['cls_loss'] + det_losses['reg_loss'],
            'tracking_loss': track_losses['id_loss'] + track_losses['motion_loss'],
            'physics_loss': physics_losses['gravity_loss'] + physics_losses['air_resistance_loss'],
            **det_losses,
            **track_losses,
            **physics_losses
        }
