"""
Physics-Informed Loss Functions
This module implements custom loss functions that incorporate physical constraints
for shuttlecock trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .physics_utils import (
    calculate_air_resistance,
    calculate_magnus_force,
    calculate_gravity_effect
)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function combining detection and physical constraints.
    """
    
    def __init__(
        self,
        mass: float = 0.005,  # kg, standard shuttlecock mass
        drag_coefficient: float = 0.6,
        lift_coefficient: float = 0.4,
        air_density: float = 1.225,  # kg/m³
        gravity: float = 9.81,  # m/s²
        fps: int = 30,  # frames per second
        lambda_physics: float = 0.1,  # weight for physics loss
        lambda_smooth: float = 0.05   # weight for smoothness loss
    ):
        super().__init__()
        self.mass = mass
        self.drag_coefficient = drag_coefficient
        self.lift_coefficient = lift_coefficient
        self.air_density = air_density
        self.gravity = gravity
        self.dt = 1.0 / fps
        self.lambda_physics = lambda_physics
        self.lambda_smooth = lambda_smooth
        
    def forward(
        self,
        pred_trajectories: torch.Tensor,  # Shape: (B, T, 3) - x, y, z coordinates
        pred_velocities: torch.Tensor,    # Shape: (B, T, 3) - vx, vy, vz
        pred_rotations: torch.Tensor,     # Shape: (B, T, 3) - wx, wy, wz
        gt_trajectories: Optional[torch.Tensor] = None,
        gt_velocities: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None  # Valid trajectory points mask
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.
        
        Args:
            pred_trajectories: Predicted 3D trajectories
            pred_velocities: Predicted velocities
            pred_rotations: Predicted angular velocities
            gt_trajectories: Ground truth trajectories (if available)
            gt_velocities: Ground truth velocities (if available)
            masks: Valid points mask
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        batch_size, seq_len, _ = pred_trajectories.shape
        
        # 1. Supervision Loss (if ground truth available)
        if gt_trajectories is not None:
            trajectory_loss = F.mse_loss(
                pred_trajectories[masks],
                gt_trajectories[masks]
            )
            losses['loss_trajectory'] = trajectory_loss
            
            if gt_velocities is not None:
                velocity_loss = F.mse_loss(
                    pred_velocities[masks],
                    gt_velocities[masks]
                )
                losses['loss_velocity'] = velocity_loss
        
        # 2. Physics Consistency Loss
        physics_loss = torch.tensor(0., device=pred_trajectories.device)
        
        for t in range(seq_len - 1):
            # Current state
            pos_t = pred_trajectories[:, t]      # (B, 3)
            vel_t = pred_velocities[:, t]        # (B, 3)
            rot_t = pred_rotations[:, t]         # (B, 3)
            
            # Next state
            pos_tp1 = pred_trajectories[:, t + 1]  # (B, 3)
            vel_tp1 = pred_velocities[:, t + 1]    # (B, 3)
            
            # Calculate forces
            air_resistance = calculate_air_resistance(
                vel_t,
                self.drag_coefficient,
                self.air_density
            )
            
            magnus_force = calculate_magnus_force(
                vel_t,
                rot_t,
                self.lift_coefficient,
                self.air_density
            )
            
            gravity_force = calculate_gravity_effect(
                batch_size,
                self.gravity,
                device=pred_trajectories.device
            )
            
            # Total acceleration
            total_force = air_resistance + magnus_force + gravity_force
            acc_t = total_force / self.mass
            
            # Predicted next state using physics
            pred_pos_tp1 = pos_t + vel_t * self.dt + 0.5 * acc_t * self.dt ** 2
            pred_vel_tp1 = vel_t + acc_t * self.dt
            
            # Physics consistency error
            pos_error = F.mse_loss(pos_tp1, pred_pos_tp1)
            vel_error = F.mse_loss(vel_tp1, pred_vel_tp1)
            
            physics_loss = physics_loss + pos_error + vel_error
            
        losses['loss_physics'] = physics_loss * self.lambda_physics
        
        # 3. Trajectory Smoothness Loss
        if seq_len > 2:
            # Second-order smoothness (acceleration smoothness)
            acc_pred = (pred_velocities[:, 2:] - 2 * pred_velocities[:, 1:-1] + 
                       pred_velocities[:, :-2]) / (self.dt ** 2)
            smoothness_loss = torch.mean(torch.square(acc_pred))
            losses['loss_smooth'] = smoothness_loss * self.lambda_smooth
        
        # Total loss
        losses['loss_total'] = sum(losses.values())
        
        return losses
    
    
class TrajectoryFeasibilityLoss(nn.Module):
    """
    Loss function to ensure physically feasible trajectories.
    """
    
    def __init__(
        self,
        max_velocity: float = 137.0,  # m/s (fastest recorded shuttlecock speed)
        max_acceleration: float = 100.0,  # m/s²
        court_bounds: Tuple[float, float, float] = (13.4, 6.1, 8.0),  # length, width, height
        fps: int = 30
    ):
        super().__init__()
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.court_bounds = court_bounds
        self.dt = 1.0 / fps
        
    def forward(
        self,
        trajectories: torch.Tensor,  # Shape: (B, T, 3)
        velocities: torch.Tensor,    # Shape: (B, T, 3)
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feasibility loss for trajectories.
        
        Args:
            trajectories: Predicted 3D trajectories
            velocities: Predicted velocities
            masks: Valid points mask
            
        Returns:
            Dictionary of constraint violation losses
        """
        losses = {}
        
        # 1. Velocity Magnitude Constraint
        velocity_magnitudes = torch.norm(velocities, dim=-1)  # (B, T)
        velocity_violation = F.relu(velocity_magnitudes - self.max_velocity)
        losses['loss_velocity_constraint'] = torch.mean(velocity_violation)
        
        # 2. Acceleration Constraint
        accelerations = (velocities[:, 1:] - velocities[:, :-1]) / self.dt  # (B, T-1, 3)
        acc_magnitudes = torch.norm(accelerations, dim=-1)  # (B, T-1)
        acc_violation = F.relu(acc_magnitudes - self.max_acceleration)
        losses['loss_acceleration_constraint'] = torch.mean(acc_violation)
        
        # 3. Court Boundary Constraint
        x_violation = F.relu(torch.abs(trajectories[..., 0]) - self.court_bounds[0] / 2)
        y_violation = F.relu(torch.abs(trajectories[..., 1]) - self.court_bounds[1] / 2)
        z_violation = F.relu(-trajectories[..., 2])  # Only constrain minimum height
        
        boundary_violation = x_violation + y_violation + z_violation
        losses['loss_boundary_constraint'] = torch.mean(boundary_violation)
        
        # Apply masks if provided
        if masks is not None:
            for key in losses:
                losses[key] = losses[key] * masks
                losses[key] = torch.sum(losses[key]) / torch.sum(masks)
        
        # Total constraint violation loss
        losses['loss_feasibility_total'] = sum(losses.values())
        
        return losses
