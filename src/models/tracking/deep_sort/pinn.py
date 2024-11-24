"""
Physics-Informed Neural Network (PINN) for shuttlecock trajectory validation and refinement.
Combines deep learning with physical constraints from aerodynamics and projectile motion.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional

class TrajectoryPINN(nn.Module):
    """
    Physics-Informed Neural Network for shuttlecock trajectory modeling.
    Enforces physical constraints while learning from data.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        physics_weight: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the PINN model.
        
        Args:
            hidden_dim: Dimension of hidden layers
            physics_weight: Weight of physics loss vs data loss
            device: Device to run computations on
        """
        super().__init__()
        self.device = device
        self.physics_weight = physics_weight
        
        # Physical constants
        self.g = 9.81  # gravity (m/s²)
        self.rho = 1.225  # air density (kg/m³)
        self.mass = 0.005  # shuttlecock mass (kg)
        self.drag_coef = 0.6  # drag coefficient
        self.lift_coef = 0.4  # lift coefficient
        
        # Neural network architecture
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Input: (t, x, y, z)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # Output: (x, y, z)
        ).to(device)
        
    def forward(
        self,
        t: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            t: Time points
            x, y, z: Optional position coordinates for training
            
        Returns:
            Predicted positions (x, y, z)
        """
        # Prepare input
        t = t.to(self.device)
        if x is not None and y is not None and z is not None:
            x = x.to(self.device)
            y = y.to(self.device)
            z = z.to(self.device)
            inputs = torch.stack([t, x, y, z], dim=1)
        else:
            inputs = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t), torch.zeros_like(t)], dim=1)
        
        # Get predictions
        predictions = self.net(inputs)
        return predictions
    
    def physics_loss(
        self,
        t: torch.Tensor,
        pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics-based loss using aerodynamics and projectile motion.
        
        Args:
            t: Time points
            pred: Predicted positions
            
        Returns:
            Physics loss term
        """
        # Compute derivatives
        pred.requires_grad_(True)
        dt = t[1] - t[0]
        
        # First derivatives (velocity)
        dx_dt = torch.gradient(pred[:, 0], spacing=(dt,))[0]
        dy_dt = torch.gradient(pred[:, 1], spacing=(dt,))[0]
        dz_dt = torch.gradient(pred[:, 2], spacing=(dt,))[0]
        
        # Second derivatives (acceleration)
        d2x_dt2 = torch.gradient(dx_dt, spacing=(dt,))[0]
        d2y_dt2 = torch.gradient(dy_dt, spacing=(dt,))[0]
        d2z_dt2 = torch.gradient(dz_dt, spacing=(dt,))[0]
        
        # Velocity magnitude
        v = torch.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
        
        # Drag force
        drag_force = -0.5 * self.rho * self.drag_coef * v * torch.stack([dx_dt, dy_dt, dz_dt], dim=1)
        
        # Physics residuals
        residual_x = self.mass * d2x_dt2 - drag_force[:, 0]
        residual_y = self.mass * d2y_dt2 - drag_force[:, 1]
        residual_z = self.mass * d2z_dt2 - drag_force[:, 2] + self.mass * self.g
        
        # Combined physics loss
        physics_loss = (
            torch.mean(residual_x**2) +
            torch.mean(residual_y**2) +
            torch.mean(residual_z**2)
        )
        
        return physics_loss
    
    def data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute data-based loss between predictions and targets.
        
        Args:
            pred: Predicted positions
            target: Target positions
            
        Returns:
            Data loss term
        """
        return torch.mean((pred - target)**2)
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: Tuple of (time_points, positions)
            
        Returns:
            Dictionary of loss terms
        """
        t, positions = batch
        t, positions = t.to(self.device), positions.to(self.device)
        
        # Forward pass
        pred_positions = self(t, positions[:, 0], positions[:, 1], positions[:, 2])
        
        # Compute losses
        phys_loss = self.physics_loss(t, pred_positions)
        data_loss = self.data_loss(pred_positions, positions)
        
        # Combined loss
        total_loss = (
            self.physics_weight * phys_loss +
            (1 - self.physics_weight) * data_loss
        )
        
        return {
            'total_loss': total_loss,
            'physics_loss': phys_loss,
            'data_loss': data_loss
        }
    
    def validate_trajectory(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
        threshold: float = 0.1
    ) -> Tuple[bool, torch.Tensor]:
        """
        Validate a trajectory using physics constraints.
        
        Args:
            positions: Position coordinates [N, 3]
            times: Time points [N]
            threshold: Maximum allowed physics violation
            
        Returns:
            Tuple of (is_valid, refined_positions)
        """
        with torch.no_grad():
            # Convert to tensors
            positions = torch.tensor(positions, dtype=torch.float32).to(self.device)
            times = torch.tensor(times, dtype=torch.float32).to(self.device)
            
            # Get predictions
            pred_positions = self(
                times,
                positions[:, 0],
                positions[:, 1],
                positions[:, 2]
            )
            
            # Compute physics violation
            physics_violation = self.physics_loss(times, pred_positions)
            
            # Check if trajectory is physically plausible
            is_valid = physics_violation.item() < threshold
            
            return is_valid, pred_positions.cpu().numpy()
    
    def refine_trajectory(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
        num_iterations: int = 100
    ) -> torch.Tensor:
        """
        Refine a trajectory to better satisfy physics constraints.
        
        Args:
            positions: Position coordinates [N, 3]
            times: Time points [N]
            num_iterations: Number of refinement iterations
            
        Returns:
            Refined positions
        """
        positions = torch.tensor(positions, dtype=torch.float32).to(self.device)
        times = torch.tensor(times, dtype=torch.float32).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': self.net.parameters(), 'lr': 1e-4}
        ])
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            pred_positions = self(
                times,
                positions[:, 0],
                positions[:, 1],
                positions[:, 2]
            )
            
            # Compute losses
            phys_loss = self.physics_loss(times, pred_positions)
            data_loss = self.data_loss(pred_positions, positions)
            
            # Combined loss
            total_loss = (
                self.physics_weight * phys_loss +
                (1 - self.physics_weight) * data_loss
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            refined_positions = self(
                times,
                positions[:, 0],
                positions[:, 1],
                positions[:, 2]
            )
        
        return refined_positions.cpu().numpy()
