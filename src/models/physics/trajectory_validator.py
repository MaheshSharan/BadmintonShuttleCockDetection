"""
Trajectory Validator Module
This module implements a comprehensive validator for shuttlecock trajectories,
ensuring they follow physical laws and game constraints.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from .physics_utils import (
    calculate_reynolds_number,
    adjust_drag_coefficient,
    calculate_air_resistance,
    calculate_magnus_force,
    validate_trajectory
)


class TrajectoryValidator:
    """
    Validates and analyzes shuttlecock trajectories for physical feasibility.
    """
    
    def __init__(
        self,
        court_bounds: Tuple[float, float, float] = (13.4, 6.1, 8.0),  # length, width, height
        max_velocity: float = 137.0,  # m/s (fastest recorded shuttlecock speed)
        min_velocity: float = 1.0,    # m/s (minimum reasonable velocity)
        max_acceleration: float = 100.0,  # m/s²
        fps: int = 30,
        mass: float = 0.005,  # kg
        drag_coefficient: float = 0.6,
        lift_coefficient: float = 0.4,
        air_density: float = 1.225  # kg/m³
    ):
        self.court_bounds = court_bounds
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_acceleration = max_acceleration
        self.dt = 1.0 / fps
        self.mass = mass
        self.base_drag_coefficient = drag_coefficient
        self.lift_coefficient = lift_coefficient
        self.air_density = air_density
        
    def validate_sequence(
        self,
        positions: torch.Tensor,  # Shape: (T, 3)
        velocities: Optional[torch.Tensor] = None,  # Shape: (T, 3)
        rotations: Optional[torch.Tensor] = None    # Shape: (T, 3)
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Validate a sequence of shuttlecock positions and motions.
        
        Args:
            positions: Position vectors
            velocities: Velocity vectors (optional)
            rotations: Angular velocity vectors (optional)
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {}
        
        # 1. Calculate velocities if not provided
        if velocities is None:
            velocities = (positions[1:] - positions[:-1]) / self.dt
            velocities = torch.cat([velocities, velocities[-1:]], dim=0)  # Repeat last velocity
            
        # 2. Basic trajectory validation
        is_valid, reason = validate_trajectory(
            positions,
            velocities,
            self.court_bounds,
            self.max_velocity,
            self.dt
        )
        validation_info['basic_validation'] = {'valid': is_valid, 'reason': reason}
        
        if not is_valid:
            return False, validation_info
        
        # 3. Physical consistency checks
        physics_metrics = self._check_physical_consistency(
            positions,
            velocities,
            rotations
        )
        validation_info['physics_metrics'] = physics_metrics
        
        # 4. Game-specific constraints
        game_valid, game_metrics = self._check_game_constraints(positions, velocities)
        validation_info['game_metrics'] = game_metrics
        
        if not game_valid:
            return False, validation_info
        
        # 5. Smoothness analysis
        smoothness_metrics = self._analyze_smoothness(positions, velocities)
        validation_info['smoothness_metrics'] = smoothness_metrics
        
        # Overall validation
        is_valid = (
            validation_info['basic_validation']['valid'] and
            physics_metrics['physically_consistent'] and
            game_metrics['game_valid']
        )
        
        return is_valid, validation_info
    
    def _check_physical_consistency(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        rotations: Optional[torch.Tensor]
    ) -> Dict[str, any]:
        """Check physical consistency of the trajectory."""
        metrics = {'physically_consistent': True}
        
        # 1. Energy conservation check
        kinetic_energy = 0.5 * self.mass * torch.sum(velocities**2, dim=-1)
        potential_energy = self.mass * 9.81 * positions[:, 2]  # z-coordinate
        total_energy = kinetic_energy + potential_energy
        
        # Allow for some energy loss due to air resistance
        energy_loss_rate = (total_energy[:-1] - total_energy[1:]) / total_energy[:-1]
        metrics['max_energy_loss_rate'] = float(torch.max(energy_loss_rate))
        
        if torch.any(energy_loss_rate < 0):  # Energy shouldn't increase
            metrics['physically_consistent'] = False
            metrics['energy_violation'] = True
        
        # 2. Air resistance effect
        velocity_magnitudes = torch.norm(velocities, dim=-1)
        reynolds_numbers = calculate_reynolds_number(velocity_magnitudes)
        drag_coefficients = adjust_drag_coefficient(reynolds_numbers, self.base_drag_coefficient)
        
        air_resistance = calculate_air_resistance(
            velocities,
            drag_coefficients.mean(),  # Use mean Cd for simplicity
            self.air_density
        )
        
        # Check if deceleration due to air resistance is reasonable
        air_deceleration = torch.norm(air_resistance, dim=-1) / self.mass
        metrics['max_air_deceleration'] = float(torch.max(air_deceleration))
        
        if torch.any(air_deceleration > self.max_acceleration):
            metrics['physically_consistent'] = False
            metrics['air_resistance_violation'] = True
        
        # 3. Magnus effect check if rotations provided
        if rotations is not None:
            magnus_force = calculate_magnus_force(
                velocities,
                rotations,
                self.lift_coefficient,
                self.air_density
            )
            
            magnus_acceleration = torch.norm(magnus_force, dim=-1) / self.mass
            metrics['max_magnus_acceleration'] = float(torch.max(magnus_acceleration))
            
            # Check if Magnus effect is within reasonable bounds
            if torch.any(magnus_acceleration > self.max_acceleration * 0.5):  # Magnus shouldn't dominate
                metrics['physically_consistent'] = False
                metrics['magnus_violation'] = True
        
        return metrics
    
    def _check_game_constraints(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[bool, Dict[str, any]]:
        """Check badminton game-specific constraints."""
        metrics = {'game_valid': True}
        
        # 1. Net height constraint (1.55m at center)
        net_height = 1.55
        net_x_position = 0.0  # Assuming net is at center
        
        # Find points where trajectory crosses net
        x_positions = positions[:, 0]
        net_crossings = torch.where((x_positions[:-1] * x_positions[1:]) <= 0)[0]
        
        if len(net_crossings) > 0:
            for crossing_idx in net_crossings:
                # Interpolate height at net crossing
                t = -x_positions[crossing_idx] / (x_positions[crossing_idx + 1] - x_positions[crossing_idx])
                height_at_net = (1 - t) * positions[crossing_idx, 2] + t * positions[crossing_idx + 1, 2]
                
                if height_at_net < net_height:
                    metrics['game_valid'] = False
                    metrics['net_violation'] = True
                    break
        
        # 2. Service constraints (simplified)
        if torch.norm(velocities[0]) > 100:  # Unrealistic service speed
            metrics['game_valid'] = False
            metrics['service_violation'] = True
        
        # 3. Court boundary constraints with margin
        margin = 0.1  # 10cm margin
        x_valid = torch.all(torch.abs(positions[:, 0]) <= self.court_bounds[0] / 2 + margin)
        y_valid = torch.all(torch.abs(positions[:, 1]) <= self.court_bounds[1] / 2 + margin)
        
        if not (x_valid and y_valid):
            metrics['game_valid'] = False
            metrics['boundary_violation'] = True
        
        return metrics['game_valid'], metrics
    
    def _analyze_smoothness(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze trajectory smoothness."""
        metrics = {}
        
        # 1. Velocity smoothness
        velocity_changes = velocities[1:] - velocities[:-1]
        velocity_smoothness = torch.mean(torch.norm(velocity_changes, dim=-1))
        metrics['velocity_smoothness'] = float(velocity_smoothness)
        
        # 2. Acceleration smoothness
        accelerations = velocity_changes / self.dt
        acceleration_changes = accelerations[1:] - accelerations[:-1]
        acceleration_smoothness = torch.mean(torch.norm(acceleration_changes, dim=-1))
        metrics['acceleration_smoothness'] = float(acceleration_smoothness)
        
        # 3. Curvature analysis
        trajectory_directions = velocities / (torch.norm(velocities, dim=-1, keepdim=True) + 1e-6)
        direction_changes = trajectory_directions[1:] - trajectory_directions[:-1]
        curvature = torch.mean(torch.norm(direction_changes, dim=-1))
        metrics['mean_curvature'] = float(curvature)
        
        return metrics
