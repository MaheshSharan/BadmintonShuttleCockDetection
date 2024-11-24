"""
Physics Utilities
This module implements various physics calculations for shuttlecock trajectory modeling.
"""

import torch
import numpy as np
from typing import Tuple


def calculate_air_resistance(
    velocity: torch.Tensor,
    drag_coefficient: float,
    air_density: float,
    cross_section: float = 0.00125  # m², approximate shuttlecock cross-section
) -> torch.Tensor:
    """
    Calculate air resistance force using the drag equation.
    F_drag = -0.5 * ρ * v² * C_d * A * v̂
    
    Args:
        velocity: Velocity vector (B, 3)
        drag_coefficient: Drag coefficient
        air_density: Air density in kg/m³
        cross_section: Shuttlecock cross-sectional area
        
    Returns:
        Air resistance force vector
    """
    velocity_magnitude = torch.norm(velocity, dim=-1, keepdim=True)
    velocity_direction = velocity / (velocity_magnitude + 1e-6)
    
    # Drag force magnitude
    force_magnitude = -0.5 * air_density * velocity_magnitude**2 * drag_coefficient * cross_section
    
    # Force vector
    return force_magnitude * velocity_direction


def calculate_magnus_force(
    velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    lift_coefficient: float,
    air_density: float,
    radius: float = 0.0325  # m, approximate shuttlecock radius
) -> torch.Tensor:
    """
    Calculate Magnus force on the spinning shuttlecock.
    F_M = 0.5 * ρ * v² * C_L * A * (ω × v̂)
    
    Args:
        velocity: Velocity vector (B, 3)
        angular_velocity: Angular velocity vector (B, 3)
        lift_coefficient: Lift coefficient
        air_density: Air density in kg/m³
        radius: Shuttlecock characteristic radius
        
    Returns:
        Magnus force vector
    """
    velocity_magnitude = torch.norm(velocity, dim=-1, keepdim=True)
    velocity_direction = velocity / (velocity_magnitude + 1e-6)
    
    # Cross product of angular velocity and velocity direction
    cross_product = torch.cross(angular_velocity, velocity_direction, dim=-1)
    
    # Magnus force magnitude
    force_magnitude = 0.5 * air_density * velocity_magnitude**2 * lift_coefficient * np.pi * radius**2
    
    # Force vector
    return force_magnitude * cross_product


def calculate_gravity_effect(
    batch_size: int,
    gravity: float = 9.81,
    device: torch.device = None
) -> torch.Tensor:
    """
    Calculate gravity force vector.
    
    Args:
        batch_size: Batch size
        gravity: Gravitational acceleration (m/s²)
        device: Torch device
        
    Returns:
        Gravity force vector (B, 3)
    """
    gravity_vector = torch.zeros(batch_size, 3, device=device)
    gravity_vector[:, 2] = -gravity  # z-axis points up
    return gravity_vector


def calculate_reynolds_number(
    velocity: torch.Tensor,
    characteristic_length: float = 0.065,  # m, shuttlecock diameter
    kinematic_viscosity: float = 1.48e-5  # m²/s, air at 20°C
) -> torch.Tensor:
    """
    Calculate Reynolds number for the flow around the shuttlecock.
    Re = (v * L) / ν
    
    Args:
        velocity: Velocity magnitude
        characteristic_length: Characteristic length (diameter)
        kinematic_viscosity: Kinematic viscosity of air
        
    Returns:
        Reynolds number
    """
    return velocity * characteristic_length / kinematic_viscosity


def adjust_drag_coefficient(
    reynolds_number: torch.Tensor,
    base_drag_coefficient: float = 0.6
) -> torch.Tensor:
    """
    Adjust drag coefficient based on Reynolds number.
    
    Args:
        reynolds_number: Reynolds number
        base_drag_coefficient: Base drag coefficient
        
    Returns:
        Adjusted drag coefficient
    """
    # Simplified adjustment based on typical Reynolds number ranges
    adjustment = torch.ones_like(reynolds_number)
    
    # Low Reynolds number regime
    mask_low = reynolds_number < 2e4
    adjustment[mask_low] = 1.2
    
    # High Reynolds number regime
    mask_high = reynolds_number > 1e5
    adjustment[mask_high] = 0.8
    
    return base_drag_coefficient * adjustment


def calculate_air_density(
    temperature: float = 20.0,  # °C
    pressure: float = 101325,   # Pa
    humidity: float = 0.5       # 50% relative humidity
) -> float:
    """
    Calculate air density based on environmental conditions.
    
    Args:
        temperature: Temperature in Celsius
        pressure: Atmospheric pressure in Pascal
        humidity: Relative humidity (0-1)
        
    Returns:
        Air density in kg/m³
    """
    # Convert temperature to Kelvin
    T = temperature + 273.15
    
    # Dry air density
    rho_dry = pressure / (287.05 * T)
    
    # Water vapor pressure (simplified August equation)
    p_sat = 610.78 * np.exp(17.2694 * temperature / (temperature + 238.3))
    p_vapor = humidity * p_sat
    
    # Moist air density
    rho = (pressure - 0.378 * p_vapor) / (287.05 * T)
    
    return rho


def validate_trajectory(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    court_bounds: Tuple[float, float, float] = (13.4, 6.1, 8.0),
    max_velocity: float = 137.0,  # m/s
    dt: float = 1/30  # time step
) -> Tuple[bool, str]:
    """
    Validate if a trajectory is physically feasible.
    
    Args:
        positions: Position vectors (T, 3)
        velocities: Velocity vectors (T, 3)
        court_bounds: Court dimensions (length, width, height)
        max_velocity: Maximum allowed velocity
        dt: Time step
        
    Returns:
        Tuple of (is_valid, reason)
    """
    # Check court boundaries
    x_valid = torch.all(torch.abs(positions[:, 0]) <= court_bounds[0] / 2)
    y_valid = torch.all(torch.abs(positions[:, 1]) <= court_bounds[1] / 2)
    z_valid = torch.all(positions[:, 2] >= 0)
    
    if not (x_valid and y_valid and z_valid):
        return False, "Trajectory exceeds court boundaries"
    
    # Check velocity constraints
    velocity_magnitudes = torch.norm(velocities, dim=-1)
    if torch.any(velocity_magnitudes > max_velocity):
        return False, f"Velocity exceeds maximum limit of {max_velocity} m/s"
    
    # Check acceleration constraints
    accelerations = (velocities[1:] - velocities[:-1]) / dt
    acc_magnitudes = torch.norm(accelerations, dim=-1)
    max_acc = 100.0  # m/s²
    
    if torch.any(acc_magnitudes > max_acc):
        return False, f"Acceleration exceeds maximum limit of {max_acc} m/s²"
    
    return True, "Trajectory is physically feasible"
