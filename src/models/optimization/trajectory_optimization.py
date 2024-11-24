"""
Optimization components for high-speed tracking, trajectory smoothing, and interpolation.
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Dict, Optional
from scipy.signal import savgol_filter
from scipy.optimize import minimize

class TrajectoryOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.smoothing_window = config.get('smoothing_window', 15)
        self.polyorder = config.get('polyorder', 3)
        self.interpolation_fps = config.get('interpolation_fps', 120)
        self.physics_weight = config.get('physics_weight', 0.7)
        
    def optimize_trajectory(
        self,
        points: np.ndarray,
        confidences: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize trajectory using multiple techniques.
        
        Args:
            points: Array of shape [N, 2] containing trajectory points
            confidences: Array of shape [N] containing detection confidences
            timestamps: Array of shape [N] containing timestamps
            
        Returns:
            Optimized points and interpolated timestamps
        """
        # 1. Apply confidence-based filtering
        valid_mask = confidences > self.config['confidence_threshold']
        filtered_points = points[valid_mask]
        filtered_timestamps = timestamps[valid_mask]
        
        if len(filtered_points) < 3:
            return points, timestamps
        
        # 2. Apply Savitzky-Golay smoothing
        smoothed_x = savgol_filter(filtered_points[:, 0], self.smoothing_window, self.polyorder)
        smoothed_y = savgol_filter(filtered_points[:, 1], self.smoothing_window, self.polyorder)
        smoothed_points = np.stack([smoothed_x, smoothed_y], axis=1)
        
        # 3. Physics-based optimization
        optimized_points = self._physics_optimize(smoothed_points, filtered_timestamps)
        
        # 4. Interpolation for high-speed trajectories
        interpolated_points, interpolated_timestamps = self._interpolate_trajectory(
            optimized_points,
            filtered_timestamps
        )
        
        return interpolated_points, interpolated_timestamps
    
    def _physics_optimize(self, points: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Apply physics-based optimization to trajectory points."""
        def physics_cost(params):
            """Cost function incorporating physics constraints."""
            # Reconstruct trajectory from parameters
            positions = params.reshape(-1, 2)
            
            # Velocity and acceleration constraints
            velocities = np.diff(positions, axis=0) / np.diff(timestamps)[:, None]
            accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[:-1])[:, None]
            
            # Physics-based costs
            gravity_cost = np.mean((accelerations[:, 1] + 9.81)**2)  # Gravity constraint
            smoothness_cost = np.mean(np.sum(accelerations**2, axis=1))  # Smoothness
            data_cost = np.mean(np.sum((positions - points)**2, axis=1))  # Data fidelity
            
            return (self.physics_weight * (gravity_cost + smoothness_cost) + 
                   (1 - self.physics_weight) * data_cost)
        
        # Optimize trajectory
        result = minimize(
            physics_cost,
            points.flatten(),
            method='BFGS',
            options={'maxiter': 100}
        )
        
        return result.x.reshape(-1, 2)
    
    def _interpolate_trajectory(
        self,
        points: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate trajectory for smooth high-speed motion."""
        # Create time points for interpolation
        t_total = timestamps[-1] - timestamps[0]
        num_points = int(t_total * self.interpolation_fps)
        t_interp = np.linspace(timestamps[0], timestamps[-1], num_points)
        
        # Fit cubic spline
        cs_x = CubicSpline(timestamps, points[:, 0])
        cs_y = CubicSpline(timestamps, points[:, 1])
        
        # Interpolate points
        interp_x = cs_x(t_interp)
        interp_y = cs_y(t_interp)
        
        return np.stack([interp_x, interp_y], axis=1), t_interp

class ConfidenceScorer:
    def __init__(self, config: Dict):
        self.velocity_threshold = config.get('velocity_threshold', 50)  # pixels per frame
        self.acceleration_threshold = config.get('acceleration_threshold', 10)
        self.physics_weight = config.get('physics_weight', 0.6)
        self.temporal_weight = config.get('temporal_weight', 0.4)
        
    def compute_confidence(
        self,
        points: np.ndarray,
        detection_scores: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Compute confidence scores using physics validation and temporal consistency.
        
        Args:
            points: Array of shape [N, 2] containing trajectory points
            detection_scores: Original detection confidence scores
            timestamps: Timestamp for each point
            
        Returns:
            Array of confidence scores for each point
        """
        if len(points) < 3:
            return detection_scores
        
        # 1. Physics-based confidence
        velocities = np.diff(points, axis=0) / np.diff(timestamps)[:, None]
        accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[:-1])[:, None]
        
        # Velocity constraints
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        velocity_scores = np.exp(-np.abs(velocity_magnitudes - self.velocity_threshold) / 
                               self.velocity_threshold)
        
        # Acceleration constraints
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        acceleration_scores = np.exp(-acceleration_magnitudes / self.acceleration_threshold)
        
        # Pad scores to match original length
        velocity_scores = np.pad(velocity_scores, (0, 1), mode='edge')
        acceleration_scores = np.pad(acceleration_scores, (0, 2), mode='edge')
        
        # 2. Temporal consistency
        temporal_scores = self._compute_temporal_consistency(points, timestamps)
        
        # 3. Combine scores
        physics_scores = (velocity_scores + acceleration_scores) / 2
        final_scores = (self.physics_weight * physics_scores +
                       self.temporal_weight * temporal_scores +
                       (1 - self.physics_weight - self.temporal_weight) * detection_scores)
        
        return final_scores
    
    def _compute_temporal_consistency(
        self,
        points: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """Compute temporal consistency scores."""
        # Use sliding window to compute local consistency
        window_size = min(5, len(points))
        scores = np.zeros(len(points))
        
        for i in range(len(points)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(points), i + window_size // 2 + 1)
            window = points[start_idx:end_idx]
            
            if len(window) < 3:
                scores[i] = 1.0
                continue
            
            # Fit local polynomial and compute residual
            t = timestamps[start_idx:end_idx] - timestamps[start_idx]
            coeffs_x = np.polyfit(t, window[:, 0], 2)
            coeffs_y = np.polyfit(t, window[:, 1], 2)
            
            # Evaluate polynomial at current point
            t_i = timestamps[i] - timestamps[start_idx]
            pred_x = np.polyval(coeffs_x, t_i)
            pred_y = np.polyval(coeffs_y, t_i)
            
            # Compute residual and convert to score
            residual = np.sqrt((pred_x - points[i, 0])**2 + (pred_y - points[i, 1])**2)
            scores[i] = np.exp(-residual / 10)  # Scale factor of 10 pixels
            
        return scores
