"""
Kalman Filter Module optimized for shuttlecock tracking.
Implements specialized motion model considering shuttlecock's unique aerodynamics.
"""

import numpy as np
from typing import Tuple, Optional
import scipy.linalg


class ShuttlecockKalmanFilter:
    """
    Custom Kalman filter for shuttlecock tracking with aerodynamic considerations.
    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    where:
        - (x, y, z): 3D position
        - (vx, vy, vz): 3D velocity
        - (ax, ay, az): 3D acceleration (including drag and Magnus effects)
    """
    
    def __init__(self):
        # State transition matrix
        self.F = np.eye(9)  # [x,y,z, vx,vy,vz, ax,ay,az]
        dt = 1/30.0  # Assume 30fps
        
        # Position update from velocity
        self.F[0:3, 3:6] = np.eye(3) * dt
        # Velocity update from acceleration
        self.F[3:6, 6:9] = np.eye(3) * dt
        
        # Process noise
        self.Q = np.eye(9) * 0.1
        self.Q[6:9, 6:9] *= 10  # Higher uncertainty in acceleration
        
        # Measurement matrix (we only observe position)
        self.H = np.zeros((3, 9))
        self.H[0:3, 0:3] = np.eye(3)
        
        # Measurement noise
        self.R = np.eye(3) * 1.0
        
        # Initial state
        self.x = np.zeros(9)
        self.P = np.eye(9) * 100
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the state forward by one timestep."""
        # Predict state
        self.x = self.F @ self.x
        
        # Update covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x, self.P
        
    def update(self, measurement: np.ndarray) -> None:
        """Update state with measurement."""
        # Compute Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(9)
        self.P = (I - K @ self.H) @ self.P
        
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = True
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        
        Args:
            mean: State vector [9]
            covariance: State covariance [9,9]
            measurements: Array of measurements [N,3]
            only_position: If True, only use position for gating
            
        Returns:
            Array of N distances
        """
        if only_position:
            mean = mean[:3]
            covariance = covariance[:3, :3]
            
        d = measurements - mean
        
        if len(d.shape) == 1:
            d = d.reshape(1, -1)
            
        # Compute Mahalanobis distance
        S = covariance + self.R
        distances = []
        
        for i in range(len(measurements)):
            dist = d[i].T @ np.linalg.inv(S) @ d[i]
            distances.append(dist)
            
        return np.array(distances)
