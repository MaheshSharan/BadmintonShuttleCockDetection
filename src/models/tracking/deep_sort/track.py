"""
Track class for shuttlecock tracking.
Maintains state information and provides prediction capabilities.
"""

import numpy as np
from typing import Optional, List, Tuple
from .kalman_filter import ShuttlecockKalmanFilter
from enum import Enum

class TrackState(Enum):
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    """
    Track class for shuttlecock tracking with specialized motion model.
    """
    
    def __init__(
        self,
        track_id: int,
        detection: np.ndarray,
        confidence: float = 1.0,
        max_age: int = 30
    ):
        """
        Initialize track with initial detection.
        
        Args:
            track_id: Unique track identifier
            detection: Initial detection [x, y, z]
            confidence: Detection confidence
            max_age: Maximum age of the track
        """
        self.track_id = track_id
        self.confidence = confidence
        self._max_age = max_age
        
        # Initialize Kalman filter state
        self.kf = ShuttlecockKalmanFilter()
        self.state = np.zeros(9)  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.state[:3] = detection
        self.covariance = np.eye(9)
        
        # Track state
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state_flags = TrackState.Tentative
        
        # Store trajectory
        self.positions = [detection]  # Store positions for visualization
        self.velocities = []  # Store velocities for analysis
        self.accelerations = []  # Store accelerations for analysis
        
    def predict(self) -> np.ndarray:
        """Propagate the state distribution to the current time step."""
        self.state, self.covariance = self.kf.predict(
            self.state, self.covariance
        )
        self.age += 1
        self.time_since_update += 1
        print(f"Track {self.track_id} predict: state={self.state[:3]}")
        return self.state[:3]
    
    def update(
        self,
        detection: np.ndarray,
        confidence: float = 1.0,
        dt: float = 1/30.0
    ) -> None:
        """
        Update this track with observed detection.
        
        Args:
            detection: Detection position [x, y, z]
            confidence: Detection confidence
            dt: Time delta since last update
        """
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        
        # Update Kalman filter state
        self.kf.update(detection)
        self.state = self.kf.x
        self.covariance = self.kf.P
        
        # Store position for visualization
        self.positions.append(detection)
        
        # Store velocity for analysis
        if len(self.positions) >= 2:
            pos_arr = np.array(self.positions[-2:])
            vel = (pos_arr[1] - pos_arr[0]) / dt
            self.velocities.append(vel)
            
            # Calculate acceleration
            if len(self.velocities) > 1:
                vel_arr = np.array(self.velocities[-2:])
                acc = (vel_arr[1] - vel_arr[0]) / dt
                self.accelerations.append(acc)
        
        # Update state flags
        self._update_state_flags()
        
        print(f"Track {self.track_id} status: hits={self.hits}, confirmed={self.is_confirmed()}")
    
    def mark_missed(self) -> None:
        """Mark this track as missed (no association at the current time step)."""
        self.time_since_update += 1
        if self.time_since_update > self._max_age:
            self.state_flags = TrackState.Deleted
            
    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (not yet confirmed)."""
        return self.state_flags == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state_flags == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """Returns True if this track is deleted."""
        return self.state_flags == TrackState.Deleted
        
    def _update_state_flags(self) -> None:
        """Update track state flags based on hits and time since update."""
        if self.hits >= 3:
            self.state_flags = TrackState.Confirmed
        elif self.time_since_update > self._max_age:
            self.state_flags = TrackState.Deleted
