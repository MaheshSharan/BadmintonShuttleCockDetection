"""
Main tracker module implementing modified DeepSORT for shuttlecock tracking.
Integrates detection, appearance features, and motion prediction.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from collections import deque

from .track import Track
from .kalman_filter import ShuttlecockKalmanFilter
from .pinn import TrajectoryPINN
from .feature_extractor import ShuttlecockFeatureExtractor, FeatureDatabase


class ShuttlecockTracker:
    """
    Modified DeepSORT tracker optimized for shuttlecock tracking.
    Integrates physics-based motion model with appearance features.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        max_acceleration: float = 100.0,  # m/s²
        max_velocity: float = 137.0,  # m/s (fastest recorded shuttlecock speed)
    ):
        """
        Initialize tracker with shuttlecock-specific parameters.
        
        Args:
            max_age: Maximum frames to keep track without detection
            n_init: Minimum detections to confirm track
            max_iou_distance: Maximum IOU distance for detection association
            max_cosine_distance: Maximum cosine distance for feature matching
            max_acceleration: Maximum allowed acceleration
            max_velocity: Maximum allowed velocity
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.max_acceleration = max_acceleration
        self.max_velocity = max_velocity
        
        self.tracks: List[Track] = []
        self.deleted_tracks: List[Track] = []
        self.track_id_count: int = 1
        
        # Track motion statistics
        self.velocity_history = deque(maxlen=100)
        self.acceleration_history = deque(maxlen=100)
        
        # Initialize PINN and feature extractor
        self.pinn = TrajectoryPINN()
        self.feature_extractor = ShuttlecockFeatureExtractor()
        self.feature_db = FeatureDatabase()
        
    def predict(self) -> None:
        """
        Predict track states and update motion statistics.
        """
        for track in self.tracks:
            predicted_pos = track.predict()
            
            # Get current velocity and acceleration
            if len(track.velocities) > 0:
                current_velocity = track.velocities[-1]
                velocity_magnitude = np.linalg.norm(current_velocity)
                self.velocity_history.append(velocity_magnitude)
                
                if len(track.accelerations) > 0:
                    current_acceleration = track.accelerations[-1]
                    acceleration_magnitude = np.linalg.norm(current_acceleration)
                    self.acceleration_history.append(acceleration_magnitude)
                    
                    # Check for physically impossible motions
                    if (velocity_magnitude > self.max_velocity or 
                        acceleration_magnitude > self.max_acceleration):
                        track.confidence *= 0.5  # Reduce confidence in track
    
    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        """
        Perform measurement update and track management.
        
        Args:
            detections: Array of detections in format [N,3] for N detections
            confidences: Array of confidence values [N] for N detections
        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._associate_detections_to_tracks(detections, confidences)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                detections[detection_idx],
                confidences[detection_idx]
            )
        
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(
                detections[detection_idx],
                confidences[detection_idx]
            )
            
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        print(f"After update: {len(self.tracks)} tracks")
        for track in self.tracks:
            print(f"  Track {track.track_id}: state={track.state[:3]}, hits={track.hits}")
    
    def _associate_detections_to_tracks(
        self,
        detections: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
        """Associate detections to existing tracks using motion and appearance."""
        if len(self.tracks) == 0:
            # If no tracks exist, all detections are unmatched
            return [], np.arange(len(detections)), np.array([])
            
        if len(detections) == 0:
            # If no detections, all tracks are unmatched
            return [], np.array([]), np.arange(len(self.tracks))

        # Calculate cost matrix
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        for i, track in enumerate(self.tracks):
            # Compute gating distance for all tracks, not just confirmed ones
            gating_dist = track.kf.gating_distance(
                track.state, track.covariance, detections
            )
            cost_matrix[:, i] = gating_dist

        # Perform matching
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(cost_matrix) > 0:
            # Use Hungarian algorithm for matching
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter matches using gating threshold
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] > self.max_iou_distance:
                    continue
                matches.append((row, col))
                unmatched_detections.remove(row)
                unmatched_tracks.remove(col)

        print(f"Matches: {matches}")
        print(f"Unmatched detections: {unmatched_detections}")
        print(f"Unmatched tracks: {unmatched_tracks}")
        
        return matches, np.array(unmatched_tracks), np.array(unmatched_detections)
    
    def _initiate_track(
        self,
        detection: np.ndarray,
        confidence: float,
    ) -> None:
        """
        Create and initialize a new track.
        
        Args:
            detection: Detection position [3]
            confidence: Detection confidence
        """
        new_track = Track(
            self.track_id_count,
            detection,
            confidence=confidence
        )
            
        self.tracks.append(new_track)
        self.track_id_count += 1
    
    def get_active_tracks(self) -> List[Track]:
        """Get list of active (confirmed) tracks."""
        return [track for track in self.tracks if track.is_confirmed()]
    
    def get_track_states(self) -> Dict[str, np.ndarray]:
        """
        Get current states of all active tracks.
        
        Returns:
            Dictionary containing positions, velocities, and accelerations
        """
        active_tracks = self.get_active_tracks()
        states = {
            'positions': np.array([t.state[:3] for t in active_tracks]),
            'velocities': np.array([t.state[3:6] for t in active_tracks]),
            'accelerations': np.array([t.state[6:] for t in active_tracks])
        }
        return states
    
    def get_motion_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get motion statistics for trajectory analysis.
        
        Returns:
            Dictionary containing velocity and acceleration statistics
        """
        velocity_stats = np.array(self.velocity_history)
        acceleration_stats = np.array(self.acceleration_history)
        
        return {
            'velocity_mean': np.mean(velocity_stats) if len(velocity_stats) > 0 else 0,
            'velocity_std': np.std(velocity_stats) if len(velocity_stats) > 0 else 0,
            'acceleration_mean': np.mean(acceleration_stats) if len(acceleration_stats) > 0 else 0,
            'acceleration_std': np.std(acceleration_stats) if len(acceleration_stats) > 0 else 0
        }
