"""
Visualization module for tracking results.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple


class TrackingVisualizer:
    """Visualize tracking results with detections, tracks, and trajectories."""
    
    def __init__(
        self,
        detection_color: Tuple[int, int, int] = (0, 255, 0),  # Green
        track_color: Tuple[int, int, int] = (255, 0, 0),      # Red
        trajectory_color: Tuple[int, int, int] = (0, 0, 255),  # Blue
        text_color: Tuple[int, int, int] = (255, 255, 255),   # White
        line_thickness: int = 2,
        font_scale: float = 0.5
    ):
        self.detection_color = detection_color
        self.track_color = track_color
        self.trajectory_color = trajectory_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes and scores."""
        frame = frame.copy()
        for det in detections:
            box = det['bbox']
            score = det['score']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.detection_color, self.line_thickness)
            
            # Draw confidence score
            text = f"{score:.2f}"
            text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
            cv2.putText(frame, text, (x1, y1 - 5), self.font, self.font_scale,
                       self.text_color, self.line_thickness)
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Draw track boxes and IDs."""
        frame = frame.copy()
        for track in tracks:
            box = track['bbox']
            track_id = track['id']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.track_color, self.line_thickness)
            
            # Draw track ID
            text = f"ID: {track_id}"
            cv2.putText(frame, text, (x1, y2 + 20), self.font, self.font_scale,
                       self.text_color, self.line_thickness)
        
        return frame
    
    def draw_trajectories(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Draw track trajectories."""
        frame = frame.copy()
        for track in tracks:
            trajectory = track['trajectory']
            if len(trajectory) < 2:
                continue
            
            # Convert trajectory to numpy array of points
            points = np.array(trajectory, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            
            # Draw trajectory line
            cv2.polylines(frame, [points], False, self.trajectory_color,
                         thickness=self.line_thickness)
        
        return frame
    
    def draw_metrics_overlay(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Draw metrics overlay on frame."""
        frame = frame.copy()
        y_offset = 30
        
        for key, value in metrics.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), self.font, self.font_scale,
                       self.text_color, self.line_thickness)
            y_offset += 25
        
        return frame
