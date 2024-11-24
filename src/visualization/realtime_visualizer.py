"""
Real-time visualization module for shuttlecock detection.
"""
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

logger = logging.getLogger(__name__)

class RealtimeVisualizer:
    """Real-time visualization of shuttlecock detection and tracking."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        save_video: bool = False,
        display: bool = True
    ):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
            output_dir: Directory to save visualizations
            save_video: Whether to save video output
            display: Whether to display visualization window
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_video = save_video
        self.display = display
        
        # Setup colors and styles
        self.colors = {
            'detection': (0, 255, 0),  # Green
            'tracking': (255, 0, 0),   # Red
            'prediction': (0, 0, 255)  # Blue
        }
        
        # Video writer setup
        self.video_writer = None
        if save_video and output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize display window
        if display:
            cv2.namedWindow('Shuttlecock Detection', cv2.WINDOW_NORMAL)
            
    def visualize_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        trajectories: List[Dict],
        predictions: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Visualize detections, tracking, and predictions on a frame.
        
        Args:
            frame: Input frame
            detections: List of detection results
            trajectories: List of tracking trajectories
            predictions: List of trajectory predictions
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            box = det['bbox']
            conf = det['confidence']
            self._draw_bbox(vis_frame, box, f'Shuttle: {conf:.2f}', self.colors['detection'])
            
        # Draw tracking trajectories
        for traj in trajectories:
            points = traj['points']
            track_id = traj['track_id']
            self._draw_trajectory(vis_frame, points, track_id, self.colors['tracking'])
            
        # Draw predictions if available
        if predictions:
            for pred in predictions:
                points = pred['points']
                conf = pred['confidence']
                self._draw_prediction(vis_frame, points, conf, self.colors['prediction'])
                
        return vis_frame
        
    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        label: str,
        color: Tuple[int, int, int]
    ):
        """Draw bounding box with label."""
        x1, y1, w, h = [int(c) for c in bbox]
        x2, y2 = x1 + w, y1 + h
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - 10), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def _draw_trajectory(
        self,
        frame: np.ndarray,
        points: List[Tuple[float, float]],
        track_id: int,
        color: Tuple[int, int, int]
    ):
        """Draw tracking trajectory."""
        points = np.array(points, dtype=np.int32)
        
        # Draw trajectory line
        if len(points) > 1:
            cv2.polylines(frame, [points], False, color, 2)
            
        # Draw current position
        if len(points) > 0:
            current_pos = tuple(points[-1])
            cv2.circle(frame, current_pos, 4, color, -1)
            cv2.putText(frame, f'ID: {track_id}', 
                       (current_pos[0] + 5, current_pos[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    def _draw_prediction(
        self,
        frame: np.ndarray,
        points: List[Tuple[float, float]],
        confidence: float,
        color: Tuple[int, int, int]
    ):
        """Draw predicted trajectory."""
        points = np.array(points, dtype=np.int32)
        
        # Draw prediction line (dashed)
        if len(points) > 1:
            for i in range(len(points) - 1):
                pt1 = tuple(points[i])
                pt2 = tuple(points[i + 1])
                # Create dashed line effect
                if i % 2 == 0:
                    cv2.line(frame, pt1, pt2, color, 1)
                    
        # Draw confidence
        if len(points) > 0:
            end_pos = tuple(points[-1])
            cv2.putText(frame, f'Conf: {confidence:.2f}',
                       (end_pos[0] + 5, end_pos[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    def display_frame(self, frame: np.ndarray):
        """Display frame in window."""
        if self.display:
            cv2.imshow('Shuttlecock Detection', frame)
            cv2.waitKey(1)
            
    def save_frame(self, frame: np.ndarray, frame_idx: int):
        """Save frame to video file."""
        if self.save_video:
            if self.video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    str(self.output_dir / 'output.mp4'),
                    fourcc, 30.0, (w, h)
                )
            self.video_writer.write(frame)
            
    def close(self):
        """Clean up resources."""
        if self.display:
            cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()
