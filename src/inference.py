"""
Real-time inference pipeline for video processing.
"""
import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
import time

from src.models.shuttlecock_tracker import ShuttlecockTracker
from src.visualization.visualizer import TrackingVisualizer

class ShuttlecockInference:
    def __init__(
        self,
        model: ShuttlecockTracker,
        config_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.config = self.load_config(config_path)
        self.frame_buffer = []
        self.buffer_size = self.config['inference']['buffer_size']
        self.visualizer = TrackingVisualizer()
        self.metrics_logger = MetricsLogger(self.config['logging']['metrics_path'])
        
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame for inference."""
        # Resize frame
        frame = cv2.resize(frame, (self.config['inference']['width'],
                                 self.config['inference']['height']))
        
        # Convert to torch tensor and normalize
        frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        
        # Apply normalization
        mean = torch.tensor(self.config['inference']['mean']).view(3, 1, 1)
        std = torch.tensor(self.config['inference']['std']).view(3, 1, 1)
        frame = (frame - mean) / std
        
        return frame

    def process_video_stream(self, video_source: int = 0) -> None:
        """Process video stream in real-time."""
        cap = cv2.VideoCapture(video_source)
        self.model.eval()
        
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                processed_frame = processed_frame.unsqueeze(0).to(self.device)
                
                # Update frame buffer
                self.frame_buffer.append(processed_frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                
                # Run inference
                if len(self.frame_buffer) == self.buffer_size:
                    batch = torch.cat(self.frame_buffer, dim=0)
                    detections, tracks = self.model(batch)
                    
                    # Log metrics
                    self.metrics_logger.log_frame_metrics(detections, tracks)
                    
                    # Visualize results
                    frame = self.visualizer.draw_detections(frame, detections)
                    frame = self.visualizer.draw_tracks(frame, tracks)
                    frame = self.visualizer.draw_trajectories(frame, tracks)
                    
                    # Add metrics overlay
                    frame = self.visualizer.draw_metrics_overlay(
                        frame, 
                        self.metrics_logger.get_current_metrics()
                    )
                
                # Display frame
                cv2.imshow('Shuttlecock Tracking', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        self.metrics_logger.save_metrics()

    def process_video_file(self, video_path: str, output_path: str = None) -> None:
        """Process a video file and optionally save the output."""
        cap = cv2.VideoCapture(video_path)
        self.model.eval()
        
        # Setup video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
        
        trajectory_points = []
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                
                if len(self.frame_buffer) >= self.buffer_size:
                    frames_tensor = torch.stack(self.frame_buffer).unsqueeze(0).to(self.device)
                    results = self.model.inference(frames_tensor)
                    
                    # Update trajectory
                    if results['tracks']:
                        point = self.get_shuttlecock_position(results['tracks'])
                        if point:
                            trajectory_points.append(point)
                    
                    # Visualize with trajectory
                    frame = self.visualize_results(frame, results, trajectory_points)
                    self.frame_buffer.pop(0)
                
                if output_path:
                    out.write(frame)
        
        cap.release()
        if output_path:
            out.release()

    def visualize_results(
        self,
        frame: np.ndarray,
        results: Dict,
        trajectory_points: List[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Visualize detection and tracking results on frame."""
        # Draw detections
        for detection in results['detections']:
            x1, y1, x2, y2 = detection[:4]
            conf = detection[4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 2)
            cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw trajectory
        if trajectory_points and len(trajectory_points) > 1:
            points = np.array(trajectory_points, dtype=np.int32)
            cv2.polylines(frame, [points], False, (0, 0, 255), 2)
        
        # Draw predictions
        for pred in results['predictions']:
            x, y = pred[:2]
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        return frame

    @staticmethod
    def get_shuttlecock_position(tracks: List) -> Tuple[int, int]:
        """Extract shuttlecock position from tracking results."""
        if not tracks:
            return None
        # Assume the first track is the shuttlecock
        track = tracks[0]
        return (int(track.mean[0]), int(track.mean[1]))

class MetricsLogger:
    """Log and save inference metrics."""
    def __init__(self, metrics_path: str):
        self.metrics_path = Path(metrics_path)
        self.metrics = {
            'frame_count': 0,
            'detection_count': 0,
            'track_count': 0,
            'fps': 0,
            'confidence_scores': [],
            'track_lengths': []
        }
        self.start_time = None
    
    def log_frame_metrics(self, detections: List[Dict], tracks: List[Dict]) -> None:
        """Log metrics for current frame."""
        if self.start_time is None:
            self.start_time = time.time()
            
        self.metrics['frame_count'] += 1
        self.metrics['detection_count'] += len(detections)
        self.metrics['track_count'] = len(tracks)
        
        # Update FPS
        elapsed_time = time.time() - self.start_time
        self.metrics['fps'] = self.metrics['frame_count'] / elapsed_time
        
        # Log confidence scores and track lengths
        for det in detections:
            self.metrics['confidence_scores'].append(det['score'])
        for track in tracks:
            self.metrics['track_lengths'].append(len(track['trajectory']))
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics for display."""
        return {
            'FPS': f"{self.metrics['fps']:.1f}",
            'Detections': self.metrics['detection_count'],
            'Active Tracks': self.metrics['track_count']
        }
    
    def save_metrics(self) -> None:
        """Save metrics to file."""
        metrics_dir = self.metrics_path.parent
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
