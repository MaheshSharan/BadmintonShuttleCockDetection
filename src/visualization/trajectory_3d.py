"""
3D trajectory visualization for shuttlecock tracking.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import torch
from pathlib import Path
import logging
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)

class Trajectory3DVisualizer:
    """3D visualization of shuttlecock trajectories."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Initialize 3D visualizer.
        
        Args:
            config: Visualization configuration
            output_dir: Directory to save visualizations
            interactive: Whether to show interactive plot
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.interactive = interactive
        
        # Setup 3D plot
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        
        # Animation properties
        self.animation = None
        self.trajectory_data = []
        
    def setup_plot(self):
        """Setup 3D plot properties."""
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title('Shuttlecock 3D Trajectory')
        
        # Set court dimensions (standard badminton court)
        self.ax.set_xlim([-3, 3])  # Half court width
        self.ax.set_ylim([0, 13.4])  # Court length
        self.ax.set_zlim([0, 8])  # Height
        
        # Draw court outline
        self._draw_court()
        
    def _draw_court(self):
        """Draw badminton court outline."""
        # Court dimensions (in meters)
        court_width = 6.1
        court_length = 13.4
        service_line = 1.98
        
        # Draw court boundaries
        x = [-court_width/2, court_width/2, court_width/2, -court_width/2, -court_width/2]
        y = [0, 0, court_length, court_length, 0]
        z = [0, 0, 0, 0, 0]
        
        self.ax.plot(x, y, z, 'k-', linewidth=1)
        
        # Draw net
        net_x = [-court_width/2, court_width/2]
        net_y = [court_length/2, court_length/2]
        net_z = [0, 0]
        net_height = 1.55
        
        self.ax.plot(net_x, net_y, [0, 0], 'k-', linewidth=2)
        self.ax.plot(net_x, net_y, [net_height, net_height], 'k-', linewidth=2)
        self.ax.plot([-court_width/2, -court_width/2], [court_length/2, court_length/2], [0, net_height], 'k-', linewidth=2)
        self.ax.plot([court_width/2, court_width/2], [court_length/2, court_length/2], [0, net_height], 'k-', linewidth=2)
        
    def visualize_trajectory(
        self,
        trajectory: List[Dict[str, np.ndarray]],
        predictions: Optional[List[Dict[str, np.ndarray]]] = None,
        physics_data: Optional[Dict] = None
    ):
        """
        Visualize 3D trajectory with optional predictions and physics data.
        
        Args:
            trajectory: List of trajectory points with 3D coordinates
            predictions: List of predicted trajectory points
            physics_data: Physics-based trajectory information
        """
        # Plot actual trajectory
        points = np.array([p['position'] for p in trajectory])
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', label='Actual', linewidth=2)
        self.ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c='b', marker='o')
        
        # Plot predictions if available
        if predictions:
            pred_points = np.array([p['position'] for p in predictions])
            self.ax.plot(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                        'r--', label='Predicted', linewidth=2)
            
        # Add physics visualization if available
        if physics_data:
            self._visualize_physics(physics_data)
            
        self.ax.legend()
        
        if self.interactive:
            plt.show()
            
    def _visualize_physics(self, physics_data: Dict):
        """Visualize physics-based information."""
        # Draw velocity vectors
        if 'velocity' in physics_data:
            vel = physics_data['velocity']
            pos = physics_data['position']
            self.ax.quiver(pos[0], pos[1], pos[2],
                         vel[0], vel[1], vel[2],
                         color='g', label='Velocity')
            
        # Draw acceleration vectors
        if 'acceleration' in physics_data:
            acc = physics_data['acceleration']
            pos = physics_data['position']
            self.ax.quiver(pos[0], pos[1], pos[2],
                         acc[0], acc[1], acc[2],
                         color='y', label='Acceleration')
            
    def animate_trajectory(
        self,
        trajectory: List[Dict[str, np.ndarray]],
        predictions: Optional[List[Dict[str, np.ndarray]]] = None,
        interval: int = 50
    ):
        """
        Create animation of trajectory evolution.
        
        Args:
            trajectory: List of trajectory points
            predictions: List of predicted points
            interval: Animation interval in milliseconds
        """
        self.trajectory_data = trajectory
        self.prediction_data = predictions
        
        def update(frame):
            self.ax.cla()
            self.setup_plot()
            
            # Plot trajectory up to current frame
            points = np.array([p['position'] for p in trajectory[:frame+1]])
            self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', label='Actual')
            
            # Plot current predictions if available
            if predictions and frame < len(predictions):
                pred_points = np.array([p['position'] for p in predictions[frame]])
                self.ax.plot(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                            'r--', label='Predicted')
                
            self.ax.legend()
            
        self.animation = FuncAnimation(
            self.fig, update,
            frames=len(trajectory),
            interval=interval,
            blit=False
        )
        
        if self.interactive:
            plt.show()
            
    def save_visualization(self, filename: str):
        """Save current visualization or animation."""
        if self.output_dir:
            save_path = self.output_dir / filename
            if self.animation:
                self.animation.save(str(save_path), writer='pillow')
            else:
                plt.savefig(str(save_path))
                
    def close(self):
        """Clean up resources."""
        plt.close(self.fig)
