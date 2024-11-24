import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrajectoryVisualizer:
    """Visualize shuttlecock trajectories and motion patterns."""
    
    def __init__(self, config: dict):
        """
        Initialize trajectory visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config['analysis']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')  # Using the updated seaborn style name
        self.dpi = config['reporting']['dpi']
        self.fig_size = tuple(config['reporting']['figure_size'])
        
        # Visualization parameters
        self.viz_config = config['analysis']['visualization']['trajectory']
    
    def visualize_trajectories(self, trajectories: List[np.ndarray], 
                             title: str = "Shuttlecock Trajectories"):
        """
        Visualize multiple trajectories in 2D.
        
        Args:
            trajectories: List of trajectory arrays (Nx2)
            title: Plot title
        """
        plt.figure(figsize=self.fig_size)
        
        for traj in trajectories:
            if len(traj) > 1:  # Only plot if we have at least 2 points
                plt.plot(traj[:, 0], traj[:, 1], 
                        alpha=self.viz_config['line_alpha'],
                        marker='o', 
                        markersize=self.viz_config['marker_size'])
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trajectories_2d.png', dpi=self.dpi)
        plt.close()
    
    def visualize_3d_trajectories(self, trajectories: List[np.ndarray], 
                                frame_indices: List[np.ndarray],
                                title: str = "3D Shuttlecock Trajectories"):
        """
        Visualize trajectories in 3D (x, y, time).
        
        Args:
            trajectories: List of trajectory arrays (Nx2)
            frame_indices: List of frame indices for each trajectory
            title: Plot title
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        for traj, frames in zip(trajectories, frame_indices):
            if len(traj) > 1:
                ax.plot3D(traj[:, 0], traj[:, 1], frames,
                         alpha=self.viz_config['line_alpha'],
                         marker='o',
                         markersize=self.viz_config['marker_size'])
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Frame Number')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trajectories_3d.png', dpi=self.dpi)
        plt.close()
    
    def plot_velocity_profile(self, trajectories: List[np.ndarray],
                            frame_indices: List[np.ndarray]):
        """
        Plot velocity profiles for trajectories.
        
        Args:
            trajectories: List of trajectory arrays (Nx2)
            frame_indices: List of frame indices for each trajectory
        """
        plt.figure(figsize=self.fig_size)
        
        for traj, frames in zip(trajectories, frame_indices):
            if len(traj) > 1:
                # Calculate velocities
                velocities = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
                plt.plot(frames[1:], velocities,
                        alpha=self.viz_config['line_alpha'])
        
        plt.xlabel('Frame Number')
        plt.ylabel('Velocity (pixels/frame)')
        plt.title('Shuttlecock Velocity Profiles')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'velocity_profiles.png', dpi=self.dpi)
        plt.close()
    
    def plot_acceleration_profile(self, trajectories: List[np.ndarray],
                                frame_indices: List[np.ndarray]):
        """
        Plot acceleration profiles for trajectories.
        
        Args:
            trajectories: List of trajectory arrays (Nx2)
            frame_indices: List of frame indices for each trajectory
        """
        plt.figure(figsize=self.fig_size)
        
        for traj, frames in zip(trajectories, frame_indices):
            if len(traj) > 2:  # Need at least 3 points for acceleration
                # Calculate velocities and accelerations
                velocities = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
                accelerations = np.diff(velocities)
                
                plt.plot(frames[2:], accelerations,
                        alpha=self.viz_config['line_alpha'])
        
        plt.xlabel('Frame Number')
        plt.ylabel('Acceleration (pixels/frame²)')
        plt.title('Shuttlecock Acceleration Profiles')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'acceleration_profiles.png', dpi=self.dpi)
        plt.close()
    
    def plot_motion_characteristics(self, trajectories: List[np.ndarray]):
        """
        Plot motion characteristics distribution.
        
        Args:
            trajectories: List of trajectory arrays (Nx2)
        """
        velocities = []
        accelerations = []
        
        for traj in trajectories:
            if len(traj) > 2:
                # Calculate velocities and accelerations
                vel = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
                acc = np.diff(vel)
                
                velocities.extend(vel)
                accelerations.extend(acc)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size)
        
        # Plot velocity distribution
        sns.histplot(velocities, bins=30, ax=ax1)
        ax1.set_title('Velocity Distribution')
        ax1.set_xlabel('Velocity (pixels/frame)')
        ax1.set_ylabel('Count')
        
        # Plot acceleration distribution
        sns.histplot(accelerations, bins=30, ax=ax2)
        ax2.set_title('Acceleration Distribution')
        ax2.set_xlabel('Acceleration (pixels/frame²)')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'motion_characteristics.png', dpi=self.dpi)
        plt.close()
