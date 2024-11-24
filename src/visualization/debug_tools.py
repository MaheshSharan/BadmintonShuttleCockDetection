"""
Debug visualization tools for model development.
"""
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

logger = logging.getLogger(__name__)

class DebugVisualizer:
    """Debug visualization tools for model development."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        save_debug: bool = True
    ):
        """
        Initialize debug visualizer.
        
        Args:
            config: Visualization configuration
            output_dir: Directory to save debug visualizations
            save_debug: Whether to save debug outputs
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_debug = save_debug
        
        if save_debug and output_dir:
            self.debug_dir = Path(output_dir) / 'debug'
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
    def visualize_feature_maps(
        self,
        feature_maps: torch.Tensor,
        layer_name: str,
        max_channels: int = 16
    ) -> plt.Figure:
        """
        Visualize feature maps from a specific layer.
        
        Args:
            feature_maps: Feature maps tensor [C, H, W]
            layer_name: Name of the layer
            max_channels: Maximum number of channels to display
            
        Returns:
            Figure object
        """
        # Convert to numpy and normalize
        features = feature_maps.detach().cpu().numpy()
        n_channels = min(features.shape[0], max_channels)
        
        # Create grid plot
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flat
        
        for i in range(n_channels):
            feature = features[i]
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
            axes[i].imshow(feature, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Channel {i}')
            
        # Turn off unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f'Feature Maps - {layer_name}')
        plt.tight_layout()
        
        if self.save_debug:
            plt.savefig(self.debug_dir / f'feature_maps_{layer_name}.png')
            
        return fig
        
    def visualize_attention(
        self,
        attention_weights: torch.Tensor,
        frame: np.ndarray,
        save_prefix: str = 'attention'
    ) -> np.ndarray:
        """
        Visualize attention weights overlaid on frame.
        
        Args:
            attention_weights: Attention weights [H, W]
            frame: Input frame
            save_prefix: Prefix for saved files
            
        Returns:
            Visualization image
        """
        # Normalize attention weights
        attention = attention_weights.detach().cpu().numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Resize to frame size
        attention = cv2.resize(attention, (frame.shape[1], frame.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attention * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay on frame
        alpha = 0.5
        vis_frame = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
        
        if self.save_debug:
            cv2.imwrite(str(self.debug_dir / f'{save_prefix}.png'), vis_frame)
            
        return vis_frame
        
    def visualize_gradients(
        self,
        gradients: torch.Tensor,
        parameter_name: str
    ) -> plt.Figure:
        """
        Visualize gradient distributions.
        
        Args:
            gradients: Gradient tensor
            parameter_name: Name of parameter
            
        Returns:
            Figure object
        """
        grads = gradients.detach().cpu().numpy().flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1.hist(grads, bins=50, alpha=0.7)
        ax1.set_title(f'Gradient Distribution - {parameter_name}')
        ax1.set_xlabel('Gradient Value')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(grads)
        ax2.set_title(f'Gradient Statistics - {parameter_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_debug:
            plt.savefig(self.debug_dir / f'gradients_{parameter_name}.png')
            
        return fig
        
    def visualize_physics_debug(
        self,
        trajectory: List[Dict],
        physics_debug: Dict,
        save_prefix: str = 'physics'
    ) -> List[plt.Figure]:
        """
        Visualize physics-based debug information.
        
        Args:
            trajectory: Trajectory points
            physics_debug: Physics debug information
            save_prefix: Prefix for saved files
            
        Returns:
            List of figure objects
        """
        figs = []
        
        # Energy conservation
        if 'energy' in physics_debug:
            fig_energy, ax_energy = plt.subplots(figsize=(10, 6))
            energy_data = physics_debug['energy']
            ax_energy.plot(energy_data['kinetic'], label='Kinetic')
            ax_energy.plot(energy_data['potential'], label='Potential')
            ax_energy.plot(energy_data['total'], label='Total')
            ax_energy.set_title('Energy Conservation')
            ax_energy.set_xlabel('Frame')
            ax_energy.set_ylabel('Energy (J)')
            ax_energy.grid(True, alpha=0.3)
            ax_energy.legend()
            figs.append(fig_energy)
            
        # Forces
        if 'forces' in physics_debug:
            fig_forces, ax_forces = plt.subplots(figsize=(10, 6))
            forces = physics_debug['forces']
            for force_name, force_data in forces.items():
                ax_forces.plot(force_data, label=force_name)
            ax_forces.set_title('Force Components')
            ax_forces.set_xlabel('Frame')
            ax_forces.set_ylabel('Force (N)')
            ax_forces.grid(True, alpha=0.3)
            ax_forces.legend()
            figs.append(fig_forces)
            
        if self.save_debug:
            for i, fig in enumerate(figs):
                fig.savefig(self.debug_dir / f'{save_prefix}_{i}.png')
                
        return figs
        
    def visualize_tracking_debug(
        self,
        frame: np.ndarray,
        tracking_debug: Dict,
        save_prefix: str = 'tracking'
    ) -> np.ndarray:
        """
        Visualize tracking debug information.
        
        Args:
            frame: Input frame
            tracking_debug: Tracking debug information
            save_prefix: Prefix for saved files
            
        Returns:
            Debug visualization frame
        """
        vis_frame = frame.copy()
        
        # Draw tracking states
        if 'states' in tracking_debug:
            states = tracking_debug['states']
            for track_id, state in states.items():
                # Draw state ellipse
                cv2.ellipse(vis_frame,
                           center=tuple(map(int, state['position'])),
                           axes=tuple(map(int, state['uncertainty'])),
                           angle=0, startAngle=0, endAngle=360,
                           color=(0, 255, 0), thickness=2)
                
                # Draw velocity vector
                if 'velocity' in state:
                    start_point = tuple(map(int, state['position']))
                    velocity = np.array(state['velocity'])
                    end_point = tuple(map(int, state['position'] + velocity * 10))
                    cv2.arrowedLine(vis_frame, start_point, end_point,
                                  (255, 0, 0), 2)
                    
        # Draw association costs
        if 'association_costs' in tracking_debug:
            costs = tracking_debug['association_costs']
            y_offset = 30
            for track_id, cost_dict in costs.items():
                text = f'Track {track_id} - Cost: {cost_dict["cost"]:.2f}'
                cv2.putText(vis_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
                
        if self.save_debug:
            cv2.imwrite(str(self.debug_dir / f'{save_prefix}.png'), vis_frame)
            
        return vis_frame
        
    def save_debug_info(self, debug_info: Dict, filename: str):
        """Save debug information to file."""
        if self.save_debug:
            save_path = self.debug_dir / filename
            np.save(str(save_path), debug_info)
            
    def close_all(self):
        """Close all open figures."""
        plt.close('all')
