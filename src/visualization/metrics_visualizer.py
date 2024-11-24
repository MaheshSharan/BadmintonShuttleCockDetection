"""
Metrics visualization tools for model analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)

class MetricsVisualizer:
    """Visualization tools for model metrics and analysis."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        style: str = 'dark'
    ):
        """
        Initialize metrics visualizer.
        
        Args:
            config: Visualization configuration
            output_dir: Directory to save visualizations
            style: Plot style ('dark' or 'light')
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Set plot style
        plt.style.use('dark_background' if style == 'dark' else 'default')
        
    def plot_precision_recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            save: Whether to save plot
            
        Returns:
            Figure and axes objects
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(recall, precision, 'b-', label=f'AP = {ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save and self.output_dir:
            fig.savefig(self.output_dir / 'precision_recall.png')
            
        return fig, ax
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: Class labels
            save: Whether to save plot
            
        Returns:
            Figure and axes objects
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        if save and self.output_dir:
            fig.savefig(self.output_dir / 'confusion_matrix.png')
            
        return fig, ax
        
    def plot_trajectory_metrics(
        self,
        trajectories: List[Dict],
        predictions: List[Dict],
        physics_metrics: Optional[Dict] = None,
        save: bool = False
    ) -> List[Tuple[plt.Figure, plt.Axes]]:
        """
        Plot trajectory-related metrics.
        
        Args:
            trajectories: Ground truth trajectories
            predictions: Predicted trajectories
            physics_metrics: Physics-based metrics
            save: Whether to save plots
            
        Returns:
            List of figure and axes objects
        """
        figs = []
        
        # Position error plot
        fig_pos, ax_pos = plt.subplots(figsize=(10, 6))
        errors = [np.linalg.norm(t['position'] - p['position'])
                 for t, p in zip(trajectories, predictions)]
        ax_pos.plot(errors, 'b-', label='Position Error')
        ax_pos.set_xlabel('Frame')
        ax_pos.set_ylabel('Error (meters)')
        ax_pos.set_title('Trajectory Position Error')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend()
        figs.append((fig_pos, ax_pos))
        
        # Velocity metrics if available
        if physics_metrics and 'velocity_error' in physics_metrics:
            fig_vel, ax_vel = plt.subplots(figsize=(10, 6))
            ax_vel.plot(physics_metrics['velocity_error'], 'r-',
                       label='Velocity Error')
            ax_vel.set_xlabel('Frame')
            ax_vel.set_ylabel('Error (m/s)')
            ax_vel.set_title('Trajectory Velocity Error')
            ax_vel.grid(True, alpha=0.3)
            ax_vel.legend()
            figs.append((fig_vel, ax_vel))
            
        if save and self.output_dir:
            for i, (fig, _) in enumerate(figs):
                fig.savefig(self.output_dir / f'trajectory_metrics_{i}.png')
                
        return figs
        
    def plot_detection_stats(
        self,
        detections: List[Dict],
        save: bool = False
    ) -> List[Tuple[plt.Figure, plt.Axes]]:
        """
        Plot detection statistics.
        
        Args:
            detections: Detection results
            save: Whether to save plots
            
        Returns:
            List of figure and axes objects
        """
        figs = []
        
        # Confidence distribution
        fig_conf, ax_conf = plt.subplots(figsize=(10, 6))
        confidences = [d['confidence'] for d in detections]
        ax_conf.hist(confidences, bins=50, alpha=0.7)
        ax_conf.set_xlabel('Confidence Score')
        ax_conf.set_ylabel('Count')
        ax_conf.set_title('Detection Confidence Distribution')
        ax_conf.grid(True, alpha=0.3)
        figs.append((fig_conf, ax_conf))
        
        # Size distribution
        fig_size, ax_size = plt.subplots(figsize=(10, 6))
        sizes = [d['bbox'][2] * d['bbox'][3] for d in detections]  # width * height
        ax_size.hist(sizes, bins=50, alpha=0.7)
        ax_size.set_xlabel('Bounding Box Area')
        ax_size.set_ylabel('Count')
        ax_size.set_title('Detection Size Distribution')
        ax_size.grid(True, alpha=0.3)
        figs.append((fig_size, ax_size))
        
        if save and self.output_dir:
            for i, (fig, _) in enumerate(figs):
                fig.savefig(self.output_dir / f'detection_stats_{i}.png')
                
        return figs
        
    def plot_training_curves(
        self,
        metrics: Dict[str, List[float]],
        save: bool = False
    ) -> List[Tuple[plt.Figure, plt.Axes]]:
        """
        Plot training curves.
        
        Args:
            metrics: Dictionary of training metrics
            save: Whether to save plots
            
        Returns:
            List of figure and axes objects
        """
        figs = []
        
        # Loss curves
        if 'train_loss' in metrics and 'val_loss' in metrics:
            fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
            ax_loss.plot(metrics['train_loss'], 'b-', label='Training Loss')
            ax_loss.plot(metrics['val_loss'], 'r-', label='Validation Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title('Training and Validation Loss')
            ax_loss.grid(True, alpha=0.3)
            ax_loss.legend()
            figs.append((fig_loss, ax_loss))
            
        # Learning rate
        if 'learning_rate' in metrics:
            fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
            ax_lr.plot(metrics['learning_rate'], 'g-')
            ax_lr.set_xlabel('Iteration')
            ax_lr.set_ylabel('Learning Rate')
            ax_lr.set_title('Learning Rate Schedule')
            ax_lr.grid(True, alpha=0.3)
            figs.append((fig_lr, ax_lr))
            
        if save and self.output_dir:
            for i, (fig, _) in enumerate(figs):
                fig.savefig(self.output_dir / f'training_curves_{i}.png')
                
        return figs
        
    def close_all(self):
        """Close all open figures."""
        plt.close('all')
