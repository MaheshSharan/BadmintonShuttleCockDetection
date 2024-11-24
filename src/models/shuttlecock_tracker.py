"""
Main ShuttlecockTracker model that orchestrates all components.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

from .backbone.csp_darknet import CSPDarknet
from .neck.fpn import FPN
from .head.detection_head import DetectionHead
from .physics.trajectory_validator import TrajectoryValidator
from .tracking.deep_sort.tracker import ShuttlecockTracker as DeepSORT
from .physics.physics_loss import PhysicsInformedLoss as PhysicsLoss
from .tracking.deep_sort.pinn import TrajectoryPINN as PINN
from .tracking.trajectory_prediction import TrajectoryPredictor
from .optimization.trajectory_optimization import TrajectoryOptimizer, ConfidenceScorer
from .optimization.batch_processor import BatchProcessor
from .optimization.gpu_optimizer import GPUOptimizer
from .optimization.memory_optimizer import MemoryOptimizer
from .optimization.model_pruner import ModelPruner

class ShuttlecockTracker(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        # Initialize core detection components
        self.backbone = CSPDarknet(**config['backbone'])
        self.neck = FPN(**config['neck'])
        self.detection_head = DetectionHead(**config['head'])
        
        # Initialize physics and tracking components
        self.trajectory_validator = TrajectoryValidator(**config['trajectory_validator'])
        self.physics_loss = PhysicsLoss(**config['physics_loss'])
        self.pinn = PINN(**config['pinn'])
        self.tracker = DeepSORT(**config['tracking'])
        
        # Initialize new trajectory prediction and optimization components
        self.trajectory_predictor = TrajectoryPredictor(**config['prediction'])
        self.trajectory_optimizer = TrajectoryOptimizer(config['optimization'])
        self.confidence_scorer = ConfidenceScorer(config['optimization'])

        # Initialize model optimization components
        self.batch_processor = BatchProcessor(**config['batch_processing'])
        self.gpu_optimizer = GPUOptimizer(self, **config['gpu_optimization'])
        memory_config = {'model': self, **config['memory_optimization']}
        self.memory_optimizer = MemoryOptimizer(**memory_config)
        self.model_pruner = ModelPruner(self, **config['model_pruning'])

    def forward(self, frames: torch.Tensor, is_training: bool = True) -> Dict:
        """
        Forward pass through the entire pipeline.
        
        Args:
            frames: Tensor of shape [batch_size, num_frames, channels, height, width]
            is_training: Whether in training mode
            
        Returns:
            Dict containing detection results, tracking results, and losses
        """
        # Apply batch processing and GPU optimization
        frames = self.batch_processor.process_batch(frames)
        frames = self.gpu_optimizer.optimize_memory(frames)
        
        batch_size, num_frames = frames.shape[:2]
        results = {}
        
        # Use memory optimization context
        with self.memory_optimizer.efficient_forward():
            # 1. Feature Extraction Pipeline
            features = self.backbone(frames.view(-1, *frames.shape[2:]))  # Flatten batch and frames
            neck_features = self.neck(features)
            
            # 2. Detection with Physics Validation
            detections = self.detection_head(neck_features)
            validated_detections = self.trajectory_validator(detections)
            results['detections'] = validated_detections
            
            # 3. Physics-Informed Neural Network Predictions
            pinn_predictions = self.pinn(validated_detections)
            results['pinn_predictions'] = pinn_predictions
            
            # 4. Trajectory Prediction
            track_sequences = []
            track_ids = []
            for batch_idx in range(batch_size):
                batch_detections = validated_detections[batch_idx]
                if len(batch_detections) > 0:
                    track_sequences.append(batch_detections[:, :2])  # Get positions
                    track_ids.extend(range(len(batch_detections)))
            
            if track_sequences:
                track_sequences = torch.stack(track_sequences)
                predictions, confidence = self.trajectory_predictor(
                    track_sequences,
                    track_ids
                )
                results['trajectory_predictions'] = predictions
                results['prediction_confidence'] = confidence
            
            # 5. Tracking Integration
            tracking_results = []
            optimized_trajectories = []
            for batch_idx in range(batch_size):
                batch_detections = validated_detections[batch_idx]
                # Update tracker with predictions if available
                if 'trajectory_predictions' in results:
                    batch_predictions = results['trajectory_predictions'][batch_idx]
                    tracks = self.tracker.update(
                        batch_detections,
                        batch_predictions
                    )
                else:
                    tracks = self.tracker.update(batch_detections)
                
                tracking_results.append(tracks)
                
                # 6. Trajectory Optimization
                if len(tracks) > 0:
                    track_points = torch.tensor([[t.mean[0], t.mean[1]] for t in tracks])
                    track_times = torch.tensor([t.time_since_update for t in tracks])
                    track_conf = results['prediction_confidence'][batch_idx] if 'prediction_confidence' in results else None
                    
                    # Optimize trajectory
                    optimized_points, _ = self.trajectory_optimizer.optimize_trajectory(
                        track_points.cpu().numpy(),
                        track_conf.cpu().numpy() if track_conf is not None else None,
                        track_times.cpu().numpy()
                    )
                    optimized_trajectories.append(torch.from_numpy(optimized_points).to(track_points.device))
            
            results['tracks'] = tracking_results
            if optimized_trajectories:
                results['optimized_trajectories'] = optimized_trajectories
            
            # 7. Calculate Losses during Training
            if is_training:
                losses = {
                    'detection_loss': self.detection_head.loss(detections),
                    'physics_loss': self.physics_loss(validated_detections, pinn_predictions),
                    'tracking_loss': self.tracker.loss(tracking_results)
                }
                
                # Add trajectory prediction loss if available
                if 'trajectory_predictions' in results:
                    losses['prediction_loss'] = self.trajectory_predictor.loss(
                        results['trajectory_predictions'],
                        validated_detections
                    )
                
                results['losses'] = losses
                
        return results

    @torch.no_grad()
    def inference(self, video_frames: torch.Tensor) -> Dict:
        """
        Inference mode forward pass optimized for real-time processing.
        
        Args:
            video_frames: Tensor of shape [num_frames, channels, height, width]
            
        Returns:
            Dict containing detection and tracking results
        """
        self.eval()
        results = self.forward(video_frames.unsqueeze(0), is_training=False)
        return {
            'detections': results['detections'][0],  # Remove batch dimension
            'tracks': results['tracks'][0],
            'predictions': results['trajectory_predictions'][0] if 'trajectory_predictions' in results else None,
            'optimized_trajectories': results['optimized_trajectories'][0] if 'optimized_trajectories' in results else None
        }
