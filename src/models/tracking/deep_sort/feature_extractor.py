"""
Custom appearance feature extractor for shuttlecock re-identification.
Uses lightweight CNN architecture optimized for small, fast-moving objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np

class ShuttlecockFeatureExtractor(nn.Module):
    """
    Custom feature extractor for shuttlecock appearance matching.
    Optimized for small, fast-moving objects with motion blur.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (64, 64),
        feature_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the feature extractor.
        
        Args:
            input_shape: Input image dimensions (H, W)
            feature_dim: Output feature dimension
            device: Device to run computations on
        """
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        
        # Lightweight CNN backbone
        self.backbone = nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Motion-aware block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Feature extraction block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Final conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(device)
        
        # Feature projection
        self.projector = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim)
        ).to(device)
        
        # Motion embedding
        self.motion_embedding = nn.Sequential(
            nn.Linear(6, 32),  # velocity and acceleration
            nn.ReLU(),
            nn.Linear(32, feature_dim)
        ).to(device)
        
    def forward(
        self,
        images: torch.Tensor,
        motion_info: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from shuttlecock images.
        
        Args:
            images: Batch of images [B, C, H, W]
            motion_info: Optional motion information [B, 6] (vx,vy,vz,ax,ay,az)
            
        Returns:
            Feature vectors [B, feature_dim]
        """
        # Process images
        x = self.backbone(images)
        x = x.view(x.size(0), -1)
        features = self.projector(x)
        
        # Incorporate motion information if available
        if motion_info is not None:
            motion_features = self.motion_embedding(motion_info)
            features = features + 0.2 * motion_features  # weighted combination
        
        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)
        return features
    
    def extract_features(
        self,
        images: np.ndarray,
        motion_info: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract features from numpy image array.
        
        Args:
            images: Image array [B, H, W, C] in uint8 format
            motion_info: Optional motion information [B, 6]
            
        Returns:
            Feature array [B, feature_dim]
        """
        # Preprocess images
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2)
        images = F.interpolate(images, size=self.input_shape)
        images = images.to(self.device) / 255.0
        
        # Convert motion info if provided
        if motion_info is not None:
            motion_info = torch.from_numpy(motion_info).float().to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self(images, motion_info)
        
        return features.cpu().numpy()
    
    def compute_distance(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine distance between feature sets.
        
        Args:
            features1: First feature set [N, feature_dim]
            features2: Second feature set [M, feature_dim]
            
        Returns:
            Distance matrix [N, M]
        """
        # Normalize features
        features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity = np.dot(features1, features2.T)
        
        # Convert to distance
        distance = 1 - similarity
        return distance
    
    def train_step(
        self,
        anchor_images: torch.Tensor,
        positive_images: torch.Tensor,
        negative_images: torch.Tensor,
        margin: float = 0.2
    ) -> torch.Tensor:
        """
        Training step using triplet loss.
        
        Args:
            anchor_images: Anchor images [B, C, H, W]
            positive_images: Positive images [B, C, H, W]
            negative_images: Negative images [B, C, H, W]
            margin: Triplet loss margin
            
        Returns:
            Loss value
        """
        # Extract features
        anchor_features = self(anchor_images)
        positive_features = self(positive_images)
        negative_features = self(negative_images)
        
        # Compute distances
        positive_dist = torch.sum((anchor_features - positive_features)**2, dim=1)
        negative_dist = torch.sum((anchor_features - negative_features)**2, dim=1)
        
        # Triplet loss
        loss = torch.mean(torch.clamp(positive_dist - negative_dist + margin, min=0))
        return loss
    
class FeatureDatabase:
    """
    Database for storing and matching shuttlecock appearance features.
    """
    
    def __init__(
        self,
        max_features: int = 1000,
        feature_dim: int = 128
    ):
        """
        Initialize feature database.
        
        Args:
            max_features: Maximum number of features to store
            feature_dim: Feature dimension
        """
        self.max_features = max_features
        self.feature_dim = feature_dim
        self.features: List[np.ndarray] = []
        self.track_ids: List[int] = []
    
    def add_feature(
        self,
        feature: np.ndarray,
        track_id: int
    ) -> None:
        """
        Add a feature to the database.
        
        Args:
            feature: Feature vector [feature_dim]
            track_id: Associated track ID
        """
        if len(self.features) >= self.max_features:
            self.features.pop(0)
            self.track_ids.pop(0)
        
        self.features.append(feature)
        self.track_ids.append(track_id)
    
    def match_features(
        self,
        query_features: np.ndarray,
        threshold: float = 0.7
    ) -> Tuple[List[int], np.ndarray]:
        """
        Match query features against database.
        
        Args:
            query_features: Query feature vectors [N, feature_dim]
            threshold: Maximum distance threshold
            
        Returns:
            Tuple of (matched_track_ids, distances)
        """
        if not self.features:
            return [], np.array([])
        
        # Stack database features
        db_features = np.stack(self.features)
        
        # Compute distances
        distances = 1 - np.dot(query_features, db_features.T)
        
        # Find best matches
        best_indices = np.argmin(distances, axis=1)
        best_distances = distances[np.arange(len(query_features)), best_indices]
        
        # Filter by threshold
        valid_matches = best_distances < threshold
        matched_ids = [self.track_ids[i] for i in best_indices[valid_matches]]
        matched_distances = best_distances[valid_matches]
        
        return matched_ids, matched_distances
