"""
Data augmentation transforms for shuttlecock detection.
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union
import random
import logging

logger = logging.getLogger(__name__)

class ShuttlecockTransform:
    """
    Transforms for shuttlecock detection training.
    Applies augmentations to both images and bounding boxes.
    """
    
    def __init__(
        self,
        train: bool = True,
        size: Tuple[int, int] = (720, 1280),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Initialize transform.
        
        Args:
            train: If True, apply training augmentations
            size: Target size (height, width)
            mean: Normalization mean
            std: Normalization std
        """
        self.train = train
        self.size = size
        self.mean = mean
        self.std = std
        
    def __call__(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to image and boxes.
        
        Args:
            image: Image tensor [C, H, W]
            boxes: Bounding boxes tensor [N, 4] in normalized coordinates
            
        Returns:
            Transformed image and boxes
        """
        if self.train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                if boxes is not None and len(boxes):
                    boxes[:, 0] = 1 - boxes[:, 0] - boxes[:, 2]
            
            # Random vertical flip
            if random.random() > 0.5:
                image = F.vflip(image)
                if boxes is not None and len(boxes):
                    boxes[:, 1] = 1 - boxes[:, 1] - boxes[:, 3]
            
            # Color jittering
            image = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )(image)
            
            # Random rotation (small angles)
            angle = random.uniform(-15, 15)
            image = F.rotate(image, angle)
            if boxes is not None and len(boxes):
                # Convert normalized to absolute coordinates
                h, w = image.shape[1:]
                boxes_abs = boxes.clone()
                boxes_abs[:, [0, 2]] *= w
                boxes_abs[:, [1, 3]] *= h
                
                # Rotate boxes
                center = torch.tensor([[w/2, h/2]])
                angle_rad = angle * np.pi / 180
                rot_mat = torch.tensor([
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]
                ])
                
                # Rotate box centers
                centers = boxes_abs[:, :2] + boxes_abs[:, 2:]/2
                centers = (centers - center) @ rot_mat.T + center
                
                # Update box coordinates
                boxes_abs[:, :2] = centers - boxes_abs[:, 2:]/2
                
                # Convert back to normalized coordinates
                boxes[:, [0, 2]] = boxes_abs[:, [0, 2]] / w
                boxes[:, [1, 3]] = boxes_abs[:, [1, 3]] / h
        
        # Resize
        image = F.resize(image, self.size)
        
        # Normalize
        image = F.normalize(image, self.mean, self.std)
        
        return image, boxes

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Handles variable number of objects per frame.
    
    Args:
        batch: List of dictionaries containing:
            - frames: [T, C, H, W]
            - labels: [T, 5] tensor
            - sequence_id: str
            
    Returns:
        Batched dictionary containing:
            - frames: [B, T, C, H, W]
            - labels: [B, T, 5] tensor
            - sequence_ids: List[B] of str
    """
    frames = torch.stack([item['frames'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    sequence_ids = [item['sequence_id'] for item in batch]
    
    return {
        'frames': frames,
        'labels': labels,
        'sequence_ids': sequence_ids
    }
