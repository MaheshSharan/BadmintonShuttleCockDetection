import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pandas as pd
import albumentations as A
from .video_processor import VideoProcessor

class ShuttlecockDataset(Dataset):
    """Dataset class for shuttlecock detection and tracking."""
    
    def __init__(self, 
                 root_dir: str,
                 config: dict,
                 split: str = 'Train',
                 transform: Optional[A.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing videos and annotations
            config: Configuration dictionary
            split: Dataset split ('Train' or 'valid')
            transform: Albumentations transformations
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.split = split
        self.transform = transform or self._get_default_transform()
        self.processor = VideoProcessor(config)
        
        # Load all annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Load all annotations from CSV files."""
        annotations = []
        split_dir = self.root_dir / self.split
        
        # Iterate through match directories
        for match_dir in split_dir.glob(self.config['data']['match_pattern']):
            csv_dir = match_dir / 'csv'
            video_dir = match_dir / 'video'
            
            # Process each CSV file
            for csv_file in csv_dir.glob(self.config['data']['csv_pattern']):
                # Load CSV data
                df = pd.read_csv(csv_file)
                
                # Get corresponding video file
                video_name = csv_file.stem.replace('_ball', '') + '.mp4'
                video_path = video_dir / video_name
                
                if not video_path.exists():
                    continue
                
                # Create annotation entry
                anno = {
                    'match_id': match_dir.name,
                    'video_path': str(video_path),
                    'csv_path': str(csv_file),
                    'frames': df['Frame'].values,
                    'visibility': df['Visibility'].values,
                    'coordinates': np.column_stack([df['X'].values, df['Y'].values])
                }
                
                annotations.append(anno)
        
        return annotations
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get dataset item.
        
        Args:
            idx: Index of item to get
            
        Returns:
            Tuple of (image tensor, annotation dict)
        """
        anno = self.annotations[idx]
        
        # Load video frame
        frames, frame_info = self.processor.extract_frames(
            anno['video_path'],
            start_frame=anno['frames'][0],
            end_frame=anno['frames'][-1]
        )
        
        # Process annotations
        boxes = []
        labels = []
        
        for frame_idx, (frame, vis, coord) in enumerate(zip(
            anno['frames'], anno['visibility'], anno['coordinates'])):
            
            if vis == 1:
                x, y = coord
                w, h = self.config['data']['box_size']
                
                # Convert to [x1, y1, x2, y2] format
                box = [
                    max(0, x - w/2),
                    max(0, y - h/2),
                    min(frame_info['width'], x + w/2),
                    min(frame_info['height'], y + h/2)
                ]
                
                boxes.append(box)
                labels.append(1)  # 1 for shuttlecock
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=frames[0],  # Use first frame
                bboxes=boxes.numpy().tolist(),
                class_labels=labels.numpy().tolist()
            )
            
            image = torch.from_numpy(transformed['image'].transpose(2, 0, 1))
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'match_id': anno['match_id'],
            'frame_indices': anno['frames']
        }
    
    def _get_default_transform(self) -> A.Compose:
        """Get default augmentation transform."""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=self.config['augmentation']['brightness_range'],
                contrast_limit=self.config['augmentation']['contrast_range'],
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=self.config['augmentation']['translation_range'],
                scale_limit=self.config['augmentation']['scale_range'],
                rotate_limit=self.config['augmentation']['rotation_range'],
                p=0.5
            ),
            A.Normalize(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
