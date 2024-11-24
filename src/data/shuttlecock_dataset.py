"""
Dataset loader for shuttlecock detection dataset.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

class ShuttlecockDataset(Dataset):
    """Dataset for loading shuttlecock video sequences and annotations."""
    
    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 16,
        transform=None,
        target_size: Tuple[int, int] = (720, 1280)
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing match folders
            sequence_length: Number of frames per sequence
            transform: Optional transforms to apply
            target_size: Target frame size (height, width)
        """
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_size = target_size
        
        # Load metadata
        metadata_path = self.root_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Please run preprocess_dataset.py first."
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.sequences = []
        for seq in metadata['sequences']:
            frames_dir = self.root_dir / seq['frames_dir']
            annotations_path = self.root_dir / seq['annotations_path']
            
            # Load annotations
            df = pd.read_csv(annotations_path)
            # Convert column names to lowercase for consistency
            df.columns = df.columns.str.lower()
            frame_count = seq['frame_count']
            
            # Create sequences
            for start_idx in range(0, frame_count - sequence_length + 1, sequence_length // 2):
                end_idx = start_idx + sequence_length
                seq_df = df[(df['frame'] >= start_idx) & (df['frame'] < end_idx)]
                
                if len(seq_df) > 0:  # Only add sequences with annotations
                    self.sequences.append({
                        'frames_dir': frames_dir,
                        'start_frame': start_idx,
                        'end_frame': end_idx,
                        'annotations': seq_df
                    })
        
        logger.info(f"Found {len(self.sequences)} sequences")

    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict:
        """Get a sequence of frames and annotations."""
        sequence = self.sequences[idx]
        frames_dir = sequence['frames_dir']
        start_frame = sequence['start_frame']
        end_frame = sequence['end_frame']
        annotations = sequence['annotations']
        
        # Load frames
        frames = []
        for frame_idx in range(start_frame, end_frame):
            frame_path = frames_dir / f"{frame_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        # Convert frames to tensor
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Get annotations for each frame
        labels = []
        for frame_idx in range(start_frame, end_frame):
            frame_annotations = annotations[annotations['frame'] == frame_idx]
            if len(frame_annotations) > 0:
                # Convert bbox coordinates to normalized format
                x = frame_annotations['x'].values[0] / self.target_size[1]
                y = frame_annotations['y'].values[0] / self.target_size[0]
                w = frame_annotations['width'].values[0] / self.target_size[1] if 'width' in frame_annotations else 0.1
                h = frame_annotations['height'].values[0] / self.target_size[0] if 'height' in frame_annotations else 0.1
                labels.append([1, x, y, w, h])  # 1 is the class ID for shuttlecock
            else:
                labels.append([0, 0, 0, 0, 0])  # No annotation for this frame
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            frames, labels = self.transform(frames, labels)
        
        return {
            'frames': frames,
            'labels': labels,
            'sequence_id': str(frames_dir.name)
        }
