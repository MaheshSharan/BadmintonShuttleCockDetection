import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and analyze dataset quality."""
    
    def __init__(self, config: dict):
        """
        Initialize validator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_size = tuple(config['data']['img_size'])
        
    def validate_video(self, video_path: str) -> Tuple[bool, Dict]:
        """
        Validate video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, metadata)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, {'error': 'Could not open video file'}
            
            metadata = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            
            # Check if video has frames
            if metadata['frame_count'] == 0:
                return False, {'error': 'Video has no frames'}
            
            # Check if video dimensions are reasonable
            if metadata['width'] < 100 or metadata['height'] < 100:
                return False, {'error': 'Video dimensions too small'}
            
            # Check if FPS is reasonable
            if metadata['fps'] < 15:
                return False, {'error': 'FPS too low'}
            
            cap.release()
            return True, metadata
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def validate_annotations(self, csv_path: str) -> Tuple[bool, Dict]:
        """
        Validate annotation file.
        
        Args:
            csv_path: Path to annotation CSV file
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['Frame', 'Visibility', 'X', 'Y']
            
            # Check required columns
            if not all(col in df.columns for col in required_columns):
                return False, {'error': 'Missing required columns'}
            
            # Basic statistics
            stats = {
                'total_frames': len(df),
                'visible_frames': df['Visibility'].sum(),
                'visibility_ratio': df['Visibility'].mean(),
                'x_range': (df['X'][df['Visibility'] == 1].min(),
                          df['X'][df['Visibility'] == 1].max()),
                'y_range': (df['Y'][df['Visibility'] == 1].min(),
                          df['Y'][df['Visibility'] == 1].max()),
            }
            
            # Validate values
            if df['Frame'].duplicated().any():
                return False, {'error': 'Duplicate frame numbers'}
            
            if not df['Frame'].is_monotonic_increasing:
                return False, {'error': 'Frame numbers not monotonic'}
            
            if not all(v in [0, 1] for v in df['Visibility'].unique()):
                return False, {'error': 'Invalid visibility values'}
            
            # Check for unreasonable coordinates
            visible_mask = df['Visibility'] == 1
            if visible_mask.any():
                if (df.loc[visible_mask, 'X'] < 0).any() or \
                   (df.loc[visible_mask, 'Y'] < 0).any():
                    return False, {'error': 'Negative coordinates'}
            
            return True, stats
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def analyze_dataset(self, root_dir: str) -> Dict:
        """
        Analyze entire dataset.
        
        Args:
            root_dir: Root directory of dataset
            
        Returns:
            Dictionary containing analysis results
        """
        root_dir = Path(root_dir)
        analysis = {
            'total_videos': 0,
            'valid_videos': 0,
            'total_annotations': 0,
            'valid_annotations': 0,
            'total_frames': 0,
            'visible_frames': 0,
            'average_visibility_ratio': [],
            'fps_distribution': [],
            'resolution_distribution': [],
            'errors': []
        }
        
        for split in ['train', 'valid', 'test']:
            split_dir = root_dir / split
            if not split_dir.exists():
                continue
                
            for match_dir in split_dir.glob('match*'):
                video_dir = match_dir / 'video'
                csv_dir = match_dir / 'csv'
                
                for video_path in video_dir.glob('*.mp4'):
                    analysis['total_videos'] += 1
                    csv_path = csv_dir / f"{video_path.stem}.csv"
                    
                    # Validate video
                    is_valid_video, video_meta = self.validate_video(str(video_path))
                    if is_valid_video:
                        analysis['valid_videos'] += 1
                        analysis['fps_distribution'].append(video_meta['fps'])
                        analysis['resolution_distribution'].append(
                            (video_meta['width'], video_meta['height'])
                        )
                    else:
                        analysis['errors'].append({
                            'file': str(video_path),
                            'error': video_meta['error']
                        })
                    
                    # Validate annotations if they exist
                    if csv_path.exists():
                        analysis['total_annotations'] += 1
                        is_valid_ann, ann_stats = self.validate_annotations(str(csv_path))
                        
                        if is_valid_ann:
                            analysis['valid_annotations'] += 1
                            analysis['total_frames'] += ann_stats['total_frames']
                            analysis['visible_frames'] += ann_stats['visible_frames']
                            analysis['average_visibility_ratio'].append(
                                ann_stats['visibility_ratio']
                            )
                        else:
                            analysis['errors'].append({
                                'file': str(csv_path),
                                'error': ann_stats['error']
                            })
        
        # Calculate averages
        if analysis['average_visibility_ratio']:
            analysis['average_visibility_ratio'] = np.mean(
                analysis['average_visibility_ratio']
            )
        
        return analysis
