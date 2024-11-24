import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from .validator import DataValidator
from .video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Comprehensive data quality checking and reporting system."""
    
    def __init__(self, config: dict):
        """Initialize quality checker with configuration."""
        self.config = config
        self.validator = DataValidator(config)
        self.processor = VideoProcessor(config)
        
    def generate_quality_report(self, dataset_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive quality report for the dataset.
        
        Args:
            dataset_path: Path to dataset root directory
            output_path: Optional path to save report JSON
            
        Returns:
            Dictionary containing quality metrics and issues
        """
        dataset_path = Path(dataset_path)
        report = {
            'dataset_statistics': {},
            'quality_metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        # Analyze dataset structure
        report['dataset_statistics'] = self._analyze_dataset_structure(dataset_path)
        
        # Validate data quality
        report['quality_metrics'] = self._validate_data_quality(dataset_path)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(
            report['dataset_statistics'],
            report['quality_metrics']
        )
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report
    
    def _analyze_dataset_structure(self, dataset_path: Path) -> Dict:
        """Analyze dataset structure and generate statistics."""
        stats = {
            'total_samples': 0,
            'split_distribution': {},
            'class_distribution': {},
            'video_statistics': {
                'duration_stats': {},
                'resolution_stats': {},
                'fps_stats': {}
            }
        }
        
        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
                
            split_stats = {
                'total_videos': 0,
                'total_frames': 0,
                'total_annotations': 0
            }
            
            video_durations = []
            video_resolutions = []
            video_fps = []
            
            for match_dir in split_path.glob('match*'):
                video_dir = match_dir / 'video'
                csv_dir = match_dir / 'csv'
                
                for video_path in video_dir.glob('*.mp4'):
                    split_stats['total_videos'] += 1
                    
                    # Get video metadata
                    is_valid, metadata = self.validator.validate_video(str(video_path))
                    if is_valid:
                        video_durations.append(metadata['frame_count'] / metadata['fps'])
                        video_resolutions.append((metadata['width'], metadata['height']))
                        video_fps.append(metadata['fps'])
                    
                    # Process corresponding annotation
                    csv_path = csv_dir / f"{video_path.stem}.csv"
                    if csv_path.exists():
                        split_stats['total_annotations'] += 1
                        is_valid, ann_stats = self.validator.validate_annotations(str(csv_path))
                        if is_valid:
                            split_stats['total_frames'] += ann_stats['total_frames']
            
            stats['split_distribution'][split] = split_stats
            stats['total_samples'] += split_stats['total_videos']
            
            # Calculate video statistics
            if video_durations:
                stats['video_statistics']['duration_stats'][split] = {
                    'mean': np.mean(video_durations),
                    'std': np.std(video_durations),
                    'min': np.min(video_durations),
                    'max': np.max(video_durations)
                }
            
            if video_fps:
                stats['video_statistics']['fps_stats'][split] = {
                    'mean': np.mean(video_fps),
                    'std': np.std(video_fps),
                    'modes': pd.Series(video_fps).mode().tolist()
                }
            
            if video_resolutions:
                unique_resolutions = set(video_resolutions)
                stats['video_statistics']['resolution_stats'][split] = {
                    'unique_resolutions': list(unique_resolutions),
                    'most_common': max(set(video_resolutions), 
                                     key=video_resolutions.count)
                }
        
        return stats
    
    def _validate_data_quality(self, dataset_path: Path) -> Dict:
        """Validate data quality and generate metrics."""
        quality_metrics = {
            'video_quality': {},
            'annotation_quality': {},
            'cross_validation': {}
        }
        
        # Validate videos in parallel
        with ThreadPoolExecutor() as executor:
            video_futures = []
            for video_path in dataset_path.rglob('*.mp4'):
                future = executor.submit(self._validate_video_quality, video_path)
                video_futures.append((video_path, future))
            
            for video_path, future in video_futures:
                quality_metrics['video_quality'][str(video_path)] = future.result()
        
        # Validate annotations
        for csv_path in dataset_path.rglob('*.csv'):
            quality_metrics['annotation_quality'][str(csv_path)] = \
                self._validate_annotation_quality(csv_path)
        
        # Cross-validate video-annotation pairs
        quality_metrics['cross_validation'] = self._cross_validate_pairs(dataset_path)
        
        return quality_metrics
    
    def _validate_video_quality(self, video_path: Path) -> Dict:
        """Validate individual video quality."""
        metrics = {
            'technical_quality': {},
            'content_quality': {}
        }
        
        # Technical quality checks
        is_valid, metadata = self.validator.validate_video(str(video_path))
        if is_valid:
            metrics['technical_quality'] = {
                'resolution': (metadata['width'], metadata['height']),
                'fps': metadata['fps'],
                'duration': metadata['frame_count'] / metadata['fps'],
                'aspect_ratio': metadata['width'] / metadata['height']
            }
        
        # Content quality checks (sample frames)
        try:
            frames, _ = self.processor.extract_frames(
                str(video_path),
                start_frame=0,
                end_frame=min(100, metadata['frame_count'])
            )
            
            metrics['content_quality'] = {
                'mean_brightness': np.mean([frame.mean() for frame in frames]),
                'mean_contrast': np.mean([frame.std() for frame in frames]),
                'blur_scores': [cv2.Laplacian(frame, cv2.CV_64F).var() 
                              for frame in frames[:10]]  # Check first 10 frames
            }
        except Exception as e:
            metrics['content_quality']['error'] = str(e)
        
        return metrics
    
    def _validate_annotation_quality(self, csv_path: Path) -> Dict:
        """Validate individual annotation quality."""
        metrics = {}
        
        is_valid, stats = self.validator.validate_annotations(str(csv_path))
        if is_valid:
            metrics.update(stats)
            
            # Additional quality checks
            df = pd.read_csv(csv_path)
            
            # Check trajectory smoothness
            if 'X' in df.columns and 'Y' in df.columns:
                visible_mask = df['Visibility'] == 1
                if visible_mask.any():
                    x_coords = df.loc[visible_mask, 'X']
                    y_coords = df.loc[visible_mask, 'Y']
                    
                    metrics['trajectory_smoothness'] = {
                        'x_smoothness': np.mean(np.abs(np.diff(x_coords))),
                        'y_smoothness': np.mean(np.abs(np.diff(y_coords)))
                    }
            
            # Check annotation consistency
            metrics['consistency'] = {
                'visibility_transitions': len(df['Visibility'].diff()[df['Visibility'].diff() != 0]),
                'max_invisible_sequence': self._get_max_sequence(df['Visibility'] == 0),
                'max_visible_sequence': self._get_max_sequence(df['Visibility'] == 1)
            }
        
        return metrics
    
    def _cross_validate_pairs(self, dataset_path: Path) -> Dict:
        """Cross-validate video-annotation pairs."""
        cross_validation = {
            'matching_pairs': 0,
            'mismatched_pairs': 0,
            'orphaned_files': [],
            'frame_count_matches': 0,
            'frame_count_mismatches': []
        }
        
        for video_path in dataset_path.rglob('*.mp4'):
            expected_csv = video_path.parent.parent / 'csv' / f"{video_path.stem}.csv"
            
            if expected_csv.exists():
                cross_validation['matching_pairs'] += 1
                
                # Validate frame counts
                is_valid, video_meta = self.validator.validate_video(str(video_path))
                if is_valid:
                    df = pd.read_csv(expected_csv)
                    if len(df) == video_meta['frame_count']:
                        cross_validation['frame_count_matches'] += 1
                    else:
                        cross_validation['frame_count_mismatches'].append({
                            'video': str(video_path),
                            'video_frames': video_meta['frame_count'],
                            'annotation_frames': len(df)
                        })
            else:
                cross_validation['mismatched_pairs'] += 1
                cross_validation['orphaned_files'].append(str(video_path))
        
        return cross_validation
    
    def _get_max_sequence(self, series: pd.Series) -> int:
        """Get maximum consecutive sequence length."""
        return max((series != series.shift()).cumsum().value_counts())
    
    def _generate_recommendations(self, statistics: Dict, metrics: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Dataset balance recommendations
        split_dist = statistics['split_distribution']
        if 'train' in split_dist and 'valid' in split_dist:
            train_size = split_dist['train']['total_videos']
            valid_size = split_dist['valid']['total_videos']
            ratio = train_size / (train_size + valid_size)
            
            if ratio < 0.7:
                recommendations.append(
                    "Consider increasing training set size. Current train-val "
                    f"ratio is {ratio:.2f}, recommended > 0.7"
                )
        
        # Video quality recommendations
        for split, fps_stats in statistics['video_statistics']['fps_stats'].items():
            if fps_stats['mean'] < 30:
                recommendations.append(
                    f"Low average FPS ({fps_stats['mean']:.1f}) in {split} split. "
                    "Consider using higher FPS videos for better tracking"
                )
        
        # Annotation quality recommendations
        annotation_issues = []
        for path, metrics in metrics['annotation_quality'].items():
            if 'consistency' in metrics:
                if metrics['consistency']['max_invisible_sequence'] > 30:
                    annotation_issues.append(path)
        
        if annotation_issues:
            recommendations.append(
                f"Found {len(annotation_issues)} annotations with long invisible "
                "sequences. Consider improving annotation coverage"
            )
        
        return recommendations
