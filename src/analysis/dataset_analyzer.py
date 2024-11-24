import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.video_processor import VideoProcessor
from typing import Dict, List, Tuple, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyze shuttlecock dataset characteristics."""
    
    def __init__(self, config: dict):
        """
        Initialize dataset analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.processor = VideoProcessor(config)
        self.output_dir = Path(config['analysis']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')  # Using the updated seaborn style name
        self.dpi = config['reporting']['dpi']
        self.fig_size = tuple(config['reporting']['figure_size'])
    
    def analyze_dataset(self, dataset_dir: str) -> Dict:
        """
        Analyze entire dataset.
        
        Args:
            dataset_dir: Root directory of dataset
            
        Returns:
            Dictionary containing analysis results
        """
        dataset_dir = Path(dataset_dir)
        results = {
            'total_matches': 0,
            'total_frames': 0,
            'visible_frames': 0,
            'invisible_frames': 0,
            'trajectories': [],
            'position_stats': {
                'x': {'mean': 0, 'std': 0},
                'y': {'mean': 0, 'std': 0}
            },
            'splits': {}
        }
        
        # Analyze each split
        for split in self.config['data']['splits']:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
                
            split_results = self._analyze_split(split_dir)
            results['splits'][split] = split_results
            
            # Update overall statistics
            results['total_matches'] += split_results['match_count']
            results['total_frames'] += split_results['frame_count']
            results['visible_frames'] += split_results['visible_frames']
            results['invisible_frames'] += split_results['invisible_frames']
            results['trajectories'].extend(split_results['trajectories'])
        
        # Calculate position statistics
        if results['trajectories']:
            all_positions = np.concatenate(results['trajectories'])
            results['position_stats']['x'] = {
                'mean': float(np.mean(all_positions[:, 0])),
                'std': float(np.std(all_positions[:, 0]))
            }
            results['position_stats']['y'] = {
                'mean': float(np.mean(all_positions[:, 1])),
                'std': float(np.std(all_positions[:, 1]))
            }
        
        return results
    
    def _analyze_split(self, split_dir: Path) -> Dict:
        """
        Analyze a dataset split.
        
        Args:
            split_dir: Directory containing the split
            
        Returns:
            Dictionary containing split analysis results
        """
        results = {
            'match_count': 0,
            'frame_count': 0,
            'visible_frames': 0,
            'invisible_frames': 0,
            'trajectories': []
        }
        
        # Process each match
        for match_dir in split_dir.glob(self.config['data']['match_pattern']):
            csv_dir = match_dir / self.config['data']['csv_dir']
            if not csv_dir.exists():
                logger.warning(f"CSV directory not found: {csv_dir}")
                continue
                
            results['match_count'] += 1
            
            # Process each CSV file
            for csv_file in csv_dir.glob(self.config['data']['annotation_pattern']):
                df = pd.read_csv(csv_file)
                
                results['frame_count'] += len(df)
                results['visible_frames'] += df['Visibility'].sum()
                results['invisible_frames'] += len(df) - df['Visibility'].sum()
                
                # Extract trajectory
                visible_points = df[df['Visibility'] == 1][['X', 'Y']].values
                if len(visible_points) > 0:
                    results['trajectories'].append(visible_points)
        
        return results
    
    def generate_visualizations(self, results: Dict):
        """
        Generate analysis visualizations.
        
        Args:
            results: Analysis results dictionary
        """
        self._plot_visibility_distribution(results)
        self._plot_position_heatmap(results)
        self._plot_trajectory_characteristics(results)
        self._plot_split_statistics(results)
    
    def _plot_visibility_distribution(self, results: Dict):
        """Plot shuttlecock visibility distribution."""
        plt.figure(figsize=self.fig_size)
        
        # Create visibility data
        splits = list(results['splits'].keys())
        visible = [results['splits'][s]['visible_frames'] for s in splits]
        invisible = [results['splits'][s]['invisible_frames'] for s in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        plt.bar(x - width/2, visible, width, label='Visible')
        plt.bar(x + width/2, invisible, width, label='Invisible')
        
        plt.xlabel('Dataset Split')
        plt.ylabel('Number of Frames')
        plt.title('Shuttlecock Visibility Distribution')
        plt.xticks(x, splits)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visibility_distribution.png', dpi=self.dpi)
        plt.close()
    
    def _plot_position_heatmap(self, results: Dict) -> None:
        """Generate and save position heatmap."""
        if not results['trajectories']:
            logger.warning("No trajectories found for heatmap generation")
            return
            
        all_positions = np.concatenate(results['trajectories'])
        
        # Create 2D histogram
        plt.hist2d(all_positions[:, 0], all_positions[:, 1], 
                  bins=self.config['analysis']['visualization']['heatmap']['resolution'],
                  cmap='viridis')
        
        plt.colorbar(label='Count')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Shuttlecock Position Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'position_heatmap.png', dpi=self.dpi)
        plt.close()
    
    def _plot_trajectory_characteristics(self, results: Dict):
        """Plot trajectory characteristics."""
        plt.figure(figsize=self.fig_size)
        
        # Calculate trajectory lengths
        lengths = [len(traj) for traj in results['trajectories']]
        
        plt.hist(lengths, bins=30)
        plt.xlabel('Trajectory Length (frames)')
        plt.ylabel('Count')
        plt.title('Distribution of Trajectory Lengths')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trajectory_lengths.png', dpi=self.dpi)
        plt.close()
    
    def _plot_split_statistics(self, results: Dict):
        """Plot statistics for each dataset split."""
        plt.figure(figsize=self.fig_size)
        
        splits = list(results['splits'].keys())
        frame_counts = [results['splits'][s]['frame_count'] for s in splits]
        
        plt.bar(splits, frame_counts)
        plt.xlabel('Dataset Split')
        plt.ylabel('Total Frames')
        plt.title('Frame Distribution Across Splits')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'split_statistics.png', dpi=self.dpi)
        plt.close()
    
    def save_results(self, results: Dict):
        """
        Save analysis results to JSON file.
        
        Args:
            results: Analysis results dictionary
        """
        output_file = self.output_dir / 'analysis_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        results = self._prepare_for_serialization(results)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    def _prepare_for_serialization(self, obj: any) -> any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_serialization(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
