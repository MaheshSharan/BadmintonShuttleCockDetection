import json
import logging
from pathlib import Path
from typing import Dict
import argparse

from dataset_analyzer import DatasetAnalyzer
from trajectory_visualizer import TrajectoryVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_analysis(config_path: str):
    """
    Run complete dataset analysis pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    dataset_dir = Path(config['data']['root_dir'])
    
    logger.info("Starting dataset analysis pipeline...")
    
    # Initialize analyzers
    dataset_analyzer = DatasetAnalyzer(config)
    trajectory_visualizer = TrajectoryVisualizer(config)
    
    # Run dataset analysis
    logger.info("Analyzing dataset statistics...")
    analysis_results = dataset_analyzer.analyze_dataset(dataset_dir)
    dataset_analyzer.generate_visualizations(analysis_results)
    dataset_analyzer.save_results(analysis_results)
    
    # Generate trajectory visualizations
    logger.info("Generating trajectory visualizations...")
    trajectories = analysis_results['trajectories']
    if trajectories:
        # Basic trajectory visualization
        trajectory_visualizer.visualize_trajectories(trajectories)
        
        # Get frame indices for each trajectory
        frame_indices = [range(len(traj)) for traj in trajectories]
        
        # Generate advanced visualizations
        trajectory_visualizer.visualize_3d_trajectories(trajectories, frame_indices)
        trajectory_visualizer.plot_velocity_profile(trajectories, frame_indices)
        trajectory_visualizer.plot_acceleration_profile(trajectories, frame_indices)
        trajectory_visualizer.plot_motion_characteristics(trajectories)
    
    logger.info("Analysis pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Run shuttlecock dataset analysis")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    run_analysis(args.config)

if __name__ == '__main__':
    main()
