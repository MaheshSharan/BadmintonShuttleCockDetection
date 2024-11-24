"""
Preprocess the shuttlecock dataset by extracting frames and preparing annotations.
"""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

def extract_frames(video_path: Path, output_dir: Path, frame_indices: list = None, sequence_id: str = None):
    """Extract frames from video."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False
            
        if frame_indices is None:
            frame_indices = range(total_frames)
            
        # Create progress bar for frame extraction
        pbar = tqdm(
            frame_indices,
            desc=f"Extracting frames from {sequence_id}",
            leave=False
        )
        
        extracted_count = 0
        for frame_idx in pbar:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
                continue
                
            # Save frame
            frame_path = output_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_count += 1
            
            # Update progress bar description
            pbar.set_description(f"Extracting frames from {sequence_id} ({extracted_count}/{len(frame_indices)})")
        
        cap.release()
        logger.info(f"Successfully extracted {extracted_count} frames from {sequence_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        return False

def process_sequence(match_dir: Path, sequence_id: str, target_size=(720, 1280)):
    """Process a single sequence."""
    try:
        # Get paths
        csv_path = match_dir / "csv" / f"{sequence_id}_ball.csv"
        video_path = match_dir / "video" / f"{sequence_id}.mp4"
        frames_dir = match_dir / "frames" / sequence_id
        
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return None
            
        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return None
            
        logger.info(f"Processing sequence: {sequence_id}")
        logger.info(f"CSV: {csv_path}")
        logger.info(f"Video: {video_path}")
        
        # Read and preprocess annotations
        df = pd.read_csv(csv_path)
        original_columns = list(df.columns)
        df.columns = df.columns.str.lower()  # Convert column names to lowercase
        logger.info(f"Converted columns from {original_columns} to {list(df.columns)}")
        
        # Extract frames
        frame_indices = df['frame'].unique().tolist()
        logger.info(f"Found {len(frame_indices)} unique frames in annotations")
        
        success = extract_frames(video_path, frames_dir, frame_indices, sequence_id)
        if not success:
            logger.error(f"Failed to extract frames for {sequence_id}")
            return None
            
        # Create metadata with correct relative paths including match directory
        metadata = {
            'sequence_id': sequence_id,
            'frame_count': len(frame_indices),
            'frame_size': target_size,
            'frames_dir': str(match_dir.name + "/" + frames_dir.relative_to(match_dir)),
            'annotations_path': str(match_dir.name + "/" + csv_path.relative_to(match_dir)),
            'total_annotations': len(df)
        }
        
        logger.info(f"Successfully processed sequence {sequence_id}")
        logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error processing sequence {sequence_id}: {e}")
        return None

def preprocess_dataset(root_dir: Path, target_size=(720, 1280)):
    """Preprocess entire dataset."""
    start_time = datetime.now()
    root_dir = Path(root_dir)
    
    logger.info(f"Starting dataset preprocessing at {start_time}")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Target size: {target_size}")
    
    # Process each split (train/valid)
    for split in ['Train', 'valid']:
        split_dir = root_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        logger.info(f"\nProcessing {split} split...")
        metadata_list = []
        
        # Process each match
        match_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('match')]
        logger.info(f"Found {len(match_dirs)} matches in {split} split")
        
        # Create progress bar for matches
        match_pbar = tqdm(
            match_dirs,
            desc=f"Processing {split} matches",
            unit="match"
        )
        
        for match_dir in match_pbar:
            logger.info(f"\nProcessing match: {match_dir.name}")
            
            # Get all CSV files
            csv_files = list((match_dir / 'csv').glob('*_ball.csv'))
            sequence_ids = [f.stem.replace('_ball', '') for f in csv_files]
            logger.info(f"Found {len(sequence_ids)} sequences in {match_dir.name}")
            
            # Process each sequence
            successful_sequences = 0
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_sequence, match_dir, seq_id, target_size)
                    for seq_id in sequence_ids
                ]
                
                for future in futures:
                    metadata = future.result()
                    if metadata:
                        metadata_list.append(metadata)
                        successful_sequences += 1
            
            # Update progress bar description
            match_pbar.set_description(
                f"Processing {split} matches ({successful_sequences}/{len(sequence_ids)} sequences processed)"
            )
        
        # Save metadata
        metadata_path = split_dir / 'metadata.json'
        metadata_dict = {
            'sequences': metadata_list,
            'target_size': target_size,
            'total_sequences': len(metadata_list),
            'processing_time': str(datetime.now() - start_time)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"\nCompleted {split} split:")
        logger.info(f"- Processed {len(metadata_list)} sequences")
        logger.info(f"- Metadata saved to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess shuttlecock dataset')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing Train and valid splits')
    args = parser.parse_args()
    
    preprocess_dataset(args.root_dir)

if __name__ == '__main__':
    main()
