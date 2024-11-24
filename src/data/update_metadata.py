"""
Update metadata.json paths without regenerating frames.
"""
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_metadata_paths(split_dir: Path):
    """Update paths in metadata.json to include match directory names."""
    metadata_path = split_dir / 'metadata.json'
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return
    
    # Read existing metadata
    logger.info(f"Reading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Update paths for each sequence
    updated_sequences = []
    for seq in metadata['sequences']:
        # Get sequence ID and find corresponding match directory
        sequence_id = seq['sequence_id']
        match_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('match')]
        
        # Find which match directory contains this sequence
        for match_dir in match_dirs:
            csv_path = match_dir / 'csv' / f"{sequence_id}_ball.csv"
            if csv_path.exists():
                # Update paths to include match directory
                seq['frames_dir'] = f"{match_dir.name}/frames/{sequence_id}"
                seq['annotations_path'] = f"{match_dir.name}/csv/{sequence_id}_ball.csv"
                updated_sequences.append(seq)
                logger.info(f"Updated paths for sequence {sequence_id} in {match_dir.name}")
                break
    
    # Update metadata with new sequences
    metadata['sequences'] = updated_sequences
    
    # Backup original metadata
    backup_path = metadata_path.with_suffix('.json.bak')
    logger.info(f"Creating backup at {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save updated metadata
    logger.info(f"Saving updated metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Successfully updated {len(updated_sequences)} sequences in metadata")

def main():
    # Update Train split
    train_dir = Path("ShuttleCockFrameDataset/Train")
    logger.info(f"Updating metadata for Train split")
    update_metadata_paths(train_dir)
    
    # Update valid split
    valid_dir = Path("ShuttleCockFrameDataset/valid")
    logger.info(f"Updating metadata for valid split")
    update_metadata_paths(valid_dir)

if __name__ == '__main__':
    main()
