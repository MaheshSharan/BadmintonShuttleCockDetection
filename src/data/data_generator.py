"""
Advanced data generator with efficient loading and caching.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import h5py
from typing import List, Tuple, Dict, Optional
import threading
from queue import Queue
import mmap
from .video_processor import VideoProcessor

class ShuttlecockDataGenerator(Dataset):
    """Efficient data generator with caching and prefetching."""
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        sequence_length: int = 16,
        frame_size: Tuple[int, int] = (512, 512),
        cache_size: int = 1000,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        use_cache: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.use_cache = use_cache
        
        # Initialize components
        self.video_processor = VideoProcessor(
            frame_size=frame_size,
            sequence_length=sequence_length,
            num_workers=num_workers,
            cache_size=cache_size
        )
        
        # Setup caching
        self.cache_dir = self.data_dir / '.cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / f'{mode}_cache.h5'
        self._setup_cache()
        
        # Load dataset structure
        self.matches = self._load_dataset_structure()
        self.sequences = self._generate_sequences()
        
        # Setup prefetching
        self.prefetch_queue = Queue(maxsize=cache_size)
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence with efficient caching."""
        sequence_info = self.sequences[idx]
        
        # Try to get from cache first
        if self.use_cache:
            cached_data = self._get_from_cache(sequence_info['cache_key'])
            if cached_data is not None:
                return cached_data
        
        # Load and process sequence
        frames, labels = self._load_sequence(sequence_info)
        
        # Cache the processed sequence
        if self.use_cache:
            self._add_to_cache(sequence_info['cache_key'], frames, labels)
        
        return {
            'frames': torch.from_numpy(frames),
            'labels': torch.from_numpy(labels)
        }
        
    def _setup_cache(self):
        """Setup HDF5 cache file."""
        if self.use_cache and not self.cache_file.exists():
            with h5py.File(self.cache_file, 'w') as f:
                f.create_group('frames')
                f.create_group('labels')
                
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get sequence from cache."""
        if not self.use_cache:
            return None
            
        try:
            with h5py.File(self.cache_file, 'r') as f:
                if cache_key in f['frames']:
                    return {
                        'frames': torch.from_numpy(f['frames'][cache_key][:]),
                        'labels': torch.from_numpy(f['labels'][cache_key][:])
                    }
        except:
            return None
            
    def _add_to_cache(self, cache_key: str, frames: np.ndarray, 
                      labels: np.ndarray):
        """Add sequence to cache."""
        if not self.use_cache:
            return
            
        try:
            with h5py.File(self.cache_file, 'a') as f:
                if cache_key not in f['frames']:
                    f['frames'].create_dataset(cache_key, data=frames)
                    f['labels'].create_dataset(cache_key, data=labels)
        except:
            pass
            
    def _load_dataset_structure(self) -> List[Dict]:
        """Load dataset structure with memory mapping."""
        matches = []
        match_dirs = sorted(self.data_dir.glob('match*'))
        
        for match_dir in match_dirs:
            video_file = next(match_dir.glob('*.mp4'))
            csv_file = next(match_dir.glob('*.csv'))
            
            # Memory map the video file
            video_map = mmap.mmap(
                video_file.open('rb').fileno(),
                0,
                access=mmap.ACCESS_READ
            )
            
            matches.append({
                'match_id': match_dir.name,
                'video_path': str(video_file),
                'csv_path': str(csv_file),
                'video_map': video_map
            })
            
        return matches
        
    def _generate_sequences(self) -> List[Dict]:
        """Generate sequence information."""
        sequences = []
        
        for match in self.matches:
            # Get video length
            cap = cv2.VideoCapture(match['video_path'])
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Generate sequences with overlap
            overlap = self.sequence_length // 2
            for start_idx in range(0, total_frames - self.sequence_length, overlap):
                sequences.append({
                    'match_id': match['match_id'],
                    'start_frame': start_idx,
                    'end_frame': start_idx + self.sequence_length,
                    'video_path': match['video_path'],
                    'csv_path': match['csv_path'],
                    'cache_key': f"{match['match_id']}_{start_idx}"
                })
                
        return sequences
        
    def _load_sequence(self, sequence_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process a sequence."""
        # Start video processing
        self.video_processor.start_processing(sequence_info['video_path'])
        
        # Get processed frames
        frames = []
        while len(frames) < self.sequence_length:
            if not self.video_processor.processed_queue.empty():
                frames.extend(self.video_processor.processed_queue.get())
                
        # Stop processing
        self.video_processor.stop_processing()
        
        # Load labels
        labels = self._load_labels(sequence_info)
        
        return np.array(frames), labels
        
    def _load_labels(self, sequence_info: Dict) -> np.ndarray:
        """Load and process sequence labels."""
        # Implementation depends on your label format
        # This is a placeholder
        return np.zeros((self.sequence_length, 4))  # [x, y, visibility, class]
        
    def _prefetch_worker(self):
        """Prefetch worker for async loading."""
        while True:
            if self.prefetch_queue.qsize() < self.cache_size:
                # Get next sequences to prefetch
                current_size = self.prefetch_queue.qsize()
                num_to_fetch = min(
                    self.cache_size - current_size,
                    self.prefetch_factor
                )
                
                for _ in range(num_to_fetch):
                    idx = np.random.randint(len(self))
                    sequence_info = self.sequences[idx]
                    
                    # Check cache first
                    cached_data = self._get_from_cache(sequence_info['cache_key'])
                    if cached_data is not None:
                        self.prefetch_queue.put(cached_data)
                        continue
                        
                    # Load and process sequence
                    frames, labels = self._load_sequence(sequence_info)
                    
                    # Cache the sequence
                    if self.use_cache:
                        self._add_to_cache(
                            sequence_info['cache_key'],
                            frames,
                            labels
                        )
                        
                    self.prefetch_queue.put({
                        'frames': torch.from_numpy(frames),
                        'labels': torch.from_numpy(labels)
                    })
                    
    def get_dataloader(self, batch_size: int = 8) -> DataLoader:
        """Get DataLoader with optimal settings."""
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )
