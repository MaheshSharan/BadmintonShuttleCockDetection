"""
Distributed training setup and utilities.
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Dict, Any, Callable
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """Handles distributed training setup and coordination."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        world_size: int = None,
        backend: str = "nccl",
        find_unused_parameters: bool = False
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            world_size: Number of processes for distributed training
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
            find_unused_parameters: Whether to find unused parameters in DDP
        """
        self.model = model
        self.world_size = world_size or torch.cuda.device_count()
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        self.ddp_model = None
        
    def setup(self, rank: int):
        """
        Setup the distributed process group.
        
        Args:
            rank: Process rank
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend=self.backend,
            init_method='env://',
            world_size=self.world_size,
            rank=rank
        )
        
        torch.cuda.set_device(rank)
        self.model.cuda(rank)
        
        self.ddp_model = DDP(
            self.model,
            device_ids=[rank],
            find_unused_parameters=self.find_unused_parameters
        )
        
    def cleanup(self):
        """Cleanup the distributed process group."""
        dist.destroy_process_group()
        
    @staticmethod
    def run_distributed(
        fn: Callable,
        world_size: int,
        *args,
        **kwargs
    ):
        """
        Run a function in a distributed manner.
        
        Args:
            fn: Function to run
            world_size: Number of processes
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn
        """
        mp.spawn(fn, args=(world_size, *args), nprocs=world_size, join=True)
        
    @property
    def is_distributed(self) -> bool:
        """Whether running in distributed mode."""
        return dist.is_initialized()
        
    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process."""
        return not self.is_distributed or dist.get_rank() == 0
