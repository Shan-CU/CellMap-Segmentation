"""
Distributed Data Parallel (DDP) utilities for multi-GPU training.

DDP is preferred over DataParallel because:
1. Better scaling - One process per GPU avoids Python's GIL bottleneck
2. More efficient gradient synchronization via NCCL all-reduce
3. Lower memory overhead - each process holds only its own gradients
4. Proper batch distribution across processes

Usage:
    # In training script:
    from cellmap_segmentation_challenge.utils import setup_ddp, cleanup_ddp, is_main_process
    
    # Initialize DDP (call at start of training)
    local_rank, world_size = setup_ddp()
    
    # Wrap model
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Only log/save on main process
    if is_main_process():
        print("Training started...")
        torch.save(model.state_dict(), "checkpoint.pth")
    
    # Cleanup at end
    cleanup_ddp()
    
Launch with torchrun:
    torchrun --nproc_per_node=3 train_script.py
"""

import os
import torch
import torch.distributed as dist
from typing import Tuple, Optional


def setup_ddp(backend: str = "nccl") -> Tuple[int, int]:
    """
    Initialize Distributed Data Parallel (DDP) process group.
    
    This function should be called at the beginning of training when using
    multiple GPUs with DDP. It reads environment variables set by torchrun
    (or torch.distributed.launch) to configure the distributed training.
    
    Parameters
    ----------
    backend : str
        The distributed backend to use. Default is "nccl" which is optimized
        for NVIDIA GPUs. Use "gloo" for CPU or as a fallback.
    
    Returns
    -------
    Tuple[int, int]
        A tuple of (local_rank, world_size) where:
        - local_rank: The GPU index for this process (0, 1, 2, ...)
        - world_size: Total number of processes/GPUs
    
    Notes
    -----
    Environment variables used:
    - LOCAL_RANK: GPU index on this node (set by torchrun)
    - RANK: Global process rank (set by torchrun)
    - WORLD_SIZE: Total number of processes (set by torchrun)
    - MASTER_ADDR: Address of rank 0 process (set by torchrun, default localhost)
    - MASTER_PORT: Port for communication (set by torchrun, default 29500)
    """
    # Get environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Only initialize if world_size > 1 (multi-GPU)
    if world_size > 1:
        # Set the device before initializing process group
        torch.cuda.set_device(local_rank)
        
        # Initialize the process group
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        
        # Synchronize all processes
        dist.barrier()
        
        if is_main_process():
            print(f"DDP initialized: {world_size} processes, backend={backend}")
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    return local_rank, world_size


def cleanup_ddp():
    """
    Clean up the DDP process group.
    
    Should be called at the end of training to properly release resources.
    Safe to call even if DDP was not initialized (single GPU case).
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_ddp_initialized() -> bool:
    """
    Check if DDP is initialized.
    
    Returns
    -------
    bool
        True if DDP process group is initialized, False otherwise.
    """
    return dist.is_initialized()


def is_main_process() -> bool:
    """
    Check if this is the main (rank 0) process.
    
    Use this to guard operations that should only happen once, like:
    - Logging
    - Saving checkpoints
    - Creating directories
    - Printing progress
    
    Returns
    -------
    bool
        True if this is rank 0 or if DDP is not initialized (single GPU).
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """
    Get the total number of processes in the DDP group.
    
    Returns
    -------
    int
        World size (number of GPUs/processes), or 1 if DDP not initialized.
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the global rank of this process.
    
    Returns
    -------
    int
        Global rank (0 to world_size-1), or 0 if DDP not initialized.
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Get the local rank (GPU index) of this process on this node.
    
    Returns
    -------
    int
        Local rank, or 0 if DDP not initialized.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def reduce_value(value: float, op: str = "mean") -> float:
    """
    Reduce a scalar value across all processes.
    
    Parameters
    ----------
    value : float
        The value to reduce from this process.
    op : str
        The reduction operation: "mean", "sum", "min", or "max".
    
    Returns
    -------
    float
        The reduced value.
    """
    if not dist.is_initialized():
        return value
    
    tensor = torch.tensor(value, dtype=torch.float32, device="cuda")
    
    if op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")
    
    return tensor.item()


def broadcast_object(obj, src: int = 0):
    """
    Broadcast a Python object from source rank to all other ranks.
    
    Parameters
    ----------
    obj : Any
        The object to broadcast (only used on source rank).
    src : int
        Source rank to broadcast from. Default is 0.
    
    Returns
    -------
    Any
        The broadcasted object (same on all ranks after call).
    """
    if not dist.is_initialized():
        return obj
    
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def sync_across_processes():
    """
    Synchronize all processes with a barrier.
    
    Use this when you need to ensure all processes have reached
    the same point before continuing (e.g., after saving a checkpoint
    that other processes might need to load).
    """
    if dist.is_initialized():
        dist.barrier()


def get_ddp_sampler(dataset, shuffle: bool = True, seed: int = 0):
    """
    Get a DistributedSampler for the dataset.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to sample from.
    shuffle : bool
        Whether to shuffle the data. Default is True.
    seed : int
        Random seed for shuffling. Default is 0.
    
    Returns
    -------
    torch.utils.data.distributed.DistributedSampler or None
        The distributed sampler, or None if DDP is not initialized.
    """
    if not dist.is_initialized():
        return None
    
    from torch.utils.data.distributed import DistributedSampler
    
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
    )
