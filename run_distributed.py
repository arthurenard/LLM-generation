#!/usr/bin/env python
import os
import sys
import logging
import hydra
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from src.generation.generator import TextGenerator
from src.utils.data_utils import load_prompts

logger = logging.getLogger(__name__)

def setup_distributed(rank, world_size):
    """
    Setup distributed training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main entry point for distributed text generation.
    
    Args:
        config: Configuration from Hydra
    """
    # Enable distributed mode
    config.runtime.distributed = True
    
    # Get world size and rank
    world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        logger.warning("Only one GPU detected, falling back to single-GPU mode")
        # Run in single-GPU mode
        run_generation(0, 1, config)
    else:
        logger.info(f"Running in distributed mode with {world_size} GPUs")
        
        # Update vLLM tensor parallel size if using vLLM
        if config.generation.vllm.enabled:
            config.generation.vllm.tensor_parallel_size = world_size
        
        # Spawn processes
        torch.multiprocessing.spawn(
            run_generation,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    
    logger.info("Distributed generation completed successfully")

def run_generation(rank, world_size, config):
    """
    Run generation on a single process.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        config: Configuration
    """
    # Setup distributed environment
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set device
    config.runtime.device = f"cuda:{rank}"
    
    # Only log on rank 0
    if rank == 0:
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.logging.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(config.output.dir, "generation.log"))
            ]
        )
        
        logger.info(f"Process {rank}/{world_size} started")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Set random seed for reproducibility
    pl.seed_everything(config.runtime.seed + rank)
    
    # Load prompts
    all_prompts = load_prompts(config)
    
    # Distribute prompts across processes
    prompts_per_process = len(all_prompts) // world_size
    start_idx = rank * prompts_per_process
    end_idx = start_idx + prompts_per_process if rank < world_size - 1 else len(all_prompts)
    process_prompts = all_prompts[start_idx:end_idx]
    
    logger.info(f"Process {rank} handling {len(process_prompts)} prompts ({start_idx} to {end_idx-1})")
    
    # Initialize generator
    generator = TextGenerator(config)
    
    # Generate text
    generator.generate(process_prompts)
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()
    
    # Log completion on rank 0
    if rank == 0:
        logger.info("Generation pipeline completed successfully")

if __name__ == "__main__":
    # Run main function
    main() 