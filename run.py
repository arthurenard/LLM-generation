#!/usr/bin/env python
import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl

from src.generation.generator import TextGenerator

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main entry point for the text generation pipeline.
    
    Args:
        config: Configuration from Hydra
    """
    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Set random seed for reproducibility
    pl.seed_everything(config.runtime.seed)
    
    # Setup device
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        config.runtime.device = "cpu"
    
    # Initialize generator
    generator = TextGenerator(config)
    
    # Generate text
    generator.generate()
    
    logger.info("Generation pipeline completed successfully")

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Run main function
    main() 