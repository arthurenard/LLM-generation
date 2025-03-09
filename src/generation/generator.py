from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
import os
import time
import json
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from src.models.model_factory import create_model
from src.utils.data_utils import load_prompts, save_batch
from src.utils.post_processing import batch_post_process
from src.utils.metrics import PerformanceTracker
from src.logging.wandb_logger import WandBLogger

logger = logging.getLogger(__name__)

class TextGenerator:
    """Text generation pipeline for synthetic data generation."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the text generator.
        
        Args:
            config: Configuration from Hydra
        """
        self.config = config
        self.model = None
        self.output_dir = Path(config.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Initialize WandB logger if enabled
        self.wandb_logger = None
        if config.logging.enabled:
            self.wandb_logger = WandBLogger(config)
    
    def setup_logging(self) -> None:
        """Setup logging for the generator."""
        # Configure basic logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "generation.log")
            ]
        )
        
        # Save configuration
        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.config))
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Configuration: {OmegaConf.to_yaml(self.config)}")
    
    def setup_model(self) -> None:
        """Setup the model for generation."""
        # Create model based on configuration
        self.model = create_model(self.config.model, self.config.generation)
        
        # Log model information
        logger.info(f"Model: {self.config.model.name}")
        logger.info(f"Generation parameters: {OmegaConf.to_yaml(self.config.generation)}")
    
    def get_model(self):
        """Get the model, initializing it if necessary."""
        if self.model is None:
            self.setup_model()
        return self.model
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of generated texts
        """
        model = self.get_model()
        
        # Start timing
        self.performance_tracker.start_batch()
        
        # Generate text
        generated_texts = model.generate(prompts, self.config.generation)
        
        # Apply post-processing if enabled
        if self.config.post_processing.enabled:
            generated_texts = batch_post_process(generated_texts, self.config)
        
        # Get tokenizer to count tokens
        tokenizer = model.get_tokenizer()
        total_tokens = sum(len(tokenizer.encode(text)) for text in generated_texts)
        
        # End timing and update metrics
        self.performance_tracker.end_batch(total_tokens, len(generated_texts))
        
        # Log metrics to WandB if enabled
        if self.wandb_logger is not None:
            metrics = self.performance_tracker.get_metrics()
            self.wandb_logger.log_metrics(metrics)
            
            # Log token diversity if enabled
            if self.config.logging.metrics.token_diversity:
                self.wandb_logger.log_token_diversity(generated_texts, tokenizer)
            
            # Log memory usage if enabled
            if self.config.logging.metrics.memory_usage:
                self.wandb_logger.log_memory_usage()
        
        return generated_texts
    
    def generate(self, prompts: Optional[List[str]] = None) -> None:
        """
        Generate text for all prompts.
        
        Args:
            prompts: Optional list of prompts. If None, prompts will be loaded from file or generated.
        """
        # Load or create prompts if not provided
        if prompts is None:
            prompts = load_prompts(self.config)
        
        # Get batch size
        batch_size = self.config.generation.batch_size
        
        # Create output file
        output_file = self.output_dir / f"generated_{int(time.time())}.{self.config.output.format}"
        
        # Generate in batches
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        logger.info(f"Starting generation for {len(prompts)} prompts with batch size {batch_size}")
        logger.info(f"Output file: {output_file}")
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", total=total_batches):
            # Get batch of prompts
            batch_prompts = prompts[i:i+batch_size]
            
            # Generate text for batch
            generated_texts = self.generate_batch(batch_prompts)
            
            # Create batch data
            batch_data = [
                {
                    "prompt": prompt,
                    "generated_text": text,
                    "timestamp": time.time(),
                    "model": self.config.model.name,
                    "parameters": {
                        "temperature": self.config.generation.temperature,
                        "top_p": self.config.generation.top_p,
                        "top_k": self.config.generation.top_k,
                        "max_length": self.config.generation.max_length,
                    }
                }
                for prompt, text in zip(batch_prompts, generated_texts)
            ]
            
            # Save batch
            save_batch(batch_data, output_file, self.config.output.format)
            
            # Log progress
            if (i // batch_size) % self.config.logging.log_interval == 0:
                metrics = self.performance_tracker.get_metrics()
                logger.info(f"Generated {i + len(batch_prompts)}/{len(prompts)} samples")
                logger.info(f"Throughput: {metrics['throughput/tokens_per_second']:.2f} tokens/sec")
                
                # Log sample if configured
                if self.config.logging.log_samples > 0 and i < self.config.logging.log_samples * batch_size:
                    sample_idx = min(self.config.logging.log_samples - 1, len(generated_texts) - 1)
                    logger.info(f"Sample prompt: {batch_prompts[sample_idx][:100]}...")
                    logger.info(f"Sample generation: {generated_texts[sample_idx][:200]}...")
                
                # Log samples to WandB if enabled
                if self.wandb_logger is not None and i < self.config.logging.log_samples * batch_size:
                    self.wandb_logger.log_samples(batch_prompts, generated_texts, i // batch_size)
        
        # Log final metrics
        metrics = self.performance_tracker.get_metrics()
        logger.info("Generation complete")
        logger.info(f"Total samples: {metrics['total_samples']}")
        logger.info(f"Total tokens: {metrics['total_tokens']}")
        logger.info(f"Total time: {metrics['total_time']:.2f} seconds")
        logger.info(f"Final throughput: {metrics['throughput/tokens_per_second']:.2f} tokens/sec")
        
        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Finish WandB logging
        if self.wandb_logger is not None:
            self.wandb_logger.finish()
    
    def generate_streaming(self, prompts: Optional[List[str]] = None, callback=None) -> Iterator[Tuple[int, str]]:
        """
        Generate text with streaming output.
        
        Args:
            prompts: Optional list of prompts
            callback: Optional callback function for streaming outputs
            
        Yields:
            Generated text chunks as they become available
        """
        # Load or create prompts if not provided
        if prompts is None:
            prompts = load_prompts(self.config)
        
        model = self.get_model()
        
        # Check if model supports streaming
        if not hasattr(model, "generate_with_streaming"):
            logger.warning("Model does not support streaming generation, falling back to batch generation")
            generated_texts = self.generate_batch(prompts)
            for i, text in enumerate(generated_texts):
                yield i, text
            return
        
        # Generate with streaming
        for request_id, response in model.generate_with_streaming(prompts, self.config.generation, callback):
            # Apply post-processing if enabled and we have the final response
            if self.config.post_processing.enabled and hasattr(response, "finished") and response.finished:
                text = response.outputs[0].text
                processed_text = batch_post_process([text], self.config)[0]
                response.outputs[0].text = processed_text
            
            yield request_id, response 