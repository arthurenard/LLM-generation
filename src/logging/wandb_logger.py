from typing import Dict, Any, List, Optional
import os
import time
import logging
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

logger = logging.getLogger(__name__)

class WandBLogger:
    """Logger for Weights & Biases integration."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the WandB logger.
        
        Args:
            config: Configuration from Hydra
        """
        self.config = config
        self.logging_config = config.logging
        self.enabled = self.logging_config.enabled
        self.run = None
        
        if self.enabled:
            self.setup_wandb()
    
    def setup_wandb(self) -> None:
        """Setup WandB logging."""
        logger.info("Setting up WandB logging")
        
        # Initialize WandB
        self.run = wandb.init(
            project=self.logging_config.project,
            entity=self.logging_config.entity,
            name=self.logging_config.name,
            tags=self.logging_config.tags,
            notes=self.logging_config.notes,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        logger.info(f"WandB initialized: {self.run.name}")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        if not self.enabled or self.run is None:
            return
        
        # Log metrics
        self.run.log(metrics)
    
    def log_generation_metrics(self, 
                              total_tokens: int, 
                              total_time: float, 
                              total_samples: int,
                              batch_size: int,
                              step: int) -> None:
        """
        Log generation metrics to WandB.
        
        Args:
            total_tokens: Total tokens generated
            total_time: Total generation time
            total_samples: Total samples generated
            batch_size: Batch size
            step: Current step
        """
        if not self.enabled or self.run is None:
            return
        
        # Calculate metrics
        throughput = total_tokens / total_time if total_time > 0 else 0
        tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0
        samples_per_second = total_samples / total_time if total_time > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            "throughput/tokens_per_second": throughput,
            "throughput/samples_per_second": samples_per_second,
            "generation/tokens_per_sample": tokens_per_sample,
            "generation/total_tokens": total_tokens,
            "generation/total_samples": total_samples,
            "generation/batch_size": batch_size,
            "time/total_seconds": total_time,
            "time/seconds_per_sample": total_time / total_samples if total_samples > 0 else 0,
        }
        
        # Log metrics
        self.log_metrics(metrics)
    
    def log_samples(self, prompts: List[str], generated_texts: List[str], step: int) -> None:
        """
        Log sample generations to WandB.
        
        Args:
            prompts: List of prompts
            generated_texts: List of generated texts
            step: Current step
        """
        if not self.enabled or self.run is None:
            return
        
        # Limit number of samples to log
        max_samples = min(self.logging_config.log_samples, len(prompts))
        
        # Create table for samples
        columns = ["prompt", "generated_text"]
        data = [[prompts[i], generated_texts[i]] for i in range(max_samples)]
        
        # Log table
        self.run.log({
            "samples": wandb.Table(columns=columns, data=data)
        })
    
    def log_token_diversity(self, generated_texts: List[str], tokenizer) -> None:
        """
        Log token diversity metrics.
        
        Args:
            generated_texts: List of generated texts
            tokenizer: Tokenizer for encoding texts
        """
        if not self.enabled or self.run is None or not self.logging_config.metrics.token_diversity:
            return
        
        # Tokenize texts
        all_token_ids = []
        for text in generated_texts:
            token_ids = tokenizer.encode(text)
            all_token_ids.extend(token_ids)
        
        # Calculate token diversity metrics
        unique_tokens = len(set(all_token_ids))
        total_tokens = len(all_token_ids)
        
        # Calculate entropy
        token_counts = {}
        for token_id in all_token_ids:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
        
        probabilities = [count / total_tokens for count in token_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        
        # Log metrics
        self.log_metrics({
            "diversity/unique_tokens": unique_tokens,
            "diversity/total_tokens": total_tokens,
            "diversity/unique_ratio": unique_tokens / total_tokens if total_tokens > 0 else 0,
            "diversity/entropy": entropy,
        })
    
    def log_memory_usage(self) -> None:
        """Log GPU memory usage."""
        if not self.enabled or self.run is None or not self.logging_config.metrics.memory_usage:
            return
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return
        
        # Get memory usage for each GPU
        memory_allocated = []
        memory_reserved = []
        
        for i in range(torch.cuda.device_count()):
            memory_allocated.append(torch.cuda.memory_allocated(i) / (1024 ** 3))  # GB
            memory_reserved.append(torch.cuda.memory_reserved(i) / (1024 ** 3))  # GB
        
        # Log metrics
        for i, (allocated, reserved) in enumerate(zip(memory_allocated, memory_reserved)):
            self.log_metrics({
                f"memory/gpu{i}/allocated_gb": allocated,
                f"memory/gpu{i}/reserved_gb": reserved,
                f"memory/gpu{i}/utilization": allocated / reserved if reserved > 0 else 0,
            })
    
    def finish(self) -> None:
        """Finish WandB logging."""
        if self.enabled and self.run is not None:
            self.run.finish() 