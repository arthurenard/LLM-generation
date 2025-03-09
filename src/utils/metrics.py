from typing import List, Dict, Any, Optional
import time
import numpy as np
import torch
from transformers import PreTrainedTokenizer

def calculate_token_diversity(texts: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
    """
    Calculate token diversity metrics for generated texts.
    
    Args:
        texts: List of generated texts
        tokenizer: Tokenizer for encoding texts
        
    Returns:
        Dictionary of diversity metrics
    """
    # Tokenize texts
    all_token_ids = []
    for text in texts:
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
    
    return {
        "unique_tokens": unique_tokens,
        "total_tokens": total_tokens,
        "unique_ratio": unique_tokens / total_tokens if total_tokens > 0 else 0,
        "entropy": entropy,
    }

def calculate_throughput(total_tokens: int, total_time: float, total_samples: int) -> Dict[str, float]:
    """
    Calculate throughput metrics.
    
    Args:
        total_tokens: Total tokens generated
        total_time: Total generation time
        total_samples: Total samples generated
        
    Returns:
        Dictionary of throughput metrics
    """
    throughput = total_tokens / total_time if total_time > 0 else 0
    tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0
    samples_per_second = total_samples / total_time if total_time > 0 else 0
    
    return {
        "tokens_per_second": throughput,
        "samples_per_second": samples_per_second,
        "tokens_per_sample": tokens_per_sample,
        "seconds_per_sample": total_time / total_samples if total_samples > 0 else 0,
    }

def calculate_memory_usage() -> Dict[str, float]:
    """
    Calculate GPU memory usage.
    
    Returns:
        Dictionary of memory usage metrics
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return {}
    
    # Get memory usage for each GPU
    memory_metrics = {}
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
        
        memory_metrics[f"gpu{i}_allocated_gb"] = allocated
        memory_metrics[f"gpu{i}_reserved_gb"] = reserved
        memory_metrics[f"gpu{i}_utilization"] = allocated / reserved if reserved > 0 else 0
    
    return memory_metrics

class PerformanceTracker:
    """Utility class for tracking generation performance."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.start_time = None
        self.end_time = None
        self.total_tokens = 0
        self.total_samples = 0
        self.batch_times = []
    
    def start_batch(self):
        """Start timing a batch."""
        self.start_time = time.time()
    
    def end_batch(self, num_tokens: int, num_samples: int):
        """
        End timing a batch and update metrics.
        
        Args:
            num_tokens: Number of tokens in the batch
            num_samples: Number of samples in the batch
        """
        self.end_time = time.time()
        batch_time = self.end_time - self.start_time
        
        self.total_tokens += num_tokens
        self.total_samples += num_samples
        self.batch_times.append((batch_time, num_tokens, num_samples))
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        total_time = sum(time for time, _, _ in self.batch_times)
        
        metrics = {
            "total_tokens": self.total_tokens,
            "total_samples": self.total_samples,
            "total_time": total_time,
        }
        
        # Add throughput metrics
        throughput_metrics = calculate_throughput(self.total_tokens, total_time, self.total_samples)
        metrics.update({f"throughput/{k}": v for k, v in throughput_metrics.items()})
        
        # Add memory metrics
        memory_metrics = calculate_memory_usage()
        metrics.update({f"memory/{k}": v for k, v in memory_metrics.items()})
        
        return metrics 