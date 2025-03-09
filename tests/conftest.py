"""
Common fixtures for testing the text generation pipeline.
"""
import os
import sys
import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def test_config_dir() -> Path:
    """Return the path to the test configuration directory."""
    return Path(__file__).parent / "fixtures" / "config"

@pytest.fixture
def sample_prompts() -> List[str]:
    """Return a list of sample prompts for testing."""
    return [
        "Write a short story about a robot.",
        "Explain quantum computing to a 5-year-old.",
        "Create a recipe for chocolate chip cookies."
    ]

@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Return a mock model configuration for testing."""
    config = {
        "name": "gpt2",
        "pretrained_model_name_or_path": "gpt2",  # Using actual gpt2 model
        "model_type": "causal_lm",
        "revision": "main",
        "quantization": {
            "enabled": False,
            "bits": 8,
            "group_size": 128
        },
        "model_loading": {
            "use_cache": True,
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": False
        },
        "tokenizer": {
            "padding_side": "right",
            "truncation_side": "right",
            "use_fast": True,
            "add_bos_token": False
        }
    }
    return OmegaConf.create(config)

@pytest.fixture
def mock_generation_config() -> Dict[str, Any]:
    """Return a mock generation configuration for testing."""
    config = {
        "temperature": 0.8,
        "top_p": 0.92,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "max_length": 128,  # Smaller for testing
        "min_length": 10,
        "num_return_sequences": 1,
        "do_sample": True,
        "early_stopping": False,
        "batch_size": 2,  # Smaller for testing
        "max_batch_size": 4,
        "chunk_size": 128,
        "prompt_template": "{prompt}",
        "prompt_file": None,
        "num_prompts": 3,
        "vllm": {
            "enabled": False,  # Disabled by default for testing
            "tensor_parallel_size": 1,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.7,
            "swap_space": 1,
            "enforce_eager": True
        }
    }
    return OmegaConf.create(config)

@pytest.fixture
def mock_logging_config() -> Dict[str, Any]:
    """Return a mock logging configuration for testing."""
    config = {
        "enabled": False,  # Disabled for testing
        "project": "test-project",
        "entity": None,
        "name": "test-run",
        "tags": ["test"],  # This is a proper list format
        "notes": "Test run",
        "log_model": False,
        "log_level": "info",
        "log_interval": 1,
        "log_samples": 1,
        "log_gradients": False,
        "metrics": {
            "throughput": True,
            "latency": True,
            "token_diversity": True,
            "memory_usage": True
        }
    }
    return OmegaConf.create(config)

@pytest.fixture
def mock_post_processing_config() -> Dict[str, Any]:
    """Return a mock post-processing configuration for testing."""
    config = {
        "enabled": True,
        "max_length": 200,
        "complete_sentences": True,
        "deduplicate": True,
        "min_length": 5
    }
    return OmegaConf.create(config)

@pytest.fixture
def mock_full_config(
    mock_model_config, 
    mock_generation_config, 
    mock_logging_config, 
    mock_post_processing_config
) -> Dict[str, Any]:
    """Return a complete mock configuration for testing."""
    config = {
        "model": mock_model_config,
        "generation": mock_generation_config,
        "logging": mock_logging_config,
        "post_processing": mock_post_processing_config,
        "output": {
            "dir": "./test_outputs",
            "format": "jsonl",
            "save_interval": 1
        },
        "runtime": {
            "seed": 42,
            "device": "cpu",  # Use CPU for testing
            "num_workers": 1,
            "distributed": False
        }
    }
    return OmegaConf.create(config)

@pytest.fixture
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()

@pytest.fixture
def skip_if_no_cuda(cuda_available):
    """Skip a test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping GPU test") 