from typing import Union, Dict, Any
import logging
from omegaconf import DictConfig
import pytorch_lightning as pl

from src.models.base_model import BaseModel
from src.models.vllm_model import VLLMModel

logger = logging.getLogger(__name__)

def create_model(model_config: DictConfig, generation_config: DictConfig) -> Union[BaseModel, VLLMModel]:
    """
    Create a model instance based on configuration.
    
    Args:
        model_config: Model configuration
        generation_config: Generation configuration
        
    Returns:
        Model instance
    """
    # Check if vLLM is enabled
    if generation_config.vllm.enabled:
        logger.info("Creating vLLM model for high-performance inference")
        return VLLMModel(model_config, generation_config)
    else:
        logger.info("Creating standard Hugging Face model")
        return BaseModel(model_config) 