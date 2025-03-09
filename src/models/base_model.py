from typing import Dict, Any, Optional, Union, List
import os
import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

class BaseModel(pl.LightningModule):
    """Base model class for text generation models."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration from Hydra
        """
        super().__init__()
        self.config = config
        self.model = None
        self.tokenizer = None
        self.save_hyperparameters(dict(config))
        
    def setup_model(self) -> None:
        """Load the model and tokenizer based on configuration."""
        model_name = self.config.pretrained_model_name_or_path
        logger.info(f"Loading model: {model_name}")
        
        # Setup quantization parameters if enabled
        quantization_kwargs = {}
        if self.config.quantization.enabled:
            if self.config.quantization.bits == 8:
                logger.info("Using 8-bit quantization")
                quantization_kwargs["load_in_8bit"] = True
            elif self.config.quantization.bits == 4:
                logger.info("Using 4-bit quantization")
                quantization_kwargs["load_in_4bit"] = True
                quantization_kwargs["bnb_4bit_compute_dtype"] = getattr(
                    torch, self.config.model_loading.torch_dtype
                ) if self.config.model_loading.torch_dtype != "auto" else torch.float16
                quantization_kwargs["bnb_4bit_quant_type"] = "nf4"
                quantization_kwargs["bnb_4bit_use_double_quant"] = True
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=self.config.revision,
            torch_dtype=getattr(torch, self.config.model_loading.torch_dtype) 
                if self.config.model_loading.torch_dtype != "auto" else "auto",
            device_map=self.config.model_loading.device_map,
            trust_remote_code=self.config.model_loading.trust_remote_code,
            use_cache=self.config.model_loading.use_cache,
            **quantization_kwargs
        )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=self.config.revision,
            use_fast=self.config.tokenizer.use_fast,
            padding_side=self.config.tokenizer.padding_side,
            truncation_side=self.config.tokenizer.truncation_side,
            trust_remote_code=self.config.model_loading.trust_remote_code,
        )
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Model loaded successfully: {model_name}")
        
    def get_model(self) -> PreTrainedModel:
        """Get the loaded model."""
        if self.model is None:
            self.setup_model()
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the loaded tokenizer."""
        if self.tokenizer is None:
            self.setup_model()
        return self.tokenizer
    
    def prepare_inputs(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Dictionary of model inputs
        """
        tokenizer = self.get_tokenizer()
        
        # Add BOS token if configured
        if self.config.tokenizer.add_bos_token:
            processed_prompts = []
            for prompt in prompts:
                if not prompt.startswith(tokenizer.bos_token):
                    prompt = tokenizer.bos_token + prompt
                processed_prompts.append(prompt)
            prompts = processed_prompts
        
        # Tokenize inputs
        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_length", 1024),
        )
        
        return inputs
    
    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.get_model()(**inputs)
    
    def generate(self, prompts: List[str], generation_config: Dict[str, Any]) -> List[str]:
        """
        Generate text based on prompts.
        
        Args:
            prompts: List of text prompts
            generation_config: Generation parameters
            
        Returns:
            List of generated texts
        """
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        inputs = self.prepare_inputs(prompts)
        
        # Move inputs to the correct device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate outputs
        outputs = model.generate(
            **inputs,
            max_length=generation_config.get("max_length", 1024),
            min_length=generation_config.get("min_length", 10),
            do_sample=generation_config.get("do_sample", True),
            temperature=generation_config.get("temperature", 0.8),
            top_p=generation_config.get("top_p", 0.92),
            top_k=generation_config.get("top_k", 50),
            repetition_penalty=generation_config.get("repetition_penalty", 1.0),
            num_return_sequences=generation_config.get("num_return_sequences", 1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Decode outputs
        generated_texts = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        return generated_texts 