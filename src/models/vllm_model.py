from typing import Dict, Any, List, Optional, Union
import time
import torch
import logging
from omegaconf import DictConfig
import pytorch_lightning as pl
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class VLLMModel(pl.LightningModule):
    """Model class for vLLM-based inference."""
    
    def __init__(self, config: DictConfig, generation_config: DictConfig):
        """
        Initialize the vLLM model.
        
        Args:
            config: Model configuration from Hydra
            generation_config: Generation configuration from Hydra
        """
        super().__init__()
        self.config = config
        self.generation_config = generation_config
        self.model = None
        self.tokenizer = None
        self.save_hyperparameters({
            "model": dict(config),
            "generation": dict(generation_config)
        })
        
    def setup_model(self) -> None:
        """Load the model and tokenizer using vLLM."""
        model_name = self.config.pretrained_model_name_or_path
        logger.info(f"Loading model with vLLM: {model_name}")
        
        # Determine quantization settings
        quantization_config = None
        if self.config.quantization.enabled:
            if self.config.quantization.bits == 4:
                quantization_config = "awq"  # vLLM supports AWQ for 4-bit
            elif self.config.quantization.bits == 8:
                quantization_config = "int8"  # 8-bit quantization
        
        # Load the model with vLLM
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=self.generation_config.vllm.tensor_parallel_size,
            gpu_memory_utilization=self.generation_config.vllm.gpu_memory_utilization,
            max_model_len=self.generation_config.vllm.max_model_len,
            quantization=quantization_config,
            trust_remote_code=self.config.model_loading.trust_remote_code,
            dtype=self.config.model_loading.torch_dtype if self.config.model_loading.torch_dtype != "auto" else "half",
            swap_space=self.generation_config.vllm.swap_space,
            enforce_eager=self.generation_config.vllm.enforce_eager,
            revision=self.config.revision,
        )
        
        # Load the tokenizer separately for preprocessing
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
            
        logger.info(f"vLLM model loaded successfully: {model_name}")
    
    def get_model(self) -> LLM:
        """Get the loaded vLLM model."""
        if self.model is None:
            self.setup_model()
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the loaded tokenizer."""
        if self.tokenizer is None:
            self.setup_model()
        return self.tokenizer
    
    def prepare_prompts(self, prompts: List[str]) -> List[str]:
        """
        Prepare prompts for generation.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            List of processed prompts
        """
        tokenizer = self.get_tokenizer()
        
        # Add BOS token if configured
        if self.config.tokenizer.add_bos_token:
            processed_prompts = []
            for prompt in prompts:
                if not prompt.startswith(tokenizer.bos_token):
                    prompt = tokenizer.bos_token + prompt
                processed_prompts.append(prompt)
            return processed_prompts
        
        return prompts
    
    def create_sampling_params(self, generation_config: Optional[Dict[str, Any]] = None) -> SamplingParams:
        """
        Create vLLM sampling parameters from generation config.
        
        Args:
            generation_config: Optional override for generation parameters
            
        Returns:
            vLLM SamplingParams object
        """
        config = generation_config or self.generation_config
        
        return SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_length,
            presence_penalty=0.0 if config.repetition_penalty == 1.0 else (config.repetition_penalty - 1.0),
            frequency_penalty=0.0,
            n=config.num_return_sequences,
            stop=None,  # Can be configured if needed
            include_prompt=True,
            use_beam_search=not config.do_sample,
            best_of=config.num_return_sequences if not config.do_sample else None,
        )
    
    def generate(self, prompts: List[str], generation_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate text based on prompts using vLLM.
        
        Args:
            prompts: List of text prompts
            generation_config: Optional override for generation parameters
            
        Returns:
            List of generated texts
        """
        model = self.get_model()
        processed_prompts = self.prepare_prompts(prompts)
        
        # Create sampling parameters
        sampling_params = self.create_sampling_params(generation_config)
        
        # Track generation time for throughput calculation
        start_time = time.time()
        
        # Generate outputs with vLLM
        outputs = model.generate(processed_prompts, sampling_params)
        
        # Calculate generation metrics
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Extract generated texts
        generated_texts = [output.outputs[0].text for output in outputs]
        
        # Calculate and log throughput
        total_tokens = sum(len(self.tokenizer.encode(text)) for text in generated_texts)
        throughput = total_tokens / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {len(generated_texts)} sequences with {total_tokens} tokens in {generation_time:.2f}s")
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        
        return generated_texts
    
    def generate_with_streaming(self, prompts: List[str], generation_config: Optional[Dict[str, Any]] = None, callback=None):
        """
        Generate text with streaming output.
        
        Args:
            prompts: List of text prompts
            generation_config: Optional override for generation parameters
            callback: Optional callback function for streaming outputs
            
        Yields:
            Generated text chunks as they become available
        """
        model = self.get_model()
        processed_prompts = self.prepare_prompts(prompts)
        
        # Create sampling parameters
        sampling_params = self.create_sampling_params(generation_config)
        
        # Generate with streaming
        for request_id, result_generator in model.generate_stream(processed_prompts, sampling_params):
            for response in result_generator:
                if callback:
                    callback(request_id, response)
                yield request_id, response 