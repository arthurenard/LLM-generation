import torch
from typing import List, Dict, Any, Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from omegaconf import OmegaConf

class Generator:
    """Text generation class using Hugging Face models."""
    
    def __init__(self, model_config: Dict[str, Any], generation_config: Dict[str, Any]):
        """
        Initialize the generator.
        
        Args:
            model_config: Model configuration
            generation_config: Generation configuration
        """
        self.model_config = model_config
        self.generation_config = generation_config
        self.model = None
        self.tokenizer = None
        
    def setup_model(self) -> None:
        """Set up the model and tokenizer."""
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config["pretrained_model_name_or_path"],
                **self.model_config["model_loading"]
            )
            
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["pretrained_model_name_or_path"],
                **self.model_config["tokenizer"]
            )
            
    def get_model(self):
        """Get the model, setting it up if necessary."""
        if self.model is None:
            self.setup_model()
        return self.model
        
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated texts
        """
        if self.model is None:
            self.setup_model()
            
        # Tokenize inputs
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            **self.generation_config
        )
        
        # Decode outputs
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
    def generate(self, prompt: str) -> str:
        """
        Generate text for a single prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        return self.generate_batch([prompt])[0]
        
    def generate_streaming(self, prompt: str) -> Iterator[str]:
        """
        Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            
        Yields:
            Generated tokens one by one
        """
        if self.model is None:
            self.setup_model()
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Check if streaming is supported
        if hasattr(self.model, "can_generate") and self.model.can_generate():
            # Use streaming generation
            for output in self.model.generate(
                **inputs,
                **self.generation_config,
                streamer=True
            ):
                yield self.tokenizer.decode(output, skip_special_tokens=True)
        else:
            # Fallback to regular generation
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
            # Decode and yield tokens one by one
            for output in outputs[0]:
                yield self.tokenizer.decode(output, skip_special_tokens=True) 