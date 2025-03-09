"""
Integration tests for the text generator.
"""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.generation.generator import TextGenerator
from src.generator import Generator

@pytest.mark.integration
class TestGenerator:
    """Integration tests for the text generator."""
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_model_config, mock_generation_config):
        """Set up test fixtures."""
        self.model_config = mock_model_config
        self.generation_config = mock_generation_config
        
        # Create mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3], [4, 5, 6]]))
        self.mock_model.can_generate = MagicMock(return_value=True)
        
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        self.mock_tokenizer.decode = MagicMock(side_effect=lambda ids, **kwargs: f"Test output {ids.item() if isinstance(ids, torch.Tensor) and ids.dim() == 0 else ids[0].item()}")
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 2
        self.mock_tokenizer.__call__ = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])})
    
    @patch("src.models.model_factory.create_model")
    def test_generator_initialization(self, mock_create_model, mock_full_config):
        """Test initializing a text generator."""
        # Create a generator
        generator = TextGenerator(mock_full_config)
        
        # Check that the generator was initialized correctly
        assert generator.config == mock_full_config
        assert generator.model is None
        
        # Check that the output directory was created
        assert Path(mock_full_config.output.dir).exists()
    
    @patch('src.generator.AutoModelForCausalLM.from_pretrained')
    @patch('src.generator.AutoTokenizer.from_pretrained')
    def test_generator_setup_model(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        """Test setting up the model."""
        mock_model_from_pretrained.return_value = self.mock_model
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        generator = Generator(self.model_config, self.generation_config)
        generator.setup_model()
        
        mock_model_from_pretrained.assert_called_once()
        mock_tokenizer_from_pretrained.assert_called_once()
        assert generator.model is not None
        assert generator.tokenizer is not None
    
    @patch('src.generator.AutoModelForCausalLM.from_pretrained')
    @patch('src.generator.AutoTokenizer.from_pretrained')
    def test_generator_get_model(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        """Test getting the model."""
        mock_model_from_pretrained.return_value = self.mock_model
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        generator = Generator(self.model_config, self.generation_config)
        model = generator.get_model()
        
        mock_model_from_pretrained.assert_called_once()
        mock_tokenizer_from_pretrained.assert_called_once()
        assert model is not None
    
    @patch('src.generator.AutoModelForCausalLM.from_pretrained')
    @patch('src.generator.AutoTokenizer.from_pretrained')
    def test_generator_generate_batch(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        """Test generating text for a batch of prompts."""
        mock_model_from_pretrained.return_value = self.mock_model
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        generator = Generator(self.model_config, self.generation_config)
        prompts = ["Test prompt 1", "Test prompt 2"]
        outputs = generator.generate_batch(prompts)
        
        assert len(outputs) == len(prompts)
        assert all(isinstance(output, str) for output in outputs)
        assert outputs[0] != outputs[1]  # Ensure different outputs for different prompts
    
    @patch('src.generator.AutoModelForCausalLM.from_pretrained')
    @patch('src.generator.AutoTokenizer.from_pretrained')
    def test_generator_generate(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        """Test generating text for a single prompt."""
        mock_model_from_pretrained.return_value = self.mock_model
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        generator = Generator(self.model_config, self.generation_config)
        output = generator.generate("Test prompt")
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    @patch('src.generator.AutoModelForCausalLM.from_pretrained')
    @patch('src.generator.AutoTokenizer.from_pretrained')
    def test_generator_generate_streaming(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        """Test generating text with streaming."""
        mock_model_from_pretrained.return_value = self.mock_model
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        generator = Generator(self.model_config, self.generation_config)
        tokens = []
        for token in generator.generate_streaming("Test prompt"):
            tokens.append(token)
            
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    @patch('src.generator.AutoModelForCausalLM.from_pretrained')
    @patch('src.generator.AutoTokenizer.from_pretrained')
    def test_generator_generate_streaming_fallback(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        """Test generating text with streaming fallback."""
        mock_model_from_pretrained.return_value = self.mock_model
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        # Simulate streaming not supported
        self.mock_model.can_generate.return_value = False
        
        generator = Generator(self.model_config, self.generation_config)
        tokens = []
        for token in generator.generate_streaming("Test prompt"):
            tokens.append(token)
            
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens) 