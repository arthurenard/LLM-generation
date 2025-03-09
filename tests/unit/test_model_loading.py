"""
Unit tests for model loading functionality.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.models.base_model import BaseModel
from src.models.vllm_model import VLLMModel
from src.models.model_factory import create_model

@pytest.mark.unit
class TestModelLoading:
    """Tests for model loading functionality."""
    
    @patch("src.models.base_model.AutoModelForCausalLM")
    @patch("src.models.base_model.AutoTokenizer")
    def test_base_model_initialization(self, mock_tokenizer, mock_model, mock_model_config):
        """Test initializing a base model."""
        # Create a model
        model = BaseModel(mock_model_config)
        
        # Check that the model was initialized correctly
        assert model.config == mock_model_config
        assert model.model is None
        assert model.tokenizer is None
    
    @patch("src.models.base_model.AutoModelForCausalLM")
    @patch("src.models.base_model.AutoTokenizer")
    def test_base_model_setup(self, mock_tokenizer, mock_model, mock_model_config):
        """Test setting up a base model."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create and setup a model
        model = BaseModel(mock_model_config)
        model.setup_model()
        
        # Check that the model and tokenizer were loaded correctly
        assert model.model == mock_model_instance
        assert model.tokenizer == mock_tokenizer_instance
        
        # Check that from_pretrained was called with the correct arguments
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
    
    @patch("src.models.base_model.AutoModelForCausalLM")
    @patch("src.models.base_model.AutoTokenizer")
    def test_base_model_get_model(self, mock_tokenizer, mock_model, mock_model_config):
        """Test getting a model."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Create a model
        model = BaseModel(mock_model_config)
        
        # Get the model
        result = model.get_model()
        
        # Check that the model was loaded and returned correctly
        assert result == mock_model_instance
        assert model.model == mock_model_instance
        
        # Check that from_pretrained was called with the correct arguments
        mock_model.from_pretrained.assert_called_once()
    
    @patch("src.models.base_model.AutoModelForCausalLM")
    @patch("src.models.base_model.AutoTokenizer")
    def test_base_model_get_tokenizer(self, mock_tokenizer, mock_model, mock_model_config):
        """Test getting a tokenizer."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create a model
        model = BaseModel(mock_model_config)
        
        # Get the tokenizer
        result = model.get_tokenizer()
        
        # Check that the tokenizer was loaded and returned correctly
        assert result == mock_tokenizer_instance
        assert model.tokenizer == mock_tokenizer_instance
        
        # Check that from_pretrained was called with the correct arguments
        mock_tokenizer.from_pretrained.assert_called_once()
    
    @patch("src.models.base_model.AutoModelForCausalLM")
    @patch("src.models.base_model.AutoTokenizer")
    def test_base_model_prepare_inputs(self, mock_tokenizer, mock_model, mock_model_config):
        """Test preparing inputs for a model."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create a model
        model = BaseModel(mock_model_config)
        
        # Prepare inputs
        prompts = ["Test prompt"]
        inputs = model.prepare_inputs(prompts)
        
        # Check that the inputs were prepared correctly
        assert isinstance(inputs, dict)
        mock_tokenizer_instance.assert_called_once()
    
    @patch("src.models.vllm_model.LLM")
    @patch("src.models.vllm_model.AutoTokenizer")
    def test_vllm_model_initialization(self, mock_tokenizer, mock_llm, mock_model_config, mock_generation_config):
        """Test initializing a vLLM model."""
        # Create a model
        model = VLLMModel(mock_model_config, mock_generation_config)
        
        # Check that the model was initialized correctly
        assert model.config == mock_model_config
        assert model.generation_config == mock_generation_config
        assert model.model is None
        assert model.tokenizer is None
    
    @patch("src.models.vllm_model.LLM")
    @patch("src.models.vllm_model.AutoTokenizer")
    def test_vllm_model_setup(self, mock_tokenizer, mock_llm, mock_model_config, mock_generation_config):
        """Test setting up a vLLM model."""
        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create and setup a model
        model = VLLMModel(mock_model_config, mock_generation_config)
        model.setup_model()
        
        # Check that the model and tokenizer were loaded correctly
        assert model.model == mock_llm_instance
        assert model.tokenizer == mock_tokenizer_instance
        
        # Check that the model and tokenizer were loaded with the correct arguments
        mock_llm.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
    
    @patch("src.models.model_factory.VLLMModel")
    @patch("src.models.model_factory.BaseModel")
    def test_create_model_base(self, mock_base_model, mock_vllm_model, mock_model_config, mock_generation_config):
        """Test creating a base model."""
        # Disable vLLM
        mock_generation_config.vllm.enabled = False
        
        # Create a model
        create_model(mock_model_config, mock_generation_config)
        
        # Check that the correct model was created
        mock_base_model.assert_called_once_with(mock_model_config)
        mock_vllm_model.assert_not_called()
    
    @patch("src.models.model_factory.VLLMModel")
    @patch("src.models.model_factory.BaseModel")
    def test_create_model_vllm(self, mock_base_model, mock_vllm_model, mock_model_config, mock_generation_config):
        """Test creating a vLLM model."""
        # Enable vLLM
        mock_generation_config.vllm.enabled = True
        
        # Create a model
        create_model(mock_model_config, mock_generation_config)
        
        # Check that the correct model was created
        mock_vllm_model.assert_called_once_with(mock_model_config, mock_generation_config)
        mock_base_model.assert_not_called() 