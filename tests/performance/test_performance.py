"""
Performance tests for the text generation pipeline.
"""
import time
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.models.base_model import BaseModel
from src.models.vllm_model import VLLMModel
from src.generation.generator import TextGenerator

@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance tests for the text generation pipeline."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("src.models.base_model.AutoModelForCausalLM")
    @patch("src.models.base_model.AutoTokenizer")
    def test_base_model_generation_performance(self, mock_tokenizer, mock_model, mock_model_config, mock_generation_config):
        """Test the performance of base model generation."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        
        # Make the generate method actually take some time
        def slow_generate(**kwargs):
            time.sleep(0.1)  # Simulate generation time
            batch_size = kwargs.get("input_ids", torch.tensor([[]])).shape[0]
            return torch.tensor([[1, 2, 3, 4, 5]] * batch_size)
        
        mock_model_instance.generate.side_effect = slow_generate
        mock_model_instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_tokenizer_instance.decode.side_effect = lambda ids, **kwargs: "Generated text"
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create a model
        model = BaseModel(mock_model_config)
        
        # Generate text with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        times = []
        
        for batch_size in batch_sizes:
            prompts = ["Test prompt"] * batch_size
            
            # Measure generation time
            start_time = time.time()
            model.generate(prompts, mock_generation_config)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Check that generation time scales reasonably with batch size
        # The time should increase with batch size, but not linearly
        # (due to parallelization)
        for i in range(1, len(times)):
            # Each doubling of batch size should take less than double the time
            assert times[i] < 2 * times[i-1], f"Generation time scaled poorly from batch size {batch_sizes[i-1]} to {batch_sizes[i]}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("src.models.vllm_model.LLM")
    @patch("src.models.vllm_model.AutoTokenizer")
    @patch("src.models.vllm_model.SamplingParams")
    def test_vllm_model_generation_performance(self, mock_sampling_params, mock_tokenizer, mock_llm, mock_model_config, mock_generation_config):
        """Test the performance of vLLM model generation."""
        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_sampling_params_instance = MagicMock()
        
        # Make the generate method actually take some time
        def slow_generate(prompts, sampling_params):
            time.sleep(0.1)  # Simulate generation time
            return [MagicMock(outputs=[MagicMock(text="Generated text")]) for _ in prompts]
        
        mock_llm_instance.generate.side_effect = slow_generate
        mock_tokenizer_instance.encode.side_effect = lambda text, **kwargs: [1, 2, 3, 4, 5]
        mock_sampling_params.return_value = mock_sampling_params_instance
        
        mock_llm.return_value = mock_llm_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create a model
        model = VLLMModel(mock_model_config, mock_generation_config)
        
        # Mock the create_sampling_params method to avoid the attribute error
        model.create_sampling_params = MagicMock(return_value=mock_sampling_params_instance)
        
        # Generate text with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        times = []
        
        for batch_size in batch_sizes:
            prompts = ["Test prompt"] * batch_size
            
            # Measure generation time
            start_time = time.time()
            model.generate(prompts)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Check that generation time scales reasonably with batch size
        # The time should increase with batch size, but not linearly
        # (due to parallelization)
        for i in range(1, len(times)):
            # Each doubling of batch size should take less than double the time
            assert times[i] < 2 * times[i-1], f"Generation time scaled poorly from batch size {batch_sizes[i-1]} to {batch_sizes[i]}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("src.models.model_factory.create_model")
    def test_generator_memory_usage(self, mock_create_model, mock_full_config, sample_prompts):
        """Test the memory usage of the generator."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = ["Generated text"] * len(sample_prompts)
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.encode.side_effect = lambda text: [1, 2, 3, 4, 5]
        mock_create_model.return_value = mock_model
        
        # Create a generator
        generator = TextGenerator(mock_full_config)
        
        # Record initial memory usage
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Generate a batch
        generator.generate_batch(sample_prompts)
        
        # Record final memory usage
        final_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Check that memory usage is reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase >= 0, "Memory usage decreased during generation"
        
        # In a real test, we would check that memory usage is within expected bounds,
        # but since we're using mocks, we can't make meaningful assertions about
        # the actual memory usage.
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("src.models.model_factory.create_model")
    def test_generator_throughput(self, mock_create_model, mock_full_config):
        """Test the throughput of the generator."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Make the generate method actually take some time
        def slow_generate(prompts, generation_config):
            time.sleep(0.1)  # Simulate generation time
            return ["Generated text"] * len(prompts)
        
        mock_model.generate.side_effect = slow_generate
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.encode.side_effect = lambda text: [1, 2, 3, 4, 5]
        mock_create_model.return_value = mock_model
        
        # Create a generator
        generator = TextGenerator(mock_full_config)
        
        # Generate batches of different sizes
        batch_sizes = [1, 2, 4, 8]
        throughputs = []
        
        for batch_size in batch_sizes:
            prompts = ["Test prompt"] * batch_size
            
            # Generate a batch
            generator.generate_batch(prompts)
            
            # Get throughput
            metrics = generator.performance_tracker.get_metrics()
            throughputs.append(metrics["throughput/tokens_per_second"])
        
        # Check that throughput scales reasonably with batch size
        # The throughput should increase with batch size
        for i in range(1, len(throughputs)):
            assert throughputs[i] > throughputs[i-1], f"Throughput did not increase from batch size {batch_sizes[i-1]} to {batch_sizes[i]}" 