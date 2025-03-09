"""
Unit tests for metrics calculation.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.utils.metrics import (
    calculate_token_diversity,
    calculate_throughput,
    calculate_memory_usage,
    PerformanceTracker
)

@pytest.mark.unit
class TestMetrics:
    """Tests for metrics calculation."""
    
    def test_calculate_token_diversity(self):
        """Test calculating token diversity metrics."""
        # Create a mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text: [1, 2, 3, 4, 5] if text == "Text 1" else [3, 4, 5, 6, 7]
        
        # Test with a list of texts
        texts = ["Text 1", "Text 2"]
        metrics = calculate_token_diversity(texts, tokenizer)
        
        # Check that the metrics were calculated correctly
        assert "unique_tokens" in metrics
        assert "total_tokens" in metrics
        assert "unique_ratio" in metrics
        assert "entropy" in metrics
        
        assert metrics["unique_tokens"] == 7  # Unique tokens: 1, 2, 3, 4, 5, 6, 7
        assert metrics["total_tokens"] == 10  # Total tokens: 5 + 5
        assert metrics["unique_ratio"] == 0.7  # 7 / 10
        assert metrics["entropy"] > 0  # Entropy should be positive
    
    def test_calculate_throughput(self):
        """Test calculating throughput metrics."""
        # Test with sample values
        total_tokens = 1000
        total_time = 10.0
        total_samples = 100
        
        metrics = calculate_throughput(total_tokens, total_time, total_samples)
        
        # Check that the metrics were calculated correctly
        assert "tokens_per_second" in metrics
        assert "samples_per_second" in metrics
        assert "tokens_per_sample" in metrics
        assert "seconds_per_sample" in metrics
        
        assert metrics["tokens_per_second"] == 100.0  # 1000 / 10
        assert metrics["samples_per_second"] == 10.0  # 100 / 10
        assert metrics["tokens_per_sample"] == 10.0  # 1000 / 100
        assert metrics["seconds_per_sample"] == 0.1  # 10 / 100
    
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    def test_calculate_memory_usage(self, mock_reserved, mock_allocated, mock_device_count, mock_is_available):
        """Test calculating memory usage metrics."""
        # Setup mocks
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        mock_allocated.side_effect = lambda device: 1 * (1024 ** 3) if device == 0 else 2 * (1024 ** 3)
        mock_reserved.side_effect = lambda device: 2 * (1024 ** 3) if device == 0 else 4 * (1024 ** 3)
        
        # Calculate memory usage
        metrics = calculate_memory_usage()
        
        # Check that the metrics were calculated correctly
        assert "gpu0_allocated_gb" in metrics
        assert "gpu0_reserved_gb" in metrics
        assert "gpu0_utilization" in metrics
        assert "gpu1_allocated_gb" in metrics
        assert "gpu1_reserved_gb" in metrics
        assert "gpu1_utilization" in metrics
        
        assert metrics["gpu0_allocated_gb"] == 1.0
        assert metrics["gpu0_reserved_gb"] == 2.0
        assert metrics["gpu0_utilization"] == 0.5
        assert metrics["gpu1_allocated_gb"] == 2.0
        assert metrics["gpu1_reserved_gb"] == 4.0
        assert metrics["gpu1_utilization"] == 0.5
    
    @patch("torch.cuda.is_available")
    def test_calculate_memory_usage_no_cuda(self, mock_is_available):
        """Test calculating memory usage metrics when CUDA is not available."""
        # Setup mocks
        mock_is_available.return_value = False
        
        # Calculate memory usage
        metrics = calculate_memory_usage()
        
        # Check that the metrics are empty
        assert metrics == {}
    
    def test_performance_tracker_initialization(self):
        """Test initializing a performance tracker."""
        tracker = PerformanceTracker()
        
        # Check that the tracker was initialized correctly
        assert tracker.start_time is None
        assert tracker.end_time is None
        assert tracker.total_tokens == 0
        assert tracker.total_samples == 0
        assert tracker.batch_times == []
    
    def test_performance_tracker_batch_timing(self):
        """Test timing a batch with the performance tracker."""
        tracker = PerformanceTracker()
        
        # Start timing
        tracker.start_batch()
        assert tracker.start_time is not None
        
        # End timing
        tracker.end_batch(100, 10)
        assert tracker.end_time is not None
        assert tracker.total_tokens == 100
        assert tracker.total_samples == 10
        assert len(tracker.batch_times) == 1
    
    @patch("src.utils.metrics.calculate_throughput")
    @patch("src.utils.metrics.calculate_memory_usage")
    def test_performance_tracker_get_metrics(self, mock_memory, mock_throughput):
        """Test getting metrics from the performance tracker."""
        # Setup mocks
        mock_throughput.return_value = {
            "tokens_per_second": 100.0,
            "samples_per_second": 10.0,
            "tokens_per_sample": 10.0,
            "seconds_per_sample": 0.1
        }
        mock_memory.return_value = {
            "gpu0_allocated_gb": 1.0,
            "gpu0_reserved_gb": 2.0,
            "gpu0_utilization": 0.5
        }
        
        # Create a tracker and add some data
        tracker = PerformanceTracker()
        tracker.batch_times = [(1.0, 100, 10), (2.0, 200, 20)]
        tracker.total_tokens = 300
        tracker.total_samples = 30
        
        # Get metrics
        metrics = tracker.get_metrics()
        
        # Check that the metrics were calculated correctly
        assert "total_tokens" in metrics
        assert "total_samples" in metrics
        assert "total_time" in metrics
        assert "throughput/tokens_per_second" in metrics
        assert "throughput/samples_per_second" in metrics
        assert "memory/gpu0_allocated_gb" in metrics
        
        assert metrics["total_tokens"] == 300
        assert metrics["total_samples"] == 30
        assert metrics["total_time"] == 3.0
        assert metrics["throughput/tokens_per_second"] == 100.0
        assert metrics["throughput/samples_per_second"] == 10.0
        assert metrics["memory/gpu0_allocated_gb"] == 1.0 