"""
Unit tests for post-processing functionality.
"""
import pytest
from omegaconf import OmegaConf

from src.utils.post_processing import (
    clean_whitespace,
    truncate_text,
    remove_incomplete_sentence,
    deduplicate_sentences,
    apply_post_processing,
    batch_post_process
)

@pytest.mark.unit
class TestPostProcessing:
    """Tests for post-processing functionality."""
    
    def test_clean_whitespace(self):
        """Test cleaning whitespace in text."""
        # Test with multiple spaces
        text = "This   has   multiple    spaces."
        result = clean_whitespace(text)
        assert result == "This has multiple spaces."
        
        # Test with leading and trailing whitespace
        text = "  \t  Leading and trailing whitespace.  \n  "
        result = clean_whitespace(text)
        assert result == "Leading and trailing whitespace."
        
        # Test with newlines and tabs
        text = "This\nhas\nnewlines\tand\ttabs."
        result = clean_whitespace(text)
        assert result == "This has newlines and tabs."
    
    def test_truncate_text(self):
        """Test truncating text to a maximum length."""
        # Test with text shorter than max_length
        text = "This is a short text."
        result = truncate_text(text, 100)
        assert result == text
        
        # Test with text longer than max_length
        text = "This is a longer text that needs to be truncated."
        result = truncate_text(text, 20)
        assert len(result) <= 20
        
        # Test truncation at sentence boundary
        text = "First sentence. Second sentence. Third sentence."
        result = truncate_text(text, 30)
        assert result.endswith(".")
        # The result might only include "First sentence." if the truncation is strict
        assert "First sentence." in result
    
    def test_remove_incomplete_sentence(self):
        """Test removing incomplete sentences."""
        # Test with complete sentences
        text = "This is a complete sentence. This is another complete sentence."
        result = remove_incomplete_sentence(text)
        assert result == text
        
        # Test with incomplete sentence at the end
        text = "This is a complete sentence. This is an incomplete"
        result = remove_incomplete_sentence(text)
        assert result == "This is a complete sentence."
        
        # Test with only an incomplete sentence
        text = "This is an incomplete"
        result = remove_incomplete_sentence(text)
        assert result == ""
    
    def test_deduplicate_sentences(self):
        """Test removing duplicate consecutive sentences."""
        # Test with no duplicates
        text = "First sentence. Second sentence. Third sentence."
        result = deduplicate_sentences(text)
        assert result == text
        
        # Test with duplicates
        text = "First sentence. First sentence. Second sentence. Second sentence."
        result = deduplicate_sentences(text)
        assert result == "First sentence. Second sentence."
        
        # Test with non-consecutive duplicates
        text = "First sentence. Second sentence. First sentence."
        result = deduplicate_sentences(text)
        assert result == text  # Non-consecutive duplicates should be preserved
    
    def test_apply_post_processing(self):
        """Test applying all post-processing steps."""
        # Create a test configuration
        config = OmegaConf.create({
            "post_processing": {
                "max_length": 50,
                "complete_sentences": True,
                "deduplicate": True
            }
        })
        
        # Test with a complex text
        text = "  This   is a test.  This   is a test.  This is an incomplete"
        result = apply_post_processing(text, config)
        
        # Check that all post-processing steps were applied
        assert "  " not in result  # Whitespace cleaned
        assert len(result) <= 50  # Truncated
        assert result.endswith(".")  # Incomplete sentence removed
        assert result.count("This is a test.") == 1  # Duplicates removed
    
    def test_batch_post_process(self):
        """Test post-processing a batch of texts."""
        # Create a test configuration
        config = OmegaConf.create({
            "post_processing": {
                "max_length": 50,
                "complete_sentences": True,
                "deduplicate": True
            }
        })
        
        # Test with a batch of texts
        texts = [
            "  This   is text 1.  ",
            "  This   is text 2. This is an incomplete",
            "  This   is text 3.  This   is text 3.  "
        ]
        
        results = batch_post_process(texts, config)
        
        # Check that all texts were processed correctly
        assert len(results) == len(texts)
        assert results[0] == "This is text 1."
        assert results[1] == "This is text 2."
        assert results[2] == "This is text 3." 