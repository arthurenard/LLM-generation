from typing import List, Dict, Any, Optional, Union
import re
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def clean_whitespace(text: str) -> str:
    """
    Clean whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length in characters
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    sentences = re.split(r'(?<=[.!?])\s+', text[:max_length + 100])
    
    result = ""
    for sentence in sentences:
        if len(result) + len(sentence) + 1 <= max_length:  # +1 for space
            if result:
                result += " " + sentence
            else:
                result = sentence
        else:
            break
    
    return result.strip()

def remove_incomplete_sentence(text: str) -> str:
    """
    Remove incomplete sentence at the end of text.
    
    Args:
        text: Input text
        
    Returns:
        Text with complete sentences only
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Check if the last sentence ends with a period, exclamation mark, or question mark
    if sentences and not re.search(r'[.!?]$', sentences[-1]):
        # Remove the last incomplete sentence
        return " ".join(sentences[:-1])
    
    return text

def deduplicate_sentences(text: str) -> str:
    """
    Remove duplicate consecutive sentences.
    
    Args:
        text: Input text
        
    Returns:
        Text with duplicates removed
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove duplicates while preserving order
    unique_sentences = []
    for sentence in sentences:
        if not unique_sentences or sentence != unique_sentences[-1]:
            unique_sentences.append(sentence)
    
    return " ".join(unique_sentences)

def apply_post_processing(text: str, config: Optional[DictConfig] = None) -> str:
    """
    Apply all post-processing steps to text.
    
    Args:
        text: Input text
        config: Optional configuration
        
    Returns:
        Processed text
    """
    # Clean whitespace
    text = clean_whitespace(text)
    
    # Apply post-processing based on configuration
    if config and hasattr(config, "post_processing"):
        # Truncate text if max_length is specified
        if hasattr(config.post_processing, "max_length"):
            text = truncate_text(text, config.post_processing.max_length)
        
        # Remove incomplete sentences if specified
        if hasattr(config.post_processing, "complete_sentences") and config.post_processing.complete_sentences:
            text = remove_incomplete_sentence(text)
        
        # Remove duplicates if specified
        if hasattr(config.post_processing, "deduplicate") and config.post_processing.deduplicate:
            text = deduplicate_sentences(text)
    
    return text

def batch_post_process(texts: List[str], config: Optional[DictConfig] = None) -> List[str]:
    """
    Apply post-processing to a batch of texts.
    
    Args:
        texts: List of input texts
        config: Optional configuration
        
    Returns:
        List of processed texts
    """
    return [apply_post_processing(text, config) for text in texts] 