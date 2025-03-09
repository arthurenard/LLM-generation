from typing import List, Dict, Any, Optional, Union
import os
import json
import random
import logging
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_prompts(config: DictConfig) -> List[str]:
    """
    Load prompts from file or generate random prompts.
    
    Args:
        config: Configuration from Hydra
        
    Returns:
        List of prompts
    """
    # Check if prompt file is specified
    if config.generation.prompt_file:
        prompt_file = Path(config.generation.prompt_file)
        logger.info(f"Loading prompts from {prompt_file}")
        
        # Check file extension
        if prompt_file.suffix == ".txt":
            # Load text file (one prompt per line)
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        elif prompt_file.suffix == ".json" or prompt_file.suffix == ".jsonl":
            # Load JSON file
            prompts = []
            with open(prompt_file, "r", encoding="utf-8") as f:
                if prompt_file.suffix == ".json":
                    # Single JSON object or array
                    data = json.load(f)
                    if isinstance(data, list):
                        prompts = [item["prompt"] if isinstance(item, dict) else item for item in data]
                    elif isinstance(data, dict) and "prompts" in data:
                        prompts = data["prompts"]
                    else:
                        prompts = [data["prompt"]]
                else:
                    # JSONL file (one JSON object per line)
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            prompts.append(item["prompt"] if isinstance(item, dict) else item)
        elif prompt_file.suffix == ".csv" or prompt_file.suffix == ".parquet":
            # Load CSV or Parquet file
            if prompt_file.suffix == ".csv":
                df = pd.read_csv(prompt_file)
            else:
                df = pd.read_parquet(prompt_file)
            
            # Get prompt column
            prompt_column = "prompt"
            if prompt_column not in df.columns:
                # Try to find a suitable column
                potential_columns = ["prompt", "text", "input", "query"]
                for col in potential_columns:
                    if col in df.columns:
                        prompt_column = col
                        break
                else:
                    # Use the first column
                    prompt_column = df.columns[0]
            
            prompts = df[prompt_column].tolist()
        else:
            raise ValueError(f"Unsupported prompt file format: {prompt_file.suffix}")
        
        logger.info(f"Loaded {len(prompts)} prompts from {prompt_file}")
    else:
        # Generate random prompts
        logger.info(f"Generating {config.generation.num_prompts} random prompts")
        
        # Simple random prompts (in a real system, you'd have better prompt generation)
        prompts = [
            f"Write a short story about {random.choice(['space travel', 'time travel', 'dragons', 'robots', 'magic'])}"
            for _ in range(config.generation.num_prompts)
        ]
        
        logger.info(f"Generated {len(prompts)} random prompts")
    
    # Apply prompt template if specified
    if config.generation.prompt_template != "{prompt}":
        prompts = [config.generation.prompt_template.format(prompt=prompt) for prompt in prompts]
    
    return prompts

def save_batch(batch_data: List[Dict[str, Any]], output_file: Union[str, Path], format: str = "jsonl") -> None:
    """
    Save a batch of generated text.
    
    Args:
        batch_data: List of dictionaries containing generated text and metadata
        output_file: Path to output file
        format: Output format (jsonl, csv, parquet)
    """
    output_file = Path(output_file)
    
    # Create parent directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on format
    if format.lower() == "jsonl":
        # Append to JSONL file
        with open(output_file, "a", encoding="utf-8") as f:
            for item in batch_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif format.lower() == "csv":
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(batch_data)
        
        # If file exists, append without header
        if output_file.exists():
            df.to_csv(output_file, mode="a", header=False, index=False)
        else:
            df.to_csv(output_file, index=False)
    elif format.lower() == "parquet":
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # If file exists, read existing file and append
        if output_file.exists():
            existing_df = pd.read_parquet(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save as Parquet
        df.to_parquet(output_file, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}")

def post_process(text: str, config: Optional[DictConfig] = None) -> str:
    """
    Apply post-processing to generated text.
    
    Args:
        text: Generated text
        config: Optional configuration
        
    Returns:
        Post-processed text
    """
    # Simple post-processing (can be expanded based on requirements)
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Truncate to max length if specified
    if config and hasattr(config, "post_processing") and hasattr(config.post_processing, "max_length"):
        max_length = config.post_processing.max_length
        if len(text) > max_length:
            text = text[:max_length]
    
    return text 