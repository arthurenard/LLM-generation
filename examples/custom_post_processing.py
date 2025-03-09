#!/usr/bin/env python3
"""
Example script demonstrating how to use custom post-processing with the text generation pipeline.
"""
import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text Generation with Custom Post-Processing")
    
    # Model selection
    parser.add_argument("--model", type=str, default="gpt2", help="Model to use (e.g., gpt2, mistral-7b)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.92, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    
    # Post-processing parameters
    parser.add_argument("--complete-sentences", action="store_true", help="Remove incomplete sentences")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicate sentences")
    parser.add_argument("--max-chars", type=int, default=2048, help="Maximum characters in output")
    
    # Input/output settings
    parser.add_argument("--prompt-file", type=str, default="examples/prompts.txt", help="Path to file containing prompts")
    parser.add_argument("--output-dir", type=str, default="./outputs/custom_post_processing", help="Output directory")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Build Hydra command
    cmd = ["python3", "run.py"]
    
    # Add model configuration
    cmd.append(f"model={args.model}")
    
    # Add generation parameters
    cmd.append(f"generation.temperature={args.temperature}")
    cmd.append(f"generation.top_p={args.top_p}")
    cmd.append(f"generation.max_length={args.max_length}")
    
    # Add post-processing parameters
    cmd.append(f"post_processing.enabled=true")
    cmd.append(f"post_processing.complete_sentences={str(args.complete_sentences).lower()}")
    cmd.append(f"post_processing.deduplicate={str(args.deduplicate).lower()}")
    cmd.append(f"post_processing.max_length={args.max_chars}")
    
    # Add input/output settings
    cmd.append(f"generation.prompt_file={args.prompt_file}")
    cmd.append(f"output.dir={args.output_dir}")
    
    # Print command
    print("Running command:", " ".join(cmd))
    
    # Run command
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 