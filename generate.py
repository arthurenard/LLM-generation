#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text Generation CLI")
    
    # Model selection
    parser.add_argument("--model", type=str, default="gpt2", help="Model to use (e.g., gpt2, mistral-7b)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.92, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation")
    
    # Input/output settings
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to file containing prompts")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to generate if no prompt file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--output-format", type=str, default="jsonl", choices=["jsonl", "csv", "parquet"], help="Output format")
    
    # Runtime settings
    parser.add_argument("--distributed", action="store_true", help="Enable distributed generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu, or specific device)")
    
    # vLLM settings
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for generation")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM")
    
    # Logging settings
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="text-generation-pipeline", help="WandB project name")
    parser.add_argument("--log-samples", type=int, default=10, help="Number of samples to log")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Build Hydra command
    cmd = ["python3"]
    
    # Choose script based on distributed flag
    if args.distributed:
        cmd.append("run_distributed.py")
    else:
        cmd.append("run.py")
    
    # Add model configuration
    cmd.append(f"model={args.model}")
    
    # Add generation parameters
    cmd.append(f"generation.temperature={args.temperature}")
    cmd.append(f"generation.top_p={args.top_p}")
    cmd.append(f"generation.max_length={args.max_length}")
    cmd.append(f"generation.batch_size={args.batch_size}")
    
    # Add input/output settings
    if args.prompt_file:
        cmd.append(f"generation.prompt_file={args.prompt_file}")
    cmd.append(f"generation.num_prompts={args.num_prompts}")
    cmd.append(f"output.dir={args.output_dir}")
    cmd.append(f"output.format={args.output_format}")
    
    # Add runtime settings
    cmd.append(f"runtime.seed={args.seed}")
    cmd.append(f"runtime.device={args.device}")
    
    # Add vLLM settings
    cmd.append(f"generation.vllm.enabled={str(args.use_vllm).lower()}")
    if args.use_vllm:
        cmd.append(f"generation.vllm.tensor_parallel_size={args.tensor_parallel_size}")
    
    # Add logging settings
    cmd.append(f"logging.enabled={str(args.wandb).lower()}")
    if args.wandb:
        cmd.append(f"logging.project={args.wandb_project}")
        cmd.append(f"logging.log_samples={args.log_samples}")
    
    # Print command
    print("Running command:", " ".join(cmd))
    
    # Run command
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 