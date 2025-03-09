#!/bin/bash
# Example commands for running the text generation pipeline

# Basic example with GPT-2
echo "Running basic example with GPT-2..."
python3 generate.py --model gpt2 --prompt-file examples/prompts.txt --output-dir ./outputs/gpt2 --use-vllm

# Example with higher temperature for more creative outputs
echo "Running example with higher temperature..."
python3 generate.py --model gpt2 --temperature 1.2 --prompt-file examples/prompts.txt --output-dir ./outputs/gpt2_high_temp --use-vllm

# Example with Mistral 7B (if available)
echo "Running example with Mistral 7B..."
python3 generate.py --model mistral-7b --prompt-file examples/prompts.txt --output-dir ./outputs/mistral --use-vllm

# Example with distributed generation (multi-GPU)
echo "Running distributed example..."
python3 generate.py --model gpt2 --prompt-file examples/prompts.txt --output-dir ./outputs/gpt2_distributed --distributed --use-vllm

# Example with WandB logging
echo "Running example with WandB logging..."
python3 generate.py --model gpt2 --prompt-file examples/prompts.txt --output-dir ./outputs/gpt2_wandb --wandb --wandb-project text-generation-demo --use-vllm

# Example with CSV output format
echo "Running example with CSV output..."
python3 generate.py --model gpt2 --prompt-file examples/prompts.txt --output-dir ./outputs/gpt2_csv --output-format csv --use-vllm

echo "All examples completed!" 