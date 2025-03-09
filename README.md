# Efficient Large-Scale Synthetic Text Generation Pipeline

A scalable and efficient synthetic text generation system leveraging PyTorch Lightning, Hugging Face Transformers, vLLM, and Weights & Biases (WandB).

## Features

- **Model Selection & Loading**: Support for pre-trained autoregressive models with quantization options (8-bit, 4-bit)
- **High-Performance Inference**: vLLM integration for optimized text generation with PagedAttention
- **PyTorch Lightning Integration**: Clean structure with LightningModules for model management
- **Configurable Generation Pipeline**: Flexible settings for temperature, top-p, sequence length, etc.
- **Logging & Monitoring**: WandB integration for tracking generation metrics and sample outputs
- **Scalability & Deployment**: Multi-GPU support via Torch Distributed and vLLM's tensor parallelism
- **Output Storage & Post-Processing**: Efficient storage formats (JSONL, CSV, Parquet) and text filtering

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a modern, fast Python package manager and resolver. To install dependencies using UV:

```bash
# Clone the repository
git clone https://github.com/yourusername/text-generation-pipeline.git
cd text-generation-pipeline

# Option 1: Use the automated setup script
./scripts/setup.sh

# Option 2: Manual setup
# Install UV if you don't have it already
curl -sSf https://astral.sh/uv/install.sh | bash

# Create a virtual environment and install dependencies
uv venv --python=python3
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements-uv.txt
```

#### Reproducible Installations with Lockfiles

For reproducible installations, you can use the lockfile:

```bash
# Generate a lockfile
./scripts/generate_lockfile.sh

# Install from the lockfile
uv pip install -r requirements-lock.txt
```

#### Programmatic Dependency Management

For automated dependency management or CI/CD pipelines, you can use the Python script:

```bash
# Setup a new environment and install dependencies
./scripts/manage_dependencies.py --action setup

# Generate a lockfile
./scripts/manage_dependencies.py --action lockfile

# Update dependencies and regenerate the lockfile
./scripts/manage_dependencies.py --action update

# Install dependencies from a specific file
./scripts/manage_dependencies.py --action install --requirements custom-requirements.txt
```

### Requirements Files Explained

The project includes several requirements files for different purposes:

- **requirements-uv.txt**: The primary requirements file for UV. Use this when installing with UV.
- **requirements.txt**: Legacy requirements file for pip. Only use this if you're not using UV.
- **requirements-lock.txt**: Generated lockfile for reproducible installations with UV (created by the setup script or generate_lockfile.sh).

### Using Pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/yourusername/text-generation-pipeline.git
cd text-generation-pipeline

# Option 1: Use the automated setup script
./scripts/setup_pip.sh

# Option 2: Manual setup
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Generation

```bash
# Generate text using GPT-2 with default settings
python3 run.py model=gpt2 output.dir=./outputs
```

### Advanced Configuration

```bash
# Generate text with custom parameters
python3 run.py model=mistral-7b generation.temperature=0.8 generation.top_p=0.92 generation.max_length=1024 generation.batch_size=32
```

### Multi-GPU Inference

```bash
# Run distributed generation across multiple GPUs
python3 run_distributed.py model=gpt2 generation.batch_size=64
```

### Using the CLI

```bash
# Use the CLI for easier parameter configuration
python3 generate.py --model gpt2 --temperature 0.8 --top-p 0.92 --max-length 1024 --prompt-file examples/prompts.txt --output-dir ./outputs/gpt2
```

## Project Structure

```
text-generation-pipeline/
├── config/                  # Hydra configuration files
│   ├── model/               # Model-specific configurations
│   ├── generation/          # Text generation parameters
│   ├── logging/             # WandB and other logging settings
│   └── post_processing/     # Post-processing configurations
├── src/                     # Source code
│   ├── models/              # Model implementations
│   │   ├── base_model.py    # Base model using Hugging Face
│   │   ├── vllm_model.py    # vLLM-based model for high-performance inference
│   │   └── model_factory.py # Factory for creating models
│   ├── generation/          # Text generation logic
│   │   └── generator.py     # Main generator class
│   ├── utils/               # Utility functions
│   │   ├── data_utils.py    # Data loading and saving utilities
│   │   ├── metrics.py       # Performance metrics calculation
│   │   └── post_processing.py # Text post-processing utilities
│   └── logging/             # Logging and monitoring
│       └── wandb_logger.py  # WandB integration
├── examples/                # Example scripts and data
│   ├── prompts.txt          # Sample prompts
│   ├── run_examples.sh      # Example usage scripts
│   └── custom_post_processing.py # Custom post-processing example
├── scripts/                 # Utility scripts
│   ├── setup.sh             # Setup script using UV
│   ├── setup_pip.sh         # Setup script using pip
│   ├── generate_lockfile.sh # Script to generate UV lockfile
│   ├── fix_missing_deps.sh  # Script to fix missing dependencies
│   └── manage_dependencies.py # Python script for dependency management
├── run.py                   # Main entry point
├── run_distributed.py       # Distributed inference script
├── generate.py              # CLI for generation
├── pyproject.toml           # Project metadata and dependencies
├── requirements-uv.txt      # UV-specific requirements
└── requirements.txt         # Pip requirements (legacy)
```

## Configuration

The pipeline uses Hydra for configuration management, allowing for flexible and modular settings.

### Model Configuration

Model configurations are stored in `config/model/` and include settings for:

- Model name and path
- Quantization options (8-bit, 4-bit)
- Model loading parameters (dtype, device mapping)
- Tokenizer settings

Example (`config/model/gpt2.yaml`):
```yaml
name: gpt2
pretrained_model_name_or_path: gpt2
model_type: causal_lm
revision: main

# Quantization settings
quantization:
  enabled: false
  bits: 8  # 8, 4
  group_size: 128  # For GPTQ quantization

# Model loading settings
model_loading:
  use_cache: true
  torch_dtype: auto  # auto, float16, bfloat16, float32
  device_map: auto  # auto, balanced, sequential, or specific mapping
  trust_remote_code: false
  
# Tokenizer settings
tokenizer:
  padding_side: right
  truncation_side: right
  use_fast: true
  add_bos_token: false  # Whether to add BOS token automatically
```

### Generation Configuration

Generation configurations are stored in `config/generation/` and include settings for:

- Sampling parameters (temperature, top-p, top-k)
- Sequence length constraints
- Batch processing settings
- vLLM-specific parameters

Example (`config/generation/default.yaml`):
```yaml
# Generation parameters
temperature: 0.8
top_p: 0.92
top_k: 50
repetition_penalty: 1.0
max_length: 1024
min_length: 10
num_return_sequences: 1
do_sample: true
early_stopping: false

# Batch processing
batch_size: 16
max_batch_size: 64  # Maximum batch size for vLLM
chunk_size: 512  # Chunk size for processing large datasets

# Prompt settings
prompt_template: "{prompt}"  # Template for formatting prompts
prompt_file: null  # Path to file containing prompts (one per line)
num_prompts: 1000  # Number of prompts to generate if no prompt file

# vLLM specific settings
vllm:
  enabled: true
  tensor_parallel_size: 1  # Number of GPUs for tensor parallelism
  max_model_len: 8192  # Maximum model sequence length
  gpu_memory_utilization: 0.9  # Target GPU memory utilization
  swap_space: 4  # Swap space in GiB
  enforce_eager: false  # Enforce eager execution (for debugging)
```

### Post-Processing Configuration

Post-processing configurations are stored in `config/post_processing/` and include settings for:

- Text cleaning and formatting
- Length constraints
- Sentence completion
- Deduplication

Example (`config/post_processing/default.yaml`):
```yaml
# Post-processing settings
enabled: true
max_length: 2048  # Maximum length in characters
complete_sentences: true  # Remove incomplete sentences
deduplicate: true  # Remove duplicate consecutive sentences
min_length: 10  # Minimum length in characters to keep
```

### Logging Configuration

Logging configurations are stored in `config/logging/` and include settings for:

- WandB integration
- Metrics tracking
- Sample logging

Example (`config/logging/wandb.yaml`):
```yaml
enabled: true
project: text-generation-pipeline
entity: null  # Your wandb username or team name
name: ${model.name}_${now:%Y-%m-%d_%H-%M-%S}
tags: [${model.name}]
notes: "Text generation with ${model.name}"

# Logging settings
log_model: false
log_level: info
log_interval: 100  # Log metrics every N samples
log_samples: 10  # Number of generated samples to log
log_gradients: false

# Metrics to track
metrics:
  throughput: true  # Tokens per second
  latency: true  # Generation latency
  token_diversity: true  # Measure diversity of generated tokens
  memory_usage: true  # Track GPU memory usage
```

## Advanced Usage

### Custom Post-Processing

You can customize post-processing by modifying the configuration or using the provided utilities:

```bash
# Run with custom post-processing settings
python examples/custom_post_processing.py --model gpt2 --complete-sentences --deduplicate --max-chars 1500
```

### Distributed Generation

For large-scale generation across multiple GPUs:

```bash
# Run with distributed processing
python run_distributed.py model=gpt2 generation.batch_size=32
```

### Streaming Generation

The pipeline supports streaming generation for real-time output:

```python
from src.generation.generator import TextGenerator
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config/config.yaml")

# Initialize generator
generator = TextGenerator(config)

# Generate with streaming
prompts = ["Write a short story about a robot."]
for request_id, response in generator.generate_streaming(prompts):
    print(f"Request {request_id}: {response.outputs[0].text}")
```

## Performance Optimization

### vLLM Integration

The pipeline leverages vLLM for high-performance inference:

- **PagedAttention**: Efficient memory management for handling long sequences
- **Tensor Parallelism**: Distribute model across multiple GPUs
- **Continuous Batching**: Process requests as they arrive for maximum throughput

Enable vLLM with:

```bash
python3 run.py model=gpt2 generation.vllm.enabled=true
```

### Quantization

Reduce memory usage with model quantization:

```bash
# Use 8-bit quantization
python3 run.py model=gpt2 model.quantization.enabled=true model.quantization.bits=8

# Use 4-bit quantization
python3 run.py model=mistral-7b model.quantization.enabled=true model.quantization.bits=4
```

## Monitoring and Metrics

The pipeline integrates with Weights & Biases for comprehensive monitoring:

- **Throughput**: Tokens per second and samples per second
- **Memory Usage**: GPU memory allocation and utilization
- **Token Diversity**: Measure diversity of generated text
- **Sample Outputs**: Log and visualize generated samples

Enable WandB logging with:

```bash
python3 run.py model=gpt2 logging.enabled=true logging.project=my-generation-project
```

## Examples

The `examples/` directory contains sample scripts and data:

- `prompts.txt`: Sample prompts for testing
- `run_examples.sh`: Shell script with example commands
- `custom_post_processing.py`: Example of custom post-processing

Run the examples with:

```bash
# Run all examples
./examples/run_examples.sh

# Run specific example
python3 examples/custom_post_processing.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT 

### Troubleshooting

If you encounter missing dependency errors like `ModuleNotFoundError: No module named 'hydra'`, you can use the provided fix script:

```bash
# Activate your virtual environment first
source .venv/bin/activate  # For UV
# OR
source venv/bin/activate   # For pip

# Run the fix script
./scripts/fix_missing_deps.sh
```

This script will check for and install any missing dependencies. 