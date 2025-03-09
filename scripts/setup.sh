#!/bin/bash
# Setup script for the text generation pipeline using UV

# Ensure the script exits if any command fails
set -e

echo "Setting up the text generation pipeline..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -sSf https://astral.sh/uv/install.sh | bash
    # Source the updated PATH to include UV
    source ~/.bashrc
fi

# Create a virtual environment
echo "Creating a virtual environment..."
uv venv --python=python3 .venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate

# Check if lockfile exists
if [ -f "requirements-lock.txt" ]; then
    echo "Installing dependencies from lockfile..."
    uv pip install -r requirements-lock.txt
else
    echo "Lockfile not found. Installing dependencies from requirements-uv.txt..."
    uv pip install -r requirements-uv.txt
    
    # Generate lockfile for future use
    echo "Generating lockfile for future use..."
    uv pip compile requirements-uv.txt --output-file requirements-lock.txt
fi

echo "Setup completed successfully!"
echo "To activate the virtual environment, run:"
echo "source .venv/bin/activate" 