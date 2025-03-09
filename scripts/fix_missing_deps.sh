#!/bin/bash
# Script to fix missing dependencies

# Ensure the script exits if any command fails
set -e

echo "Checking and installing missing dependencies..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: You are not in a virtual environment."
    echo "It's recommended to activate your virtual environment first."
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Check for hydra-core
if ! python3 -c "import hydra" &> /dev/null; then
    echo "Installing hydra-core..."
    pip install hydra-core
fi

# Check for other common dependencies
for package in torch transformers vllm wandb omegaconf; do
    if ! python3 -c "import $package" &> /dev/null; then
        echo "Installing $package..."
        pip install $package
    fi
done

echo "Checking if all dependencies from requirements.txt are installed..."
pip install -r requirements.txt

echo "Dependencies check completed."
echo "You should now be able to run the pipeline without missing dependency errors." 