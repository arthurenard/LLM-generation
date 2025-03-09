#!/bin/bash
# Setup script for the text generation pipeline using pip

# Ensure the script exits if any command fails
set -e

echo "Setting up the text generation pipeline using pip..."

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup completed successfully!"
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate" 