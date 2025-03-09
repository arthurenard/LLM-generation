#!/bin/bash
# Script to generate a UV lockfile for reproducible installations

# Ensure the script exits if any command fails
set -e

echo "Generating UV lockfile..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -sSf https://astral.sh/uv/install.sh | bash
    # Source the updated PATH to include UV
    source ~/.bashrc
fi

# Generate the lockfile
echo "Creating lockfile from requirements-uv.txt..."
uv pip compile requirements-uv.txt --output-file requirements-lock.txt

echo "Lockfile generated successfully at requirements-lock.txt"
echo "To install dependencies using the lockfile, run:"
echo "uv pip install -r requirements-lock.txt" 