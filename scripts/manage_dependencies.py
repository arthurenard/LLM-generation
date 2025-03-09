#!/usr/bin/env python3
"""
Script to demonstrate how to use UV for dependency management programmatically.
This can be useful for automated dependency updates or CI/CD pipelines.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_uv_installed():
    """Check if UV is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv():
    """Install UV."""
    print("Installing UV...")
    subprocess.run(
        ["curl", "-sSf", "https://astral.sh/uv/install.sh"],
        stdout=subprocess.PIPE,
        check=True,
    )
    subprocess.run(["bash", "-c", "$(curl -sSf https://astral.sh/uv/install.sh)"], check=True)
    print("UV installed successfully.")


def create_venv(venv_path=".venv"):
    """Create a virtual environment using UV."""
    print(f"Creating virtual environment at {venv_path}...")
    subprocess.run(["uv", "venv", "--python=python3", venv_path], check=True)
    print(f"Virtual environment created at {venv_path}")


def install_dependencies(requirements_file="requirements-uv.txt", venv_path=".venv"):
    """Install dependencies using UV."""
    print(f"Installing dependencies from {requirements_file}...")
    
    # Determine the Python executable path in the virtual environment
    if sys.platform == "win32":
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Install dependencies
    subprocess.run(
        ["uv", "pip", "install", "-r", requirements_file],
        env={"PATH": os.environ["PATH"], "VIRTUAL_ENV": os.path.abspath(venv_path)},
        check=True,
    )
    print("Dependencies installed successfully.")


def generate_lockfile(requirements_file="requirements-uv.txt", lockfile="requirements-lock.txt"):
    """Generate a lockfile using UV."""
    print(f"Generating lockfile from {requirements_file}...")
    subprocess.run(
        ["uv", "pip", "compile", requirements_file, "--output-file", lockfile],
        check=True,
    )
    print(f"Lockfile generated at {lockfile}")


def update_dependencies(requirements_file="requirements-uv.txt", lockfile="requirements-lock.txt"):
    """Update dependencies and regenerate the lockfile."""
    print("Updating dependencies...")
    subprocess.run(
        ["uv", "pip", "compile", requirements_file, "--output-file", lockfile, "--upgrade"],
        check=True,
    )
    print("Dependencies updated and lockfile regenerated.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Manage dependencies using UV")
    parser.add_argument(
        "--action",
        choices=["setup", "install", "update", "lockfile"],
        default="setup",
        help="Action to perform",
    )
    parser.add_argument(
        "--venv", default=".venv", help="Path to virtual environment"
    )
    parser.add_argument(
        "--requirements",
        default="requirements-uv.txt",
        help="Path to requirements file",
    )
    parser.add_argument(
        "--lockfile",
        default="requirements-lock.txt",
        help="Path to lockfile",
    )
    
    args = parser.parse_args()
    
    # Check if UV is installed
    if not check_uv_installed():
        install_uv()
    
    # Perform the requested action
    if args.action == "setup":
        create_venv(args.venv)
        install_dependencies(args.requirements, args.venv)
    elif args.action == "install":
        install_dependencies(args.requirements, args.venv)
    elif args.action == "update":
        update_dependencies(args.requirements, args.lockfile)
    elif args.action == "lockfile":
        generate_lockfile(args.requirements, args.lockfile)


if __name__ == "__main__":
    main() 