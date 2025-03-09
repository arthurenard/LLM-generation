#!/usr/bin/env python3
"""
Python script to run all tests for the text generation pipeline.
This is an alternative to the shell script for better cross-platform compatibility.
"""
import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def check_pytest_installed():
    """Check if pytest is installed."""
    try:
        import pytest
        return True
    except ImportError:
        return False

def install_pytest():
    """Install pytest and pytest-cov."""
    print("Installing pytest and pytest-cov...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], check=True)

def run_tests(marker, name):
    """Run tests with a specific marker."""
    print("\n" + "=" * 40)
    print(f"Running {name} tests")
    print("=" * 40)
    
    # Create the test reports directory
    Path("test_reports").mkdir(exist_ok=True)
    
    # Run the tests
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests", "-m", marker, "-v"
        ],
        capture_output=True,
        text=True
    )
    
    # Print the output
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    print(f"Done running {name} tests")
    print("=" * 40)
    
    return result.returncode == 0

def validate_configs():
    """Validate YAML configurations."""
    print("\n" + "=" * 40)
    print("Validating YAML configurations")
    print("=" * 40)
    
    try:
        from src.utils.config_validator import check_config_files
        result = check_config_files()
        print("Done validating configurations")
        print("=" * 40)
        return result
    except ImportError:
        print("Error: Could not import config_validator. Make sure the project is properly installed.")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for the text generation pipeline")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--config-only", action="store_true", help="Run only configuration validation")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Set default values
    run_unit = True
    run_integration = True
    run_performance = False
    run_config = True
    
    # Override with command line arguments
    if args.unit_only:
        run_unit = True
        run_integration = False
        run_performance = False
        run_config = False
    elif args.integration_only:
        run_unit = False
        run_integration = True
        run_performance = False
        run_config = False
    elif args.config_only:
        run_unit = False
        run_integration = False
        run_performance = False
        run_config = True
    elif args.all:
        run_unit = True
        run_integration = True
        run_performance = True
        run_config = True
    
    if args.performance:
        run_performance = True
    
    # Print header
    print("=" * 40)
    print("Running tests for text generation pipeline")
    print("=" * 40)
    
    # Check if pytest is installed
    if not check_pytest_installed():
        install_pytest()
    
    # Run the tests
    results = {}
    
    if run_unit:
        results["unit"] = run_tests("unit", "unit")
    
    if run_integration:
        results["integration"] = run_tests("integration", "integration")
    
    if run_performance:
        results["performance"] = run_tests("performance", "performance")
    
    if run_config:
        results["config"] = validate_configs()
    
    # Print summary
    print("\n" + "=" * 40)
    print("Test Summary")
    print("=" * 40)
    for test_type, result in results.items():
        status = "Passed" if result else "Failed"
        print(f"{test_type.capitalize()} tests: {status}")
    
    for test_type in ["unit", "integration", "performance", "config"]:
        if test_type not in results:
            print(f"{test_type.capitalize()} tests: Skipped")
    
    print("\nTest reports are available in the test_reports directory")
    print("=" * 40)
    
    # Return success if all run tests passed
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 