#!/bin/bash
# Script to run all tests for the text generation pipeline

# Set the Python executable
PYTHON="python3"

# Ensure the script exits if any command fails
set -e

# Print header
echo "========================================"
echo "Running tests for text generation pipeline"
echo "========================================"

# Check if pytest is installed
if ! $PYTHON -c "import pytest" &> /dev/null; then
    echo "pytest is not installed. Installing..."
    pip install pytest pytest-cov
fi

# Check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "It's recommended to run tests in a virtual environment."
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Create a directory for test reports
mkdir -p test_reports

# Function to run tests with a specific marker
run_tests() {
    local marker=$1
    local name=$2
    
    echo ""
    echo "========================================"
    echo "Running $name tests"
    echo "========================================"
    
    # Run pytest with the marker
    $PYTHON -m pytest tests -m "$marker" -v
    
    echo "Done running $name tests"
    echo "========================================"
}

# Function to validate YAML configurations
validate_configs() {
    echo ""
    echo "========================================"
    echo "Validating YAML configurations"
    echo "========================================"
    
    # Create a simple script to validate configurations
    cat > validate_configs.py << 'EOF'
import sys
from src.utils.config_validator import check_config_files

if __name__ == "__main__":
    config_dir = sys.argv[1] if len(sys.argv) > 1 else "config"
    if not check_config_files(config_dir):
        sys.exit(1)
EOF
    
    # Run the script
    $PYTHON validate_configs.py
    
    # Clean up
    rm validate_configs.py
    
    echo "Done validating configurations"
    echo "========================================"
}

# Parse command line arguments
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_PERFORMANCE=false
RUN_CONFIG=true
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)
            RUN_UNIT=true
            RUN_INTEGRATION=false
            RUN_PERFORMANCE=false
            RUN_CONFIG=false
            shift
            ;;
        --integration-only)
            RUN_UNIT=false
            RUN_INTEGRATION=true
            RUN_PERFORMANCE=false
            RUN_CONFIG=false
            shift
            ;;
        --performance)
            RUN_PERFORMANCE=true
            shift
            ;;
        --config-only)
            RUN_UNIT=false
            RUN_INTEGRATION=false
            RUN_PERFORMANCE=false
            RUN_CONFIG=true
            shift
            ;;
        --all)
            RUN_ALL=true
            RUN_UNIT=true
            RUN_INTEGRATION=true
            RUN_PERFORMANCE=true
            RUN_CONFIG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--unit-only] [--integration-only] [--performance] [--config-only] [--all]"
            exit 1
            ;;
    esac
done

# Run the tests
if $RUN_UNIT; then
    run_tests "unit" "unit"
fi

if $RUN_INTEGRATION; then
    run_tests "integration" "integration"
fi

if $RUN_PERFORMANCE; then
    run_tests "performance" "performance"
fi

if $RUN_CONFIG; then
    validate_configs
fi

# Print summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Unit tests: $(if $RUN_UNIT; then echo "Run"; else echo "Skipped"; fi)"
echo "Integration tests: $(if $RUN_INTEGRATION; then echo "Run"; else echo "Skipped"; fi)"
echo "Performance tests: $(if $RUN_PERFORMANCE; then echo "Run"; else echo "Skipped"; fi)"
echo "Configuration validation: $(if $RUN_CONFIG; then echo "Run"; else echo "Skipped"; fi)"
echo ""
echo "Test reports are available in the test_reports directory"
echo "========================================" 