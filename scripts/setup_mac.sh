#!/bin/bash

# Neural-LSH Setup Script for macOS
# Run this script to set up the project on Mac

echo "=========================================="
echo "Neural-LSH Setup for macOS"
echo "=========================================="

# Check Python3
echo "✓ Checking Python3..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 not found. Please install Python3 first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"

# Check pip3
echo "✓ Checking pip3..."
if ! command -v pip3 &> /dev/null; then
    echo "✗ pip3 not found. Please install pip3."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo ""
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "✓ Virtual environment created and activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Pre-compute ground truth caches (recommended):"
echo "   python3 search/create_brute_force_cache.py --sift 1000 --mnist 1000"
echo ""
echo "3. Build index (one time):"
echo "   python3 src/nlsh_build.py --data dataset/sift_base.fvecs --type sift --m 100 --knn 10 --layers 3 --epochs 10"
echo ""
echo "4. Run experiments:"
echo "   ./experimental_study/run_experiment_scenario.sh"
echo ""
