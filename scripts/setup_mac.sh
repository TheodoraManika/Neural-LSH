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
echo "2. Build index (one time):"
echo "   cd src/"
echo "   python3 nlsh_build.py -d ../dataset/sift_learn.fvecs -i ../model_index/sift_index.bin -type sift -m 100 --knn 10 --layers 3 --epochs 10"
echo ""
echo "3. Search with model:"
echo "   python3 nlsh_search.py -d ../dataset/sift_base.fvecs -q ../dataset/sift_query.fvecs -i ../model_index/sift_index.bin -o ../results/sift_results.txt -type sift -N 10 -T 5 -R 2800"
echo ""
