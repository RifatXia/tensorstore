#!/bin/bash

# setup script - run this first on login node

echo "=========================================="
echo "model checkpointing setup"
echo "========================================="

# load modules
echo "loading modules..."
module load python/3.11.9-zg4555e
module load python-venv/1.0-u5vf2gn

# create venv
echo "creating virtual environment..."
mkdir -p /mnt/common/$USER/venvs
cd /mnt/common/$USER/venvs
python -m venv venv
source venv/bin/activate

# install dependencies
echo "installing dependencies..."
pip install --upgrade pip
pip install torch transformers tensorstore numpy tqdm matplotlib

echo ""
echo "=========================================="
echo "setup complete!"
echo "=========================================="
echo ""
echo "next steps:"
echo "1. run: bash download.sh (on login node)"
echo "2. run: sbatch run_all.sh (submit to cluster)"
