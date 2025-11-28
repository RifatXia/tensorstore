#!/bin/bash

# download model - run this on login node (has internet access)

echo "=========================================="
echo "downloading model"
echo "run this on login node only!"
echo "=========================================="

# set cache to shared storage
export HF_HOME=/mnt/common/$USER/huggingface_cache
mkdir -p $HF_HOME

echo "cache location: $HF_HOME"

# load modules
module load python/3.11.9-zg4555e
module load python-venv/1.0-u5vf2gn

# activate venv
source /mnt/common/$USER/venvs/venv/bin/activate

# download model
cd src
python -u download_model.py

echo ""
echo "=========================================="
echo "download complete!"
echo "=========================================="
echo ""
echo "next step: sbatch run_all.sh"
