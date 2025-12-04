#!/bin/bash

# download model - run this on login node (has internet access)
#
# usage:
#   bash download.sh                                    # download default model (public)
#   MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" bash download.sh  # download specific model
#   HF_TOKEN="hf_xxx" MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" bash download.sh  # private model
#
# for private/gated models:
#   1. get your token from https://huggingface.co/settings/tokens
#   2. export HF_TOKEN="hf_your_token_here" before running
#   3. or pass it inline: HF_TOKEN="hf_xxx" bash download.sh

echo "=========================================="
echo "downloading model"
echo "run this on login node only!"
echo "=========================================="

# set cache to shared storage
export HF_HOME=/mnt/common/$USER/huggingface_cache
mkdir -p $HF_HOME

echo "cache location: $HF_HOME"
echo "hf token: ${HF_TOKEN:+set (private models enabled)}${HF_TOKEN:-not set (public models only)}"

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
