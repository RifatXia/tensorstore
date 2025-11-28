#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=model-pytorch
#SBATCH --output=logs/pytorch-%j.out

# run pytorch checkpointing only

echo "=========================================="
echo "pytorch checkpointing"
echo "=========================================="
echo "job started: $(date)"
echo "node: $(hostname)"
echo "job id: $SLURM_JOB_ID"
echo "=========================================="

export HF_HOME=/mnt/common/$USER/huggingface_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

module load python/3.11.9-zg4555e
module load python-venv/1.0-u5vf2gn
source /mnt/common/$USER/venvs/venv/bin/activate

mkdir -p logs
cd src
python -u save_pytorch.py

echo ""
echo "=========================================="
echo "job completed: $(date)"
echo "=========================================="
