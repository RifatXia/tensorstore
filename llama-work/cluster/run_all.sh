#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=llama-checkpoint
#SBATCH --output=logs/checkpoint-%j.out

# main script to run all checkpointing phases

echo "=========================================="
echo "llama checkpointing - all phases"
echo "=========================================="
echo "job started: $(date)"
echo "node: $(hostname)"
echo "job id: $SLURM_JOB_ID"
echo "=========================================="

# use shared storage cache
export HF_HOME=/mnt/common/$USER/huggingface_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "cache: $HF_HOME"
echo ""

# load modules
echo "loading modules..."
module load python/3.11.9-zg4555e
module load python-venv/1.0-u5vf2gn

# activate venv
echo "activating venv..."
source /mnt/common/$USER/venvs/venv/bin/activate

echo "python version: $(python --version)"
echo ""

# create logs directory
mkdir -p logs

# run all phases
cd src

echo "=========================================="
echo "running all checkpointing phases"
echo "=========================================="
python -u run_all_phases.py
echo ""

echo "=========================================="
echo "all phases complete!"
echo "job completed: $(date)"
echo "=========================================="
