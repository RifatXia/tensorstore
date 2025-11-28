#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --job-name=model-t5x
#SBATCH --output=logs/t5x-%j.out

# run t5x-optimized tensorstore checkpointing only

echo "=========================================="
echo "t5x-optimized tensorstore checkpointing"
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
python -u save_t5x.py

echo ""
echo "=========================================="
echo "job completed: $(date)"
echo "=========================================="
