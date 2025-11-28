#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=llama-checkpoint
#SBATCH --output=logs/checkpoint-%j.out

# main script to run all checkpointing phases with configurable options
#
# usage:
#   sbatch run_all.sh                                    # run all phases with defaults
#   sbatch run_all.sh meta-llama/Llama-2-7b-hf          # specify model
#   MODEL_NAME=... PHASES=1,2,3 sbatch run_all.sh       # specify model and phases
#
# environment variables:
#   MODEL_NAME      - huggingface model name (default: openlm-research/open_llama_3b)
#   PHASES          - comma-separated phase numbers to run (default: 1,2,3,4a,4b,4c)
#   CHUNK_SIZE_MB   - chunk size in megabytes (default: 64)
#   DEVICE          - device to use (default: cpu)
#   SKIP_PLOTS      - set to 1 to skip plot generation (default: 0)

echo "=========================================="
echo "llama checkpointing - configurable run"
echo "=========================================="
echo "job started: $(date)"
echo "node: $(hostname)"
echo "job id: $SLURM_JOB_ID"
echo "=========================================="

# parse command line arguments
if [ -n "$1" ]; then
    export MODEL_NAME="$1"
fi

# set defaults if not provided
export MODEL_NAME="${MODEL_NAME:-openlm-research/open_llama_3b}"
export PHASES="${PHASES:-1,2,3,4a,4b,4c}"
export CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-64}"
export DEVICE="${DEVICE:-cpu}"
export SKIP_PLOTS="${SKIP_PLOTS:-0}"

# use shared storage cache
export HF_HOME=/mnt/common/$USER/huggingface_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo ""
echo "configuration:"
echo "  model: $MODEL_NAME"
echo "  phases: $PHASES"
echo "  chunk size: ${CHUNK_SIZE_MB} mb"
echo "  device: $DEVICE"
echo "  skip plots: $SKIP_PLOTS"
echo "  cache: $HF_HOME"
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

# run phases
cd src

echo "=========================================="
echo "running checkpointing phases: $PHASES"
echo "=========================================="
python -u run_all_phases.py \
    --model "$MODEL_NAME" \
    --phases "$PHASES" \
    --chunk-size "$CHUNK_SIZE_MB" \
    --device "$DEVICE" \
    $([ "$SKIP_PLOTS" = "1" ] && echo "--skip-plots")
echo ""

echo "=========================================="
echo "phases complete!"
echo "job completed: $(date)"
echo "=========================================="
