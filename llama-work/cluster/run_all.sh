#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=model-checkpoint
#SBATCH --output=logs/checkpoint-%j.out
#SBATCH --exclusive

# main script to run all checkpointing phases with configurable options
#
# usage:
#   sbatch run_all.sh                                    # run all phases with defaults
#   sbatch run_all.sh meta-llama/Llama-2-7b-hf          # specify model
#   MODEL_NAME=... PHASES=1,2,3 sbatch run_all.sh       # specify model and phases
#
# environment variables:
#   MODEL_NAME      - huggingface model name (default: openlm-research/open_llama_3b)
#   PHASES          - comma-separated phase numbers to run (default: 1,2,3)
#   CHUNK_SIZE_MB   - chunk size in megabytes (default: 64)
#   CONCURRENCY     - tensorstore concurrency limit (default: tensorstore default)
#   DTYPE           - data type: auto, float16, float32, bfloat16 (default: auto)
#   DEVICE          - device to use (default: cpu)
#   NUM_RUNS        - number of runs per phase for reliability (default: 3)
#   SKIP_PLOTS      - set to 1 to skip plot generation (default: 0)
#   HF_TOKEN        - huggingface token for private/gated models (optional)
#   RUN_TIMESTAMP   - timestamp for this run (default: auto-generated)

echo "=========================================="
echo "model checkpointing - configurable run"
echo "========================================="
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
export PHASES="${PHASES:-1,2,3}"
export CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-64}"
export CONCURRENCY="${CONCURRENCY:-}"  # empty = tensorstore default
export DTYPE="${DTYPE:-auto}"
export DEVICE="${DEVICE:-cpu}"
export NUM_RUNS="${NUM_RUNS:-3}"
export SKIP_PLOTS="${SKIP_PLOTS:-0}"

# generate timestamp for this run if not provided
if [ -z "$RUN_TIMESTAMP" ]; then
    export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi

# use shared storage cache
export HF_HOME=/mnt/common/$USER/huggingface_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo ""
echo "configuration:"
echo "  model: $MODEL_NAME"
echo "  run timestamp: $RUN_TIMESTAMP"
echo "  phases: $PHASES"
echo "  chunk size: ${CHUNK_SIZE_MB} mb"
echo "  concurrency: ${CONCURRENCY:-default (tensorstore)}"
echo "  dtype: $DTYPE"
echo "  device: $DEVICE"
echo "  num runs: $NUM_RUNS (for reliability)"
echo "  skip plots: $SKIP_PLOTS"
echo "  cache: $HF_HOME"
echo "  hf token: ${HF_TOKEN:+set (private models enabled)}${HF_TOKEN:-not set (public models only)}"
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
    $([ -n "$CONCURRENCY" ] && echo "--concurrency $CONCURRENCY") \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --num-runs "$NUM_RUNS" \
    $([ "$SKIP_PLOTS" = "1" ] && echo "--skip-plots")
echo ""

echo "=========================================="
echo "phases complete!"
echo "job completed: $(date)"
echo "=========================================="
