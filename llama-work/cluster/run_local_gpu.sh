#!/bin/bash

# local gpu run script - for running on local machine with nvidia gpu
#
# usage:
#   bash run_local_gpu.sh                                    # run all phases with defaults
#   MODEL_NAME="Qwen/Qwen2.5-7B" bash run_local_gpu.sh      # specify model
#   DTYPE="float16" bash run_local_gpu.sh                    # specify dtype
#
# environment variables:
#   MODEL_NAME      - huggingface model name (default: openlm-research/open_llama_3b)
#   PHASES          - comma-separated phase numbers to run (default: 1,2,3,4a,4b,4c)
#   CHUNK_SIZE_MB   - chunk size in megabytes (default: 64)
#   DTYPE           - data type: auto, float16, float32, bfloat16 (default: auto)
#   SKIP_PLOTS      - set to 1 to skip plot generation (default: 0)
#   HF_TOKEN        - huggingface token for private/gated models (optional)
#   RUN_TIMESTAMP   - timestamp for this run (default: auto-generated)
#   CLEAR_CACHE     - set to 1 to clear system cache before each operation (default: 1)

echo "=========================================="
echo "model checkpointing - local gpu run"
echo "=========================================="
echo "started: $(date)"
echo "hostname: $(hostname)"
echo "=========================================="

# parse command line arguments
if [ -n "$1" ]; then
    export MODEL_NAME="$1"
fi

# set defaults if not provided
export MODEL_NAME="${MODEL_NAME:-openlm-research/open_llama_3b}"
export PHASES="${PHASES:-1,2,3,4a,4b,4c}"
export CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-64}"
export DTYPE="${DTYPE:-auto}"
export DEVICE="cuda"  # force cuda for local gpu
export SKIP_PLOTS="${SKIP_PLOTS:-0}"
export CLEAR_CACHE="${CLEAR_CACHE:-1}"

# generate timestamp for this run if not provided
if [ -z "$RUN_TIMESTAMP" ]; then
    export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi

# use local cache (adjust path as needed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo ""
echo "configuration:"
echo "  model: $MODEL_NAME"
echo "  run timestamp: $RUN_TIMESTAMP"
echo "  phases: $PHASES"
echo "  chunk size: ${CHUNK_SIZE_MB} mb"
echo "  dtype: $DTYPE"
echo "  device: $DEVICE (gpu)"
echo "  skip plots: $SKIP_PLOTS"
echo "  clear cache: $CLEAR_CACHE"
echo "  cache: $HF_HOME"
echo "  hf token: ${HF_TOKEN:+set (private models enabled)}${HF_TOKEN:-not set (public models only)}"
echo ""

# check for nvidia gpu
if ! command -v nvidia-smi &> /dev/null; then
    echo "error: nvidia-smi not found. ensure nvidia drivers are installed."
    exit 1
fi

echo "gpu information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# check python and cuda availability
echo "python version: $(python --version)"
python -c "import torch; print(f'pytorch version: {torch.__version__}'); print(f'cuda available: {torch.cuda.is_available()}'); print(f'cuda version: {torch.version.cuda if torch.cuda.is_available() else \"n/a\"}')"
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
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    $([ "$SKIP_PLOTS" = "1" ] && echo "--skip-plots") \
    $([ "$CLEAR_CACHE" = "1" ] && echo "--clear-cache")
echo ""

echo "=========================================="
echo "phases complete!"
echo "completed: $(date)"
echo "=========================================="
