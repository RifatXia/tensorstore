#!/bin/bash

# local run script - for running on local machine (cpu or gpu)
#
# usage:
#   ./run_local_gpu.sh                                       # run all phases with defaults (cpu)
#   DEVICE="cuda" ./run_local_gpu.sh                         # use gpu
#   MODEL_NAME="Qwen/Qwen2.5-7B" ./run_local_gpu.sh          # specify model
#   DTYPE="float16" ./run_local_gpu.sh                       # specify dtype
#
# environment variables:
#   MODEL_NAME      - huggingface model name (default: openlm-research/open_llama_3b)
#   DEVICE          - device to use: cpu or cuda (default: cpu)
#   PHASES          - comma-separated phase numbers to run (default: 1,2,3)
#   CHUNK_SIZE_MB   - chunk size in megabytes (default: 64)
#   CONCURRENCY     - tensorstore concurrency limit (default: none)
#   DTYPE           - data type: auto, float16, float32, bfloat16 (default: auto)
#   SKIP_PLOTS      - set to 1 to skip plot generation (default: 0)
#   CLEAR_CACHE     - set to 1 to enable cache clearing (default: 0, requires sudo)
#   HF_TOKEN        - huggingface token for private/gated models (optional)
#   RUN_TIMESTAMP   - timestamp for this run (default: auto-generated)

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
export PHASES="${PHASES:-1,2,3}"
export CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-64}"
export CONCURRENCY="${CONCURRENCY:-}"  # empty = tensorstore default
export DTYPE="${DTYPE:-auto}"
export DEVICE="${DEVICE:-cpu}"  # use cpu by default to avoid memory issues
export SKIP_PLOTS="${SKIP_PLOTS:-0}"
export CLEAR_CACHE="${CLEAR_CACHE:-0}"  # disable cache clearing by default (requires sudo)

# generate timestamp for this run if not provided
if [ -z "$RUN_TIMESTAMP" ]; then
    export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi

# use local cache (adjust path as needed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# activate virtual environment if it exists
VENV_PATH="../llama-venv"
if [ -d "$VENV_PATH" ]; then
    echo "activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "warning: virtual environment not found at $VENV_PATH"
    echo "make sure pytorch and dependencies are installed"
fi

echo ""
echo "configuration:"
echo "  model: $MODEL_NAME"
echo "  run timestamp: $RUN_TIMESTAMP"
echo "  phases: $PHASES"
echo "  chunk size: ${CHUNK_SIZE_MB} mb"
echo "  concurrency: ${CONCURRENCY:-default (tensorstore)}"
echo "  dtype: $DTYPE"
echo "  device: $DEVICE"
echo "  skip plots: $SKIP_PLOTS"
echo "  clear cache: $CLEAR_CACHE (disabled by default - no sudo needed)"
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
echo "python version: $(python3 --version)"
python3 -c "import torch; print(f'pytorch version: {torch.__version__}'); print(f'cuda available: {torch.cuda.is_available()}'); print(f'cuda version: {torch.version.cuda if torch.cuda.is_available() else \"n/a\"}')"
echo ""

# create logs directory
mkdir -p logs

# run phases
cd src

echo "=========================================="
echo "running checkpointing phases: $PHASES"
echo "=========================================="
python3 -u run_all_phases.py \
    --model "$MODEL_NAME" \
    --phases "$PHASES" \
    --chunk-size "$CHUNK_SIZE_MB" \
    $([ -n "$CONCURRENCY" ] && echo "--concurrency $CONCURRENCY") \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    $([ "$SKIP_PLOTS" = "1" ] && echo "--skip-plots") \
    $([ "$CLEAR_CACHE" = "0" ] && echo "--no-clear-cache")
echo ""

echo "=========================================="
echo "phases complete!"
echo "completed: $(date)"
echo "=========================================="
