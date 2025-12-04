#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=param-sweep
#SBATCH --output=logs/sweep-%j.out
#SBATCH --exclusive

# tensorstore parameter sweep - optimize tensorstore configuration
#
# sweeps only phase 2 (basic tensorstore) to find optimal settings
# compares: chunk size, dtype, and file_io concurrency
#
# usage:
#   sbatch run_sweep.sh chunk 1,4,16,64                       # sweep chunk sizes
#   sbatch run_sweep.sh dtype float16,bfloat16,float32        # sweep dtypes
#   sbatch run_sweep.sh concurrency 1,4,16,64,128             # sweep file_io concurrency
#   MODEL_NAME="Qwen/Qwen2.5-7B" sbatch run_sweep.sh chunk 1,4,16,64
#
# arguments:
#   $1 - parameter to sweep: 'chunk', 'dtype', or 'concurrency'
#   $2 - comma-separated values to test
#
# environment variables:
#   MODEL_NAME      - huggingface model name (default: openlm-research/open_llama_3b)
#   PHASES          - phases to run (default: 2 - tensorstore only)
#   DEVICE          - device to use (default: cpu)
#   HF_TOKEN        - huggingface token for private/gated models (optional)

echo "=========================================="
echo "parameter sweep - cluster run"
echo "=========================================="
echo "job started: $(date)"
echo "node: $(hostname)"
echo "job id: $SLURM_JOB_ID"
echo "=========================================="

if [ $# -lt 2 ]; then
    echo "usage: sbatch run_sweep.sh <chunk|dtype|concurrency> <comma-separated-values>"
    echo ""
    echo "examples:"
    echo "  sbatch run_sweep.sh chunk 1,4,16,64"
    echo "  sbatch run_sweep.sh dtype float16,bfloat16,float32"
    echo "  sbatch run_sweep.sh concurrency 1,4,16,64,128"
    echo "  MODEL_NAME=\"Qwen/Qwen2.5-7B\" sbatch run_sweep.sh chunk 1,4,16,64"
    exit 1
fi

SWEEP_PARAM=$1
SWEEP_VALUES=$2

# validate sweep parameter
if [ "$SWEEP_PARAM" != "chunk" ] && [ "$SWEEP_PARAM" != "dtype" ] && [ "$SWEEP_PARAM" != "concurrency" ]; then
    echo "error: sweep parameter must be 'chunk', 'dtype', or 'concurrency'"
    exit 1
fi

echo "=========================================="
echo "parameter sweep: $SWEEP_PARAM"
echo "=========================================="

# set defaults
export MODEL_NAME="${MODEL_NAME:-openlm-research/open_llama_3b}"
export PHASES="${PHASES:-2}"  # sweep only runs phase 2 (tensorstore) by default
export DEVICE="${DEVICE:-cpu}"

# use shared storage cache
export HF_HOME=/mnt/common/$USER/huggingface_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

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
SWEEP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# extract model id for directory name
MODEL_ID=$(echo "$MODEL_NAME" | sed 's/.*\///')
export SWEEP_ID="${SWEEP_TIMESTAMP}_sweep_${SWEEP_PARAM}_${MODEL_ID}"

echo ""
echo "configuration:"
echo "  sweep id: $SWEEP_ID"
echo "  phases: $PHASES"
echo "  device: $DEVICE"
echo ""

# convert comma-separated values to array
IFS=',' read -ra VALUES <<< "$SWEEP_VALUES"

echo "running ${#VALUES[@]} experiments..."
echo ""

# run sweep
for VALUE in "${VALUES[@]}"; do
    echo "=========================================="
    echo "running with $SWEEP_PARAM = $VALUE"
    echo "=========================================="
    
    # set run timestamp for this specific run - nested under sweep directory
    export RUN_TIMESTAMP="${SWEEP_ID}/${SWEEP_PARAM}${VALUE}"
    
    if [ "$SWEEP_PARAM" = "chunk" ]; then
        export CHUNK_SIZE_MB=$VALUE
        export DTYPE="${DTYPE:-auto}"
        export CONCURRENCY="${CONCURRENCY:-}"
    elif [ "$SWEEP_PARAM" = "dtype" ]; then
        export DTYPE=$VALUE
        export CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-64}"
        export CONCURRENCY="${CONCURRENCY:-}"
    else
        export CONCURRENCY=$VALUE
        export CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-64}"
        export DTYPE="${DTYPE:-auto}"
    fi
    
    # run experiment on cluster using tensorstore-only sweep runner
    cd src
    python -u run_tensorstore_sweep.py \
        --model "$MODEL_NAME" \
        --chunk-size "$CHUNK_SIZE_MB" \
        $([ -n "$CONCURRENCY" ] && echo "--concurrency $CONCURRENCY") \
        --dtype "$DTYPE" \
        --device "$DEVICE"
    cd ..
    
    echo ""
done

echo "=========================================="
echo "sweep complete!"
echo "=========================================="
echo ""
echo "generating comparison plots..."

# generate comparison plots
cd src
python -u compare_sweep.py \
    --sweep-id "$SWEEP_ID" \
    --sweep-param "$SWEEP_PARAM" \
    --sweep-values "$SWEEP_VALUES"
cd ..

echo ""
echo "results saved in: results/${SWEEP_ID}/"
echo "comparison plots: results/${SWEEP_ID}/sweep_comparison.png"
