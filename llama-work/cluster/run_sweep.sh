#!/bin/bash

# parameter sweep script - compare different chunk sizes or dtypes
#
# usage:
#   bash run_sweep.sh chunk 1,4,16,64                       # sweep chunk sizes
#   bash run_sweep.sh dtype float16,bfloat16,float32        # sweep dtypes
#   MODEL_NAME="Qwen/Qwen2.5-7B" bash run_sweep.sh chunk 1,4,16,64
#
# arguments:
#   $1 - parameter to sweep: 'chunk' or 'dtype'
#   $2 - comma-separated values to test
#
# environment variables:
#   MODEL_NAME      - huggingface model name (default: openlm-research/open_llama_3b)
#   PHASES          - comma-separated phase numbers to run (default: 1,2,3,4a,4b,4c)
#   DEVICE          - device: cpu or cuda (default: cuda if available, else cpu)
#   HF_TOKEN        - huggingface token for private/gated models (optional)

if [ $# -lt 2 ]; then
    echo "usage: bash run_sweep.sh <chunk|dtype> <comma-separated-values>"
    echo ""
    echo "examples:"
    echo "  bash run_sweep.sh chunk 1,4,16,64"
    echo "  bash run_sweep.sh dtype float16,bfloat16,float32"
    echo "  MODEL_NAME=\"Qwen/Qwen2.5-7B\" bash run_sweep.sh chunk 1,4,16,64"
    exit 1
fi

SWEEP_PARAM=$1
SWEEP_VALUES=$2

# validate sweep parameter
if [ "$SWEEP_PARAM" != "chunk" ] && [ "$SWEEP_PARAM" != "dtype" ]; then
    echo "error: sweep parameter must be 'chunk' or 'dtype'"
    exit 1
fi

echo "=========================================="
echo "parameter sweep: $SWEEP_PARAM"
echo "values: $SWEEP_VALUES"
echo "=========================================="

# set defaults
export MODEL_NAME="${MODEL_NAME:-openlm-research/open_llama_3b}"
export PHASES="${PHASES:-1,2,3,4a,4b,4c}"

# auto-detect device
if command -v nvidia-smi &> /dev/null && python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    export DEVICE="${DEVICE:-cuda}"
else
    export DEVICE="${DEVICE:-cpu}"
fi

# generate sweep timestamp
SWEEP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export SWEEP_ID="${SWEEP_TIMESTAMP}_${SWEEP_PARAM}_sweep"

echo ""
echo "configuration:"
echo "  model: $MODEL_NAME"
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
    
    # set run timestamp for this specific run
    export RUN_TIMESTAMP="${SWEEP_ID}_${VALUE}"
    
    if [ "$SWEEP_PARAM" = "chunk" ]; then
        export CHUNK_SIZE_MB=$VALUE
        export DTYPE="${DTYPE:-auto}"
    else
        export DTYPE=$VALUE
        export CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-64}"
    fi
    
    # run experiment
    if [ "$DEVICE" = "cuda" ]; then
        bash run_local_gpu.sh
    else
        # for cluster/cpu
        cd src
        python -u run_all_phases.py \
            --model "$MODEL_NAME" \
            --phases "$PHASES" \
            --chunk-size "$CHUNK_SIZE_MB" \
            --dtype "$DTYPE" \
            --device "$DEVICE" \
            --clear-cache
        cd ..
    fi
    
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
