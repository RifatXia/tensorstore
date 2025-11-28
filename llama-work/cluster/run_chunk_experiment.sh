#!/bin/bash

# experiment script to test different chunk sizes
# usage: bash run_chunk_experiment.sh

MODEL_NAME="${MODEL_NAME:-openlm-research/open_llama_3b}"
CHUNK_SIZES=(1 4 16 64)

echo "=========================================="
echo "chunk size experiment"
echo "model: $MODEL_NAME"
echo "chunk sizes: ${CHUNK_SIZES[@]} mb"
echo "=========================================="

# download model first (if not already cached)
echo "ensuring model is downloaded..."
MODEL_NAME="$MODEL_NAME" bash download.sh

# run experiment for each chunk size
for chunk_size in "${CHUNK_SIZES[@]}"; do
    echo ""
    echo "=========================================="
    echo "running with chunk size: ${chunk_size} mb"
    echo "=========================================="
    
    # run only phase 2 (basic tensorstore) with specific chunk size
    MODEL_NAME="$MODEL_NAME" \
    CHUNK_SIZE_MB="$chunk_size" \
    PHASES="2" \
    sbatch run_all.sh
    
    echo "job submitted for chunk size ${chunk_size} mb"
    sleep 2  # small delay between submissions
done

echo ""
echo "=========================================="
echo "all experiments submitted!"
echo "check logs/ directory for results"
echo "=========================================="
