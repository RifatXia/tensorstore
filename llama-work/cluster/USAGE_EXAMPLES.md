# Usage Examples - Configurable Checkpointing

The system now accepts command-line arguments and environment variables for flexible configuration.

## Basic Usage

### 1. Run All Phases (Default)
```bash
sbatch run_all.sh
```
- Model: `openlm-research/open_llama_3b` (default)
- Phases: All 6 phases (1,2,3,4a,4b,4c)
- Chunk size: 64 MB
- Device: CPU
- Plots: Generated

### 2. Specify Model via Argument
```bash
sbatch run_all.sh meta-llama/Llama-2-7b-hf
```
- Model: `meta-llama/Llama-2-7b-hf`
- Phases: All 6 phases
- Other settings: defaults

### 3. Specify Model via Environment Variable
```bash
MODEL_NAME="mistralai/Mistral-7B-v0.1" sbatch run_all.sh
```

## Advanced Usage

### 4. Run Specific Phases Only
```bash
# Run only PyTorch and basic TensorStore
PHASES="1,2" sbatch run_all.sh

# Run only phase 4 variants
PHASES="4a,4b,4c" sbatch run_all.sh

# Run PyTorch and T5X only
PHASES="1,3" sbatch run_all.sh
```

### 5. Change Chunk Size
```bash
# Use 128 MB chunks
CHUNK_SIZE_MB=128 sbatch run_all.sh

# Use 32 MB chunks
CHUNK_SIZE_MB=32 sbatch run_all.sh
```

### 6. Skip Plot Generation
```bash
# Skip plots to save time
SKIP_PLOTS=1 sbatch run_all.sh
```

### 7. Combined Configuration
```bash
# Custom model, specific phases, custom chunk size
MODEL_NAME="meta-llama/Llama-2-7b-hf" \
PHASES="1,2,3" \
CHUNK_SIZE_MB=128 \
sbatch run_all.sh
```

## Direct Python Execution

You can also run the Python script directly (useful for testing):

### 8. Run Locally with Arguments
```bash
cd src
python run_all_phases.py \
    --model "openlm-research/open_llama_3b" \
    --phases "1,2" \
    --chunk-size 64 \
    --device cpu
```

### 9. Skip Plots
```bash
python run_all_phases.py --skip-plots
```

### 10. Run Single Phase
```bash
# Only PyTorch
python run_all_phases.py --phases "1"

# Only T5X
python run_all_phases.py --phases "3"
```

## Common Scenarios

### Scenario 1: Quick Test (PyTorch Only)
```bash
PHASES="1" sbatch run_all.sh
```
- Fastest execution
- Only baseline checkpoint
- No TensorStore overhead

### Scenario 2: TensorStore Comparison
```bash
PHASES="2,3,4a,4b,4c" sbatch run_all.sh
```
- Skip PyTorch (already have baseline)
- Compare all TensorStore variants
- Analyze optimization impacts

### Scenario 3: Multiple Models Sequential
```bash
# Run model 1
MODEL_NAME="openlm-research/open_llama_3b" sbatch run_all.sh

# Run model 2
MODEL_NAME="meta-llama/Llama-2-7b-hf" sbatch run_all.sh

# Run model 3
MODEL_NAME="mistralai/Mistral-7B-v0.1" sbatch run_all.sh
```
- Each model in separate directory
- No conflicts
- Can run in parallel on different nodes

### Scenario 4: Chunk Size Experiment
```bash
# Test different chunk sizes
CHUNK_SIZE_MB=16 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=32 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=64 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=128 PHASES="2" sbatch run_all.sh
```
- Isolate chunk size impact
- Compare performance
- Find optimal setting

### Scenario 5: Production Run (All Phases, Custom Model)
```bash
MODEL_NAME="meta-llama/Llama-2-13b-hf" \
CHUNK_SIZE_MB=64 \
sbatch run_all.sh
```
- Complete 6-phase analysis
- Custom model
- Full visualizations

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `openlm-research/open_llama_3b` | HuggingFace model identifier |
| `PHASES` | `1,2,3,4a,4b,4c` | Comma-separated phase numbers |
| `CHUNK_SIZE_MB` | `64` | Chunk size in megabytes |
| `DEVICE` | `cpu` | Device to use (cpu/cuda) |
| `SKIP_PLOTS` | `0` | Set to 1 to skip plot generation |

## Phase Numbers

- `1` - PyTorch (baseline)
- `2` - TensorStore basic (64 MB chunks)
- `3` - T5X-optimized (compression + concurrency)
- `4a` - Concurrency only
- `4b` - 1 MB chunks
- `4c` - Compression only

## Output Organization

All outputs are organized by model:
```
saved_models/
└── <model_name>/
    ├── pytorch.pth              # Phase 1
    ├── tensorstore/             # Phase 2
    ├── t5x_tensorstore/         # Phase 3
    ├── phase4a_concurrency/     # Phase 4a
    ├── phase4b_chunks/          # Phase 4b
    ├── phase4c_compression/     # Phase 4c
    ├── plots/                   # Visualizations
    └── all_phases_results.json  # Metrics
```

## Tips

1. **Start small**: Test with `PHASES="1"` first
2. **Use skip-plots**: Add `SKIP_PLOTS=1` for faster iteration
3. **Monitor logs**: `tail -f logs/checkpoint-JOBID.out`
4. **Check results**: `cat saved_models/<model>/all_phases_results.json`
5. **Multiple models**: Each gets its own directory, no conflicts

## Troubleshooting

**Wrong phases running?**
```bash
# Check your PHASES variable
echo $PHASES

# Explicitly set it
PHASES="1,2,3" sbatch run_all.sh
```

**Model not found?**
```bash
# Verify model name
MODEL_NAME="meta-llama/Llama-2-7b-hf" bash download.sh

# Then run
MODEL_NAME="meta-llama/Llama-2-7b-hf" sbatch run_all.sh
```

**Out of memory?**
```bash
# Run fewer phases at once
PHASES="1" sbatch run_all.sh
PHASES="2" sbatch run_all.sh
# etc.
```
