# Usage

## Setup (first time only)
```bash
ssh ares
cd /home/zchowdhury1/work/cluster
bash setup.sh
```

## Download Model (on login node)
```bash
# Default model
bash download.sh

# Specific model
MODEL_NAME="Qwen/Qwen2.5-7B" bash download.sh
MODEL_NAME="mistralai/Mistral-7B-v0.1" bash download.sh
MODEL_NAME="meta-llama/Meta-Llama-3-8B" bash download.sh
```

## Run All Phases
```bash
# Default (all 6 phases)
sbatch run_all.sh

# Specific model
MODEL_NAME="Qwen/Qwen2.5-7B" sbatch run_all.sh

# Specific phases only
PHASES="1,2,3" sbatch run_all.sh

# Custom chunk size
CHUNK_SIZE_MB=16 sbatch run_all.sh
```

## Test Different Chunk Sizes
```bash
# Automated (tests 1, 4, 16, 64 MB)
bash run_chunk_experiment.sh

# Manual
CHUNK_SIZE_MB=1 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=4 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=16 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=64 PHASES="2" sbatch run_all.sh
```

## Check Results
```bash
# Monitor job
tail -f logs/checkpoint-JOBID.out

# View results
cat saved_models/<model_id>/all_phases_results.json

# View configuration
cat saved_models/<model_id>/all_phases_results.json | jq '.phases[].configuration'

# View plots
ls saved_models/<model_id>/plots/
```

## Phase Options
- `1` = PyTorch
- `2` = TensorStore basic
- `3` = T5X optimized (compression + concurrency)
- `4a` = Concurrency only
- `4b` = 1 MB chunks
- `4c` = Compression only

## Environment Variables
- `MODEL_NAME` - model to use (default: openlm-research/open_llama_3b)
- `PHASES` - which phases to run (default: 1,2,3,4a,4b,4c)
- `CHUNK_SIZE_MB` - chunk size (default: 64)
- `DEVICE` - cpu or cuda (default: cpu)
- `SKIP_PLOTS` - set to 1 to skip plots (default: 0)

## Examples

### Run specific model with custom settings
```bash
MODEL_NAME="Qwen/Qwen2.5-7B" CHUNK_SIZE_MB=32 PHASES="2,3" sbatch run_all.sh
```

### Test multiple chunk sizes
```bash
for chunk in 1 4 16 64; do
    CHUNK_SIZE_MB=$chunk PHASES="2" sbatch run_all.sh
done
```

### Compare two models
```bash
MODEL_NAME="openlm-research/open_llama_3b" bash download.sh
MODEL_NAME="Qwen/Qwen2.5-7B" bash download.sh

MODEL_NAME="openlm-research/open_llama_3b" sbatch run_all.sh
MODEL_NAME="Qwen/Qwen2.5-7B" sbatch run_all.sh
```

## Output Structure
```
saved_models/<model_id>/
├── pytorch.pth
├── tensorstore/
├── t5x_tensorstore/
├── phase4a_concurrency/
├── phase4b_chunks/
├── phase4c_compression/
├── plots/
│   ├── 6way_comparison.png
│   └── tensorstore_variants.png
└── all_phases_results.json
```

## JSON Output Includes
- Save/load times
- File sizes
- Configuration: chunk_size_mb, compression, concurrency, dtype
- Model metadata: model_name, model_type, device, timestamp
