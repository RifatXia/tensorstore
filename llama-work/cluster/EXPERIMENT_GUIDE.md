# Experiment Guide - Running Custom Configurations

## 🎯 Overview

You can now run experiments with custom configurations and get detailed results including all configuration parameters in the JSON output.

## 📊 Enhanced JSON Output

The results JSON now includes:

```json
{
  "model_name": "openlm-research/open_llama_3b",
  "model_id": "open_llama_3b",
  "model_type": "llama",
  "device": "cpu",
  "timestamp": "2025-11-28 03:30:00",
  "phases": {
    "pytorch": {
      "save_time_ms": 6377.5,
      "load_time_ms": 3616.3,
      "file_size_bytes": 2730000000,
      "file_size_gb": 2.54,
      "configuration": {
        "method": "torch.save",
        "dtype": "float16",
        "compression": "none",
        "format": "pytorch"
      }
    },
    "tensorstore": {
      "save_time_ms": 140185.7,
      "load_time_ms": 18534.2,
      "file_size_bytes": 2580000000,
      "file_size_gb": 2.40,
      "configuration": {
        "chunk_size_mb": 64,
        "compression": "none",
        "concurrency": 1,
        "dtype": "float16",
        "parameters_saved": 100
      }
    }
  }
}
```

## 🔧 Running Custom Experiments

### 1. Test Different Chunk Sizes

**Automated Script (Recommended):**
```bash
# Run experiments with 1, 4, 16, 64 MB chunks
bash run_chunk_experiment.sh
```

**Manual Method:**
```bash
# Run each chunk size individually
CHUNK_SIZE_MB=1 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=4 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=16 PHASES="2" sbatch run_all.sh
CHUNK_SIZE_MB=64 PHASES="2" sbatch run_all.sh
```

### 2. Test Different Configurations

**Only Compression:**
```bash
PHASES="4c" CHUNK_SIZE_MB=64 sbatch run_all.sh
```

**Only Concurrency:**
```bash
PHASES="4a" CHUNK_SIZE_MB=64 sbatch run_all.sh
```

**Only Chunk Size Variation:**
```bash
PHASES="4b" sbatch run_all.sh  # Uses 1 MB chunks
```

**Full T5X Optimization:**
```bash
PHASES="3" CHUNK_SIZE_MB=64 sbatch run_all.sh
```

### 3. Compare Multiple Chunk Sizes for Same Model

```bash
# Download model once
MODEL_NAME="openlm-research/open_llama_3b" bash download.sh

# Run with different chunk sizes
for chunk in 1 4 16 64; do
    MODEL_NAME="openlm-research/open_llama_3b" \
    CHUNK_SIZE_MB=$chunk \
    PHASES="2" \
    sbatch run_all.sh
done
```

### 4. Test Different Models with Same Configuration

```bash
# Download all models
MODEL_NAME="openlm-research/open_llama_3b" bash download.sh
MODEL_NAME="Qwen/Qwen2.5-7B" bash download.sh

# Run same configuration on both
for model in "openlm-research/open_llama_3b" "Qwen/Qwen2.5-7B"; do
    MODEL_NAME="$model" \
    CHUNK_SIZE_MB=64 \
    PHASES="2,3" \
    sbatch run_all.sh
done
```

## 📋 Configuration Parameters

### Available Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `CHUNK_SIZE_MB` | 1, 4, 16, 64, 128, etc. | Chunk size in megabytes |
| `PHASES` | 1, 2, 3, 4a, 4b, 4c | Which phases to run |
| `DEVICE` | cpu, cuda | Device to use |
| `SKIP_PLOTS` | 0, 1 | Skip plot generation |
| `MODEL_NAME` | Any HF model | Model to test |

### Phase Configurations

| Phase | Chunk Size | Compression | Concurrency | Purpose |
|-------|------------|-------------|-------------|---------|
| 1 | N/A | No | N/A | PyTorch baseline |
| 2 | Custom | No | No | Basic TensorStore |
| 3 | Custom | gzip-1 | 128 | T5X optimized |
| 4a | Custom | No | 128 | Test concurrency |
| 4b | 1 MB | No | No | Test small chunks |
| 4c | Custom | gzip-1 | No | Test compression |

## 📊 Analyzing Results

### View Results JSON

```bash
# View complete results
cat saved_models/<model_id>/all_phases_results.json

# Pretty print with jq
cat saved_models/<model_id>/all_phases_results.json | jq '.'

# Extract specific phase
cat saved_models/<model_id>/all_phases_results.json | jq '.phases.tensorstore'

# Compare configurations
cat saved_models/<model_id>/all_phases_results.json | jq '.phases[].configuration'
```

### Example Analysis

```bash
# Get chunk sizes used
jq '.phases[].configuration.chunk_size_mb' saved_models/*/all_phases_results.json

# Get save times
jq '.phases[].save_time_ms' saved_models/*/all_phases_results.json

# Compare compression impact
jq '.phases | to_entries | map({phase: .key, compression: .value.configuration.compression, save_time: .value.save_time_ms})' saved_models/*/all_phases_results.json
```

## 🔬 Common Experiment Scenarios

### Scenario 1: Chunk Size Optimization

**Goal:** Find optimal chunk size for your model

```bash
# Test multiple chunk sizes
for chunk in 1 4 8 16 32 64 128; do
    CHUNK_SIZE_MB=$chunk PHASES="2" sbatch run_all.sh
done

# Compare results
for chunk in 1 4 8 16 32 64 128; do
    echo "Chunk size: ${chunk} MB"
    # Results will be in separate runs
done
```

### Scenario 2: Compression vs No Compression

**Goal:** Measure compression impact

```bash
# No compression (Phase 2)
PHASES="2" CHUNK_SIZE_MB=64 sbatch run_all.sh

# With compression (Phase 4c)
PHASES="4c" CHUNK_SIZE_MB=64 sbatch run_all.sh

# Compare
jq '.phases.tensorstore, .phases.phase4c_compression' saved_models/*/all_phases_results.json
```

### Scenario 3: Concurrency Impact

**Goal:** Measure concurrency benefit

```bash
# No concurrency (Phase 2)
PHASES="2" CHUNK_SIZE_MB=64 sbatch run_all.sh

# With concurrency (Phase 4a)
PHASES="4a" CHUNK_SIZE_MB=64 sbatch run_all.sh

# Compare
jq '.phases.tensorstore, .phases.phase4a_concurrency' saved_models/*/all_phases_results.json
```

### Scenario 4: Full Optimization Stack

**Goal:** Test each optimization individually and combined

```bash
# Baseline
PHASES="2" CHUNK_SIZE_MB=64 sbatch run_all.sh

# Individual optimizations
PHASES="4a" CHUNK_SIZE_MB=64 sbatch run_all.sh  # Concurrency
PHASES="4b" sbatch run_all.sh                    # Small chunks
PHASES="4c" CHUNK_SIZE_MB=64 sbatch run_all.sh  # Compression

# All combined
PHASES="3" CHUNK_SIZE_MB=64 sbatch run_all.sh

# Compare all
jq '.phases' saved_models/*/all_phases_results.json
```

## 📈 Result Comparison Script

Create a simple Python script to compare results:

```python
import json
import glob

# Load all results
results = []
for file in glob.glob('saved_models/*/all_phases_results.json'):
    with open(file) as f:
        results.append(json.load(f))

# Compare chunk sizes
for result in results:
    for phase, data in result['phases'].items():
        config = data.get('configuration', {})
        chunk_size = config.get('chunk_size_mb', 'N/A')
        save_time = data['save_time_ms']
        file_size = data['file_size_gb']
        print(f"{phase}: chunk={chunk_size}MB, save={save_time:.1f}ms, size={file_size:.2f}GB")
```

## 🎯 Best Practices

### 1. Systematic Testing
- Test one variable at a time
- Keep other parameters constant
- Run multiple iterations for reliability

### 2. Documentation
- Note the job IDs for each experiment
- Keep a log of what you're testing
- Save the JSON results with descriptive names

### 3. Resource Management
- Use `SKIP_PLOTS=1` for faster iteration
- Run only necessary phases
- Clean up old checkpoints between experiments

### 4. Result Organization
```bash
# Create experiment directory
mkdir -p experiments/chunk_size_test/

# Copy results after each run
cp saved_models/*/all_phases_results.json \
   experiments/chunk_size_test/results_${CHUNK_SIZE_MB}mb.json
```

## 🚀 Quick Reference

### Run Single Configuration
```bash
CHUNK_SIZE_MB=16 PHASES="2" sbatch run_all.sh
```

### Run Multiple Configurations
```bash
bash run_chunk_experiment.sh
```

### Check Results
```bash
cat saved_models/<model_id>/all_phases_results.json | jq '.phases[].configuration'
```

### Compare Configurations
```bash
jq '.phases | to_entries | map({phase: .key, config: .value.configuration, save_time: .value.save_time_ms})' \
   saved_models/*/all_phases_results.json
```

## ✅ Summary

**New Features:**
- ✅ Configuration details in JSON output
- ✅ Chunk size, compression, concurrency tracked
- ✅ Timestamp and model type included
- ✅ Easy comparison between runs
- ✅ Automated experiment scripts

**Usage:**
```bash
# Simple: Run with custom chunk size
CHUNK_SIZE_MB=16 PHASES="2" sbatch run_all.sh

# Advanced: Run full experiment
bash run_chunk_experiment.sh

# Analysis: View configurations
jq '.phases[].configuration' saved_models/*/all_phases_results.json
```

**All configuration parameters are now tracked in the JSON output for easy comparison!**
