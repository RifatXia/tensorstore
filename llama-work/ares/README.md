# LLaMA Model Checkpointing - Ares Cluster

CPU-only comparison of PyTorch vs TensorStore checkpointing approaches.

## Quick Start

### Virtual Environment
```bash
chmod +x setup_venv.sh
./setup_venv.sh
source llama-venv/bin/activate
jupyter notebook checkpoint_comparison.ipynb
```

### Docker
```bash
docker-compose up -d
# access jupyter at http://localhost:8888
```


## What It Does

- Loads OpenLLaMA-3B model (CPU-only)
- Saves all 237 parameters using 8 different approaches:
  1. PyTorch (baseline)
  2. TensorStore (basic)
  3. T5X-optimized TensorStore
  4. TensorStore + Concurrency (128 ops)
  5. TensorStore + Large Chunks (1MB)
  6. TensorStore + Compression (gzip)
  7. TensorStore + Float16 (2 bytes)
  8. TensorStore + OCDBT driver
- Measures save/load times and file sizes
- Generates comprehensive performance visualizations:
  - 8-way comprehensive analysis (6 graphs)
  - TensorStore variants comparison (4 graphs)
  - Optimization impact analysis (2 graphs)

## Requirements

- Python 3.10+
- 32GB+ RAM
- 50GB+ disk space
- **CPU-only** (no GPU/CUDA required)
- **Optimized installation** (no NVIDIA dependencies)

## Output

Results saved in `saved_models/`:
- `openllama_3b_pytorch.pth`
- `openllama_3b_tensorstore/`
- `openllama_3b_t5x_tensorstore/`
- `phase4a_concurrency/`
- `phase4b_chunks/`
- `phase4c_compression/`
- `phase4d_float16/`
- `phase4e_ocdbt/`
- `8way_comprehensive_analysis.png`
- `tensorstore_variants_comparison.png`
- `optimization_impact_analysis.png`

## Notes

- First run downloads ~13GB model
- CPU-only, no GPU required
- No model inference performed
- All 237 parameters saved consistently
