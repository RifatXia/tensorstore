# llama model checkpointing on ares cluster

modular checkpointing framework comparing pytorch, tensorstore, and t5x-optimized approaches for any llama-compatible model.

## ⚡ fully dynamic

- **no hardcoded model names** - all code uses `MODEL_NAME` from config
- **automatic file naming** - output files use `MODEL_ID` extracted from model name
- **works with any llama model** - just set `MODEL_NAME` and go
- **all paths dynamic** - no manual filename changes needed

## project structure

```
cluster/
├── src/                      # python source files
│   ├── config.py            # configuration settings
│   ├── utils.py             # utility functions
│   ├── download_model.py    # download model from huggingface
│   ├── load_model.py        # load model into memory
│   ├── save_pytorch.py      # phase 1: pytorch checkpointing
│   ├── save_tensorstore.py  # phase 2: tensorstore checkpointing
│   ├── save_t5x.py          # phase 3: t5x-optimized tensorstore
│   └── compare_results.py   # compare all methods
├── setup.sh                 # initial setup (run once)
├── download.sh              # download model (run on login node)
├── run_all.sh               # run all phases (slurm)
├── run_pytorch.sh           # run pytorch only (slurm)
├── run_tensorstore.sh       # run tensorstore only (slurm)
├── run_t5x.sh               # run t5x only (slurm)
├── logs/                    # job output logs
└── saved_models/            # checkpoints saved here
```

## quick start

### 1. copy to cluster

```bash
# from local machine
ssh ares "mkdir -p /home/zchowdhury1/work"
scp -r . ares:/home/zchowdhury1/work/cluster/
```

### 2. setup (first time only)

run on login node:

```bash
ssh ares
cd /home/zchowdhury1/work/cluster
bash setup.sh
```

this will:
- load required modules
- create virtual environment at `/mnt/common/$USER/venvs/venv`
- install all dependencies (torch, transformers, tensorstore, tqdm, matplotlib)

### 3. set model name (required)

**option a: environment variable (recommended)**

edit `run_all.sh`, add after line 40:

```bash
cd src

# set your model
export MODEL_NAME="meta-llama/Llama-2-7b-hf"

echo "=========================================="
```

**option b: edit config**

edit `src/config.py` line 15:

```python
MODEL_NAME = os.environ.get('MODEL_NAME', "meta-llama/Llama-2-7b-hf")
```

**supported models:**
- `openlm-research/open_llama_3b` (default, 3.4b params, ~6.4gb)
- `meta-llama/Llama-2-7b-hf` (7b params, ~13gb)
- `meta-llama/Llama-2-13b-hf` (13b params, ~25gb)
- `mistralai/Mistral-7B-v0.1` (7b params, ~14gb)
- any huggingface llama-compatible model

### 4. download model (first time only)

**⚠️ IMPORTANT**: Compute nodes don't have internet access. You **must** download on login node first:

```bash
bash download.sh
```

model will be cached at `/mnt/common/$USER/huggingface_cache`

**Why this matters**: The code uses `local_files_only=True` to prevent internet access on compute nodes. If the model isn't cached, the job will fail with "Model not found" error.

### 5. run checkpointing

**option a: run all phases**
```bash
sbatch run_all.sh
```

runs all three checkpointing methods sequentially and compares results.

**option b: run individual phases**
```bash
# pytorch only
sbatch run_pytorch.sh

# tensorstore only
sbatch run_tensorstore.sh

# t5x-optimized only
sbatch run_t5x.sh
```

### 6. monitor progress

```bash
# check job status
squeue -u $USER

# view live output (replace JOBID with your job number)
tail -f logs/checkpoint-JOBID.out

# or for individual phases
tail -f logs/pytorch-JOBID.out
tail -f logs/tensorstore-JOBID.out
tail -f logs/t5x-JOBID.out
```

## checkpointing phases (6 total)

### phase 1: pytorch (baseline)
- uses pytorch's native `torch.save()` and `torch.load()`
- fastest save/load times
- single .pth file
- baseline for comparison
- saves all 237 model parameters

### phase 2: tensorstore (basic)
- tensorstore with zarr format
- dynamic 64 mib chunking (adaptive per tensor)
- separate zarr file per parameter (237 files)
- no compression, no concurrency
- baseline for tensorstore variants

### phase 3: t5x-optimized tensorstore
- t5x-style optimizations
- dynamic 64 mib chunks + gzip compression (level 1)
- high concurrency (128 concurrent file i/o ops)
- optimized for distributed systems
- ~23% smaller due to compression

### phase 4a: tensorstore + concurrency only
- tests impact of concurrency alone
- 64 mib chunks, no compression
- 128 concurrent operations
- isolates concurrency benefit

### phase 4b: tensorstore + 1 mib chunks
- tests impact of smaller chunks
- 1 mib chunks (vs 64 mib baseline)
- no compression, no concurrency
- isolates chunk size impact

### phase 4c: tensorstore + compression only
- tests impact of compression alone
- 64 mib chunks + gzip compression
- no concurrency
- isolates compression benefit

## output files

after running, you'll find:

```
saved_models/
└── {model_name}/                              # e.g., open_llama_3b/
    ├── pytorch.pth                            # phase 1: pytorch checkpoint
    ├── tensorstore/                           # phase 2: basic tensorstore
    │   ├── *.zarr                            # 237 parameter files
    │   └── metadata.json
    ├── t5x_tensorstore/                       # phase 3: t5x-optimized
    │   ├── *.zarr                            # 237 parameter files (compressed)
    │   └── metadata.json
    ├── phase4a_concurrency/                   # phase 4a: concurrency only
    │   ├── *.zarr
    │   └── metadata.json
    ├── phase4b_chunks/                        # phase 4b: 1 mib chunks
    │   ├── *.zarr
    │   └── metadata.json
    ├── phase4c_compression/                   # phase 4c: compression only
    │   ├── *.zarr
    │   └── metadata.json
    ├── plots/                                 # visualization charts
    │   ├── 6way_comparison.png               # comprehensive 6-way comparison
    │   └── tensorstore_variants.png          # tensorstore variants analysis
    ├── all_phases_results.json               # complete results data
    └── comparison_results.json               # file size comparison
```

**automatic organization:**

`{model_name}` is extracted from `MODEL_NAME`:
- `openlm-research/open_llama_3b` → `saved_models/open_llama_3b/`
- `meta-llama/Llama-2-7b-hf` → `saved_models/Llama-2-7b-hf/`
- `mistralai/Mistral-7B-v0.1` → `saved_models/Mistral-7B-v0.1/`

each model gets its own directory with all checkpoints and plots

## configuration

all settings in `src/config.py`:

```python
# model (set via environment or edit default)
MODEL_NAME = os.environ.get('MODEL_NAME', "openlm-research/open_llama_3b")
DEVICE = "cpu"

# tensorstore settings
CHUNK_SIZE_MB = 64               # chunk size for dynamic chunking
T5X_CHUNK_SIZE_MB = 64           # chunk size for t5x phase
CONCURRENCY_LIMIT = 128          # concurrent file operations
COMPRESSION_LEVEL = 1            # gzip compression level (1-9)

# paths (automatic)
SAVED_MODELS_DIR = "saved_models/"
HF_CACHE = "/mnt/common/$USER/huggingface_cache"
MODEL_ID = MODEL_NAME.split('/')[-1]  # extracted for filenames
```

**to change model:** see step 3 in quick start above

## how it works

### 6-phase comparison workflow

1. **load model** - loads model from huggingface with float16 precision
2. **run all 6 phases** - sequentially saves and loads with each method
3. **collect metrics** - records save time, load time, file size for each phase
4. **generate visualizations** - creates 2 comprehensive comparison charts
5. **save results** - stores all data in JSON for analysis

### dynamic model handling

1. **set model name** - via environment variable or config file
2. **automatic extraction** - `MODEL_ID` extracted from `MODEL_NAME`
3. **organized structure** - each model gets its own directory
4. **no conflicts** - different models don't interfere

example:
```bash
# run 1
export MODEL_NAME="openlm-research/open_llama_3b"
sbatch run_all.sh
# creates: saved_models/open_llama_3b/ with all 6 phases + plots

# run 2
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
sbatch run_all.sh
# creates: saved_models/Llama-2-7b-hf/ with all 6 phases + plots
```

### visualization charts

**chart 1: 6-way comprehensive comparison** (2x3 grid)
- save time comparison
- load time comparison
- file size comparison
- save speedup vs pytorch
- load speedup vs pytorch
- overall efficiency score

**chart 2: tensorstore variants** (2x2 grid)
- save time for all tensorstore variants
- load time for all tensorstore variants
- file size for all tensorstore variants
- improvement vs basic tensorstore

### code structure

all python files import from `config.py`:

```python
# src/config.py
MODEL_NAME = os.environ.get('MODEL_NAME', "openlm-research/open_llama_3b")
MODEL_ID = MODEL_NAME.split('/')[-1]
MODEL_DIR = f"saved_models/{MODEL_ID}/"
PLOTS_DIR = f"saved_models/{MODEL_ID}/plots/"

# all files use these dynamic paths
from config import MODEL_NAME, MODEL_ID, MODEL_DIR, PLOTS_DIR
```

**no hardcoded model names or paths anywhere in the code.**

## features

- **6-phase comparison**: comprehensive analysis of pytorch vs tensorstore variants
- **fully dynamic**: all model names and paths from config
- **organized structure**: each model in its own directory with plots
- **automatic visualizations**: generates 2 comprehensive comparison charts
- **detailed metrics**: save time, load time, file size for all phases
- **isolation testing**: phases 4a-4c isolate individual optimizations
- **dynamic chunking**: automatic optimal chunk size per tensor
- **easy to modify**: change model via environment variable or config
- **slurm integration**: ready-to-use batch scripts
- **multiple models**: run different models without conflicts

## testing multiple models

### option 1: sequential runs

```bash
# edit run_all.sh, set MODEL_NAME
export MODEL_NAME="openlm-research/open_llama_3b"
sbatch run_all.sh

# wait for completion, then change model
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
sbatch run_all.sh

# files won't conflict - different MODEL_ID
```

### option 2: command line

```bash
# download models
MODEL_NAME="openlm-research/open_llama_3b" python src/download_model.py
MODEL_NAME="meta-llama/Llama-2-7b-hf" python src/download_model.py

# run phases
MODEL_NAME="openlm-research/open_llama_3b" python src/run_all_phases.py
MODEL_NAME="meta-llama/Llama-2-7b-hf" python src/run_all_phases.py
```

## troubleshooting

**job not starting?**
- check queue: `squeue -u $USER`
- ares uses first-come-first-served scheduling

**module not found errors?**
- ensure you ran `setup.sh` first
- activate venv: `source /mnt/common/$USER/venvs/venv/bin/activate`

**model download failing?**
- run `download.sh` on login node (not compute node)
- compute nodes don't have internet access
- for gated models (llama-2), ensure huggingface access token

**out of memory?**
- model size varies: 3b (~6gb), 7b (~13gb), 13b (~25gb)
- check available memory: `free -h`
- use smaller model or request more memory

**wrong model being used?**
- check `MODEL_NAME` in `src/config.py`
- verify environment variable: `echo $MODEL_NAME`
- check job output: model name printed at start

**files not created?**
- check logs: `cat logs/checkpoint-JOBID.out`
- ensure `saved_models/` directory exists
- verify disk space: `df -h`

**need more time?**
- edit `#SBATCH --time=` in .sh files
- default: 4 hours for all phases, 1-2 hours for individual
- max: 48 hours on ares

## verification

check that everything is dynamic:

```bash
# all python files use MODEL_NAME/MODEL_ID from config
grep -r "MODEL_NAME\|MODEL_ID" src/*.py

# no hardcoded model names in code (only in config default)
grep -r "open_llama_3b" src/*.py
# should only find in config.py as default value

# test with different model
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
python src/download_model.py
# downloads Llama-2-7b-hf, not open_llama_3b
```

## ares cluster info

- **login node**: ares.cs.iit.edu
- **compute nodes**: ares-comp-[01-32]
- **shared storage**: `/mnt/common/$USER`
- **local ssd**: `/mnt/ssd/$USER` (on compute nodes)
- **modules**: python/3.11.9-zg4555e, python-venv/1.0-u5vf2gn

## summary

**everything is dynamic:**
1. set `MODEL_NAME` (environment variable or config)
2. `MODEL_ID` automatically extracted
3. all files use `MODEL_ID` for naming
4. run scripts normally - no manual changes needed

**no hardcoded values anywhere in the code.**
