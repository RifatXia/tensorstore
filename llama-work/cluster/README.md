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

compute nodes don't have internet access. download on login node:

```bash
bash download.sh
```

model will be cached at `/mnt/common/$USER/huggingface_cache`

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

## checkpointing phases

### phase 1: pytorch
- uses pytorch's native `torch.save()` and `torch.load()`
- fastest save/load times
- single .pth file
- baseline for comparison
- saves all model parameters

### phase 2: tensorstore (basic)
- uses tensorstore with zarr format
- dynamic 64mib chunking (adaptive per tensor)
- separate zarr file per parameter
- no compression, default concurrency
- metadata.json for parameter info

### phase 3: t5x-optimized tensorstore
- t5x-style optimizations
- dynamic 64mib chunks with gzip compression (level 1)
- high concurrency (128 concurrent file i/o ops)
- optimized for distributed systems
- ~23% smaller due to compression
- saves all model parameters successfully

## output files

after running, you'll find:

```
saved_models/
├── {model_name}_pytorch.pth                    # pytorch checkpoint
├── {model_name}_tensorstore/                   # tensorstore checkpoint
│   ├── *.zarr                                  # parameter files (237 files)
│   └── metadata.json                           # parameter metadata
├── {model_name}_t5x_tensorstore/               # t5x checkpoint
│   ├── *.zarr                                  # parameter files (237 files)
│   └── metadata.json                           # parameter metadata
├── comparison_results.json                     # performance comparison
└── 3way_comparison.png                         # visualization chart
```

**automatic file naming:**

`{model_name}` is extracted from `MODEL_NAME`:
- `openlm-research/open_llama_3b` → `open_llama_3b_pytorch.pth`
- `meta-llama/Llama-2-7b-hf` → `Llama-2-7b-hf_pytorch.pth`
- `mistralai/Mistral-7B-v0.1` → `Mistral-7B-v0.1_pytorch.pth`

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

### dynamic model handling

1. **set model name** - via environment variable or config file
2. **automatic extraction** - `MODEL_ID` extracted from `MODEL_NAME`
3. **all files adapt** - checkpoints, logs, plots use `MODEL_ID`
4. **no conflicts** - different models create different files

example:
```bash
# run 1
export MODEL_NAME="openlm-research/open_llama_3b"
sbatch run_all.sh
# creates: open_llama_3b_pytorch.pth, open_llama_3b_tensorstore/, etc.

# run 2
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
sbatch run_all.sh
# creates: Llama-2-7b-hf_pytorch.pth, Llama-2-7b-hf_tensorstore/, etc.
```

### code structure

all python files import from `config.py`:

```python
# src/config.py
MODEL_NAME = os.environ.get('MODEL_NAME', "openlm-research/open_llama_3b")
MODEL_ID = MODEL_NAME.split('/')[-1]

# all other files
from config import MODEL_NAME, MODEL_ID

# automatic usage
save_path = f"{MODEL_ID}_pytorch.pth"  # dynamic!
```

**no hardcoded model names anywhere in the code.**

## features

- **fully dynamic**: all model names and paths from config
- **modular design**: each phase is a separate python script
- **automatic plots**: generates 4-subplot comparison visualization
- **detailed logging**: timestamps and performance metrics
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
