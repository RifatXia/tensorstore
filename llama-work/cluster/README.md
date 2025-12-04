# llama model checkpointing on ares cluster

modular checkpointing framework comparing pytorch, tensorstore, and t5x-optimized approaches for any llama-compatible model.

## ⚡ key features

- **private model support** - use gated models with `HF_TOKEN` (llama 3.2, llama 3, etc.)
- **auto dtype detection** - automatically uses model's default precision from config.json
- **timestamped runs** - each run creates unique directory, enabling multiple experiments
- **no hardcoded model names** - all code uses `MODEL_NAME` from config
- **works with any llama model** - public or private, just set `MODEL_NAME` and go

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
├── saved_models/            # model checkpoints (gitignored)
└── results/                 # plots, json results (tracked in git)
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

**public models:**
- `openlm-research/open_llama_3b` (default, 3.4b params, ~6.4gb)
- `meta-llama/Llama-2-7b-hf` (7b params, ~13gb)
- `meta-llama/Llama-2-13b-hf` (13b params, ~25gb)
- `mistralai/Mistral-7B-v0.1` (7b params, ~14gb)
- `Qwen/Qwen2.5-7B` (7b params, ~14gb)

**private/gated models (requires HF token):**
- `meta-llama/Llama-3.2-3B-Instruct` (3b params, ~6gb)
- `meta-llama/Meta-Llama-3-8B` (8b params, ~16gb)
- any huggingface llama-compatible model (public or private)

### 4. download model (first time only)

**⚠️ IMPORTANT**: Compute nodes don't have internet access. You **must** download on login node first.

**for public models:**
```bash
bash download.sh
# or specify model
MODEL_NAME="Qwen/Qwen2.5-7B" bash download.sh
```

**for private/gated models:**

first, get your huggingface token:
1. go to https://huggingface.co/settings/tokens
2. create a new token (read access is sufficient)
3. copy the token (starts with `hf_`)

then download:
```bash
# set token and download
export HF_TOKEN="hf_your_token_here"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" bash download.sh

# or inline
HF_TOKEN="hf_xxx" MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" bash download.sh
```

model will be cached at `/mnt/common/$USER/huggingface_cache`

**Why this matters**: The code uses `local_files_only=True` to prevent internet access on compute nodes. If the model isn't cached, the job will fail with "Model not found" error.

### 5. run checkpointing

**option a: run all phases (public models)**
```bash
sbatch run_all.sh
```

**option b: run with private models**
```bash
# set token first
export HF_TOKEN="hf_your_token_here"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" sbatch run_all.sh

# or inline
HF_TOKEN="hf_xxx" MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" sbatch run_all.sh
```

**option c: specify dtype**
```bash
# use model's default dtype (recommended)
DTYPE="auto" sbatch run_all.sh

# or force specific dtype
DTYPE="float16" sbatch run_all.sh
DTYPE="bfloat16" sbatch run_all.sh
```

runs all six checkpointing methods sequentially and compares results.

**multiple runs with timestamps:**

each run creates timestamped directories, allowing multiple experiments:
```bash
# run 1
sbatch run_all.sh
# creates: saved_models/20251204_020230_open_llama_3b/
#          results/20251204_020230_open_llama_3b/

# run 2 (different model)
MODEL_NAME="Qwen/Qwen2.5-7B" sbatch run_all.sh
# creates: saved_models/20251204_030145_Qwen2.5-7B/
#          results/20251204_030145_Qwen2.5-7B/

# run 3 (same model, different settings)
DTYPE="bfloat16" sbatch run_all.sh
# creates: saved_models/20251204_040512_open_llama_3b/
#          results/20251204_040512_open_llama_3b/
```

### 6. monitor progress

```bash
# check job status
squeue -u $USER

# view live output (replace JOBID with your job number)
tail -f logs/checkpoint-JOBID.out

# check results
ls results/*/all_phases_results.json
ls results/*/plots/*.png
```

---

## quick reference

### environment variables

| variable | default | description |
|----------|---------|-------------|
| `MODEL_NAME` | openlm-research/open_llama_3b | huggingface model name |
| `HF_TOKEN` | none | token for private/gated models |
| `DTYPE` | auto | data type (auto/float16/float32/bfloat16) |
| `PHASES` | 1,2,3,4a,4b,4c | phases to run |
| `CHUNK_SIZE_MB` | 64 | chunk size in mb |
| `DEVICE` | cpu | device (cpu/cuda) |
| `SKIP_PLOTS` | 0 | skip plots (0/1) |
| `RUN_TIMESTAMP` | auto | custom timestamp for run |

### common commands

```bash
# download public model
MODEL_NAME="Qwen/Qwen2.5-7B" bash download.sh

# download private model (get token from https://huggingface.co/settings/tokens)
HF_TOKEN="hf_xxx" MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" bash download.sh

# run with defaults (auto dtype, all phases)
sbatch run_all.sh

# run specific model
MODEL_NAME="Qwen/Qwen2.5-7B" sbatch run_all.sh

# run private model
HF_TOKEN="hf_xxx" MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" sbatch run_all.sh

# custom dtype (override auto-detection)
DTYPE="bfloat16" sbatch run_all.sh

# specific phases only
PHASES="1,2,3" sbatch run_all.sh

# larger chunks
CHUNK_SIZE_MB=128 sbatch run_all.sh

# skip plots (faster)
SKIP_PLOTS=1 sbatch run_all.sh

# custom timestamp for organized experiments
RUN_TIMESTAMP="experiment1" sbatch run_all.sh
```

### troubleshooting

**"model not found" error:**
```bash
# download first on login node
MODEL_NAME="your-model" bash download.sh
```

**"unauthorized" or "access denied":**
```bash
# set token (get from https://huggingface.co/settings/tokens)
export HF_TOKEN="hf_your_token_here"
# verify you have access on huggingface.co and accepted model terms
```

**dtype not detected:**
```bash
# manually specify dtype
DTYPE="float16" sbatch run_all.sh
```

---

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
saved_models/                                  # model checkpoints (gitignored)
└── {timestamp}_{model_name}/                  # e.g., 20251204_020230_open_llama_3b/
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
    └── phase4c_compression/                   # phase 4c: compression only
        ├── *.zarr
        └── metadata.json

results/                                       # results (tracked in git)
└── {timestamp}_{model_name}/                  # e.g., 20251204_020230_open_llama_3b/
    ├── plots/                                 # visualization charts
    │   ├── 6way_comparison.png               # comprehensive 6-way comparison
    │   └── tensorstore_variants.png          # tensorstore variants analysis
    ├── all_phases_results.json               # complete results data
    └── comparison_results.json               # file size comparison
```

**automatic organization with timestamps:**

each run gets a unique timestamped directory:
- `openlm-research/open_llama_3b` → `saved_models/20251204_020230_open_llama_3b/` + `results/20251204_020230_open_llama_3b/`
- `meta-llama/Llama-2-7b-hf` → `saved_models/20251204_030145_Llama-2-7b-hf/` + `results/20251204_030145_Llama-2-7b-hf/`
- `Qwen/Qwen2.5-7B` → `saved_models/20251204_040512_Qwen2.5-7B/` + `results/20251204_040512_Qwen2.5-7B/`

**benefits:**
- **multiple runs** - run same model multiple times without conflicts
- **saved_models/** - large checkpoint files (gitignored, not pushed to github)
- **results/** - plots and json files (tracked in git, pushed to github)
- **easy comparison** - compare different runs by timestamp

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

# huggingface authentication
HF_TOKEN = os.environ.get('HF_TOKEN', None)  # for private/gated models
HF_CACHE = "/mnt/common/$USER/huggingface_cache"

# paths (automatic with timestamps)
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_ID = MODEL_NAME.split('/')[-1]  # extracted for filenames
RUN_ID = f"{RUN_TIMESTAMP}_{MODEL_ID}"  # e.g., "20251204_020230_Qwen2.5-7B"
SAVED_MODELS_DIR = f"saved_models/{RUN_ID}/"  # checkpoints (gitignored)
RESULTS_DIR = f"results/{RUN_ID}/"           # plots, json (tracked)
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
# creates: saved_models/20251204_020230_open_llama_3b/ + results/20251204_020230_open_llama_3b/

# run 2 (different model)
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
sbatch run_all.sh
# creates: saved_models/20251204_030145_Llama-2-7b-hf/ + results/20251204_030145_Llama-2-7b-hf/

# run 3 (same model, different dtype)
export DTYPE="bfloat16"
sbatch run_all.sh
# creates: saved_models/20251204_040512_open_llama_3b/ + results/20251204_040512_open_llama_3b/
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
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_ID = MODEL_NAME.split('/')[-1]
RUN_ID = f"{RUN_TIMESTAMP}_{MODEL_ID}"       # e.g., "20251204_020230_Qwen2.5-7B"
MODEL_DIR = f"saved_models/{RUN_ID}/"        # checkpoints
RESULTS_DIR = f"results/{RUN_ID}/"           # plots, json
PLOTS_DIR = f"results/{RUN_ID}/plots/"       # visualization charts

# all files use these dynamic paths
from config import MODEL_NAME, MODEL_ID, RUN_ID, MODEL_DIR, RESULTS_DIR, PLOTS_DIR
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
