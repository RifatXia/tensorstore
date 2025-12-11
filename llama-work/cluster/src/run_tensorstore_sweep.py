# tensorstore sweep runner - optimized for parameter sweeps
# runs only phase 2 (basic tensorstore) to find optimal configuration

import sys
import os
import argparse
import time
import json
import gc
import torch
import tensorstore as ts
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from utils import calculate_chunk_shape, format_size, get_directory_size, get_model_default_dtype, clear_system_cache, clear_gpu_cache

sys.stdout.reconfigure(line_buffering=True)

# parse command line arguments
parser = argparse.ArgumentParser(description='run tensorstore sweep with configurable options')
parser.add_argument('--model', type=str, default=None, help='huggingface model name')
parser.add_argument('--chunk-size', type=int, default=64, help='chunk size in megabytes (default: 64)')
parser.add_argument('--concurrency', type=int, default=None, help='tensorstore concurrency limit (default: tensorstore default, unlimited)')
parser.add_argument('--device', type=str, default='cpu', help='device to use (default: cpu)')
parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float16', 'float32', 'bfloat16'], help='data type for model and storage (default: auto - uses model default)')
parser.add_argument('--compression', type=str, default='none', choices=['none', 'gzip'], help='compression type (default: none)')
parser.add_argument('--num-runs', type=int, default=1, help='number of runs (default: 1 for sweeps)')
parser.add_argument('--no-clear-cache', action='store_true', help='disable cache clearing (enabled by default)')
args = parser.parse_args()

# number of runs (fixed to 1 for sweeps to save time)
NUM_RUNS = 1
if args.num_runs != 1:
    print(f"\nwarning: sweeps are optimized for single runs. ignoring --num-runs={args.num_runs}, using 1 instead")
print(f"\nrunning {NUM_RUNS} time (sweep mode)")

# set environment variables before importing config
if args.model:
    os.environ['MODEL_NAME'] = args.model
if args.device:
    os.environ['DEVICE'] = args.device

# now import config (after setting env vars)
from config import MODEL_NAME, MODEL_ID, MODEL_TYPE, DEVICE, MODEL_DIR, RESULTS_DIR, PLOTS_DIR, HF_CACHE, HF_TOKEN, RUN_ID

# determine dtype - use model default if 'auto'
if args.dtype == 'auto':
    DTYPE = get_model_default_dtype(MODEL_NAME, HF_CACHE)
    print(f"\nauto-detected dtype: {DTYPE}")
else:
    DTYPE = args.dtype
    print(f"\nusing specified dtype: {DTYPE}")

dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
torch_dtype = dtype_map[DTYPE]

print("="*70)
print(f"TENSORSTORE SWEEP: {MODEL_NAME}")
print(f"model type: {MODEL_TYPE}")
print(f"run id: {RUN_ID}")
print(f"dtype: {DTYPE}")
print(f"chunk size: {args.chunk_size} MB")
print(f"concurrency: {args.concurrency if args.concurrency else 'default (tensorstore)'}")
print(f"compression: {args.compression}")
print(f"num runs: {NUM_RUNS}")
print(f"clear cache: {not args.no_clear_cache}")
print("="*70)

# cache clearing helper - enabled by default
def clear_caches_if_enabled():
    if not args.no_clear_cache:
        clear_system_cache()
        if DEVICE == 'cuda':
            clear_gpu_cache()

# create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"\nmodel checkpoints directory: {MODEL_DIR}")
print(f"results directory: {RESULTS_DIR}")
print(f"note: individual plots skipped, use compare_sweep.py for final comparison")

# load model
print(f"\nloading model: {MODEL_NAME}")
print(f"device: {DEVICE}")

# force offline mode to avoid internet access on compute nodes
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
    local_files_only=True,
    trust_remote_code=True,
    token=HF_TOKEN
)
model = model.to(DEVICE)
print(f"✓ model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m parameters")

# storage for results
results = {
    'model_name': MODEL_NAME,
    'run_id': RUN_ID,
    'model_type': MODEL_TYPE,
    'device': DEVICE,
    'dtype': DTYPE,
    'chunk_size_mb': args.chunk_size,
    'concurrency': args.concurrency if args.concurrency else 'default',
    'compression': args.compression,
    'num_runs': NUM_RUNS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'phases': {}
}

# tensorstore save function with compression support
def save_tensorstore(model_state, save_dir, chunk_size_mb=64, concurrency_limit=None, compression='none'):
    """save model using tensorstore with dynamic chunking and optional compression"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"TENSORSTORE: saving (compression: {compression})")
    print(f"{'='*70}")
    
    # only set context if concurrency is enabled AND a limit is specified
    if concurrency_limit is not None:
        context = ts.Context({'file_io_concurrency': {'limit': concurrency_limit}})
    else:
        context = None
    
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # clear cache before save
    clear_caches_if_enabled()
    start_time = time.time()
    saved_count = 0
    
    # dtype conversion based on global DTYPE setting
    # dtype conversion
    # note: bfloat16 is converted to float16 for tensorstore compatibility
    # this maintains similar storage size (2 bytes) while ensuring numpy compatibility
    dtype_conversion = {
        'float16': (lambda x: x.detach().cpu().half().numpy(), '<f2'),
        'float32': (lambda x: x.detach().cpu().float().numpy(), '<f4'),
        'bfloat16': (lambda x: x.detach().cpu().half().numpy(), '<f2')  # convert bfloat16 to float16
    }
    convert_fn, zarr_dtype = dtype_conversion[DTYPE]
    
    for param_name, param_tensor in tqdm(model_state.items(), desc="saving"):
        try:
            param_np = convert_fn(param_tensor)
            safe_name = param_name.replace('.', '_').replace('/', '_')
            param_path = os.path.join(save_dir, f"{safe_name}.zarr")
            
            target_elements = chunk_size_bytes // param_np.dtype.itemsize
            chunk_shape = calculate_chunk_shape(list(param_np.shape), target_elements)
            
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': param_path},
                'metadata': {
                    'shape': list(param_np.shape),
                    'dtype': zarr_dtype,
                    'chunks': chunk_shape
                }
            }
            
            # add compression if enabled
            if compression == 'gzip':
                spec['metadata']['compressor'] = {
                    'id': 'gzip',
                    'level': 1
                }
            
            if context:
                store = ts.open(spec, create=True, delete_existing=True, context=context).result()
            else:
                store = ts.open(spec, create=True, delete_existing=True).result()
            
            store.write(param_np).result()
            saved_count += 1
        except Exception as e:
            print(f"error saving {param_name}: {e}")
            continue
    
    save_time = (time.time() - start_time) * 1000
    dir_size = get_directory_size(save_dir)
    
    print(f"✓ saved {saved_count} parameters in {save_time:.1f} ms")
    print(f"✓ total size: {format_size(dir_size)}")
    
    return save_time, dir_size, saved_count

# tensorstore load function
def load_tensorstore(save_dir):
    """load model from tensorstore"""
    print(f"\nTENSORSTORE: loading")
    
    # clear cache before load
    clear_caches_if_enabled()
    start_time = time.time()
    loaded_count = 0
    
    zarr_files = [f for f in os.listdir(save_dir) if f.endswith('.zarr')]
    
    for zarr_file in tqdm(zarr_files, desc="loading"):
        try:
            param_path = os.path.join(save_dir, zarr_file)
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': param_path}
            }
            dataset = ts.open(spec, open=True).result()
            _ = dataset[:].read().result()
            loaded_count += 1
        except Exception as e:
            print(f"error loading {zarr_file}: {e}")
    
    load_time = (time.time() - start_time) * 1000
    print(f"✓ loaded {loaded_count} parameters in {load_time:.1f} ms")
    
    return load_time

# run tensorstore phase (single run for sweep)
print(f"\n{'='*70}")
print(f"PHASE 2: TENSORSTORE (BASIC) - SWEEP MODE")
print(f"{'='*70}")

model_state = model.state_dict()
ts_dir = os.path.join(MODEL_DIR, "tensorstore")

ts_save_time, ts_size, ts_count = save_tensorstore(
    model_state, ts_dir, 
    chunk_size_mb=args.chunk_size, 
    concurrency_limit=args.concurrency,
    compression=args.compression
)

ts_load_time = load_tensorstore(ts_dir)

gc.collect()

print(f"\n{'='*70}")
print(f"TENSORSTORE RESULTS")
print(f"{'='*70}")
print(f"save time: {ts_save_time:.1f} ms")
print(f"load time: {ts_load_time:.1f} ms")
print(f"file size: {format_size(ts_size)}")
print(f"parameters: {ts_count}")

results['phases']['tensorstore'] = {
    'save_time_ms': ts_save_time,
    'load_time_ms': ts_load_time,
    'file_size_bytes': ts_size,
    'file_size_gb': ts_size / (1024**3),
    'parameters_saved': ts_count,
    'configuration': {
        'chunk_size_mb': args.chunk_size,
        'concurrency': args.concurrency if args.concurrency else 'default',
        'dtype': DTYPE,
        'compression': args.compression,
        'dynamic_chunking': True
    }
}

# cleanup
del model
del model_state
gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

# save results
results_path = os.path.join(RESULTS_DIR, "all_phases_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("SWEEP RUN COMPLETE")
print(f"{'='*70}")
print(f"\nresults: {results_path}")
print(f"checkpoints: {MODEL_DIR}")
print(f"\n{'='*70}")
