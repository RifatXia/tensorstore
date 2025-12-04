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
parser.add_argument('--clear-cache', action='store_true', help='clear system cache before each operation for accurate timing')
parser.add_argument('--skip-plots', action='store_true', help='skip plot generation')
args = parser.parse_args()

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
print(f"clear cache: {args.clear_cache}")
print("="*70)

# cache clearing helper
def clear_caches_if_enabled():
    if args.clear_cache:
        clear_system_cache()
        if DEVICE == 'cuda':
            clear_gpu_cache()

# create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"\nmodel checkpoints directory: {MODEL_DIR}")
print(f"results directory: {RESULTS_DIR}")

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
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'phases': {}
}

# tensorstore save function
def save_tensorstore(model_state, save_dir, chunk_size_mb=64, concurrency_limit=None):
    """save model using tensorstore with dynamic chunking"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"TENSORSTORE: saving")
    print(f"{'='*70}")
    
    # set context if concurrency is specified
    if concurrency_limit is not None:
        context = ts.Context({'file_io_concurrency': {'limit': concurrency_limit}})
    else:
        context = ts.Context()
    
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # clear cache before save
    clear_caches_if_enabled()
    start_time = time.time()
    saved_count = 0
    
    # dtype conversion based on global DTYPE setting
    # tensorstore supports bfloat16 via ts.bfloat16
    dtype_conversion = {
        'float16': (lambda x: x.detach().cpu().half().numpy(), ts.float16),
        'float32': (lambda x: x.detach().cpu().float().numpy(), ts.float32),
        'bfloat16': (lambda x: x.detach().cpu().to(torch.bfloat16).numpy().view(np.uint16), ts.bfloat16)
    }
    convert_fn, zarr_dtype = dtype_conversion[DTYPE]
    
    for param_name, param_tensor in tqdm(model_state.items(), desc="saving"):
        try:
            param_np = convert_fn(param_tensor)
            safe_name = param_name.replace('.', '_').replace('/', '_')
            param_path = os.path.join(save_dir, f"{safe_name}.zarr")
            
            # dynamic chunking based on parameter shape
            target_elements = chunk_size_bytes // param_np.dtype.itemsize
            chunk_shape = calculate_chunk_shape(list(param_np.shape), target_elements)
            
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': param_path},
                'metadata': {
                    'shape': list(param_np.shape),
                    'dtype': zarr_dtype,
                    'chunks': chunk_shape
                },
                'context': context
            }
            
            dataset = ts.open(spec, create=True, delete_existing=True).result()
            dataset[:] = param_np
            saved_count += 1
            
        except Exception as e:
            print(f"error saving {param_name}: {e}")
    
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

# run tensorstore phase
print(f"\n{'='*70}")
print("PHASE 2: TENSORSTORE (BASIC)")
print(f"{'='*70}")

model_state = model.state_dict()
ts_dir = os.path.join(MODEL_DIR, "tensorstore")

ts_save_time, ts_size, ts_count = save_tensorstore(
    model_state, ts_dir, 
    chunk_size_mb=args.chunk_size, 
    concurrency_limit=args.concurrency
)
ts_load_time = load_tensorstore(ts_dir)

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
        'compression': 'none',
        'dynamic_chunking': True
    }
}

print(f"\n{'='*70}")
print("TENSORSTORE RESULTS")
print(f"{'='*70}")
print(f"save time: {ts_save_time:.1f} ms")
print(f"load time: {ts_load_time:.1f} ms")
print(f"file size: {format_size(ts_size)}")
print(f"parameters: {ts_count}")

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

# generate plots if not skipped
if not args.skip_plots:
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        # create simple bar chart for tensorstore metrics
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        color = '#2E86AB'
        
        # save time
        ax1.bar(['TensorStore'], [ts_save_time], color=color, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Time (ms)', fontweight='bold')
        ax1.set_title('Save Time', fontweight='bold')
        ax1.text(0, ts_save_time, f'{ts_save_time:.0f}ms', ha='center', va='bottom', fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')
        
        # load time
        ax2.bar(['TensorStore'], [ts_load_time], color=color, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Time (ms)', fontweight='bold')
        ax2.set_title('Load Time', fontweight='bold')
        ax2.text(0, ts_load_time, f'{ts_load_time:.0f}ms', ha='center', va='bottom', fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # file size
        size_gb = ts_size / (1024**3)
        ax3.bar(['TensorStore'], [size_gb], color=color, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Size (GB)', fontweight='bold')
        ax3.set_title('File Size', fontweight='bold')
        ax3.text(0, size_gb, f'{size_gb:.2f}GB', ha='center', va='bottom', fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # add overall title
        config_str = f"Chunk: {args.chunk_size}MB, Dtype: {DTYPE}, Concurrency: {args.concurrency if args.concurrency else 'default'}"
        fig.suptitle(f'TensorStore Performance - {config_str}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, "comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ plot saved: {plot_path}")
        
    except Exception as e:
        print(f"warning: could not generate plots: {e}")

print(f"\n{'='*70}")
print("SWEEP RUN COMPLETE")
print(f"{'='*70}")
print(f"\nresults: {results_path}")
if not args.skip_plots:
    print(f"plots: {PLOTS_DIR}")
print(f"checkpoints: {MODEL_DIR}")
print(f"\n{'='*70}")
