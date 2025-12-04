#!/usr/bin/env python3
# run all 6 checkpointing phases and generate comprehensive 6-way comparison

import sys
import time
import os
import json
import gc
import argparse
import torch
import tensorstore as ts
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from utils import calculate_chunk_shape, format_size, get_directory_size, get_model_default_dtype, clear_system_cache, clear_gpu_cache

sys.stdout.reconfigure(line_buffering=True)

# parse command line arguments
parser = argparse.ArgumentParser(description='run checkpointing phases with configurable options')
parser.add_argument('--model', type=str, default=None, help='huggingface model name')
parser.add_argument('--phases', type=str, default='1,2,3,4a,4b,4c', help='comma-separated phases to run (e.g., 1,2,3 or 1,4a,4c)')
parser.add_argument('--chunk-size', type=int, default=64, help='chunk size in megabytes (default: 64)')
parser.add_argument('--concurrency', type=int, default=None, help='tensorstore concurrency limit (default: tensorstore default, unlimited)')
parser.add_argument('--device', type=str, default='cpu', help='device to use (default: cpu)')
parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float16', 'float32', 'bfloat16'], help='data type for model and storage (default: auto - uses model default)')
parser.add_argument('--skip-plots', action='store_true', help='skip plot generation')
parser.add_argument('--clear-cache', action='store_true', help='clear system cache before each operation for accurate timing')
args = parser.parse_args()

# update config with command line args
if args.model:
    os.environ['MODEL_NAME'] = args.model
if args.chunk_size:
    os.environ['CHUNK_SIZE_MB'] = str(args.chunk_size)
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

# parse which phases to run
phases_to_run = set(args.phases.split(','))
print(f"phases to run: {sorted(phases_to_run)}")

print("="*70)
print(f"6-WAY CHECKPOINTING COMPARISON: {MODEL_NAME}")
print(f"model type: {MODEL_TYPE}")
print(f"run id: {RUN_ID}")
print(f"dtype: {DTYPE}")
print(f"clear cache: {args.clear_cache}")
print("="*70)

# helper function for cache clearing
def clear_caches_if_enabled():
    """clear system and gpu caches if enabled"""
    if args.clear_cache:
        clear_system_cache()
        if DEVICE == 'cuda':
            clear_gpu_cache()

# create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"\nmodel checkpoints directory: {MODEL_DIR}")
print(f"results directory: {RESULTS_DIR}")
print(f"plots directory: {PLOTS_DIR}")

# load model
print(f"\nloading model...")
print(f"using cache: {os.environ.get('HF_HOME', 'default')}")

# force offline mode to avoid internet access on compute nodes
# use AutoModelForCausalLM for universal support (llama, qwen, mistral, etc.)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
    local_files_only=True,  # critical: prevents internet access
    trust_remote_code=True,  # required for qwen and some other models
    token=HF_TOKEN  # for private/gated models
)
model = model.to(DEVICE)
print(f"✓ model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m parameters")

# storage for results with configuration metadata
results = {
    'model_name': MODEL_NAME,
    'model_id': MODEL_ID,
    'run_id': RUN_ID,
    'model_type': MODEL_TYPE,
    'device': DEVICE,
    'dtype': DTYPE,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'phases': {}
}

# helper function for tensorstore variants
def save_tensorstore_variant(model_state, save_dir, phase_name, use_compression=False, 
                             use_concurrency=False, chunk_size_mb=64, concurrency_limit=None):
    """save model using tensorstore with specific configuration"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"{phase_name}: saving")
    print(f"{'='*70}")
    
    # only set context if concurrency is enabled AND a limit is specified
    if use_concurrency and concurrency_limit is not None:
        context = ts.Context({'file_io_concurrency': {'limit': concurrency_limit}})
    elif use_concurrency:
        context = ts.Context()  # use tensorstore default
    else:
        context = None
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # clear cache before save
    clear_caches_if_enabled()
    start_time = time.time()
    saved_count = 0
    
    # dtype conversion based on global DTYPE setting
    dtype_conversion = {
        'float16': (lambda x: x.detach().cpu().half().numpy(), '<f2'),
        'float32': (lambda x: x.detach().cpu().float().numpy(), '<f4'),
        'bfloat16': (lambda x: x.detach().cpu().to(torch.bfloat16).numpy(), '<f2')
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
            
            if use_compression:
                spec['metadata']['compressor'] = {'id': 'gzip', 'level': 1}
            
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
    
    # return configuration details along with metrics
    config = {
        'chunk_size_mb': chunk_size_mb,
        'compression': 'gzip-1' if use_compression else 'none',
        'concurrency': concurrency_limit if (use_concurrency and concurrency_limit) else ('default' if use_concurrency else 1),
        'dtype': DTYPE,
        'parameters_saved': saved_count
    }
    
    return save_time, dir_size, config

def load_tensorstore_variant(save_dir, phase_name):
    """load model from tensorstore variant"""
    print(f"\n{phase_name}: loading")
    
    # clear cache before load
    clear_caches_if_enabled()
    start_time = time.time()
    loaded_count = 0
    
    for file in tqdm(os.listdir(save_dir), desc="loading"):
        if file.endswith('.zarr'):
            param_path = os.path.join(save_dir, file)
            spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': param_path}}
            store = ts.open(spec, open=True).result()
            _ = store.read().result()
            loaded_count += 1
    
    load_time = (time.time() - start_time) * 1000
    print(f"✓ loaded {loaded_count} parameters in {load_time:.1f} ms")
    
    return load_time

# ============================================================================
# PHASE 1: PYTORCH
# ============================================================================
if '1' in phases_to_run:
    print(f"\n{'='*70}")
    print("PHASE 1: PYTORCH (BASELINE)")
    print(f"{'='*70}")

    pytorch_path = os.path.join(MODEL_DIR, "pytorch.pth")
    
    # clear cache before save
    clear_caches_if_enabled()
    start_time = time.time()
    torch.save(model.state_dict(), pytorch_path)
    pytorch_save_time = (time.time() - start_time) * 1000
    pytorch_size = os.path.getsize(pytorch_path)

    print(f"✓ saved in {pytorch_save_time:.1f} ms")
    print(f"✓ size: {format_size(pytorch_size)}")

    # clear cache before load
    clear_caches_if_enabled()
    start_time = time.time()
    state_dict = torch.load(pytorch_path, map_location='cpu', weights_only=True)
    pytorch_load_time = (time.time() - start_time) * 1000
    print(f"✓ loaded in {pytorch_load_time:.1f} ms")

    results['phases']['pytorch'] = {
        'save_time_ms': pytorch_save_time,
        'load_time_ms': pytorch_load_time,
        'file_size_bytes': pytorch_size,
        'file_size_gb': pytorch_size / (1024**3),
        'configuration': {
            'method': 'torch.save',
            'dtype': DTYPE,
            'compression': 'none',
            'format': 'pytorch'
        }
    }

    del state_dict
    gc.collect()
else:
    print(f"\n{'='*70}")
    print("PHASE 1: PYTORCH (SKIPPED)")
    print(f"{'='*70}")

# ============================================================================
# PHASE 2: TENSORSTORE (BASIC)
# ============================================================================
model_state = model.state_dict()

if '2' in phases_to_run:
    ts_dir = os.path.join(MODEL_DIR, "tensorstore")
    ts_save_time, ts_size, ts_config = save_tensorstore_variant(
        model_state, ts_dir, "PHASE 2: TENSORSTORE (BASIC)",
        use_compression=False, use_concurrency=False, chunk_size_mb=args.chunk_size, concurrency_limit=args.concurrency
    )
    ts_load_time = load_tensorstore_variant(ts_dir, "PHASE 2")

    results['phases']['tensorstore'] = {
        'save_time_ms': ts_save_time,
        'load_time_ms': ts_load_time,
        'file_size_bytes': ts_size,
        'file_size_gb': ts_size / (1024**3),
        'configuration': ts_config
    }
    gc.collect()
else:
    print(f"\n{'='*70}")
    print("PHASE 2: TENSORSTORE (SKIPPED)")
    print(f"{'='*70}")

# ============================================================================
# PHASE 3: T5X-OPTIMIZED
# ============================================================================
if '3' in phases_to_run:
    t5x_dir = os.path.join(MODEL_DIR, "t5x_tensorstore")
    t5x_save_time, t5x_size, t5x_config = save_tensorstore_variant(
        model_state, t5x_dir, "PHASE 3: T5X-OPTIMIZED",
        use_compression=True, use_concurrency=True, chunk_size_mb=args.chunk_size, concurrency_limit=args.concurrency
    )
    t5x_load_time = load_tensorstore_variant(t5x_dir, "PHASE 3")

    results['phases']['t5x'] = {
        'save_time_ms': t5x_save_time,
        'load_time_ms': t5x_load_time,
        'file_size_bytes': t5x_size,
        'file_size_gb': t5x_size / (1024**3),
        'configuration': t5x_config
    }
    gc.collect()
else:
    print(f"\n{'='*70}")
    print("PHASE 3: T5X-OPTIMIZED (SKIPPED)")
    print(f"{'='*70}")

# ============================================================================
# PHASE 4A: CONCURRENCY ONLY
# ============================================================================
if '4a' in phases_to_run:
    p4a_dir = os.path.join(MODEL_DIR, "phase4a_concurrency")
    p4a_save_time, p4a_size, p4a_config = save_tensorstore_variant(
        model_state, p4a_dir, "PHASE 4A: CONCURRENCY ONLY",
        use_compression=False, use_concurrency=True, chunk_size_mb=args.chunk_size, concurrency_limit=args.concurrency
    )
    p4a_load_time = load_tensorstore_variant(p4a_dir, "PHASE 4A")

    results['phases']['phase4a_concurrency'] = {
        'save_time_ms': p4a_save_time,
        'load_time_ms': p4a_load_time,
        'file_size_bytes': p4a_size,
        'file_size_gb': p4a_size / (1024**3),
        'configuration': p4a_config
    }
    gc.collect()
else:
    print(f"\n{'='*70}")
    print("PHASE 4A: CONCURRENCY ONLY (SKIPPED)")
    print(f"{'='*70}")

# ============================================================================
# PHASE 4B: 1 MIB CHUNKS
# ============================================================================
if '4b' in phases_to_run:
    p4b_dir = os.path.join(MODEL_DIR, "phase4b_chunks")
    p4b_save_time, p4b_size, p4b_config = save_tensorstore_variant(
        model_state, p4b_dir, "PHASE 4B: 1 MIB CHUNKS",
        use_compression=False, use_concurrency=False, chunk_size_mb=1, concurrency_limit=args.concurrency
    )
    p4b_load_time = load_tensorstore_variant(p4b_dir, "PHASE 4B")

    results['phases']['phase4b_chunks'] = {
        'save_time_ms': p4b_save_time,
        'load_time_ms': p4b_load_time,
        'file_size_bytes': p4b_size,
        'file_size_gb': p4b_size / (1024**3),
        'configuration': p4b_config
    }
    gc.collect()
else:
    print(f"\n{'='*70}")
    print("PHASE 4B: 1 MIB CHUNKS (SKIPPED)")
    print(f"{'='*70}")

# ============================================================================
# PHASE 4C: COMPRESSION ONLY
# ============================================================================
if '4c' in phases_to_run:
    p4c_dir = os.path.join(MODEL_DIR, "phase4c_compression")
    p4c_save_time, p4c_size, p4c_config = save_tensorstore_variant(
        model_state, p4c_dir, "PHASE 4C: COMPRESSION ONLY",
        use_compression=True, use_concurrency=False, chunk_size_mb=args.chunk_size, concurrency_limit=args.concurrency
    )
    p4c_load_time = load_tensorstore_variant(p4c_dir, "PHASE 4C")

    results['phases']['phase4c_compression'] = {
        'save_time_ms': p4c_save_time,
        'load_time_ms': p4c_load_time,
        'file_size_bytes': p4c_size,
        'file_size_gb': p4c_size / (1024**3),
        'configuration': p4c_config
    }
    gc.collect()
else:
    print(f"\n{'='*70}")
    print("PHASE 4C: COMPRESSION ONLY (SKIPPED)")
    print(f"{'='*70}")

del model_state
gc.collect()

# save results
results_path = os.path.join(RESULTS_DIR, "all_phases_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

# prepare data for visualization
methods = ['pytorch', 'tensorstore(ts)', 't5x', 'ts+concurrency', 'ts+1mib', 'ts+compression']
save_times = [
    results['phases']['pytorch']['save_time_ms'],
    results['phases']['tensorstore']['save_time_ms'],
    results['phases']['t5x']['save_time_ms'],
    results['phases']['phase4a_concurrency']['save_time_ms'],
    results['phases']['phase4b_chunks']['save_time_ms'],
    results['phases']['phase4c_compression']['save_time_ms']
]
load_times = [
    results['phases']['pytorch']['load_time_ms'],
    results['phases']['tensorstore']['load_time_ms'],
    results['phases']['t5x']['load_time_ms'],
    results['phases']['phase4a_concurrency']['load_time_ms'],
    results['phases']['phase4b_chunks']['load_time_ms'],
    results['phases']['phase4c_compression']['load_time_ms']
]
file_sizes = [
    results['phases']['pytorch']['file_size_gb'],
    results['phases']['tensorstore']['file_size_gb'],
    results['phases']['t5x']['file_size_gb'],
    results['phases']['phase4a_concurrency']['file_size_gb'],
    results['phases']['phase4b_chunks']['file_size_gb'],
    results['phases']['phase4c_compression']['file_size_gb']
]

# print summary
print(f"\n{'method':<20} {'save(ms)':<12} {'load(ms)':<12} {'size(gb)':<10}")
print("-" * 54)
for i, method in enumerate(methods):
    print(f"{method:<20} {save_times[i]:<12.1f} {load_times[i]:<12.1f} {file_sizes[i]:<10.2f}")

# ============================================================================
# GENERATE 6-WAY COMPARISON CHART
# ============================================================================
if not args.skip_plots and len(results['phases']) > 0:
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'6-Way Checkpointing Comparison - {MODEL_ID}', fontsize=18, fontweight='bold')

# 1. save time
ax = axes[0, 0]
ax.bar(range(len(methods)), save_times, color=colors)
ax.set_title("save time (lower is better)", fontsize=14, fontweight='bold')
ax.set_ylabel("time (ms)", fontsize=12)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(save_times):
    ax.text(i, v*1.02, f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")

# 2. load time
ax = axes[0, 1]
ax.bar(range(len(methods)), load_times, color=colors)
ax.set_title("load time (lower is better)", fontsize=14, fontweight='bold')
ax.set_ylabel("time (ms)", fontsize=12)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(load_times):
    ax.text(i, v*1.02, f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")

# 3. file size
ax = axes[0, 2]
ax.bar(range(len(methods)), file_sizes, color=colors)
ax.set_title("file size (lower is better)", fontsize=14, fontweight='bold')
ax.set_ylabel("size (gb)", fontsize=12)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(file_sizes):
    ax.text(i, v*1.02, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

# 4. save speedup vs pytorch
ax = axes[1, 0]
pytorch_save = save_times[0]
speedup_save = [(pytorch_save / t) for t in save_times]
ax.bar(range(len(methods)), speedup_save, color=colors)
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="pytorch baseline")
ax.set_title("save speedup vs pytorch (higher is better)", fontsize=14, fontweight='bold')
ax.set_ylabel("speedup factor", fontsize=12)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.legend()
for i, v in enumerate(speedup_save):
    ax.text(i, v*1.02, f"{v:.2f}x", ha="center", fontsize=8, fontweight="bold")

# 5. load speedup vs pytorch
ax = axes[1, 1]
pytorch_load = load_times[0]
speedup_load = [(pytorch_load / t) for t in load_times]
ax.bar(range(len(methods)), speedup_load, color=colors)
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="pytorch baseline")
ax.set_title("load speedup vs pytorch (higher is better)", fontsize=14, fontweight='bold')
ax.set_ylabel("speedup factor", fontsize=12)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.legend()
for i, v in enumerate(speedup_load):
    ax.text(i, v*1.02, f"{v:.2f}x", ha="center", fontsize=8, fontweight="bold")

# 6. efficiency score
ax = axes[1, 2]
total_times = [s + l for s, l in zip(save_times, load_times)]
best_total = min(total_times)
efficiency = [(best_total / t) * 100 for t in total_times]
ax.bar(range(len(methods)), efficiency, color=colors)
ax.set_title("overall efficiency score (higher is better)", fontsize=14, fontweight='bold')
ax.set_ylabel("efficiency %", fontsize=12)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(efficiency):
    ax.text(i, v*1.02, f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "6way_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ 6-way comparison chart saved: {plot_path}")

# ============================================================================
# GENERATE TENSORSTORE VARIANTS CHART
# ============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle(f'TensorStore Variants Comparison - {MODEL_ID}', fontsize=18, fontweight='bold')

ts_methods = methods[1:]
ts_save = save_times[1:]
ts_load = load_times[1:]
ts_sizes = file_sizes[1:]
ts_colors = colors[1:]

# save time
axes2[0, 0].bar(range(len(ts_methods)), ts_save, color=ts_colors)
axes2[0, 0].set_title("save time", fontsize=14, fontweight='bold')
axes2[0, 0].set_ylabel("time (ms)", fontsize=12)
axes2[0, 0].set_xticks(range(len(ts_methods)))
axes2[0, 0].set_xticklabels(ts_methods, rotation=45, ha="right", fontsize=9)
axes2[0, 0].grid(axis="y", alpha=0.3)
for i, v in enumerate(ts_save):
    axes2[0, 0].text(i, v*1.02, f"{v:.0f}", ha="center", fontsize=8)

# load time
axes2[0, 1].bar(range(len(ts_methods)), ts_load, color=ts_colors)
axes2[0, 1].set_title("load time", fontsize=14, fontweight='bold')
axes2[0, 1].set_ylabel("time (ms)", fontsize=12)
axes2[0, 1].set_xticks(range(len(ts_methods)))
axes2[0, 1].set_xticklabels(ts_methods, rotation=45, ha="right", fontsize=9)
axes2[0, 1].grid(axis="y", alpha=0.3)
for i, v in enumerate(ts_load):
    axes2[0, 1].text(i, v*1.02, f"{v:.0f}", ha="center", fontsize=8)

# file size
axes2[1, 0].bar(range(len(ts_methods)), ts_sizes, color=ts_colors)
axes2[1, 0].set_title("file size", fontsize=14, fontweight='bold')
axes2[1, 0].set_ylabel("size (gb)", fontsize=12)
axes2[1, 0].set_xticks(range(len(ts_methods)))
axes2[1, 0].set_xticklabels(ts_methods, rotation=45, ha="right", fontsize=9)
axes2[1, 0].grid(axis="y", alpha=0.3)
for i, v in enumerate(ts_sizes):
    axes2[1, 0].text(i, v*1.02, f"{v:.2f}", ha="center", fontsize=8)

# improvement vs basic tensorstore
baseline_save = ts_save[0]
baseline_load = ts_load[0]
improvements_save = [((baseline_save - s) / baseline_save * 100) for s in ts_save[1:]]
improvements_load = [((baseline_load - l) / baseline_load * 100) for l in ts_load[1:]]
opt_methods = ts_methods[1:]

x_pos = np.arange(len(opt_methods))
width = 0.35
axes2[1, 1].bar(x_pos - width/2, improvements_save, width, label="save time", color="lightcoral", alpha=0.8)
axes2[1, 1].bar(x_pos + width/2, improvements_load, width, label="load time", color="lightblue", alpha=0.8)
axes2[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.5)
axes2[1, 1].set_title("improvement vs basic tensorstore", fontsize=14, fontweight='bold')
axes2[1, 1].set_ylabel("improvement %", fontsize=12)
axes2[1, 1].set_xticks(x_pos)
axes2[1, 1].set_xticklabels(opt_methods, rotation=45, ha="right", fontsize=9)
axes2[1, 1].legend()
axes2[1, 1].grid(axis="y", alpha=0.3)
for i, (s, l) in enumerate(zip(improvements_save, improvements_load)):
    axes2[1, 1].text(i - width/2, s + (2 if s > 0 else -2), f"{s:+.0f}%", ha="center", fontsize=8)
    axes2[1, 1].text(i + width/2, l + (2 if l > 0 else -2), f"{l:+.0f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, "tensorstore_variants.png")
    plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ tensorstore variants chart saved: {plot_path2}")
else:
    if args.skip_plots:
        print(f"\n{'='*70}")
        print("VISUALIZATIONS SKIPPED (--skip-plots flag)")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("VISUALIZATIONS SKIPPED (no phases run)")
        print(f"{'='*70}")

print(f"\n{'='*70}")
print("✓ ALL PHASES COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"\nresults: {results_path}")
if not args.skip_plots:
    print(f"plots: {PLOTS_DIR}")
print(f"checkpoints: {MODEL_DIR}")
