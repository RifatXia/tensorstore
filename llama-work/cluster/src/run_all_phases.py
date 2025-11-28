# run all checkpointing phases and generate comparison

import sys
import time
import os
import json
import gc
import torch
import tensorstore as ts
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
from transformers import LlamaForCausalLM

from config import MODEL_NAME, MODEL_ID, DEVICE, SAVED_MODELS_DIR
from utils import calculate_chunk_shape, format_time, format_size, get_directory_size

sys.stdout.reconfigure(line_buffering=True)

# create output directory
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
print("created saved_models directory")

# load model
print(f"\nloading model: {MODEL_NAME}")
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
model = model.to(DEVICE)

print(f"model loaded successfully")
print(f"total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m")
print(f"parameter count: {len(list(model.named_parameters()))}")

# storage for results
results = {}

# ============================================================================
# phase 1: pytorch
# ============================================================================
print("\n" + "="*50)
print("phase 1: pytorch saving")
print("="*50)

pytorch_save_path = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_pytorch.pth")
start_time = time.time()
torch.save(model.state_dict(), pytorch_save_path)
pytorch_save_time = (time.time() - start_time) * 1000
pytorch_file_size = os.path.getsize(pytorch_save_path) / (1024**3)

print(f"pytorch save completed in {pytorch_save_time:.1f} ms")
print(f"file size: {pytorch_file_size:.2f} gb")
print(f"saved to: {pytorch_save_path}")

# pytorch load
print("\n=== phase 1: pytorch loading ===")
start_time = time.time()
state_dict = torch.load(pytorch_save_path, map_location='cpu', weights_only=True)
pytorch_load_time = (time.time() - start_time) * 1000
print(f"pytorch load completed in {pytorch_load_time:.1f} ms")
print(f"loaded {len(state_dict)} parameters successfully")
del state_dict
gc.collect()

results['pytorch'] = {
    'save_time': pytorch_save_time,
    'load_time': pytorch_load_time,
    'file_size': pytorch_file_size
}

# ============================================================================
# phase 2: tensorstore basic
# ============================================================================
print("\n" + "="*50)
print("phase 2: tensorstore saving")
print("="*50)

tensorstore_save_dir = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_tensorstore/")
os.makedirs(tensorstore_save_dir, exist_ok=True)

start_time = time.time()
model_state = model.state_dict()
print(f"processing {len(model_state)} parameters...")

_DESIRED_CHUNK_SIZE_BYTES = 64 * 1024 * 1024
saved_count = 0

for param_name, param_tensor in model_state.items():
    try:
        param_np = param_tensor.detach().cpu().half().numpy()
        safe_name = param_name.replace('.', '_').replace('/', '_')
        
        target_elements = _DESIRED_CHUNK_SIZE_BYTES // param_np.dtype.itemsize
        chunk_shape = calculate_chunk_shape(list(param_np.shape), target_elements)
        
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': f"{tensorstore_save_dir}{safe_name}.zarr"
            },
            'metadata': {
                'shape': list(param_np.shape),
                'dtype': '<f2',
                'chunks': chunk_shape
            }
        }
        
        store = ts.open(spec, create=True, delete_existing=True).result()
        store.write(param_np).result()
        saved_count += 1
    except Exception as e:
        print(f"error saving {param_name}: {e}")
        continue

# save metadata
metadata = {
    'param_names': list(model_state.keys()),
    'total_params': len(model_state)
}
with open(f"{tensorstore_save_dir}metadata.json", 'w') as f:
    json.dump(metadata, f)

tensorstore_save_time = (time.time() - start_time) * 1000
tensorstore_file_size = get_directory_size(tensorstore_save_dir) / (1024**3)

print(f"tensorstore save completed in {tensorstore_save_time:.1f} ms")
print(f"saved {saved_count} parameters")
print(f"total size: {tensorstore_file_size:.2f} gb")

del model_state
gc.collect()

# tensorstore load
print("\n=== phase 2: tensorstore loading ===")
start_time = time.time()

with open(f"{tensorstore_save_dir}metadata.json", 'r') as f:
    metadata = json.load(f)

loaded_state = {}
loaded_count = 0

for param_name in metadata['param_names']:
    try:
        safe_name = param_name.replace('.', '_').replace('/', '_')
        zarr_path = f"{tensorstore_save_dir}{safe_name}.zarr"
        
        if os.path.exists(zarr_path):
            spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': zarr_path
                }
            }
            
            store = ts.open(spec).result()
            param_np = store.read().result()
            loaded_state[param_name] = torch.from_numpy(param_np.copy())
            loaded_count += 1
    except Exception as e:
        print(f"error loading {param_name}: {e}")
        continue

tensorstore_load_time = (time.time() - start_time) * 1000
print(f"tensorstore load completed in {tensorstore_load_time:.1f} ms")
print(f"loaded {loaded_count} parameters successfully")

del loaded_state
gc.collect()

results['tensorstore'] = {
    'save_time': tensorstore_save_time,
    'load_time': tensorstore_load_time,
    'file_size': tensorstore_file_size
}

# ============================================================================
# phase 3: t5x-optimized tensorstore
# ============================================================================
print("\n" + "="*50)
print("phase 3: t5x-optimized tensorstore saving")
print("="*50)

t5x_tensorstore_save_dir = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_t5x_tensorstore/")
os.makedirs(t5x_tensorstore_save_dir, exist_ok=True)

start_time = time.time()
model_state = model.state_dict()
print(f"processing {len(model_state)} parameters with t5x optimizations...")

_TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})
_T5X_DESIRED_CHUNK_SIZE_BYTES = 64 * 1024 * 1024
saved_count = 0

for param_name, param_tensor in model_state.items():
    try:
        param_np = param_tensor.detach().cpu().half().numpy()
        safe_name = param_name.replace('.', '_').replace('/', '_')
        
        target_elements = _T5X_DESIRED_CHUNK_SIZE_BYTES // param_np.dtype.itemsize
        chunk_shape = calculate_chunk_shape(list(param_np.shape), target_elements)
        
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': f"{t5x_tensorstore_save_dir}{safe_name}.zarr"
            },
            'metadata': {
                'shape': list(param_np.shape),
                'dtype': '<f2',
                'chunks': chunk_shape,
                'compressor': {
                    'id': 'gzip',
                    'level': 1
                }
            }
        }
        
        store = ts.open(spec, create=True, delete_existing=True, context=_TS_CONTEXT).result()
        store.write(param_np).result()
        saved_count += 1
    except Exception as e:
        print(f"error saving {param_name}: {e}")
        continue

# save metadata
metadata = {
    'param_names': list(model_state.keys()),
    'total_params': len(model_state)
}
with open(f"{t5x_tensorstore_save_dir}metadata.json", 'w') as f:
    json.dump(metadata, f)

t5x_tensorstore_save_time = (time.time() - start_time) * 1000
t5x_tensorstore_file_size = get_directory_size(t5x_tensorstore_save_dir) / (1024**3)

print(f"t5x-tensorstore save completed in {t5x_tensorstore_save_time:.1f} ms")
print(f"saved {saved_count} parameters successfully")
print(f"total size: {t5x_tensorstore_file_size:.2f} gb")

del model_state
gc.collect()

# t5x-tensorstore load
print("\n=== phase 3: t5x-optimized tensorstore loading ===")
start_time = time.time()

with open(f"{t5x_tensorstore_save_dir}metadata.json", 'r') as f:
    metadata = json.load(f)

loaded_state = {}
loaded_count = 0

for param_name in metadata['param_names']:
    try:
        safe_name = param_name.replace('.', '_').replace('/', '_')
        zarr_path = f"{t5x_tensorstore_save_dir}{safe_name}.zarr"
        
        if os.path.exists(zarr_path):
            spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': zarr_path
                }
            }
            
            store = ts.open(spec, context=_TS_CONTEXT).result()
            param_np = store.read().result()
            loaded_state[param_name] = torch.from_numpy(param_np.copy())
            loaded_count += 1
    except Exception as e:
        print(f"error loading {param_name}: {e}")
        continue

t5x_tensorstore_load_time = (time.time() - start_time) * 1000
print(f"t5x-tensorstore load completed in {t5x_tensorstore_load_time:.1f} ms")
print(f"loaded {loaded_count} parameters successfully")

del loaded_state
gc.collect()

results['t5x'] = {
    'save_time': t5x_tensorstore_save_time,
    'load_time': t5x_tensorstore_load_time,
    'file_size': t5x_tensorstore_file_size
}

# ============================================================================
# save results and generate plots
# ============================================================================
print("\n" + "="*50)
print("generating comparison plots")
print("="*50)

# save results to json
results_path = os.path.join(SAVED_MODELS_DIR, "comparison_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"results saved to: {results_path}")

# prepare data for plotting
methods = ['pytorch', 'tensorstore', 't5x']
save_times = [results[m]['save_time'] for m in methods]
load_times = [results[m]['load_time'] for m in methods]
file_sizes = [results[m]['file_size'] for m in methods]
colors = ['blue', 'orange', 'green']

# create 3-way comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. save time
ax1.bar(range(len(methods)), save_times, color=colors)
ax1.set_title("save time comparison", fontsize=14, fontweight="bold")
ax1.set_ylabel("time (ms)", fontsize=12)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=11)
ax1.grid(axis="y", alpha=0.3)
for i, v in enumerate(save_times):
    ax1.text(i, v*1.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

# 2. load time
ax2.bar(range(len(methods)), load_times, color=colors)
ax2.set_title("load time comparison", fontsize=14, fontweight="bold")
ax2.set_ylabel("time (ms)", fontsize=12)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=11)
ax2.grid(axis="y", alpha=0.3)
for i, v in enumerate(load_times):
    ax2.text(i, v*1.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

# 3. file size
ax3.bar(range(len(methods)), file_sizes, color=colors)
ax3.set_title("file size comparison", fontsize=14, fontweight="bold")
ax3.set_ylabel("size (gb)", fontsize=12)
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, fontsize=11)
ax3.grid(axis="y", alpha=0.3)
for i, v in enumerate(file_sizes):
    ax3.text(i, v*1.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

# 4. combined efficiency
total_times = [s + l for s, l in zip(save_times, load_times)]
best_total = min(total_times)
efficiency = [(best_total / t) * 100 for t in total_times]
ax4.bar(range(len(methods)), efficiency, color=colors)
ax4.set_title("overall efficiency (higher is better)", fontsize=14, fontweight="bold")
ax4.set_ylabel("efficiency %", fontsize=12)
ax4.set_xticks(range(len(methods)))
ax4.set_xticklabels(methods, fontsize=11)
ax4.grid(axis="y", alpha=0.3)
for i, v in enumerate(efficiency):
    ax4.text(i, v*1.02, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plot_path = os.path.join(SAVED_MODELS_DIR, "3way_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"✓ comparison chart saved to: {plot_path}")

print("\n" + "="*50)
print("all phases complete!")
print("="*50)
