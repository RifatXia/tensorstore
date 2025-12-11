#!/usr/bin/env python3
"""
generate side-by-side comparison plots for multiple models
compares pytorch, tensorstore, and t5x phases across different models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# data directory
data_dir = Path("plot2")

# load all phase comparison data
models_data = {}
for result_dir in data_dir.iterdir():
    if result_dir.is_dir() and result_dir.name.startswith("202512"):
        results_file = result_dir / "all_phases_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                model_id = data['model_id']
                
                # store by model id
                if model_id not in models_data:
                    models_data[model_id] = []
                models_data[model_id].append(data)

print(f"loaded {len(models_data)} models")
for model_id, runs in models_data.items():
    print(f"  {model_id}: {len(runs)} run(s)")

# use the latest run for each model
latest_runs = {}
for model_id, runs in models_data.items():
    # sort by timestamp and take the latest
    latest_runs[model_id] = sorted(runs, key=lambda x: x['timestamp'])[-1]

models = list(latest_runs.keys())  # use all available models
print(f"\ncomparing {len(models)} models: {models}")

# ============================================================================
# combined plot: all models, all phases (3 subplots: save, load, file size)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('3-Phase Checkpointing: Model Comparison', fontsize=16, fontweight='bold', y=0.98)

phase_names = ['pytorch', 'tensorstore', 't5x']
phase_labels = ['PyTorch', 'TensorStore', 'T5X']
model_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # 4 colors for 4 models

# prepare data for all models
all_save_times = {phase: [] for phase in phase_names}
all_save_stds = {phase: [] for phase in phase_names}
all_load_times = {phase: [] for phase in phase_names}
all_load_stds = {phase: [] for phase in phase_names}
all_file_sizes = {phase: [] for phase in phase_names}

for model_id in models:
    data = latest_runs[model_id]
    for phase in phase_names:
        if phase in data['phases']:
            all_save_times[phase].append(data['phases'][phase]['save_time_ms']['mean'] / 1000)  # convert to seconds
            all_save_stds[phase].append(data['phases'][phase]['save_time_ms']['std'] / 1000)
            all_load_times[phase].append(data['phases'][phase]['load_time_ms']['mean'] / 1000)
            all_load_stds[phase].append(data['phases'][phase]['load_time_ms']['std'] / 1000)
            all_file_sizes[phase].append(data['phases'][phase]['file_size_gb'])
        else:
            all_save_times[phase].append(0)
            all_save_stds[phase].append(0)
            all_load_times[phase].append(0)
            all_load_stds[phase].append(0)
            all_file_sizes[phase].append(0)

# subplot 1: save time
ax_save = axes[0]
x_pos = np.arange(len(phase_labels))
width = 0.2
for idx, model_id in enumerate(models):
    save_times = [all_save_times[phase][idx] for phase in phase_names]
    
    bars = ax_save.bar(x_pos + (idx - len(models)/2 + 0.5) * width, save_times, width,
                       label=model_id, color=model_colors[idx], alpha=0.8)
    
    # add value labels on bars
    for bar, mean in zip(bars, save_times):
        height = bar.get_height()
        if height > 0:
            ax_save.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax_save.set_title('save time comparison', fontsize=14, fontweight='bold')
ax_save.set_xlabel('checkpointing method', fontsize=13, fontweight='bold')
ax_save.set_ylabel('time (seconds)', fontsize=13, fontweight='bold')
ax_save.set_xticks(x_pos)
ax_save.set_xticklabels(phase_labels, fontsize=11, fontweight='bold')
ax_save.tick_params(axis='y', labelsize=11)
for label in ax_save.get_yticklabels():
    label.set_fontweight('bold')
ax_save.legend(fontsize=9, loc='upper left')
ax_save.grid(axis='y', alpha=0.3)

# subplot 2: load time
ax_load = axes[1]
for idx, model_id in enumerate(models):
    load_times = [all_load_times[phase][idx] for phase in phase_names]
    
    bars = ax_load.bar(x_pos + (idx - len(models)/2 + 0.5) * width, load_times, width,
                       label=model_id, color=model_colors[idx], alpha=0.8)
    
    # add value labels on bars
    for bar, mean in zip(bars, load_times):
        height = bar.get_height()
        if height > 0:
            ax_load.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax_load.set_title('load time comparison', fontsize=14, fontweight='bold')
ax_load.set_xlabel('checkpointing method', fontsize=13, fontweight='bold')
ax_load.set_ylabel('time (seconds)', fontsize=13, fontweight='bold')
ax_load.set_xticks(x_pos)
ax_load.set_xticklabels(phase_labels, fontsize=11, fontweight='bold')
ax_load.tick_params(axis='y', labelsize=11)
for label in ax_load.get_yticklabels():
    label.set_fontweight('bold')
ax_load.legend(fontsize=9, loc='upper left')
ax_load.grid(axis='y', alpha=0.3)

# subplot 3: file size
ax_size = axes[2]
for idx, model_id in enumerate(models):
    file_sizes = [all_file_sizes[phase][idx] for phase in phase_names]
    
    bars = ax_size.bar(x_pos + (idx - len(models)/2 + 0.5) * width, file_sizes, width,
                       label=model_id, color=model_colors[idx], alpha=0.8)
    
    # add value labels on bars
    for bar, size in zip(bars, file_sizes):
        height = bar.get_height()
        if height > 0:
            ax_size.text(bar.get_x() + bar.get_width()/2., height,
                        f'{size:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax_size.set_title('file size comparison', fontsize=14, fontweight='bold')
ax_size.set_xlabel('checkpointing method', fontsize=13, fontweight='bold')
ax_size.set_ylabel('file size (gb)', fontsize=13, fontweight='bold')
ax_size.set_xticks(x_pos)
ax_size.set_xticklabels(phase_labels, fontsize=11, fontweight='bold')
ax_size.tick_params(axis='y', labelsize=11)
for label in ax_size.get_yticklabels():
    label.set_fontweight('bold')
ax_size.legend(fontsize=9, loc='upper left')
ax_size.grid(axis='y', alpha=0.3)

# add configuration metadata text box
num_runs = latest_runs[models[0]].get('num_runs', 3)
config_text = (
    f"Configuration (constant across all runs):\n"
    f"• Runs per phase: {num_runs}\n"
    f"• Device: CPU\n"
    f"• Dtype: float16\n"
    f"• PyTorch: no compression | TensorStore: no compression | T5X: gzip compression"
)
fig.text(0.5, -0.02, config_text, ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
         family='monospace', transform=fig.transFigure)

plt.tight_layout(rect=[0, 0.12, 1, 0.96])
output_file = data_dir / "combined_phase_comparison.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ combined phase comparison saved: {output_file}")
plt.close()

print("\n" + "="*70)
print("✓ combined phase comparison plot generated successfully!")
print("="*70)
print(f"\noutput file:")
print(f"  - {data_dir}/combined_phase_comparison.png")
print(f"\nmodels compared: {', '.join(models)}")
print(f"phases: {', '.join(phase_labels)}")
