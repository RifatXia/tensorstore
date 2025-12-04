#!/usr/bin/env python3
# compare results from parameter sweep experiments

import sys
import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description='compare sweep results')
parser.add_argument('--sweep-id', type=str, required=True, help='sweep identifier')
parser.add_argument('--sweep-param', type=str, required=True, choices=['chunk', 'dtype'], help='parameter that was swept')
parser.add_argument('--sweep-values', type=str, required=True, help='comma-separated values')
args = parser.parse_args()

# parse sweep values
sweep_values = args.sweep_values.split(',')

print("=" * 70)
print(f"parameter sweep comparison: {args.sweep_param}")
print(f"sweep id: {args.sweep_id}")
print(f"values: {sweep_values}")
print("=" * 70)

# collect results from all runs
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_base = os.path.join(base_dir, "results")

sweep_results = []
for value in sweep_values:
    run_id = f"{args.sweep_id}_{value}"
    results_file = os.path.join(results_base, run_id, "all_phases_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            sweep_results.append({
                'value': value,
                'data': data
            })
        print(f"✓ loaded results for {args.sweep_param}={value}")
    else:
        print(f"✗ missing results for {args.sweep_param}={value}")

if not sweep_results:
    print("error: no results found")
    sys.exit(1)

# extract metrics for each phase
phases = ['pytorch', 'tensorstore', 't5x', 'phase4a', 'phase4b', 'phase4c']
phase_labels = ['PyTorch', 'TensorStore', 'T5X', 'Concurrency', 'Chunks', 'Compression']

# prepare data for plotting
param_values = [r['value'] for r in sweep_results]
save_times = {phase: [] for phase in phases}
load_times = {phase: [] for phase in phases}
file_sizes = {phase: [] for phase in phases}

for result in sweep_results:
    for phase in phases:
        if phase in result['data']['phases']:
            save_times[phase].append(result['data']['phases'][phase]['save_time_ms'])
            load_times[phase].append(result['data']['phases'][phase]['load_time_ms'])
            file_sizes[phase].append(result['data']['phases'][phase]['file_size_gb'])
        else:
            save_times[phase].append(0)
            load_times[phase].append(0)
            file_sizes[phase].append(0)

# create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Parameter Sweep: {args.sweep_param.upper()} Comparison\nModel: {sweep_results[0]["data"]["model_name"]}', 
             fontsize=16, fontweight='bold')

# x-axis setup
x = np.arange(len(param_values))
width = 0.12
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

# plot 1: save time comparison
ax1 = axes[0, 0]
for i, (phase, label) in enumerate(zip(phases, phase_labels)):
    if any(save_times[phase]):
        ax1.bar(x + i*width, save_times[phase], width, label=label, color=colors[i])
ax1.set_xlabel(f'{args.sweep_param.capitalize()} Value', fontweight='bold')
ax1.set_ylabel('Save Time (ms)', fontweight='bold')
ax1.set_title('Save Time Comparison', fontweight='bold')
ax1.set_xticks(x + width * 2.5)
ax1.set_xticklabels(param_values)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# plot 2: load time comparison
ax2 = axes[0, 1]
for i, (phase, label) in enumerate(zip(phases, phase_labels)):
    if any(load_times[phase]):
        ax2.bar(x + i*width, load_times[phase], width, label=label, color=colors[i])
ax2.set_xlabel(f'{args.sweep_param.capitalize()} Value', fontweight='bold')
ax2.set_ylabel('Load Time (ms)', fontweight='bold')
ax2.set_title('Load Time Comparison', fontweight='bold')
ax2.set_xticks(x + width * 2.5)
ax2.set_xticklabels(param_values)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# plot 3: file size comparison
ax3 = axes[1, 0]
for i, (phase, label) in enumerate(zip(phases, phase_labels)):
    if any(file_sizes[phase]):
        ax3.bar(x + i*width, file_sizes[phase], width, label=label, color=colors[i])
ax3.set_xlabel(f'{args.sweep_param.capitalize()} Value', fontweight='bold')
ax3.set_ylabel('File Size (GB)', fontweight='bold')
ax3.set_title('File Size Comparison', fontweight='bold')
ax3.set_xticks(x + width * 2.5)
ax3.set_xticklabels(param_values)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# plot 4: speedup comparison (relative to first value)
ax4 = axes[1, 1]
for i, (phase, label) in enumerate(zip(phases, phase_labels)):
    if any(save_times[phase]) and save_times[phase][0] > 0:
        speedups = [save_times[phase][0] / t if t > 0 else 0 for t in save_times[phase]]
        ax4.plot(param_values, speedups, marker='o', label=label, color=colors[i], linewidth=2)
ax4.set_xlabel(f'{args.sweep_param.capitalize()} Value', fontweight='bold')
ax4.set_ylabel('Speedup (relative to first)', fontweight='bold')
ax4.set_title('Save Time Speedup', fontweight='bold')
ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()

# save plot
sweep_dir = os.path.join(results_base, args.sweep_id)
os.makedirs(sweep_dir, exist_ok=True)
plot_path = os.path.join(sweep_dir, "sweep_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ comparison plot saved: {plot_path}")

# save summary json
summary = {
    'sweep_param': args.sweep_param,
    'sweep_values': param_values,
    'model_name': sweep_results[0]['data']['model_name'],
    'results': {}
}

for phase in phases:
    if any(save_times[phase]):
        summary['results'][phase] = {
            'save_times_ms': save_times[phase],
            'load_times_ms': load_times[phase],
            'file_sizes_gb': file_sizes[phase]
        }

summary_path = os.path.join(sweep_dir, "sweep_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ summary saved: {summary_path}")

print("\n" + "=" * 70)
print("sweep comparison complete!")
print("=" * 70)
