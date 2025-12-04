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
parser.add_argument('--sweep-param', type=str, required=True, choices=['chunk', 'dtype', 'concurrency'], help='parameter that was swept')
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

# extract metrics for tensorstore phase only
phases = ['tensorstore']
phase_labels = ['TensorStore']

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

# x-axis setup
x = np.arange(len(param_values))
width = 0.35  # single color for tensorstore

# create 3-panel comparison plot for tensorstore
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
color = '#2E86AB'  # single color for tensorstore

# plot 1: save time
if any(save_times['tensorstore']):
    ax1.bar(x, save_times['tensorstore'], width, color=color, alpha=0.7, edgecolor='black')
    ax1.set_xlabel(f'{args.sweep_param.capitalize()} Value', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Save Time (ms)', fontweight='bold', fontsize=12)
    ax1.set_title('TensorStore Save Time', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_values)
    ax1.grid(alpha=0.3, axis='y')
    # add value labels on bars
    for i, v in enumerate(save_times['tensorstore']):
        ax1.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

# plot 2: load time
if any(load_times['tensorstore']):
    ax2.bar(x, load_times['tensorstore'], width, color=color, alpha=0.7, edgecolor='black')
    ax2.set_xlabel(f'{args.sweep_param.capitalize()} Value', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Load Time (ms)', fontweight='bold', fontsize=12)
    ax2.set_title('TensorStore Load Time', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_values)
    ax2.grid(alpha=0.3, axis='y')
    # add value labels on bars
    for i, v in enumerate(load_times['tensorstore']):
        ax2.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

# plot 3: file size
if any(file_sizes['tensorstore']):
    ax3.bar(x, file_sizes['tensorstore'], width, color=color, alpha=0.7, edgecolor='black')
    ax3.set_xlabel(f'{args.sweep_param.capitalize()} Value', fontweight='bold', fontsize=12)
    ax3.set_ylabel('File Size (GB)', fontweight='bold', fontsize=12)
    ax3.set_title('TensorStore File Size', fontweight='bold', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_values)
    ax3.grid(alpha=0.3, axis='y')
    # add value labels on bars
    for i, v in enumerate(file_sizes['tensorstore']):
        ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

# add overall title
fig.suptitle(f'TensorStore {args.sweep_param.capitalize()} Sweep Comparison', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# save plot
sweep_dir = os.path.join(results_base, args.sweep_id)
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
