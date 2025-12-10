#!/usr/bin/env python3
"""
generate side-by-side comparison plots for multiple models
compares concurrency and chunk size sweeps across different models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# data directory
data_dir = Path("plot1")

# load all sweep data
sweeps = {}
for sweep_dir in data_dir.iterdir():
    if sweep_dir.is_dir() and sweep_dir.name.startswith("202512"):
        summary_file = sweep_dir / "sweep_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                sweep_param = data['sweep_param']
                model_name = data['model_name']
                
                # extract model id
                model_id = model_name.split('/')[-1]
                
                # organize by sweep type and model
                if sweep_param not in sweeps:
                    sweeps[sweep_param] = {}
                sweeps[sweep_param][model_id] = data

print(f"loaded {len(sweeps)} sweep types")
for sweep_type, models in sweeps.items():
    print(f"  {sweep_type}: {list(models.keys())}")

# ============================================================================
# plot 1: concurrency comparison (save time and load time side-by-side)
# ============================================================================
if 'concurrency' in sweeps:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Concurrency Impact: Model Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    models = list(sweeps['concurrency'].keys())
    model_colors = ['#3498db', '#e74c3c']
    
    # collect data for both models
    all_data = {}
    for model_id in models:
        data = sweeps['concurrency'][model_id]
        all_data[model_id] = {
            'concurrency_values': [int(x) for x in data['sweep_values']],
            'save_times': data['results']['tensorstore']['save_times_ms'],  # keep in ms
            'load_times': data['results']['tensorstore']['load_times_ms']   # keep in ms
        }
    
    concurrency_values = all_data[models[0]]['concurrency_values']
    x_pos = np.arange(len(concurrency_values))
    width = 0.35
    
    # left plot: save time comparison
    ax_save = axes[0]
    for idx, model_id in enumerate(models):
        save_times = all_data[model_id]['save_times']
        bars = ax_save.bar(x_pos + (idx - 0.5) * width, save_times, width, 
                          label=model_id, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_save.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax_save.set_title('save time comparison', fontsize=14, fontweight='bold')
    ax_save.set_xlabel('concurrency limit', fontsize=13, fontweight='bold')
    ax_save.set_ylabel('time (ms)', fontsize=13, fontweight='bold')
    ax_save.set_xticks(x_pos)
    ax_save.set_xticklabels(concurrency_values, fontsize=11, fontweight='bold')
    ax_save.tick_params(axis='y', labelsize=11)
    for label in ax_save.get_yticklabels():
        label.set_fontweight('bold')
    ax_save.legend(fontsize=11)
    ax_save.grid(axis='y', alpha=0.3)
    
    # right plot: load time comparison
    ax_load = axes[1]
    for idx, model_id in enumerate(models):
        load_times = all_data[model_id]['load_times']
        bars = ax_load.bar(x_pos + (idx - 0.5) * width, load_times, width,
                          label=model_id, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_load.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax_load.set_title('load time comparison', fontsize=14, fontweight='bold')
    ax_load.set_xlabel('concurrency limit', fontsize=13, fontweight='bold')
    ax_load.set_ylabel('time (ms)', fontsize=13, fontweight='bold')
    ax_load.set_xticks(x_pos)
    ax_load.set_xticklabels(concurrency_values, fontsize=11, fontweight='bold')
    ax_load.tick_params(axis='y', labelsize=11)
    for label in ax_load.get_yticklabels():
        label.set_fontweight('bold')
    ax_load.legend(fontsize=11)
    ax_load.grid(axis='y', alpha=0.3)
    
    # add configuration metadata text box (positioned below axes)
    config_text = (
        "Configuration (constant across all runs):\n"
        "• Chunk Size: 64 MB\n"
        "• Compression: None\n"
        "• Device: CPU\n"
        "• Variable: Concurrency limit (1, 4, 16, 64, 128)"
    )
    fig.text(0.5, -0.02, config_text, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    output_file = data_dir / "concurrency_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ concurrency comparison saved: {output_file}")
    plt.close()

# ============================================================================
# plot 2: chunk size comparison (side-by-side models)
# ============================================================================
if 'chunk' in sweeps:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Chunk Size Impact: Model Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    models = list(sweeps['chunk'].keys())
    colors = ['#2ecc71', '#f39c12']
    
    # if there are multiple runs for same model, use the latest one
    unique_models = {}
    for model_id in models:
        if model_id not in unique_models:
            unique_models[model_id] = sweeps['chunk'][model_id]
    
    models = list(unique_models.keys())[:2]  # take first 2 unique models
    
    for idx, model_id in enumerate(models):
        data = unique_models[model_id]
        chunk_values = [int(x) for x in data['sweep_values']]
        save_times = [t / 1000 for t in data['results']['tensorstore']['save_times_ms']]  # convert to seconds
        load_times = [t / 1000 for t in data['results']['tensorstore']['load_times_ms']]  # convert to seconds
        
        ax = axes[idx]
        
        # plot save and load times
        x_pos = np.arange(len(chunk_values))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, save_times, width, label='save time', color=colors[0], alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, load_times, width, label='load time', color=colors[1], alpha=0.8)
        
        ax.set_title(f'{model_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('chunk size (mb)', fontsize=13, fontweight='bold')
        ax.set_ylabel('time (seconds)', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(chunk_values, fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # add configuration metadata text box (positioned below axes)
    config_text = (
        "Configuration (constant across all runs):\n"
        "• Concurrency: 1 (no concurrency)\n"
        "• Compression: None\n"
        "• Device: CPU\n"
        "• Variable: Chunk size (1, 4, 16, 64, 128, 256 MB)"
    )
    fig.text(0.5, -0.02, config_text, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
             family='monospace', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    output_file = data_dir / "chunk_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ chunk size comparison saved: {output_file}")
    plt.close()

print("\n" + "="*70)
print("✓ all comparison plots generated successfully!")
print("="*70)
print(f"\noutput files:")
print(f"  - {data_dir}/concurrency_comparison.png")
print(f"  - {data_dir}/chunk_comparison.png")
