#!/usr/bin/env python3
"""
generate phase comparison plots for multiple models
compares pytorch, tensorstore, and t5x phases across different models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# data directory
data_dir = Path("plot2")

# load all phase comparison data
phase_data = {}
model_full_names = {}  # store full model names
for model_dir in data_dir.iterdir():
    if model_dir.is_dir() and model_dir.name.startswith("202512"):
        results_file = model_dir / "all_phases_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                model_id = data.get('model_id', model_dir.name.split('_')[-1])
                phase_data[model_id] = data
                model_full_names[model_id] = data['model_name']

print(f"loaded {len(phase_data)} models")
for model_id in phase_data.keys():
    print(f"  {model_id}: {list(phase_data[model_id]['phases'].keys())}")

# ============================================================================
# phase comparison plot (save time, load time, and file size)
# ============================================================================
if len(phase_data) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Phase Comparison: Model Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    models = list(phase_data.keys())[:4]  # take up to 4 models
    model_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    phases = ['pytorch', 'tensorstore', 't5x']
    phase_labels = ['PyTorch', 'TensorStore (ours)', 'T5X using TensorStore']
    
    # collect data for all models
    all_data = {}
    for model_id in models:
        save_times = []
        load_times = []
        file_sizes = []
        
        for phase in phases:
            if phase in phase_data[model_id]['phases']:
                phase_info = phase_data[model_id]['phases'][phase]
                # handle both mean values and direct values, convert ms to seconds
                if isinstance(phase_info['save_time_ms'], dict):
                    save_times.append(phase_info['save_time_ms']['mean'] / 1000)
                    load_times.append(phase_info['load_time_ms']['mean'] / 1000)
                else:
                    save_times.append(phase_info['save_time_ms'] / 1000)
                    load_times.append(phase_info['load_time_ms'] / 1000)
                file_sizes.append(phase_info['file_size_gb'])
            else:
                save_times.append(0)
                load_times.append(0)
                file_sizes.append(0)
        
        all_data[model_id] = {
            'save_times': save_times,
            'load_times': load_times,
            'file_sizes': file_sizes
        }
    
    x_pos = np.arange(len(phases))
    width = 0.18  # narrower bars for 4 models
    
    # left plot: save time comparison
    ax_save = axes[0]
    for idx, model_id in enumerate(models):
        save_times = all_data[model_id]['save_times']
        full_name = model_full_names.get(model_id, model_id)
        offset = (idx - len(models)/2 + 0.5) * width
        bars = ax_save.bar(x_pos + offset, save_times, width, 
                          label=full_name, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_save.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax_save.set_title('save time comparison', fontsize=14, fontweight='bold')
    ax_save.set_xlabel('phase', fontsize=13, fontweight='bold')
    ax_save.set_ylabel('time (s)', fontsize=13, fontweight='bold')
    ax_save.set_xticks(x_pos)
    ax_save.set_xticklabels(phase_labels, fontsize=11, fontweight='bold')
    ax_save.tick_params(axis='y', labelsize=11)
    # format y-axis to show actual values without scientific notation
    ax_save.ticklabel_format(style='plain', axis='y')
    for label in ax_save.get_yticklabels():
        label.set_fontweight('bold')
    ax_save.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    ax_save.grid(axis='y', alpha=0.3)
    # add some padding at the top for labels
    ax_save.set_ylim(top=max([max(all_data[m]['save_times']) for m in models]) * 1.3)
    
    # middle plot: load time comparison
    ax_load = axes[1]
    for idx, model_id in enumerate(models):
        load_times = all_data[model_id]['load_times']
        full_name = model_full_names.get(model_id, model_id)
        offset = (idx - len(models)/2 + 0.5) * width
        bars = ax_load.bar(x_pos + offset, load_times, width,
                          label=full_name, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_load.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax_load.set_title('load time comparison', fontsize=14, fontweight='bold')
    ax_load.set_xlabel('phase', fontsize=13, fontweight='bold')
    ax_load.set_ylabel('time (s)', fontsize=13, fontweight='bold')
    ax_load.set_xticks(x_pos)
    ax_load.set_xticklabels(phase_labels, fontsize=11, fontweight='bold')
    ax_load.tick_params(axis='y', labelsize=11)
    # format y-axis to show actual values without scientific notation
    ax_load.ticklabel_format(style='plain', axis='y')
    for label in ax_load.get_yticklabels():
        label.set_fontweight('bold')
    ax_load.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    ax_load.grid(axis='y', alpha=0.3)
    # add some padding at the top for labels
    ax_load.set_ylim(top=max([max(all_data[m]['load_times']) for m in models]) * 1.3)
    
    # right plot: file size comparison
    ax_size = axes[2]
    for idx, model_id in enumerate(models):
        file_sizes = all_data[model_id]['file_sizes']
        full_name = model_full_names.get(model_id, model_id)
        offset = (idx - len(models)/2 + 0.5) * width
        bars = ax_size.bar(x_pos + offset, file_sizes, width,
                          label=full_name, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_size.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax_size.set_title('file size comparison', fontsize=14, fontweight='bold')
    ax_size.set_xlabel('phase', fontsize=13, fontweight='bold')
    ax_size.set_ylabel('file size (gb)', fontsize=13, fontweight='bold')
    ax_size.set_xticks(x_pos)
    ax_size.set_xticklabels(phase_labels, fontsize=11, fontweight='bold')
    ax_size.tick_params(axis='y', labelsize=11)
    for label in ax_size.get_yticklabels():
        label.set_fontweight('bold')
    ax_size.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    ax_size.grid(axis='y', alpha=0.3)
    # add some padding at the top for labels
    ax_size.set_ylim(top=max([max(all_data[m]['file_sizes']) for m in models]) * 1.3)
    
    # add configuration metadata text box (positioned below axes)
    config_text = (
        "Configuration:\n"
        "• PyTorch: torch.save, no compression\n"
        "• TensorStore: 64 MB chunks, no compression, concurrency=1\n"
        "• T5X: 64 MB chunks, gzip-1 compression, default concurrency\n"
        "• Device: CPU"
    )
    fig.text(0.5, -0.02, config_text, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    output_file = data_dir / "phase_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ phase comparison saved: {output_file}")
    plt.close()

print("\n" + "="*70)
print("✓ phase comparison plot generated successfully!")
print("="*70)
print(f"\noutput file:")
print(f"  - {data_dir}/phase_comparison.png")
print(f"\nmodels compared: {', '.join([model_full_names.get(m, m) for m in models])}")
print(f"phases: {', '.join(phases)}")
