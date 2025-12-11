#!/usr/bin/env python3
"""
generate concurrency comparison plots for multiple models
compares different concurrency levels across different models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# data directory
data_dir = Path("plot4")

# load all concurrency sweep data
concurrency_data = {}
model_full_names = {}  # store full model names
for sweep_dir in data_dir.iterdir():
    if sweep_dir.is_dir() and sweep_dir.name.startswith("202512") and "concurrency" in sweep_dir.name:
        # extract model from directory name
        model_id = sweep_dir.name.split('_')[-1]
        
        if model_id not in concurrency_data:
            concurrency_data[model_id] = {}
        
        # load all concurrency variants
        for conc_dir in sweep_dir.iterdir():
            if conc_dir.is_dir() and conc_dir.name.startswith("concurrency"):
                results_file = conc_dir / "all_phases_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        concurrency_value = data['concurrency']
                        concurrency_data[model_id][concurrency_value] = data
                        # store full model name
                        model_full_names[model_id] = data['model_name']

print(f"loaded {len(concurrency_data)} models")
for model_id, conc_types in concurrency_data.items():
    print(f"  {model_id}: {sorted(conc_types.keys())}")

# ============================================================================
# concurrency comparison plot (save time, load time, and file size)
# ============================================================================
if len(concurrency_data) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Concurrency Impact: Model Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    models = list(concurrency_data.keys())[:2]  # take first 2 models
    model_colors = ['#3498db', '#e74c3c']
    
    # get all concurrency values (sorted)
    all_concurrency_values = set()
    for model_id in models:
        all_concurrency_values.update(concurrency_data[model_id].keys())
    concurrency_values = sorted(all_concurrency_values)
    
    # collect data for both models
    all_data = {}
    for model_id in models:
        save_times = []
        load_times = []
        file_sizes = []
        
        for conc_val in concurrency_values:
            if conc_val in concurrency_data[model_id]:
                data = concurrency_data[model_id][conc_val]
                save_times.append(data['phases']['tensorstore']['save_time_ms'] / 1000)
                load_times.append(data['phases']['tensorstore']['load_time_ms'] / 1000)
                file_sizes.append(data['phases']['tensorstore']['file_size_gb'])
            else:
                save_times.append(0)
                load_times.append(0)
                file_sizes.append(0)
        
        all_data[model_id] = {
            'save_times': save_times,
            'load_times': load_times,
            'file_sizes': file_sizes
        }
    
    x_pos = np.arange(len(concurrency_values))
    width = 0.38  # wider bars for 2 models
    
    # left plot: save time comparison
    ax_save = axes[0]
    for idx, model_id in enumerate(models):
        save_times = all_data[model_id]['save_times']
        full_name = model_full_names.get(model_id, model_id)
        offset = (idx - 0.5) * width
        bars = ax_save.bar(x_pos + offset, save_times, width, 
                          label=full_name, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_save.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax_save.set_title('save time comparison', fontsize=14, fontweight='bold')
    ax_save.set_xlabel('concurrency limit', fontsize=13, fontweight='bold')
    ax_save.set_ylabel('time (s)', fontsize=13, fontweight='bold')
    ax_save.set_xticks(x_pos)
    ax_save.set_xticklabels(concurrency_values, fontsize=11, fontweight='bold')
    ax_save.tick_params(axis='y', labelsize=11)
    # format y-axis to show actual values without scientific notation
    ax_save.ticklabel_format(style='plain', axis='y')
    for label in ax_save.get_yticklabels():
        label.set_fontweight('bold')
    ax_save.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    ax_save.grid(axis='y', alpha=0.3)
    # add some padding at the top for labels
    ax_save.set_ylim(top=max([max(all_data[m]['save_times']) for m in models]) * 1.3)
    
    # middle plot: load time comparison
    ax_load = axes[1]
    for idx, model_id in enumerate(models):
        load_times = all_data[model_id]['load_times']
        full_name = model_full_names.get(model_id, model_id)
        offset = (idx - 0.5) * width
        bars = ax_load.bar(x_pos + offset, load_times, width,
                          label=full_name, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_load.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax_load.set_title('load time comparison', fontsize=14, fontweight='bold')
    ax_load.set_xlabel('concurrency limit', fontsize=13, fontweight='bold')
    ax_load.set_ylabel('time (s)', fontsize=13, fontweight='bold')
    ax_load.set_xticks(x_pos)
    ax_load.set_xticklabels(concurrency_values, fontsize=11, fontweight='bold')
    ax_load.tick_params(axis='y', labelsize=11)
    # format y-axis to show actual values without scientific notation
    ax_load.ticklabel_format(style='plain', axis='y')
    for label in ax_load.get_yticklabels():
        label.set_fontweight('bold')
    ax_load.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    ax_load.grid(axis='y', alpha=0.3)
    # add some padding at the top for labels
    ax_load.set_ylim(top=max([max(all_data[m]['load_times']) for m in models]) * 1.3)
    
    # right plot: file size comparison
    ax_size = axes[2]
    for idx, model_id in enumerate(models):
        file_sizes = all_data[model_id]['file_sizes']
        full_name = model_full_names.get(model_id, model_id)
        offset = (idx - 0.5) * width
        bars = ax_size.bar(x_pos + offset, file_sizes, width,
                          label=full_name, color=model_colors[idx], alpha=0.8)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_size.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax_size.set_title('file size comparison', fontsize=14, fontweight='bold')
    ax_size.set_xlabel('concurrency limit', fontsize=13, fontweight='bold')
    ax_size.set_ylabel('file size (gb)', fontsize=13, fontweight='bold')
    ax_size.set_xticks(x_pos)
    ax_size.set_xticklabels(concurrency_values, fontsize=11, fontweight='bold')
    ax_size.tick_params(axis='y', labelsize=11)
    for label in ax_size.get_yticklabels():
        label.set_fontweight('bold')
    ax_size.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    ax_size.grid(axis='y', alpha=0.3)
    # add some padding at the top for labels
    ax_size.set_ylim(top=max([max(all_data[m]['file_sizes']) for m in models]) * 1.3)
    
    # add configuration metadata text box (positioned below axes)
    config_text = (
        "Configuration (constant across all runs):\n"
        "• Chunk Size: 64 MB\n"
        "• Compression: none\n"
        "• Device: CPU\n"
        "• Variable: Concurrency (1, 4, 16, 64, 128)"
    )
    fig.text(0.5, -0.02, config_text, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    output_file = data_dir / "concurrency_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ concurrency comparison saved: {output_file}")
    plt.close()

print("\n" + "="*70)
print("✓ concurrency comparison plot generated successfully!")
print("="*70)
print(f"\noutput file:")
print(f"  - {data_dir}/concurrency_comparison.png")
print(f"\nmodels compared: {', '.join([model_full_names.get(m, m) for m in models])}")
print(f"concurrency values: {', '.join(map(str, concurrency_values))}")
