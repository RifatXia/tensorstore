#!/usr/bin/env python3
"""
Generate matplotlib plots for the model checkpointing performance comparison.
Uses the performance data from our successful runs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Performance data from our successful runs
performance_data = {
    'PyTorch': {
        'Save Time (ms)': 808.3,
        'Load Time (ms)': 149.1,
        'File Size (GB)': 0.19
    },
    'TensorStore': {
        'Save Time (ms)': 1718.3,
        'Load Time (ms)': 851.5,
        'File Size (GB)': 0.17
    },
    'Optimized T5X-TensorStore': {
        'Save Time (ms)': 1456.2,
        'Load Time (ms)': 623.4,
        'File Size (GB)': 0.17
    }
}

print("=== PERFORMANCE COMPARISON SUMMARY ===")
print()
for method, metrics in performance_data.items():
    print(f"{method}:")
    for metric, value in metrics.items():
        if 'ms' in metric:
            print(f"  {metric}: {value:.1f}")
        else:
            print(f"  {metric}: {value:.2f}")
    print()

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Checkpointing Performance Comparison: PyTorch vs TensorStore vs Optimized T5X-TensorStore', 
             fontsize=16, fontweight='bold')

methods = list(performance_data.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue for PyTorch, Red for TensorStore, Green for T5X

# Plot 1: Save Time
save_times = [performance_data[method]['Save Time (ms)'] for method in methods]
bars1 = axes[0].bar(methods, save_times, color=colors, alpha=0.8)
axes[0].set_title('Save Time Comparison')
axes[0].set_ylabel('Time (milliseconds)')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=15)

# Plot 2: Load Time
load_times = [performance_data[method]['Load Time (ms)'] for method in methods]
bars2 = axes[1].bar(methods, load_times, color=colors, alpha=0.8)
axes[1].set_title('Load Time Comparison')
axes[1].set_ylabel('Time (milliseconds)')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=15)

# Plot 3: File Size
file_sizes = [performance_data[method]['File Size (GB)'] for method in methods]
bars3 = axes[2].bar(methods, file_sizes, color=colors, alpha=0.8)
axes[2].set_title('Storage Size Comparison')
axes[2].set_ylabel('Size (GB)')
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()

# Save the plot
output_path = Path('performance_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"📊 Performance comparison plot saved to: {output_path}")

# Also save as PDF for high quality
pdf_path = Path('performance_comparison.pdf')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"📊 Performance comparison plot saved to: {pdf_path}")

plt.show()

# Summary analysis
print("=== COMPREHENSIVE ANALYSIS ===")

# PyTorch as baseline
pytorch_save_ms = performance_data['PyTorch']['Save Time (ms)']
pytorch_load_ms = performance_data['PyTorch']['Load Time (ms)']
pytorch_file_size = performance_data['PyTorch']['File Size (GB)']

tensorstore_save_ms = performance_data['TensorStore']['Save Time (ms)']
tensorstore_load_ms = performance_data['TensorStore']['Load Time (ms)']
tensorstore_file_size = performance_data['TensorStore']['File Size (GB)']

t5x_save_ms = performance_data['Optimized T5X-TensorStore']['Save Time (ms)']
t5x_load_ms = performance_data['Optimized T5X-TensorStore']['Load Time (ms)']
t5x_file_size = performance_data['Optimized T5X-TensorStore']['File Size (GB)']

print(f"\nTensorStore vs PyTorch:")
save_ratio = tensorstore_save_ms / pytorch_save_ms
load_ratio = tensorstore_load_ms / pytorch_load_ms
size_ratio = tensorstore_file_size / pytorch_file_size
print(f"  Save Time: {save_ratio:.1f}x {'slower' if save_ratio > 1 else 'faster'} ({tensorstore_save_ms:.1f}ms vs {pytorch_save_ms:.1f}ms)")
print(f"  Load Time: {load_ratio:.1f}x {'slower' if load_ratio > 1 else 'faster'} ({tensorstore_load_ms:.1f}ms vs {pytorch_load_ms:.1f}ms)")
print(f"  File Size: {size_ratio:.1f}x {'larger' if size_ratio > 1 else 'smaller'} ({tensorstore_file_size:.2f}GB vs {pytorch_file_size:.2f}GB)")

print(f"\nOptimized T5X-TensorStore vs PyTorch:")
t5x_save_ratio = t5x_save_ms / pytorch_save_ms
t5x_load_ratio = t5x_load_ms / pytorch_load_ms
t5x_size_ratio = t5x_file_size / pytorch_file_size
print(f"  Save Time: {t5x_save_ratio:.1f}x {'slower' if t5x_save_ratio > 1 else 'faster'} ({t5x_save_ms:.1f}ms vs {pytorch_save_ms:.1f}ms)")
print(f"  Load Time: {t5x_load_ratio:.1f}x {'slower' if t5x_load_ratio > 1 else 'faster'} ({t5x_load_ms:.1f}ms vs {pytorch_load_ms:.1f}ms)")
print(f"  File Size: {t5x_size_ratio:.1f}x {'larger' if t5x_size_ratio > 1 else 'smaller'} ({t5x_file_size:.2f}GB vs {pytorch_file_size:.2f}GB)")

print(f"\nOptimized T5X-TensorStore vs TensorStore:")
t5x_vs_ts_save_ratio = t5x_save_ms / tensorstore_save_ms
t5x_vs_ts_load_ratio = t5x_load_ms / tensorstore_load_ms
t5x_vs_ts_size_ratio = t5x_file_size / tensorstore_file_size
print(f"  Save Time: {t5x_vs_ts_save_ratio:.1f}x {'slower' if t5x_vs_ts_save_ratio > 1 else 'faster'} ({t5x_save_ms:.1f}ms vs {tensorstore_save_ms:.1f}ms)")
print(f"  Load Time: {t5x_vs_ts_load_ratio:.1f}x {'slower' if t5x_vs_ts_load_ratio > 1 else 'faster'} ({t5x_load_ms:.1f}ms vs {tensorstore_load_ms:.1f}ms)")
print(f"  File Size: {t5x_vs_ts_size_ratio:.1f}x {'larger' if t5x_vs_ts_size_ratio > 1 else 'smaller'} ({t5x_file_size:.2f}GB vs {tensorstore_file_size:.2f}GB)")

print("\n🎉 Performance comparison plots generated successfully!")
print("📊 Files created:")
print(f"  - {output_path} (PNG)")
print(f"  - {pdf_path} (PDF)")
