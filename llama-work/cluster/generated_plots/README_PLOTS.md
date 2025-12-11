# Plot Generation Scripts - Summary

## ✅ All Plots Generated Successfully!

All plot scripts have been recreated with **100% consistency** following the exact structure of plot2.

---

## 📊 Generated Plot Scripts

### Plot 2 - Phase Comparison (4 models)
- **Script:** `plot2_phase_comparison.py`
- **Output:** `plot2/phase_comparison.png`
- **Models:** Qwen2.5-7B, open_llama_3b, Llama-3.2-3B, Mistral-7B
- **Comparison:** PyTorch vs TensorStore vs T5X
- **Bar Width:** 0.18 (4 models)

### Plot 3 - Compression Comparison (2 models)
- **Script:** `plot3_compression_comparison.py`
- **Output:** `plot3/compression_comparison.png`
- **Models:** open_llama_3b, Llama-3.2-3B
- **Comparison:** none vs gzip compression
- **Bar Width:** 0.38 (2 models)

### Plot 4 - Concurrency Comparison (2 models)
- **Script:** `plot4_concurrency_comparison.py`
- **Output:** `plot4/concurrency_comparison.png`
- **Models:** Mistral-7B, Qwen2.5-7B
- **Comparison:** Concurrency levels (1, 4, 16, 64, 128)
- **Bar Width:** 0.38 (2 models)

### Plot 5 - Concurrency Comparison (2 models)
- **Script:** `plot5_concurrency_comparison.py`
- **Output:** `plot5/concurrency_comparison.png`
- **Models:** Llama-3.2-3B, open_llama_3b
- **Comparison:** Concurrency levels (1, 4, 16, 64, 128)
- **Bar Width:** 0.38 (2 models)

### Plot 6 - Chunk Size Comparison (4 models)
- **Script:** `plot6_chunk_size_comparison.py`
- **Output:** `plot6/chunk_size_comparison.png`
- **Models:** open_llama_3b, Qwen2.5-7B, Llama-3.2-3B, Mistral-7B
- **Comparison:** Chunk sizes (1, 4, 16, 64, 128, 256 MB)
- **Bar Width:** 0.18 (4 models)

### Plot 7 - Chunk Size Comparison (2 models)
- **Script:** `plot7_chunk_size_comparison.py`
- **Output:** `plot7/chunk_size_comparison.png`
- **Models:** open_llama_3b, Llama-3.2-3B
- **Comparison:** Chunk sizes (1, 4, 16, 64, 128, 256 MB)
- **Bar Width:** 0.38 (2 models)

---

## 🎨 Consistent Properties Across ALL Plots

### ✅ Time Units
- **Y-axis label:** `time (s)` (seconds, NOT milliseconds)
- **Conversion:** All ms values divided by 1000

### ✅ Value Labels
- **Position:** On top of each bar
- **Format:** 
  - Time: `{value:.1f}` (1 decimal place)
  - File size: `{value:.2f}` (2 decimal places)
- **Font size:** 8pt
- **Alignment:** Center, bottom

### ✅ Bar Widths
- **4-model plots (plot2, plot6):** width = 0.18
- **2-model plots (plot3, plot4, plot5, plot7):** width = 0.38

### ✅ Layout & Styling
- **Figure size:** 20x7 inches
- **Subplots:** 3 (save time, load time, file size)
- **Title font:** 16pt, bold
- **Axis labels:** 13pt, bold
- **Tick labels:** 11pt, bold
- **Legend:** 
  - 4-model plots: fontsize=10
  - 2-model plots: fontsize=11
  - Position: upper left, outside plot area
  - Transparency: framealpha=0.9

### ✅ Colors
- **Model 1:** #3498db (blue)
- **Model 2:** #e74c3c (red)
- **Model 3:** #2ecc71 (green)
- **Model 4:** #f39c12 (orange)

### ✅ Other Features
- **Y-axis format:** Plain (no scientific notation)
- **Grid:** Y-axis only, alpha=0.3
- **Vertical padding:** 30% extra space (ylim * 1.3)
- **Configuration box:** Bottom center, wheat background
- **DPI:** 150
- **Full model names:** Always displayed in legends

---

## 🚀 How to Regenerate All Plots

```bash
cd /home/rifatxia/Desktop/TensorstoreWork/tensorstore/llama-work/cluster/generated_plots

# Run all plot scripts
python3 plot2_phase_comparison.py
python3 plot3_compression_comparison.py
python3 plot4_concurrency_comparison.py
python3 plot5_concurrency_comparison.py
python3 plot6_chunk_size_comparison.py
python3 plot7_chunk_size_comparison.py
```

Or run all at once:
```bash
python3 plot2_phase_comparison.py && \
python3 plot3_compression_comparison.py && \
python3 plot4_concurrency_comparison.py && \
python3 plot5_concurrency_comparison.py && \
python3 plot6_chunk_size_comparison.py && \
python3 plot7_chunk_size_comparison.py
```

---

## ✅ Verification Checklist

All plots have been verified to include:
- [x] Time in seconds (not milliseconds)
- [x] Value labels on top of bars
- [x] Correct bar widths (0.18 for 4 models, 0.38 for 2 models)
- [x] Full model names in legends
- [x] Consistent fonts and styling
- [x] Configuration boxes
- [x] 30% vertical padding
- [x] No overlapping labels
- [x] Clean, professional appearance

---

**Generated:** 2025-12-11
**Status:** ✅ All plots consistent and ready to use!
