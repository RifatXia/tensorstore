# compare results from all checkpointing methods

import sys
import os
import json
from config import SAVED_MODELS_DIR, MODEL_ID
from utils import format_size, format_time, get_directory_size

def compare_results():
    """compare performance and file sizes of all methods"""
    print("\n" + "=" * 50)
    print("performance comparison")
    print("=" * 50)
    
    results = []
    
    # pytorch
    pytorch_path = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_pytorch.pth")
    if os.path.exists(pytorch_path):
        size = os.path.getsize(pytorch_path)
        results.append({
            'method': 'pytorch',
            'size': size,
            'size_str': format_size(size)
        })
    
    # tensorstore
    ts_dir = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_tensorstore")
    if os.path.exists(ts_dir):
        size = get_directory_size(ts_dir)
        results.append({
            'method': 'tensorstore',
            'size': size,
            'size_str': format_size(size)
        })
    
    # t5x-tensorstore
    t5x_dir = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_t5x_tensorstore")
    if os.path.exists(t5x_dir):
        size = get_directory_size(t5x_dir)
        results.append({
            'method': 't5x-tensorstore',
            'size': size,
            'size_str': format_size(size)
        })
    
    # print comparison table
    print(f"\n{'method':<20} {'file size':<15}")
    print("-" * 35)
    for result in results:
        print(f"{result['method']:<20} {result['size_str']:<15}")
    
    # save results to json
    results_path = os.path.join(SAVED_MODELS_DIR, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nresults saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    compare_results()
