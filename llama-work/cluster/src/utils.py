# utility functions for checkpointing

import time
import os
import json
import numpy as np
from tqdm import tqdm

def format_time(ms):
    """format milliseconds to readable string"""
    return f"{ms:.1f} ms"

def format_size(bytes_size):
    """format bytes to readable string"""
    for unit in ['b', 'kb', 'mb', 'gb']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} tb"

def get_directory_size(directory):
    """calculate total size of directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def calculate_chunk_shape(shape, target_elements):
    """
    calculate optimal chunk shape to target a specific number of elements
    used to create ~64mib chunks for efficient i/o
    """
    # ensure minimum chunk size of 1 element
    if target_elements < 1:
        target_elements = 1
    if not shape:
        return [1]
    
    chunk_shape = list(shape)
    
    # iteratively halve the largest dimension until chunk size fits target
    while np.prod(chunk_shape) > target_elements and max(chunk_shape) > 1:
        max_idx = chunk_shape.index(max(chunk_shape))
        chunk_shape[max_idx] = max(1, chunk_shape[max_idx] // 2)
    
    return chunk_shape

class Timer:
    """simple timer context manager and static timer"""
    _last_time = None
    
    def __init__(self, name="operation"):
        self.name = name
        self.start_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        Timer._last_time = self.elapsed_ms
        print(f"{self.name} completed in {format_time(self.elapsed_ms)}")
    
    @staticmethod
    def start():
        """start a timer and return start time"""
        return time.time()
    
    @staticmethod
    def end(start_time):
        """end timer and return elapsed time in ms"""
        elapsed_ms = (time.time() - start_time) * 1000
        Timer._last_time = elapsed_ms
        return elapsed_ms
    
    @staticmethod
    def get_last_time():
        """get the last recorded time"""
        return Timer._last_time

def get_model_default_dtype(model_name, cache_dir):
    """
    get the default dtype for a model from its config.json
    
    args:
        model_name: huggingface model name (e.g., "openlm-research/open_llama_3b")
        cache_dir: huggingface cache directory
    
    returns:
        dtype string: 'float16', 'float32', 'bfloat16', or 'float16' as fallback
    """
    try:
        # construct path to config.json in cache
        # huggingface cache structure: models--org--model/snapshots/hash/config.json
        model_cache_name = model_name.replace('/', '--')
        model_cache_path = os.path.join(cache_dir, f"models--{model_cache_name}")
        
        # find the latest snapshot directory
        if not os.path.exists(model_cache_path):
            print(f"warning: model cache not found at {model_cache_path}, using default dtype")
            return 'float16'
        
        snapshots_dir = os.path.join(model_cache_path, "snapshots")
        if not os.path.exists(snapshots_dir):
            print(f"warning: snapshots directory not found, using default dtype")
            return 'float16'
        
        # get the most recent snapshot
        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if not snapshots:
            print(f"warning: no snapshots found, using default dtype")
            return 'float16'
        
        # use the first snapshot (usually there's only one)
        snapshot_dir = os.path.join(snapshots_dir, snapshots[0])
        config_path = os.path.join(snapshot_dir, "config.json")
        
        if not os.path.exists(config_path):
            print(f"warning: config.json not found at {config_path}, using default dtype")
            return 'float16'
        
        # read config.json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # check for torch_dtype field
        torch_dtype = config.get('torch_dtype', None)
        
        if torch_dtype:
            # map torch dtype names to our format
            dtype_mapping = {
                'float16': 'float16',
                'float32': 'float32',
                'bfloat16': 'bfloat16',
                'torch.float16': 'float16',
                'torch.float32': 'float32',
                'torch.bfloat16': 'bfloat16',
            }
            dtype = dtype_mapping.get(torch_dtype, 'float16')
            print(f"✓ detected model default dtype: {dtype} (from config.json)")
            return dtype
        else:
            print(f"warning: torch_dtype not found in config.json, using float16 as default")
            return 'float16'
            
    except Exception as e:
        print(f"warning: error reading model config: {e}, using float16 as default")
        return 'float16'
