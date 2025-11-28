# utility functions for checkpointing

import time
import os
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
    """simple timer context manager"""
    def __init__(self, name="operation"):
        self.name = name
        self.start_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        print(f"{self.name} completed in {format_time(self.elapsed_ms)}")
