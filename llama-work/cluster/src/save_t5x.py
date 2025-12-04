# save model using t5x-optimized tensorstore

import sys
import os
import torch
import tensorstore as ts
import numpy as np
from tqdm import tqdm
from load_model import load_model
from config import SAVED_MODELS_DIR, MODEL_ID, T5X_CHUNK_SIZE_MB, CONCURRENCY_LIMIT, DEVICE
from utils import Timer, format_size, get_directory_size, calculate_chunk_shape, clear_system_cache, clear_gpu_cache

def save_t5x_tensorstore(model, dtype='float16'):
    """save model using t5x-optimized tensorstore approach"""
    save_dir = os.path.join(SAVED_MODELS_DIR, MODEL_ID, "t5x_tensorstore")
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("phase 3: t5x-optimized tensorstore saving")
    print(f"dtype: {dtype}")
    print("=" * 50)
    
    model_state = model.state_dict()
    metadata = {}
    
    # t5x constants
    t5x_chunk_bytes = T5X_CHUNK_SIZE_MB * 1024 * 1024
    
    # high concurrency context
    context = ts.Context({'file_io_concurrency': {'limit': CONCURRENCY_LIMIT}})
    
    # dtype conversion
    dtype_conversion = {
        'float16': (lambda x: x.detach().cpu().half().numpy(), '<f2'),
        'float32': (lambda x: x.detach().cpu().float().numpy(), '<f4'),
        'bfloat16': (lambda x: x.detach().cpu().float().numpy(), '<f4')  # convert bfloat16 to float32
    }
    convert_fn, ts_dtype = dtype_conversion.get(dtype, dtype_conversion['float16'])
    
    # clear cache before save
    clear_system_cache()
    if DEVICE == 'cuda':
        clear_gpu_cache()
    
    try:
        with Timer("t5x-tensorstore save"):
            saved_count = 0
            
            for param_name, param_tensor in tqdm(model_state.items(), desc="saving parameters"):
                # convert to numpy
                param_np = convert_fn(param_tensor)
                
                # create safe filename
                safe_name = param_name.replace('.', '_').replace('/', '_')
                param_path = os.path.join(save_dir, f"{safe_name}.zarr")
                
                # calculate t5x chunk shape
                dtype_size = np.dtype(param_np.dtype).itemsize
                target_elements = t5x_chunk_bytes // dtype_size
                chunk_shape = calculate_chunk_shape(param_np.shape, target_elements)
                
                # build t5x-style tensorstore spec
                spec = {
                    'driver': 'zarr',
                    'kvstore': {
                        'driver': 'file',
                        'path': param_path
                    },
                    'metadata': {
                        'shape': list(param_np.shape),
                        'dtype': ts_dtype,
                        'chunks': chunk_shape,
                        'compressor': {
                            'id': 'gzip',
                            'level': 1
                        }
                    }
                }
                
                # open and write with context
                store = ts.open(spec, create=True, delete_existing=True, context=context).result()
                store.write(param_np).result()
                
                # store metadata
                metadata[param_name] = {
                    'shape': list(param_np.shape),
                    'dtype': str(param_np.dtype),
                    'file': f"{safe_name}.zarr"
                }
                
                saved_count += 1
        
        # save metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        dir_size = get_directory_size(save_dir)
        print(f"saved {saved_count} parameters successfully")
        print(f"total size: {format_size(dir_size)}")
        print(f"saved to: {save_dir}")
        
        return save_dir, dir_size
    except Exception as e:
        print(f"\nerror saving with t5x-tensorstore: {e}")
        sys.exit(1)

def load_t5x_tensorstore(save_dir):
    """load model from t5x-tensorstore checkpoint"""
    print("\n" + "=" * 50)
    print("phase 3: t5x-optimized tensorstore loading")
    print("=" * 50)
    
    try:
        # load metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        with Timer("t5x-tensorstore load"):
            loaded_params = {}
            
            for param_name, param_info in tqdm(metadata.items(), desc="loading parameters"):
                param_path = os.path.join(save_dir, param_info['file'])
                
                spec = {
                    'driver': 'zarr',
                    'kvstore': {'driver': 'file', 'path': param_path}
                }
                
                dataset = ts.open(spec, open=True).result()
                param_np = dataset[:].read().result()
                loaded_params[param_name] = param_np
        
        print(f"loaded {len(loaded_params)} parameters successfully")
        return True
    except Exception as e:
        print(f"\nerror loading t5x-tensorstore checkpoint: {e}")
        return False

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    
    # load model
    model = load_model()
    
    # save with t5x-tensorstore
    save_dir, _ = save_t5x_tensorstore(model)
    
    # verify by loading
    load_t5x_tensorstore(save_dir)
    
    print("\nt5x-tensorstore checkpoint complete!")
