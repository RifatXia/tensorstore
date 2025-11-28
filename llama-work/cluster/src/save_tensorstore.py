# save model using tensorstore

import sys
import os
import json
import torch
import tensorstore as ts
import numpy as np
from tqdm import tqdm
from load_model import load_model
from config import SAVED_MODELS_DIR, MODEL_ID, CHUNK_SIZE_MB
from utils import Timer, format_size, get_directory_size, calculate_chunk_shape

def save_tensorstore(model, use_compression=False, use_concurrency=False, dtype='float16'):
    """save model using tensorstore with zarr format"""
    save_dir = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_tensorstore")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("phase 2: tensorstore saving")
    print(f"compression: {use_compression}, concurrency: {use_concurrency}, dtype: {dtype}")
    print("=" * 50)
    
    model_state = model.state_dict()
    metadata = {}
    
    # setup context if using concurrency
    context = ts.Context({'file_io_concurrency': {'limit': 128}}) if use_concurrency else None
    
    # calculate chunk size
    chunk_size_bytes = CHUNK_SIZE_MB * 1024 * 1024
    
    # dtype conversion
    dtype_conversion = {
        'float16': lambda x: x.detach().cpu().half().numpy(),
        'float32': lambda x: x.detach().cpu().float().numpy(),
        'bfloat16': lambda x: x.detach().cpu().to(torch.bfloat16).numpy()
    }
    convert_fn = dtype_conversion.get(dtype, dtype_conversion['float16'])
    
    try:
        with Timer("tensorstore save"):
            saved_count = 0
            
            for param_name, param_tensor in tqdm(model_state.items(), desc="saving parameters"):
                # convert to numpy
                param_np = convert_fn(param_tensor)
                
                # create safe filename
                safe_name = param_name.replace('.', '_').replace('/', '_')
                param_path = os.path.join(save_dir, f"{safe_name}.zarr")
                
                # calculate chunk shape
                dtype_size = np.dtype(param_np.dtype).itemsize
                target_elements = chunk_size_bytes // dtype_size
                chunk_shape = calculate_chunk_shape(param_np.shape, target_elements)
                
                # build tensorstore spec
                spec = {
                    'driver': 'zarr',
                    'kvstore': {
                        'driver': 'file',
                        'path': param_path
                    },
                    'metadata': {
                        'shape': list(param_np.shape),
                        'dtype': '<f2',  # float16 in zarr format
                        'chunks': chunk_shape
                    }
                }
                
                # add compression if requested
                if use_compression:
                    spec['metadata']['compressor'] = {'id': 'gzip', 'level': 1}
                
                # open and write
                if context:
                    store = ts.open(spec, create=True, delete_existing=True, context=context).result()
                else:
                    store = ts.open(spec, create=True, delete_existing=True).result()
                
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
        print(f"saved {saved_count} parameters")
        print(f"total size: {format_size(dir_size)}")
        print(f"saved to: {save_dir}")
        
        return save_dir, dir_size
    except Exception as e:
        print(f"\nerror saving with tensorstore: {e}")
        sys.exit(1)

def load_tensorstore(save_dir):
    """load model from tensorstore checkpoint"""
    print("\n" + "=" * 50)
    print("phase 2: tensorstore loading")
    print("=" * 50)
    
    try:
        # load metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        with Timer("tensorstore load"):
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
        print(f"\nerror loading tensorstore checkpoint: {e}")
        return False

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    
    # load model
    model = load_model()
    
    # save with tensorstore
    save_dir, _ = save_tensorstore(model, use_compression=False, use_concurrency=False)
    
    # verify by loading
    load_tensorstore(save_dir)
    
    print("\ntensorstore checkpoint complete!")
