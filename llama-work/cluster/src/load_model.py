# load model into memory

import sys
import torch
import os
from transformers import AutoModelForCausalLM
from config import MODEL_NAME, MODEL_TYPE, DEVICE
from utils import Timer

def load_model(dtype='float16'):
    """load model from huggingface (supports llama, qwen, mistral, etc.)"""
    # convert dtype string to torch dtype
    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    print("=" * 50)
    print(f"loading model: {MODEL_NAME}")
    print(f"model type: {MODEL_TYPE}")
    print(f"dtype: {dtype}")
    print(f"device: {DEVICE}")
    print(f"cpu count: {os.cpu_count()}")
    print("=" * 50)
    
    sys.stdout.flush()
    
    try:
        with Timer("model loading"):
            # use AutoModelForCausalLM for universal model loading
            # works with llama, qwen, mistral, and other architectures
            # use local_files_only to prevent internet access on compute nodes
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True  # required for qwen and some other models
            )
            
            # move to device
            model = model.to(DEVICE)
        
        print(f"\nmodel loaded successfully!")
        print(f"total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m")
        print(f"parameter count: {len(list(model.named_parameters()))}")
        
        return model
    except Exception as e:
        print(f"\nerror loading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    model = load_model()
