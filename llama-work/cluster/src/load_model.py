# load model into memory

import sys
import torch
import os
from transformers import LlamaForCausalLM
from config import MODEL_NAME, DEVICE
from utils import Timer

def load_model():
    """load llama model from huggingface"""
    print("=" * 50)
    print(f"loading model: {MODEL_NAME}")
    print(f"device: {DEVICE}")
    print(f"cpu count: {os.cpu_count()}")
    print("=" * 50)
    
    sys.stdout.flush()
    
    try:
        with Timer("model loading"):
            # load model with optimizations
            # use local_files_only to prevent internet access on compute nodes
            model = LlamaForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True
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
