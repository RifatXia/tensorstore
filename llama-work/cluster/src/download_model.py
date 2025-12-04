# download model from huggingface

import sys
import os
from transformers import AutoModelForCausalLM, AutoConfig
from config import MODEL_NAME, MODEL_TYPE, HF_CACHE, HF_TOKEN
from utils import Timer

def download_model():
    """download model from huggingface hub (supports llama, qwen, mistral, etc.)"""
    print("=" * 50)
    print(f"downloading model: {MODEL_NAME}")
    print(f"model type: {MODEL_TYPE}")
    print(f"cache location: {HF_CACHE}")
    print(f"hf token: {'set' if HF_TOKEN else 'not set (public models only)'}")
    print("=" * 50)
    
    try:
        # first, download config.json explicitly
        print("\ndownloading model config...")
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            token=HF_TOKEN  # use token for private/gated models
        )
        print(f"✓ config downloaded")
        if hasattr(config, 'torch_dtype'):
            print(f"  default dtype: {config.torch_dtype}")
        
        # then download the full model
        print("\ndownloading model weights...")
        with Timer("model download"):
            # use AutoModelForCausalLM for universal model loading
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,  # required for qwen and some other models
                token=HF_TOKEN  # use token for private/gated models
            )
        
        print(f"\nmodel downloaded successfully!")
        print(f"total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m")
        print(f"parameter count: {len(list(model.named_parameters()))}")
        
        return True
    except Exception as e:
        print(f"\nerror downloading model: {e}")
        return False

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    success = download_model()
    sys.exit(0 if success else 1)
