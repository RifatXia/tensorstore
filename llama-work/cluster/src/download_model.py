# download model from huggingface

import sys
from transformers import AutoModelForCausalLM
from config import MODEL_NAME, MODEL_TYPE, HF_CACHE
from utils import Timer

def download_model():
    """download model from huggingface hub (supports llama, qwen, mistral, etc.)"""
    print("=" * 50)
    print(f"downloading model: {MODEL_NAME}")
    print(f"model type: {MODEL_TYPE}")
    print(f"cache location: {HF_CACHE}")
    print("=" * 50)
    
    try:
        with Timer("model download"):
            # use AutoModelForCausalLM for universal model loading
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True  # required for qwen and some other models
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
