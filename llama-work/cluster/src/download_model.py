# download model from huggingface

import sys
from transformers import LlamaForCausalLM
from config import MODEL_NAME, HF_CACHE
from utils import Timer

def download_model():
    """download model from huggingface hub"""
    print("=" * 50)
    print(f"downloading model: {MODEL_NAME}")
    print(f"cache location: {HF_CACHE}")
    print("=" * 50)
    
    try:
        with Timer("model download"):
            model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
        
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
