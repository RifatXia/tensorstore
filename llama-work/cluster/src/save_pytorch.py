# save model using pytorch native format

import sys
import torch
import os
from load_model import load_model
from config import SAVED_MODELS_DIR, MODEL_ID
from utils import Timer, format_size, get_directory_size

def save_pytorch(model):
    """save model using pytorch's native serialization"""
    save_path = os.path.join(SAVED_MODELS_DIR, f"{MODEL_ID}_pytorch.pth")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("phase 1: pytorch saving")
    print("=" * 50)
    
    try:
        with Timer("pytorch save"):
            torch.save(model.state_dict(), save_path)
        
        file_size = os.path.getsize(save_path)
        print(f"file size: {format_size(file_size)}")
        print(f"saved to: {save_path}")
        
        return save_path, file_size
    except Exception as e:
        print(f"\nerror saving with pytorch: {e}")
        sys.exit(1)

def load_pytorch(save_path, model):
    """load model from pytorch checkpoint"""
    print("\n" + "=" * 50)
    print("phase 1: pytorch loading")
    print("=" * 50)
    
    try:
        with Timer("pytorch load"):
            state_dict = torch.load(save_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        print(f"loaded {len(state_dict)} parameters successfully")
        return True
    except Exception as e:
        print(f"\nerror loading pytorch checkpoint: {e}")
        return False

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    
    # load model
    model = load_model()
    
    # save with pytorch
    save_path, _ = save_pytorch(model)
    
    # verify by loading
    load_pytorch(save_path, model)
    
    print("\npytorch checkpoint complete!")
