# configuration for model checkpointing project

import os

# ============================================================================
# model configuration
# ============================================================================
# set MODEL_NAME via environment variable or edit the default below
# examples:
#   - "openlm-research/open_llama_3b" (llama)
#   - "meta-llama/Llama-2-7b-hf" (llama)
#   - "meta-llama/Llama-2-13b-hf" (llama)
#   - "mistralai/Mistral-7B-v0.1" (mistral)
#   - "Qwen/Qwen2.5-7B" (qwen2)
# ============================================================================
MODEL_NAME = os.environ.get('MODEL_NAME', "openlm-research/open_llama_3b")
DEVICE = "cpu"

# auto-detect model type from name
if 'qwen' in MODEL_NAME.lower():
    MODEL_TYPE = 'qwen'
elif 'mistral' in MODEL_NAME.lower():
    MODEL_TYPE = 'mistral'
else:
    MODEL_TYPE = 'llama'

# extract model identifier for filenames (e.g., "open_llama_3b" from "openlm-research/open_llama_3b")
MODEL_ID = MODEL_NAME.split('/')[-1] if '/' in MODEL_NAME else MODEL_NAME

# paths - organized by model name
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_DIR = os.path.join(SAVED_MODELS_DIR, MODEL_ID)  # saved_models/<model_name>/
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")           # saved_models/<model_name>/plots/

# cache configuration (for cluster)
HF_CACHE = os.environ.get('HF_HOME', '/mnt/common/$USER/huggingface_cache')

# tensorstore configuration
CHUNK_SIZE_MB = 64
T5X_CHUNK_SIZE_MB = 64
CONCURRENCY_LIMIT = 128

# compression
USE_COMPRESSION = True
COMPRESSION_LEVEL = 1  # gzip level 1 as in notebook
