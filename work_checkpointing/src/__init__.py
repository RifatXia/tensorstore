"""
TensorStore Checkpointing Package

This package provides utilities for saving and loading PyTorch models
using TensorStore instead of traditional pickle-based checkpointing.

Key modules:
- model_loader: Utilities for loading pre-trained models (OPT-125M)
- tensorstore_saver: Functions for saving models to TensorStore format
- tensorstore_loader: Functions for loading models from TensorStore format
- utils: Common utilities and helper functions
"""

__version__ = "0.1.0"
__author__ = "TensorStore Checkpointing Project"

# Import main functions for easy access
from .model_loader import load_opt_125m, get_model_info
from .tensorstore_saver import save_model_to_tensorstore, save_layer_to_tensorstore
from .tensorstore_loader import load_model_from_tensorstore, load_layer_from_tensorstore

__all__ = [
    'load_opt_125m',
    'get_model_info', 
    'save_model_to_tensorstore',
    'save_layer_to_tensorstore',
    'load_model_from_tensorstore', 
    'load_layer_from_tensorstore'
]
