#!/usr/bin/env python3
"""
simple opt-125m model loader

this script loads meta's opt-125m model using the transformers library.
no training or complex operations - just basic model loading for now.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_opt_125m():
    """
    load meta's opt-125m model and tokenizer.
    
    returns:
        tuple: (model, tokenizer)
    """
    model_name = "facebook/opt-125m"
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✓ Tokenizer loaded successfully")
        
        # load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("✓ Model loaded successfully")
        
        # set model to evaluation mode
        model.eval()
        
        # print model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def main():
    """main function to load and display model information."""
    logger.info("Starting OPT-125M model loader...")
    
    try:
        # load the model
        model, tokenizer = load_opt_125m()
        
        # display basic model information
        logger.info("=" * 50)
        logger.info("MODEL INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
        # show model structure
        logger.info("\nModel structure:")
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                logger.info(f"  {name}: {module.weight.shape}")
        
        logger.info("\n✓ Model loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
