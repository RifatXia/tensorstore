# TensorStore Checkpointing with OPT-125M

This project demonstrates how to use TensorStore for model checkpointing instead of traditional PyTorch checkpointing methods, using the OPT-125M model as an example.

## Overview

Traditional PyTorch checkpointing uses `torch.save()` and pickle serialization. This project explores using TensorStore for:
- More efficient storage of large model parameters
- Layer-by-layer parameter organization
- Better support for distributed/sharded models
- Cross-platform compatibility
- Partial model loading capabilities

## Project Structure

```
work_checkpointing/
├── README.md
├── requirements.txt
├── environment.yml
├── src/
│   ├── __init__.py
│   ├── model_loader.py          # OPT-125M model loading utilities
│   ├── tensorstore_saver.py     # TensorStore saving functionality
│   ├── tensorstore_loader.py    # TensorStore loading functionality
│   └── utils.py                 # Common utilities
├── examples/
│   ├── basic_save_load.py       # Basic save/load example
│   ├── layer_analysis.py        # Layer-by-layer analysis
│   └── comparison_benchmark.py  # Compare with torch.save()
├── tests/
│   ├── test_save_load.py        # Unit tests
│   └── test_model_integrity.py  # Model integrity tests
└── checkpoints/                 # Directory for saved checkpoints
    └── opt_125m_tensorstore/    # TensorStore format checkpoints
```

## Features

- **No Training Required**: Focus purely on model storage and retrieval
- **Layer-wise Storage**: Each model layer saved separately in TensorStore
- **Efficient Loading**: Load only specific layers when needed
- **Format Comparison**: Compare TensorStore vs traditional PyTorch checkpointing
- **Model Analysis**: Analyze model structure and parameter distribution

## Getting Started

### 1. Basic Usage

```python
from src.model_loader import load_opt_125m
from src.tensorstore_saver import save_model_to_tensorstore
from src.tensorstore_loader import load_model_from_tensorstore

# Load pre-trained OPT-125M model
model, tokenizer = load_opt_125m()

# Save to TensorStore format
save_model_to_tensorstore(model, "checkpoints/opt_125m_tensorstore/")

# Load back from TensorStore
reconstructed_model = load_model_from_tensorstore("checkpoints/opt_125m_tensorstore/")
```

### 3. Run Examples

```bash
# Basic save and load example
python examples/basic_save_load.py

# Analyze model layers
python examples/layer_analysis.py

# Benchmark against torch.save()
python examples/comparison_benchmark.py
```

## Key Concepts

### Traditional PyTorch Checkpointing
```python
# Traditional approach
torch.save(model.state_dict(), 'model.pth')
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)
```

### TensorStore Checkpointing
```python
# TensorStore approach - layer by layer
for layer_name, param in model.named_parameters():
    save_parameter_to_tensorstore(layer_name, param, checkpoint_dir)
```

## Benefits of TensorStore Approach

1. **Efficient Storage**: Better compression and chunking for large arrays
2. **Partial Loading**: Load only specific layers or parameters
3. **Scalability**: Handles very large models efficiently
4. **Cross-platform**: Works across different systems and languages
5. **Analysis-friendly**: Each layer is separately accessible

## Model: OPT-125M

We use Facebook's OPT-125M model because it's:
- Small enough for quick experimentation (125M parameters)
- Representative of transformer architecture
- Pre-trained and readily available
- Good for demonstrating layer-wise storage concepts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is for educational purposes, demonstrating TensorStore usage with PyTorch models.
