# Checkpoints Directory

This directory contains model checkpoints saved in TensorStore format.

## Structure

```
checkpoints/
└── opt_125m_tensorstore/          # OPT-125M model in TensorStore format
    ├── model_metadata.json        # Model architecture and configuration
    ├── decoder/                   # Decoder layer parameters
    │   ├── layers.0.self_attn.k_proj.weight.zarr
    │   ├── layers.0.self_attn.v_proj.weight.zarr
    │   └── ...
    ├── embed_tokens/              # Embedding layer parameters
    └── embed_positions/           # Position embedding parameters
```

## Usage

The checkpoints in this directory are created and loaded using the TensorStore checkpointing utilities:

```python
from src import save_model_to_tensorstore, load_model_from_tensorstore

# Save model
save_model_to_tensorstore(model, "checkpoints/opt_125m_tensorstore/")

# Load model
model = load_model_from_tensorstore("checkpoints/opt_125m_tensorstore/")
```

## Benefits

- **Layer-wise access**: Each parameter is stored separately
- **Efficient storage**: Compressed and chunked storage
- **Partial loading**: Load only specific layers when needed
- **Cross-platform**: Compatible across different systems
