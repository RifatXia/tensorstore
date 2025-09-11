# OPT-125M Checkpointing with TensorStore

This folder contains a Jupyter notebook that:
- Downloads `facebook/opt-125m` from Hugging Face and saves a local checkpoint (config + weights + tokenizer).
- Exports the PyTorch `state_dict()` to TensorStore-backed Zarr arrays (one array per parameter) for portable reuse.
- Reloads the model from the TensorStore export and verifies parity with the original weights.

## Environment

You can use conda or venv. Example with conda:

```bash
conda create -n opt125 python=3.10 -y
conda activate opt125
pip install torch torchvision torchaudio transformers notebook tensorstore
jupyter notebook
```

Alternatively with venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio transformers notebook tensorstore
jupyter notebook
```

## Notebook: What it does

1) Save Hugging Face checkpoint (relative paths)
- Loads `facebook/opt-125m` and tokenizer.
- Saves to `./opt125_checkpoint` using `model.save_pretrained` and `tokenizer.save_pretrained`.

2) Export weights via TensorStore (PyTorch → Zarr)
- Iterates over `model.state_dict()` and writes each parameter as a Zarr array:
  - Driver: `zarr`
  - Storage: local filesystem via TensorStore file driver
  - Directory layout: `./opt125_ts/weights/{param_name}` (nested directories allowed)
  - Metadata includes dtype, shape, optional chunks, and Zstd compression
- Writes a minimal `./opt125_ts/config.json` mirroring `save_pretrained` so the model can be re-instantiated offline.

3) Reload from TensorStore
- Recursively finds all Zarr arrays under `./opt125_ts/weights`, reads them into NumPy, and builds a PyTorch `state_dict`.
- Instantiates a new model from `./opt125_ts/config.json` and loads the `state_dict`.
- Compares generated text and hashes of the state dicts to confirm exact parity.

## Step-by-step: How saving works

- Extract state:
  - Call `model.state_dict()` and move tensors to CPU for serialization.
- Prepare per-parameter target:
  - For each parameter name (e.g., `model.decoder.embed_tokens.weight`), compute a filesystem path under `./opt125_ts/weights/{param_name}`.
  - Create parent directories as needed; parameter names can form nested folders.
- Choose array settings:
  - Determine `dtype` and `shape` from the NumPy view of the tensor.
  - Optionally compute `chunks` (a simple heuristic reduces last-dimension chunks to target ~1MB per chunk).
  - Use Zstd compression with moderate level (e.g., 3) and row-major (`order: "C"`).
- Open a TensorStore array:
  - `ts.open` with `driver: zarr` and `kvstore: {driver: file, path: <param_path>}`.
  - Pass `metadata` (dtype, shape, chunks, compressor) or specify `dtype`/`shape` via `ts.open` arguments depending on API style.
- Write data:
  - Call `.write(np_array).result()` (or await in async style) to persist the parameter.
- Save minimal config:
  - Write `./opt125_ts/config.json` from `model.config` so a fresh model can be constructed without internet access.

## Step-by-step: How loading works

- Recreate empty model:
  - Load `cfg = AutoConfig.from_pretrained("./opt125_ts")` and build an empty model with `AutoModelForCausalLM.from_config(cfg)`.
- Discover arrays:
  - Walk `./opt125_ts/weights` and detect Zarr arrays by presence of `.zarray` files.
  - For each array directory, compute the relative parameter key (directory path relative to `weights` root, with `/` separators).
- Read tensors:
  - `ts.open({driver: zarr, kvstore: {driver: file, path: <array_dir>}}, read=True)` then `.read().result()` to get a NumPy array.
  - Convert to `torch.from_numpy(np_array)` and add to a `state_dict` mapping using the computed key.
- Load weights:
  - Call `missing, unexpected = model.load_state_dict(state_dict, strict=False)` to populate parameters (reporting any key mismatches).
- Verify parity (optional):
  - Compare a small generation output from the original vs. reloaded model.
  - Hash the tensors of both `state_dict`s to verify exact equality.

## Hash verification

- A deterministic SHA-256 digest is computed over the `state_dict` by iterating keys in sorted order, mixing in each parameter name and the raw bytes of its CPU, contiguous tensor value.
- Any difference in dtype, shape, layout, or values changes the digest. Identical hashes imply byte-for-byte identical `state_dict` contents.
- In the notebook, the hash is computed for the original and the reloaded model; equality confirms a lossless round trip through TensorStore.

## File layout

- `./opt125_checkpoint/` — Standard Hugging Face checkpoint (config, tokenizer files, PyTorch bin if applicable).
- `./opt125_ts/`
  - `config.json` — Model config saved for re-instantiation.
  - `weights/` — One Zarr store per parameter (directories contain `.zarray`, `.zattrs`, chunk files, etc.).

## Why TensorStore/Zarr?

- Chunked, compressed arrays that scale to large weights.
- Backend-agnostic storage specs (can switch drivers or backends without changing model code).
- Fine-grained I/O (load or update specific parameters without monolithic archives).

## How to run

- Open `opt125_demo.ipynb` in Jupyter and run cells in order.
- Ensure you have network access for the initial model download (Hugging Face will cache files locally).
- After the export, you can work fully offline using `./opt125_checkpoint` or `./opt125_ts`.

## Notes

- The notebook uses only relative paths (e.g., `./opt125_checkpoint`, `./opt125_ts`).
- TensorStore Zarr metadata is created via TensorStore; dtype/shape are provided explicitly. Chunks use a simple heuristic in the notebook; tune as needed.