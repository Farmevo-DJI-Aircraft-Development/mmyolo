# MMYOLO Project Context

## Environment

- **Conda env**: `mmyolo` (Python 3.8)
- **Activate**: `conda activate mmyolo`
- **Working dir**: `/home/ubuntu/dev/mmyolo`
- **Branch**: `drone-model-training`

## Installed Package Versions

| Package   | Version |
|-----------|---------|
| Python    | 3.8     |
| PyTorch   | 2.4.1 + CUDA 12.4 |
| mmcv      | 2.1.0 (built from source) |
| mmdet     | 3.3.0   |
| mmengine  | 0.10.7  |
| mmyolo    | 0.6.0 (editable install) |

## Installation Notes

### Problem with `mim install -v -e .`

The docs instruct you to run `mim install -v -e .` to install mmyolo and its dependencies. This fails on this machine due to three layered incompatibilities:

1. **No pre-built mmcv binary wheels for CUDA 12.4** — OpenMMLab's CDN only provides pre-built wheels for CUDA 11.8 (`cu118`) and CUDA 12.1 (`cu121`). CUDA 12.4 (`cu124`) has no index.
2. **mmcv 2.0.1 source build uses `-std=c++14`** — falls back to building from source, but mmcv 2.0.1's `setup.py` hardcodes `-std=c++14` which is incompatible with PyTorch 2.4.x headers (which require C++17).
3. **Version cap conflicts** — the original mmyolo code capped `mmcv<2.1.0` and mmdet 3.3.0 caps `mmcv<2.2.0`. Only mmcv 2.1.0 satisfies both.

### Fixes Applied

Two source files were modified to allow mmcv 2.1.0:

**`requirements/mminstall.txt`**:
```
# Before: mmcv>=2.0.0rc4,<2.1.0
# After:
mmcv>=2.0.0rc4,<2.2.0
```

**`mmyolo/__init__.py`** (line 10):
```python
# Before: mmcv_maximum_version = '2.1.0'
# After:
mmcv_maximum_version = '2.2.0'
```

### Correct Installation Sequence

If you need to rebuild the environment from scratch:

```bash
conda create -n mmyolo python=3.8 -y
conda activate mmyolo

# Install PyTorch (CUDA 12.4)
conda install pytorch torchvision -c pytorch

# Install OpenMIM
pip install -U openmim

# Install mim dependencies (mminstall.txt now allows mmcv<2.2.0)
mim install -r requirements/mminstall.txt

# Install ninja to speed up C++ extension builds
conda install -c conda-forge ninja

# Build mmcv 2.1.0 from source (mmcv 2.1.0 correctly uses C++17 for torch > 1.12.1)
# This takes ~15-20 minutes
pip install mmcv==2.1.0 --no-binary mmcv

# Install albumentations
mim install -r requirements/albu.txt

# Install mmyolo in editable mode (mim will skip mmcv since it's already installed)
mim install -v -e .
```

### Why mmcv 2.1.0 (not 2.0.1 or 2.2.0)?

- **mmcv 2.0.1**: `setup.py` unconditionally uses `-std=c++14` → build fails against PyTorch 2.4.x headers
- **mmcv 2.1.0**: `setup.py` uses `-std=c++17` when `torch > 1.12.1` → builds successfully ✓
- **mmcv 2.2.0**: Pre-built wheel available but mmdet 3.3.0 has a hard runtime check rejecting it (`AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0.`)

## Stray Files in Project Root

The files `=0.7.1`, `=2.0.0rc4,`, and `=3.0.0` in the project root are harmless artifacts from malformed pip install commands (e.g., `pip install mmyolo=0.7.1` instead of `==0.7.1`). They can be safely deleted.

## Project Overview

This is the `drone-model-training` branch of MMYOLO, with a DJI patch applied (`patch: apply dji patch to the repo for drone compatible model`). MMYOLO is built on OpenMMLab 2.0 and supports YOLO-series algorithms (YOLOv5, YOLOv6, YOLOv7, YOLOv8, RTMDet, etc.).

Key directories:
- `configs/` — model configs (YOLOv5, YOLOv8, RTMDet, etc.)
- `mmyolo/` — source code
- `tools/` — training, testing, and utility scripts
- `data/` — datasets (not committed)
- `requirements/` — dependency files
