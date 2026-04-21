# MMYOLO Installation Guide

## Package Versions

| Package   | Version |
|-----------|---------|
| Python    | 3.8     |
| PyTorch   | 2.4.1 + CUDA 12.4 |
| mmcv      | 2.1.0 (built from source) |
| mmdet     | 3.3.0   |
| mmengine  | 0.10.7  |
| mmyolo    | 0.6.0 (editable install) |

## Why not just `mim install -v -e .`?

The standard docs command fails due to three layered incompatibilities:

1. **No pre-built mmcv binary wheels for CUDA 12.4** — OpenMMLab's CDN only provides pre-built wheels for CUDA 11.8 (`cu118`) and CUDA 12.1 (`cu121`). CUDA 12.4 (`cu124`) has no index.
2. **mmcv 2.0.1 source build uses `-std=c++14`** — incompatible with PyTorch 2.4.x headers (which require C++17).
3. **Version cap conflicts** — the original mmyolo code capped `mmcv<2.1.0` and mmdet 3.3.0 caps `mmcv<2.2.0`. Only mmcv 2.1.0 satisfies both.

### Why mmcv 2.1.0 specifically?

- **mmcv 2.0.1**: `setup.py` unconditionally uses `-std=c++14` → build fails against PyTorch 2.4.x headers
- **mmcv 2.1.0**: `setup.py` uses `-std=c++17` when `torch > 1.12.1` → builds successfully ✓
- **mmcv 2.2.0**: Pre-built wheel available but mmdet 3.3.0 has a hard runtime check rejecting it

## Source Patches Required

Two files in the repo must be patched before installing (already applied on `drone-model-training` branch):

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

## Installation Steps

There are two options.

### Option A: From `environment.yml` (faster, but mmcv still needs a manual build step)

> **Note:** `environment.yml` captures all packages except mmcv. mmcv has been intentionally
> excluded because no pre-built wheel exists for CUDA 12.4 — it must be built from source
> using the explicit `--no-binary mmcv` flag below, otherwise the install will fail.

```bash
# 1. Create and activate the environment from the exported package list
conda env create -f environment.yml
conda activate mmyolo

# 2. Build mmcv 2.1.0 from source (~15–20 minutes)
#    --no-binary forces a source build; required because no pre-built wheel exists for CUDA 12.4
pip install mmcv==2.1.0 --no-binary mmcv

# 3. Install mmyolo in editable mode
mim install -v -e .
```

### Option B: From scratch

> **Note:** Building mmcv from source takes ~15–20 minutes.

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

# Build mmcv 2.1.0 from source
pip install mmcv==2.1.0 --no-binary mmcv

# Install albumentations
mim install -r requirements/albu.txt

# Install mmyolo in editable mode
mim install -v -e .
```
