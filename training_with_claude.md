# YOLOv8 Training Setup Log

## Dataset
- Location: `/home/ubuntu/dev/mmyolo/data/cn_coty/`
- Structure:
  - `images/train/` — training images (DJI drone photos)
  - `images/test/` — test images
  - `annotations/train.json` — COCO format annotations
  - `annotations/test.json` — COCO format annotations
- 504 images, 2,693 annotations
- 1 class: `cn_coty`

## Config File
`configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`

Changes made:
- `data_root` → `/home/ubuntu/dev/mmyolo/data/cn_coty/`
- `train_ann_file` → `annotations/train.json`
- `train_data_prefix` → `images/train/`
- `val_ann_file` → `annotations/test.json`
- `val_data_prefix` → `images/test/`
- `num_classes` → `1`

## Issues Encountered & Fixes

### 1. ONNX export before training (DJI patch)
`tools/train.py` lines 120-122 (added by DJI patch) try to export the untrained model to ONNX before `runner.train()` is called.

**Fix:** Install onnx:
```bash
conda run -n mmyolo pip install onnx
```

### 2. Albumentations incompatibility
`albumentations 1.4.18` raises `ValueError: Key img_path is not in available keys` during data loading.

**Fix:** Downgrade albumentations:
```bash
conda run -n mmyolo pip install albumentations==1.3.1
```

## Training Command

```bash
conda activate mmyolo
cd /home/ubuntu/dev/mmyolo
python tools/train.py configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py --work-dir ./work_dirs/yolov8s_cn_coty
```

## Monitor Progress

```bash
tail -f work_dirs/yolov8s_cn_coty/<run_timestamp>/<run_timestamp>.log
```

## Installed Packages (added during this session)
- `onnx==1.17.0`
- `albumentations==1.3.1` (downgraded from 1.4.18)

---

## Annotation Verification

Used `tools/visualize_annotations.py` to overlay bounding boxes on a random sample of images:

```bash
python tools/visualize_annotations.py \
    --ann data/cn_coty/annotations/train.json \
    --img-dir data/cn_coty/images/train \
    --out-dir vis_annotations/train \
    --num-images 20 \
    --scale 0.25
```

Output written to `vis_annotations/train/`. Annotations confirmed correct.

---

## Image Tiling (Critical Fix)

### Problem

Training loss showed `loss_bbox: 0.0000` and `loss_dfl: 0.0000` from epoch 1 — only `loss_cls` was active and abnormally large (~3200 at epoch 1). The model made no positive assignments.

**Root cause:** DJI images are 8064×6048. A typical annotation box is ~43×40px. At YOLOv8's default 640×640 input:

```
scale = 640 / 8064 ≈ 0.079
43px × 0.079 ≈ 3.4px  →  smaller than stride-8 cells (8×8px)
```

The task-aligned assigner found no anchor points within any box, so bbox and DFL losses were permanently zero.

Secondary issue: `data_time: 5.5s` vs compute `~0.8s` — loading 8K images was the bottleneck.

### Fix: Tile the dataset

Created `tools/tile_coco_dataset.py` to slice large images into 1024×1024 tiles with 20% overlap, adjusting COCO annotations to tile-local coordinates.

```bash
# Tile training set
python tools/tile_coco_dataset.py \
    --ann      data/cn_coty/annotations/train.json \
    --img-dir  data/cn_coty/images/train \
    --out-dir  data/cn_coty_tiled \
    --split    train \
    --tile-size 1024 \
    --overlap   0.2 \
    --min-bbox-area 16

# Tile test set
python tools/tile_coco_dataset.py \
    --ann      data/cn_coty/annotations/test.json \
    --img-dir  data/cn_coty/images/test \
    --out-dir  data/cn_coty_tiled \
    --split    test \
    --tile-size 1024 \
    --overlap   0.2 \
    --min-bbox-area 16
```

Results:
- Train: 504 images → 4,410 tiles, 2,693 → 5,565 annotations
- Test: 126 images → 987 tiles, ~683 → 1,209 annotations

At 1024→640 resize, boxes are now ~27×25px — well within the stride-8 detection range.

### Config changes after tiling

In `configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`:

```python
# Changed data_root to tiled dataset
data_root = '/home/ubuntu/dev/mmyolo/data/cn_coty_tiled/'

# Lowered min_size filter (was 32, would drop valid small boxes)
# Also enabled filter_empty_gt since tiling already excluded empty tiles
filter_cfg=dict(filter_empty_gt=True, min_size=8)

# Required for custom class name — must be set on both train and val dataset dicts
metainfo=dict(classes=('cn_coty',))
```

### 3. ValueError: need at least one array to concatenate

After switching to the tiled dataset, training crashed immediately with:

```
ValueError: need at least one array to concatenate
```

**Root cause:** `YOLOv5CocoDataset` inherits from mmdet's `CocoDataset`, which has 80 COCO class names hardcoded in its `METAINFO`. During `load_data_list()`, it calls:

```python
self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
```

Since `cn_coty` is not in the 80 COCO class names, `cat_ids` is empty. All annotations are dropped, `filter_empty_gt=True` then removes all images, and `_serialize_data()` receives an empty list → crash.

**Fix:** Add `metainfo=dict(classes=('cn_coty',))` to both the `train_dataloader` and `val_dataloader` dataset dicts in the config. This overrides the inherited COCO class list so the category lookup finds `cn_coty`.

**Note:** `filter_empty_gt=True` is correct and should stay. It was doing its job — it exposed the metainfo bug instead of silently training on images with no annotations.

### 4. IndexError: list index out of range (val evaluation)

At epoch 10 validation, `coco_metric.py` crashed with:

```
data['category_id'] = self.cat_ids[label]
IndexError: list index out of range
```

**Root cause:** `val_dataloader` was missing `metainfo=dict(classes=('cn_coty',))`. The inherited 80-class COCO list produced an empty `cat_ids` for `cn_coty`, so indexing into it with any label failed.

**Fix:** Add `metainfo=dict(classes=('cn_coty',))` to the `val_dataloader` dataset dict (same as `train_dataloader`).

---

## Run 1 Results — YOLOv8-S, 50 epochs, from scratch

**Checkpoint:** `work_dirs/yolov8s_cn_coty_tiled/best_coco_bbox_mAP_epoch_46.pth`

**Config:** `configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`

Key settings:
- Model: YOLOv8-S (`deepen_factor=0.33`, `widen_factor=0.5`)
- Pretrained weights: none (trained from scratch)
- Epochs: 50 (`close_mosaic_epochs=10`)
- Batch size: 32, `base_lr=0.01`
- Input: 640×640, tiled dataset (1024px tiles, 20% overlap)
- Rotation augmentation: disabled (`max_rotate_degree=0.0`)

**COCO metrics (epoch 46 best):**

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 (all) | 0.477 |
| mAP@0.50 | 0.730 |
| mAP@0.75 | 0.550 |
| mAP_s (small) | 0.100 |
| mAP_m (medium) | 0.441 |
| mAP_l (large) | 0.589 |
| AR@100 (all) | 0.638 |
| AR@100 (small) | 0.244 |

**Key observations:**
- Small object detection is the main weakness (mAP_s = 0.100, AR_s = 0.244)
- Large/medium objects detected well
- Model likely still improving at epoch 50 (training from scratch needs more epochs)

---

## Suggested Improvements for Run 2

Ranked by expected impact:

1. **COCO pretrained weights** — add to config:
   ```python
   load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'
   ```
   Expected gain: +5–15 mAP. Head weights ignored (class count mismatch); backbone/neck fine-tunes from strong starting point.

2. **Rotation augmentation** — drone targets appear at any angle, currently disabled:
   ```python
   max_rotate_degree = 180.0  # in both train_pipeline and train_pipeline_stage2
   ```

3. **More epochs** — loss still descending at epoch 50:
   ```python
   max_epochs = 150
   close_mosaic_epochs = 15
   ```

4. **Upgrade to YOLOv8-M** — larger capacity, typically +3–5 mAP:
   ```python
   deepen_factor = 0.67
   widen_factor = 0.75
   ```
   Use the matching pretrained checkpoint for YOLOv8-M.

5. **Increase tile overlap** — reduces objects cut at tile boundaries (addresses mAP_s):
   ```bash
   python tools/tile_coco_dataset.py ... --overlap 0.35
   ```

6. **Enable `batch_shapes_cfg`** — ~+0.02 mAP at validation (uncomment block in config).

7. **Test-Time Augmentation (TTA)** — free mAP boost at inference, no retraining needed:
   ```bash
   python tools/test.py <config> <checkpoint> --tta
   ```

---

## Visualizing Predictions

### On tiled test images
```bash
conda run -n mmyolo python demo/image_demo.py \
    data/cn_coty_tiled/images/test/ \
    work_dirs/yolov8s_cn_coty_tiled/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    work_dirs/yolov8s_cn_coty_tiled/best_coco_bbox_mAP_epoch_46.pth \
    --out-dir vis_predictions/ \
    --score-thr 0.3
```

### On full-resolution DJI images (recommended)
`large_image_demo.py` handles tiling internally and stitches detections back onto the original image:
```bash
conda run -n mmyolo python demo/large_image_demo.py \
    data/cn_coty/images/test/ \
    work_dirs/yolov8s_cn_coty_tiled/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    work_dirs/yolov8s_cn_coty_tiled/best_coco_bbox_mAP_epoch_46.pth \
    --out-dir vis_predictions_full/ \
    --score-thr 0.3 \
    --patch-size 1024 \
    --patch-overlap-ratio 0.2
```

### Shell gotcha
`image_demo.py` does accept directories — the `img` argument supports "image file, dir and URL".
Line-continuation `\` must have **no trailing space** after it, or bash treats each line as a separate command.

### Verifying predictions are real (not ground truth replayed)
`image_demo.py` uses `draw_gt=False` (line 148) and runs `inference_detector()` per image — outputs are
genuine model predictions. If results look suspiciously good, check for train/test leakage (shared filenames
between `annotations/train.json` and `annotations/test.json`).

---

## Feature Map Visualization

`demo/featmap_vis_demo.py` visualizes internal activations overlaid on the prediction image.
Useful for understanding what the model attends to.

```bash
conda run -n mmyolo python demo/featmap_vis_demo.py \
    data/cn_coty_tiled/images/test/some_tile.jpg \
    work_dirs/yolov8s_cn_coty_tiled/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    work_dirs/yolov8s_cn_coty_tiled/best_coco_bbox_mAP_epoch_46.pth \
    --out-dir vis_featmaps/ \
    --target-layers backbone
```

Key options:
- `--preview-model` — print all layer names without running inference
- `--target-layers` — layer(s) to hook (default: `backbone`; can be e.g. `backbone.stage4`, `neck`)
- `--channel-reduction select_max` (default) — picks highest-activation channel; `squeeze_mean` averages all
- `--topk 4 --arrangement 2 2` — show top-4 channels in a 2×2 grid (when `--channel-reduction None`)

---

## GT vs Prediction Overlay

`tools/infer_with_gt.py` — runs inference on N random test images and draws **both GT and predicted boxes on the same image** for visual verification.

- Green boxes = ground truth
- Red boxes = model predictions (with confidence score)

```bash
conda run -n mmyolo python tools/infer_with_gt.py \
    --config work_dirs/yolov8s_cn_coty_tiled/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    --checkpoint work_dirs/yolov8s_cn_coty_tiled/best_coco_bbox_mAP_epoch_46.pth \
    --ann data/cn_coty_tiled/annotations/test.json \
    --img-dir data/cn_coty_tiled/images/test \
    --out-dir vis_gt_vs_pred \
    --n 5 --score-thr 0.3
```

Results saved to `vis_gt_vs_pred/`. Use `--seed` to control which images are sampled.
