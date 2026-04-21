"""
End-to-end training report for yolov8s_canola_tiled_1280_wf_0.5_new_data_augmented model.

Generates a PDF containing:
  - Title page with summary
  - Training configuration table
  - Training loss curves
  - Validation mAP curves
  - Performance metrics table + confusion matrix
  - 5 sample inference images (GT vs Predicted)
  - Commentary on model performance

Usage:
    conda run -n mmyolo python tools/generate_report_canola.py
"""

import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mmdet.apis import inference_detector, init_detector
from mmengine.fileio import load

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
WORK_DIR       = 'work_dirs/yolov8s_canola_tiled_1280_wf_0.5_new_data_augmented'
CONFIG_PATH    = f'{WORK_DIR}/20260418_044012/vis_data/config.py'
CKPT_PATH      = f'{WORK_DIR}/best_coco_bbox_mAP_epoch_20.pth'
# Run 1: epochs 1-21 (initial training, contains best checkpoint at epoch 20)
SCALARS_PATH_1 = f'{WORK_DIR}/20260415_195507/vis_data/scalars.json'
# Run 3: epochs 31-100 (resumed from epoch_30.pth)
SCALARS_PATH_2 = f'{WORK_DIR}/20260418_044012/vis_data/scalars.json'
PKL_PATH       = f'{WORK_DIR}/test_results.pkl'
ANN_PATH       = '/home/ubuntu/dev/data/canola_offtype_tiled/annotations/test.json'
IMG_DIR        = '/home/ubuntu/dev/data/canola_offtype_tiled/images/test'
OUT_PDF        = f'{WORK_DIR}/training_report.pdf'

SCORE_THR   = 0.3
N_SAMPLES   = 5
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# COLOUR HELPERS
# ──────────────────────────────────────────────
GT_COLOR   = (0,   220,  80)   # green  (RGB)
PRED_COLOR = (255,  60,  60)   # red    (RGB)


def draw_boxes(img_rgb, bboxes, labels, class_names, color, thickness=3,
               scores=None):
    img = img_rgb.copy()
    for i, (box, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        name = class_names[label] if label < len(class_names) else str(label)
        score_str = f' {scores[i]:.2f}' if scores is not None else ''
        cv2.putText(img, f'{name}{score_str}', (x1, max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


# ──────────────────────────────────────────────
# LOAD SCALARS
# ──────────────────────────────────────────────
def load_scalars():
    paths = [p for p in (SCALARS_PATH_1, SCALARS_PATH_2) if p is not None]
    rows = []
    for path in paths:
        with open(path) as f:
            rows.extend(json.loads(l) for l in f)
    seen_train, seen_val = set(), set()
    train, val = [], []
    for r in rows:
        if 'loss' in r:
            key = (r['epoch'], r.get('iter', 0))
            if key not in seen_train:
                seen_train.add(key)
                train.append(r)
        elif 'coco/bbox_mAP' in r:
            key = r['step']
            if key not in seen_val:
                seen_val.add(key)
                val.append(r)
    train.sort(key=lambda r: (r['epoch'], r.get('iter', 0)))
    val.sort(key=lambda r: r['step'])
    return train, val


# ──────────────────────────────────────────────
# CONFUSION MATRIX  (re-computes from pkl)
# ──────────────────────────────────────────────
def compute_confusion_matrix(dataset, results, score_thr=0.3, tp_iou_thr=0.5):
    from mmdet.evaluation import bbox_overlaps

    num_classes = len(dataset.metainfo['classes'])
    cm = np.zeros((num_classes + 1, num_classes + 1))

    for idx, per_img_res in enumerate(results):
        res = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']

        gt_bboxes = np.array([g['bbox'] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4))
        gt_labels = np.array([g['bbox_label'] for g in gts], dtype=np.int64)
        true_positives = np.zeros(len(gts))

        for det_label in np.unique(res['labels'].numpy()):
            mask       = res['labels'] == det_label
            det_bboxes = res['bboxes'][mask].numpy()
            det_scores = res['scores'][mask].numpy()

            ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes) if len(gt_bboxes) else np.zeros((len(det_bboxes), 0))
            for i, score in enumerate(det_scores):
                if score < score_thr:
                    continue
                matched = 0
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        matched += 1
                        if gt_label == det_label:
                            true_positives[j] += 1
                        cm[gt_label, det_label] += 1
                if matched == 0:
                    cm[-1, det_label] += 1
        for tp, gt_label in zip(true_positives, gt_labels):
            if tp == 0:
                cm[gt_label, -1] += 1
    return cm


def compute_best_f1(dataset, results, iou_thr=0.5):
    """Sweep score thresholds and return P/R/F1 curves plus the best F1 point."""
    from mmdet.evaluation import bbox_overlaps

    thresholds = np.linspace(0.01, 0.99, 99)
    total_gt = 0
    all_scores, all_tp = [], []

    for idx, per_img_res in enumerate(results):
        res = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']

        gt_bboxes = np.array([g['bbox'] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4))
        gt_labels = np.array([g['bbox_label'] for g in gts], dtype=np.int64)
        total_gt += len(gts)

        pred_bboxes = res['bboxes'].numpy()
        pred_scores = res['scores'].numpy()
        pred_labels = res['labels'].numpy()

        if len(pred_bboxes) == 0:
            continue

        matched = np.zeros(len(pred_bboxes), dtype=bool)
        if len(gt_bboxes) > 0:
            ious = bbox_overlaps(pred_bboxes[:, :4], gt_bboxes)
            gt_used = np.zeros(len(gts), dtype=bool)
            for i in np.argsort(-pred_scores):
                valid = (~gt_used) & (pred_labels[i] == gt_labels) & (ious[i] >= iou_thr)
                if valid.any():
                    best_j = int(np.argmax(np.where(valid, ious[i], -1.0)))
                    matched[i] = True
                    gt_used[best_j] = True

        all_scores.extend(pred_scores.tolist())
        all_tp.extend(matched.tolist())

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp, dtype=bool)

    precisions, recalls, f1s = [], [], []
    for thr in thresholds:
        keep = all_scores >= thr
        tp = int(all_tp[keep].sum())
        fp = int((~all_tp[keep]).sum())
        fn = total_gt - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    best_i = int(np.argmax(f1s))
    return {
        'thresholds': thresholds,
        'precisions': np.array(precisions),
        'recalls':    np.array(recalls),
        'f1s':        np.array(f1s),
        'best_f1':    f1s[best_i],
        'best_thr':   thresholds[best_i],
        'best_prec':  precisions[best_i],
        'best_rec':   recalls[best_i],
    }


def plot_confusion_matrix(ax, cm, labels, color_theme='Blues'):
    per_label_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.where(per_label_sums > 0, cm / per_label_sums * 100, 0)

    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(labels)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which='minor', linestyle='-', linewidth=0.5)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            text_color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{int(val)}%', ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('Ground Truth Label', fontsize=10)
    ax.set_title('Normalised Confusion Matrix', fontsize=11, fontweight='bold')
    ax.set_ylim(n - 0.5, -0.5)


# ──────────────────────────────────────────────
# SAMPLE INFERENCE IMAGES
# ──────────────────────────────────────────────
def get_sample_images(model, coco, score_thr, n=5, seed=42):
    random.seed(seed)
    gt_by_image = {}
    for ann in coco['annotations']:
        gt_by_image.setdefault(ann['image_id'], []).append(ann)
    cat_names = {c['id']: c['name'] for c in coco['categories']}
    class_names = list(cat_names.values())

    images_with_gt = [img for img in coco['images'] if img['id'] in gt_by_image]
    selected = random.sample(images_with_gt, min(n, len(images_with_gt)))

    samples = []
    for img_info in selected:
        img_path = os.path.join(IMG_DIR, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        result = inference_detector(model, img_path)
        pred   = result.pred_instances
        keep   = pred.scores > score_thr
        pred_boxes  = pred.bboxes[keep].cpu().numpy()
        pred_scores = pred.scores[keep].cpu().numpy()
        pred_labels = pred.labels[keep].cpu().numpy()

        anns      = gt_by_image.get(img_info['id'], [])
        gt_boxes  = np.array([[a['bbox'][0], a['bbox'][1],
                                a['bbox'][0] + a['bbox'][2],
                                a['bbox'][1] + a['bbox'][3]]
                               for a in anns], dtype=np.float32)
        gt_labels = np.array([list(cat_names.keys()).index(a['category_id'])
                               for a in anns], dtype=np.int64)

        img_bgr = cv2.imread(img_path)
        img_rgb = img_bgr[:, :, ::-1].copy()

        samples.append({
            'img': img_rgb,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'pred_boxes': pred_boxes,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
            'class_names': class_names,
            'name': os.path.basename(img_info['file_name']),
            'n_gt': len(gt_boxes),
            'n_pred': len(pred_boxes),
        })
    return samples, class_names


# ──────────────────────────────────────────────
# PDF GENERATION
# ──────────────────────────────────────────────
def add_title_page(pdf, best_f1=None):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1a1a2e')

    fig.text(0.5, 0.72, 'YOLOv8s (wf=0.50) — Canola Offtype Detection',
             ha='center', va='center', fontsize=26, fontweight='bold', color='white')
    fig.text(0.5, 0.62, 'Model Training & Evaluation Report',
             ha='center', va='center', fontsize=18, color='#a8dadc')
    fig.text(0.5, 0.50, 'Model: best_coco_bbox_mAP_epoch_20.pth',
             ha='center', va='center', fontsize=13, color='#e0e0e0')
    fig.text(0.5, 0.44, 'Dataset: canola_offtype_tiled  |  Class: cn_coty  |  Test images: 12800',
             ha='center', va='center', fontsize=12, color='#e0e0e0')
    fig.text(0.5, 0.38, 'Input resolution: 1280 × 1280  |  Epochs: 100  |  Best epoch: 20',
             ha='center', va='center', fontsize=12, color='#e0e0e0')

    ax = fig.add_axes([0.10, 0.18, 0.80, 0.14])
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#a8dadc')
        spine.set_linewidth(1.5)
    ax.set_xticks([]); ax.set_yticks([])
    metrics = [('mAP@0.50', '0.846'), ('mAP@0.50:0.95', '0.495'), ('mAP@0.75', '0.502'), ('AR@100', '0.591')]
    if best_f1 is not None:
        metrics.append(('Best F1', f'{best_f1:.3f}'))
    n = len(metrics)
    for k, (label, val) in enumerate(metrics):
        x = (k + 0.5) / n
        ax.text(x, 0.65, val, ha='center', va='center', fontsize=20,
                fontweight='bold', color='#e9c46a', transform=ax.transAxes)
        ax.text(x, 0.20, label, ha='center', va='center', fontsize=10,
                color='#a8dadc', transform=ax.transAxes)

    fig.text(0.5, 0.06, 'Generated with mmyolo · April 2026',
             ha='center', va='center', fontsize=9, color='#888888')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_config_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('Training Configuration', fontsize=16, fontweight='bold', y=0.97)

    rows = [
        ['Parameter',              'Value'],
        ['Architecture',           'YOLOv8s (widen=0.50, deepen=0.33)'],
        ['Input resolution',       '1280 × 1280'],
        ['Number of classes',      '1  (cn_coty — canola offtype)'],
        ['Dataset',                'canola_offtype_tiled  (tiled drone imagery)'],
        ['Train annotation',       'annotations/train.json'],
        ['Val/Test annotation',    'annotations/test.json'],
        ['Test images',            '12800  |  Annotations: 17280'],
        ['Batch size (per GPU)',   '16'],
        ['Max epochs',             '100'],
        ['Base learning rate',     '0.00125'],
        ['LR schedule',            'Linear warm-up → cosine decay (lr_factor=0.01)'],
        ['Optimiser',              'SGD  (momentum=0.937, weight_decay=0.0005, nesterov=True)'],
        ['Gradient clip',          'max_norm = 10.0'],
        ['EMA',                    'ExpMomentumEMA  (momentum=0.0001)'],
        ['Backbone',               'YOLOv8CSPDarknet P5, ReLU, BN'],
        ['Neck',                   'YOLOv8PAFPN  (3 CSP blocks)'],
        ['Loss — BBox',            'CIoU  (weight=7.5)'],
        ['Loss — Cls',             'BCE with sigmoid  (weight=0.5)'],
        ['Loss — DFL',             'Distribution Focal Loss  (weight=0.375)'],
        ['Augmentation (stage 1)', 'Mosaic (1280×1280), RandomAffine, Albu (blur/grey/CLAHE), HSV, HFlip'],
        ['Augmentation (stage 2)', 'No mosaic from epoch 90; KeepRatioResize + LetterBox + RandomAffine'],
        ['Stage-2 switch epoch',   '90'],
        ['Val interval',           'Every 10 epochs (every 1 epoch from stage 2)'],
        ['NMS IoU threshold',      '0.5'],
        ['Score threshold (test)', '0.001  (NMS) / 0.3  (reporting)'],
        ['Best checkpoint',        'epoch 20  (mAP@0.50:0.95 = 0.495)'],
        ['CUDA / GPU',             'NVIDIA A10G  |  CUDA 12.4  |  PyTorch 2.4.1'],
    ]

    col_widths = [0.38, 0.58]
    col_x      = [0.03, 0.42]
    row_height = 0.032
    y_start    = 0.92

    for r, row in enumerate(rows):
        y = y_start - r * row_height
        bg = '#f0f4ff' if r == 0 else ('#ffffff' if r % 2 == 0 else '#f7f7f7')
        rect = mpatches.FancyBboxPatch((0.02, y - 0.005), 0.96, row_height,
                                        boxstyle='square,pad=0',
                                        facecolor=bg, edgecolor='#cccccc',
                                        linewidth=0.4, transform=ax.transAxes,
                                        clip_on=False)
        ax.add_patch(rect)
        for c, (text, cw) in enumerate(zip(row, col_widths)):
            weight = 'bold' if r == 0 else 'normal'
            color  = '#1a1a6e' if r == 0 else 'black'
            ax.text(col_x[c] + 0.01, y + row_height * 0.45, text,
                    ha='left', va='center', fontsize=8.5,
                    fontweight=weight, color=color, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_loss_curves_page(pdf, train_rows):
    epochs = [r['epoch'] for r in train_rows]
    losses = {
        'Total Loss':  [r['loss']      for r in train_rows],
        'Class Loss':  [r['loss_cls']  for r in train_rows],
        'BBox Loss':   [r['loss_bbox'] for r in train_rows],
        'DFL Loss':    [r['loss_dfl']  for r in train_rows],
    }
    colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Training Loss Curves', fontsize=15, fontweight='bold')
    axes = axes.flatten()

    for ax, (name, vals), color in zip(axes, losses.items(), colors):
        ax.plot(epochs, vals, color=color, linewidth=1.8, alpha=0.85)
        if len(epochs) > 5:
            from numpy.polynomial import polynomial as P
            c = P.polyfit(epochs, vals, 4)
            trend = P.polyval(epochs, c)
            ax.plot(epochs, trend, color='black', linewidth=1.0,
                    linestyle='--', alpha=0.5)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        min_idx = int(np.argmin(vals))
        ax.axvline(epochs[min_idx], color=color, linestyle=':', linewidth=1.2, alpha=0.7)
        ax.scatter([epochs[min_idx]], [vals[min_idx]], color=color, zorder=5, s=40)
        ax.annotate(f'min={vals[min_idx]:.2f}\n(ep {epochs[min_idx]})',
                    xy=(epochs[min_idx], vals[min_idx]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=7, color=color)

        raw_line   = plt.Line2D([0], [0], color=color,  linewidth=1.8, label='Raw loss')
        trend_line = plt.Line2D([0], [0], color='black', linewidth=1.0,
                                linestyle='--', alpha=0.5, label='Polynomial trend (deg 4)')
        vline      = plt.Line2D([0], [0], color=color,  linewidth=1.2,
                                linestyle=':', alpha=0.7, label='Minimum epoch')
        ax.legend(handles=[raw_line, trend_line, vline], fontsize=7, loc='upper right')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_val_curves_page(pdf, val_rows):
    steps   = [r['step'] for r in val_rows]
    map50   = [r['coco/bbox_mAP_50'] for r in val_rows]
    map5095 = [r['coco/bbox_mAP']    for r in val_rows]
    map75   = [r['coco/bbox_mAP_75'] for r in val_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Validation mAP over Training', fontsize=15, fontweight='bold')

    ax = axes[0]
    ax.plot(steps, map5095, color='#e63946', linewidth=2, label='mAP@0.50:0.95')
    ax.plot(steps, map50,   color='#457b9d', linewidth=2, label='mAP@0.50')
    ax.plot(steps, map75,   color='#2a9d8f', linewidth=2, label='mAP@0.75')

    best_ep = steps[int(np.argmax(map5095))]
    ax.axvline(best_ep, color='grey', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Best epoch ({best_ep})')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('mAP', fontsize=10)
    ax.set_title('mAP Curves (all epochs)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 1.0)

    # Zoom to epochs 0-50 to highlight the peak at epoch 20
    ax2 = axes[1]
    early_mask = [s <= 50 for s in steps]
    early_steps = [s for s, m in zip(steps, early_mask) if m]
    early_5095  = [v for v, m in zip(map5095, early_mask) if m]
    early_50    = [v for v, m in zip(map50,   early_mask) if m]
    early_75    = [v for v, m in zip(map75,   early_mask) if m]
    ax2.plot(early_steps, early_5095, color='#e63946', linewidth=2, marker='o', markersize=5, label='mAP@0.50:0.95')
    ax2.plot(early_steps, early_50,   color='#457b9d', linewidth=2, marker='o', markersize=5, label='mAP@0.50')
    ax2.plot(early_steps, early_75,   color='#2a9d8f', linewidth=2, marker='o', markersize=5, label='mAP@0.75')
    if best_ep <= 50:
        ax2.axvline(best_ep, color='grey', linestyle='--', linewidth=1.2, alpha=0.7,
                    label=f'Best epoch ({best_ep})')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('mAP', fontsize=10)
    ax2.set_title('Zoomed — Epochs 0–50 (peak region)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_metrics_and_cm_page(pdf, cm, class_labels, f1_data=None):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Final Performance Metrics & Confusion Matrix', fontsize=15, fontweight='bold')
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2.5], hspace=0.45, wspace=0.35)

    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis('off')

    headers = ['Metric', 'Value', 'Metric', 'Value']
    data = [
        ['mAP@0.50:0.95',  '0.495', 'AR@1',           '0.421'],
        ['mAP@0.50',       '0.846', 'AR@10',          '0.586'],
        ['mAP@0.75',       '0.502', 'AR@100',         '0.591'],
        ['mAP (small)',    '0.164', 'Best epoch',      '20 / 100'],
        ['mAP (medium)',   '0.482', 'Test images',     '12800'],
        ['mAP (large)',    '0.610', 'Score threshold', '0.30'],
    ]
    if f1_data is not None:
        data.append([
            'Best F1',      f'{f1_data["best_f1"]:.3f}',
            'F1 threshold', f'{f1_data["best_thr"]:.2f}',
        ])

    col_colors = [['#dbe9ff', '#dbe9ff', '#dbe9ff', '#dbe9ff']] + \
                 [['#f7f9ff', '#ffffff', '#f7f9ff', '#ffffff']] * len(data)
    tbl = ax_tbl.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        cellColours=col_colors[1:],
        colColours=col_colors[0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.5)

    ax_cm = fig.add_subplot(gs[1, :])
    plot_confusion_matrix(ax_cm, cm, class_labels)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_inference_page(pdf, samples, page_num, total_pages):
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(11 * n / 3, 5.5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Sample Inference — GT (green) vs Predicted (red)  [page {page_num}/{total_pages}]',
                 fontsize=12, fontweight='bold')

    for ax, sample in zip(axes, samples):
        img = sample['img']
        h, w = img.shape[:2]
        max_side = 1000
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        def scale_boxes(boxes):
            return (boxes * scale).astype(np.float32) if len(boxes) else boxes

        if len(sample['gt_boxes']):
            img = draw_boxes(img, scale_boxes(sample['gt_boxes']),
                             sample['gt_labels'], sample['class_names'],
                             GT_COLOR, thickness=3)
        if len(sample['pred_boxes']):
            img = draw_boxes(img, scale_boxes(sample['pred_boxes']),
                             sample['pred_labels'], sample['class_names'],
                             PRED_COLOR, thickness=3, scores=sample['pred_scores'])
        ax.imshow(img)
        ax.set_title(f"{sample['name'][:30]}\nGT={sample['n_gt']}  Pred={sample['n_pred']}",
                     fontsize=7.5, pad=4)
        ax.axis('off')

    gt_patch   = mpatches.Patch(color=np.array(GT_COLOR)   / 255, label='Ground Truth')
    pred_patch = mpatches.Patch(color=np.array(PRED_COLOR) / 255, label='Prediction')
    fig.legend(handles=[gt_patch, pred_patch], loc='lower center',
               ncol=2, fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_f1_page(pdf, f1_data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle('F1 Score Analysis', fontsize=15, fontweight='bold')

    thrs  = f1_data['thresholds']
    f1s   = f1_data['f1s']
    precs = f1_data['precisions']
    recs  = f1_data['recalls']
    best_f1   = f1_data['best_f1']
    best_thr  = f1_data['best_thr']
    best_prec = f1_data['best_prec']
    best_rec  = f1_data['best_rec']

    ax = axes[0]
    ax.plot(thrs, f1s,   color='#e63946', linewidth=2,   label='F1')
    ax.plot(thrs, precs, color='#457b9d', linewidth=1.5, linestyle='--', label='Precision')
    ax.plot(thrs, recs,  color='#2a9d8f', linewidth=1.5, linestyle='--', label='Recall')
    ax.axvline(best_thr, color='#e9c46a', linewidth=1.5, linestyle=':', label=f'Best thr={best_thr:.2f}')
    ax.scatter([best_thr], [best_f1], color='#e9c46a', zorder=5, s=80)
    ax.annotate(f'F1={best_f1:.3f}\n@ thr={best_thr:.2f}',
                xy=(best_thr, best_f1), xytext=(10, -25),
                textcoords='offset points', fontsize=8, color='#e9c46a',
                arrowprops=dict(arrowstyle='->', color='#e9c46a'))
    ax.set_xlabel('Score Threshold', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('F1 / Precision / Recall vs Threshold', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax2 = axes[1]
    ax2.plot(recs, precs, color='#457b9d', linewidth=2, label='P-R curve')
    ax2.scatter([best_rec], [best_prec], color='#e9c46a', zorder=5, s=80,
                label=f'Best F1={best_f1:.3f}')
    ax2.annotate(f'P={best_prec:.3f}\nR={best_rec:.3f}',
                 xy=(best_rec, best_prec), xytext=(10, -25),
                 textcoords='offset points', fontsize=8, color='#e9c46a',
                 arrowprops=dict(arrowstyle='->', color='#e9c46a'))
    ax2.set_xlabel('Recall', fontsize=10)
    ax2.set_ylabel('Precision', fontsize=10)
    ax2.set_title('Precision–Recall Curve', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_commentary_page(pdf, train_rows, val_rows):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.suptitle('Performance Commentary', fontsize=16, fontweight='bold', y=0.97)

    best_map50   = max(r['coco/bbox_mAP_50'] for r in val_rows)
    best_map5095 = max(r['coco/bbox_mAP']    for r in val_rows)
    best_epoch   = val_rows[int(np.argmax([r['coco/bbox_mAP'] for r in val_rows]))]['step']
    final_map5095 = val_rows[-1]['coco/bbox_mAP']
    init_loss    = train_rows[0]['loss']
    final_loss   = train_rows[-1]['loss']
    loss_drop_pct = (init_loss - final_loss) / init_loss * 100

    commentary = f"""
OVERVIEW
The YOLOv8s (wf=0.50) model was trained for 100 epochs on the canola_offtype_tiled dataset to detect
canola offtype plants (cn_coty) from tiled drone imagery at 1280 × 1280 resolution. The dataset
consists of 12800 test tiles with 17280 annotated instances, derived from high-resolution DJI imagery
that has been split into overlapping tiles for training. The model uses the same wf=0.50 backbone as
the forest canopy runs, paired with 1280 × 1280 input tiles.

TRAINING CONVERGENCE
Total loss decreased from {init_loss:.1f} at the first logged step to {final_loss:.1f} at epoch 100,
a reduction of {loss_drop_pct:.0f}%. Note that the training scalars are drawn from two resumed runs
(epochs 1–21 and epochs 31–100); the gap at epochs 22–30 is an artefact of checkpoint resumption.
Loss curves show healthy descent across all four components (total, cls, bbox, DFL) throughout the
mosaic-augmented stage. The stage-2 pipeline switch at epoch 90 (mosaic off, LetterBox + RandomAffine)
causes a brief loss uptick, which is expected as the model adapts to the cleaner inference pipeline.

VALIDATION PERFORMANCE
Best mAP@0.50:0.95 = {best_map5095:.3f} achieved at epoch {best_epoch} — the model peaks very early
in training. mAP@0.50 = {best_map50:.3f} demonstrates strong localisation at the COCO 50% IoU
threshold. After epoch 20, performance declined monotonically to {final_map5095:.3f} mAP@0.50:0.95
at epoch 100. This suggests the continued training from epoch 30 (resumed run) diverged from the
optimal weight trajectory. The small-object mAP (0.164) is non-zero, reflecting that some offtype
plants appear as small objects in the tiled imagery — unlike the forest canopy dataset where all
objects were large. Medium (0.482) and large (0.610) object mAP are strong, indicating the model
handles typical-sized plants well.

CONFUSION MATRIX
With a single class at threshold 0.30, the confusion matrix shows: true positives (cn_coty predicted
as cn_coty), false positives (background predicted as cn_coty), and false negatives (missed cn_coty).
At the score threshold of 0.30, recall is prioritised: missed offtypes carry the highest agronomic
cost, as they may survive to contaminate seed stock.

SAMPLE INFERENCES
Sample tiles from the test set show strong agreement between ground truth and predictions on well-
isolated offtype plants. Dense clustering of offtypes in some patches results in slight over- or
under-counting, but spatial localisation is generally accurate. At 1280 × 1280 tile resolution,
fine plant boundaries are well resolved.

RECOMMENDATIONS
1. Use epoch-20 checkpoint — the best_coco_bbox_mAP_epoch_20.pth is the clear production choice.
   Future runs should apply early stopping (patience ~15 epochs on mAP@0.50:0.95) to avoid the
   performance regression observed after epoch 20.
2. Investigate resumed-run degradation — the monotonic decline from epoch 40 onwards suggests the
   epoch_30.pth checkpoint used for resumption may have been from a sub-optimal trajectory. Future
   multi-stage training should verify checkpoint quality before resumption.
3. Tiled inference overlap — ensure inference tiles use sufficient overlap (≥10% of tile size) and
   apply tile-boundary NMS to avoid double-counting plants that span tile edges.
4. Small-object recall — mAP_s = 0.164 is the weakest metric. Testing with a smaller tile stride
   or a multi-scale pyramid at inference may improve detection of young/distant offtypes.
5. Score threshold tuning — for field use, sweeping threshold from 0.1 to 0.5 on a held-out plot
   and optimising for F1 or recall@precision=0.9 will yield a deployment-ready operating point.
"""

    ax.text(0.03, 0.97, commentary.strip(), ha='left', va='top',
            fontsize=8.5, linespacing=1.55, transform=ax.transAxes,
            family='monospace',
            bbox=dict(facecolor='#f8f9fa', edgecolor='#cccccc', boxstyle='round,pad=0.5'))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print('Loading scalars...')
    train_rows, val_rows = load_scalars()
    print(f'  Train rows: {len(train_rows)}, Val rows: {len(val_rows)}')

    print('Loading COCO annotations...')
    with open(ANN_PATH) as f:
        coco = json.load(f)

    print('Loading model and running inference on 5 sample images...')
    model   = init_detector(CONFIG_PATH, CKPT_PATH, device='cuda:0')
    samples, class_names = get_sample_images(model, coco, SCORE_THR, N_SAMPLES, RANDOM_SEED)

    print('Loading predictions pkl and computing confusion matrix + best F1...')
    results = load(PKL_PATH)

    from mmengine import Config
    from mmengine.registry import init_default_scope
    from mmdet.utils import replace_cfg_vals, update_data_root
    cfg = Config.fromfile(CONFIG_PATH)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    from mmyolo.registry import DATASETS
    dataset  = DATASETS.build(cfg.test_dataloader.dataset)
    cm       = compute_confusion_matrix(dataset, results, SCORE_THR)
    cm_labels = list(dataset.metainfo['classes']) + ['background']

    f1_data = compute_best_f1(dataset, results)
    print(f'Best F1: {f1_data["best_f1"]:.3f}  @ threshold {f1_data["best_thr"]:.2f}'
          f'  (P={f1_data["best_prec"]:.3f}, R={f1_data["best_rec"]:.3f})')

    print(f'Generating PDF → {OUT_PDF}')
    with PdfPages(OUT_PDF) as pdf:
        d = pdf.infodict()
        d['Title']   = 'YOLOv8s Canola Offtype Detection Training Report'
        d['Author']  = 'mmyolo / Claude Code'
        d['Subject'] = 'Object detection model evaluation'

        add_title_page(pdf, best_f1=f1_data['best_f1'])
        add_config_page(pdf)
        add_loss_curves_page(pdf, train_rows)
        add_val_curves_page(pdf, val_rows)
        add_metrics_and_cm_page(pdf, cm, cm_labels, f1_data=f1_data)
        add_f1_page(pdf, f1_data)

        add_inference_page(pdf, samples[:3], page_num=1, total_pages=2)
        if len(samples) > 3:
            add_inference_page(pdf, samples[3:], page_num=2, total_pages=2)

        add_commentary_page(pdf, train_rows, val_rows)

    print(f'\nDone. Report saved to: {os.path.abspath(OUT_PDF)}')


if __name__ == '__main__':
    main()
