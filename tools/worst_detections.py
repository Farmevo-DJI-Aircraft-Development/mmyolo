"""
Run inference on all validation images, score each image by detection quality,
and save the top-K worst images with GT (green) + prediction (red) boxes overlaid.

Scoring per image:
  score = n_missed_gt + n_false_positives
  where:
    n_missed_gt    = GT boxes with max IoU against any prediction < iou_thr
    n_false_pos    = predicted boxes with max IoU against any GT < iou_thr
  Higher score = worse detection.

Usage:
    python tools/worst_detections.py \\
        --config  work_dirs/yolov8s_forest_canopy/yolov8_s_syncbn_fast_8xb16-500e_coco.py \\
        --checkpoint work_dirs/yolov8s_forest_canopy/best_coco_bbox_mAP_epoch_91.pth \\
        --ann     /home/ubuntu/dev/data/forest_canopy_OD_v1/annotations/instances_val.json \\
        --img-dir /home/ubuntu/dev/data/forest_canopy_OD_v1 \\
        --out-dir work_dirs/yolov8s_forest_canopy/topk50_worst_detections \\
        --topk 50 \\
        --score-thr 0.3 \\
        --iou-thr 0.5
"""
import argparse
import json
import os

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Top-K worst detection visualiser')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann', required=True, help='COCO annotation JSON (val set)')
    parser.add_argument('--img-dir', required=True,
                        help='Root directory containing images (file_name is relative to this)')
    parser.add_argument('--out-dir', default='topk50_worst_detections')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou-thr', type=float, default=0.5,
                        help='IoU threshold for matching GT <-> pred')
    parser.add_argument('--thickness', type=int, default=8,
                        help='Bounding box line thickness in pixels')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def iou_matrix(boxes_a, boxes_b):
    """Compute IoU between every pair of boxes.
    boxes_a: (N, 4) xyxy, boxes_b: (M, 4) xyxy
    Returns: (N, M) IoU matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a[:, None] + area_b[None, :] - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


def compute_score(gt_boxes, pred_boxes, iou_thr):
    """Return (score, n_missed_gt, n_fp, per_gt_max_iou).

    score = n_missed_gt + n_fp
    """
    iou = iou_matrix(gt_boxes, pred_boxes)  # (G, P)

    if len(gt_boxes) == 0:
        n_missed_gt = 0
        per_gt_max_iou = np.array([])
    else:
        per_gt_max_iou = iou.max(axis=1) if iou.size > 0 else np.zeros(len(gt_boxes))
        n_missed_gt = int((per_gt_max_iou < iou_thr).sum())

    if len(pred_boxes) == 0:
        n_fp = 0
    else:
        per_pred_max_iou = iou.max(axis=0) if iou.size > 0 else np.zeros(len(pred_boxes))
        n_fp = int((per_pred_max_iou < iou_thr).sum())

    score = n_missed_gt + n_fp
    return score, n_missed_gt, n_fp, per_gt_max_iou


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_boxes(img_rgb, bboxes, labels, class_names, color, thickness=2,
               scores=None, ious=None):
    """Draw bounding boxes in-place on a copy of img_rgb."""
    img = img_rgb.copy()
    for i, (box, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        name = class_names[label] if label < len(class_names) else str(label)
        parts = [name]
        if scores is not None:
            parts.append(f'{scores[i]:.2f}')
        if ious is not None:
            parts.append(f'iou={ious[i]:.2f}')
        text = ' '.join(parts)
        cv2.putText(img, text, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def add_stats_banner(img_rgb, rank, img_name, n_gt, n_pred, n_missed, n_fp, score):
    """Prepend a black banner with per-image stats."""
    h, w = img_rgb.shape[:2]
    banner_h = 36
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    text = (f'#{rank}  {img_name}  |  '
            f'GT={n_gt}  Pred={n_pred}  '
            f'MissedGT={n_missed}  FP={n_fp}  score={score}')
    cv2.putText(banner, text, (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([banner, img_rgb], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load annotations
    with open(args.ann) as f:
        coco = json.load(f)

    gt_by_image = {}
    for ann in coco['annotations']:
        gt_by_image.setdefault(ann['image_id'], []).append(ann)

    cat_names = {c['id']: c['name'] for c in coco['categories']}
    class_names = list(cat_names.values())
    cat_id_list = list(cat_names.keys())  # for label index lookup

    images_with_gt = [img for img in coco['images'] if img['id'] in gt_by_image]
    print(f'Val images with GT: {len(images_with_gt)}')

    # Load model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # --- Pass 1: score every image ---
    results = []
    for idx, img_info in enumerate(images_with_gt):
        img_path = os.path.join(args.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f'[SKIP] not found: {img_path}')
            continue

        result = inference_detector(model, img_path)
        pred = result.pred_instances
        keep = pred.scores > args.score_thr
        pred_boxes  = pred.bboxes[keep].cpu().numpy()
        pred_scores = pred.scores[keep].cpu().numpy()
        pred_labels = pred.labels[keep].cpu().numpy()

        anns = gt_by_image.get(img_info['id'], [])
        if len(anns):
            gt_boxes = np.array([[a['bbox'][0], a['bbox'][1],
                                  a['bbox'][0] + a['bbox'][2],
                                  a['bbox'][1] + a['bbox'][3]] for a in anns],
                                dtype=np.float32)
            gt_labels = np.array(
                [cat_id_list.index(a['category_id']) for a in anns], dtype=np.int64)
        else:
            gt_boxes  = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros(0, dtype=np.int64)

        score, n_missed, n_fp, per_gt_iou = compute_score(
            gt_boxes, pred_boxes, args.iou_thr)

        results.append(dict(
            img_info=img_info,
            img_path=img_path,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            per_gt_iou=per_gt_iou,
            score=score,
            n_missed=n_missed,
            n_fp=n_fp,
        ))

        if (idx + 1) % 50 == 0:
            print(f'  Processed {idx + 1}/{len(images_with_gt)}')

    print(f'Processed {len(results)} images total.')

    # --- Sort by score descending, take top-K ---
    results.sort(key=lambda x: x['score'], reverse=True)
    topk = results[:args.topk]

    print(f'\nTop-{args.topk} worst images (score = missed_GT + FP):')

    # --- Pass 2: draw and save ---
    for rank, r in enumerate(topk, start=1):
        img_bgr = cv2.imread(r['img_path'])
        img_rgb = img_bgr[:, :, ::-1].copy()

        # Draw GT boxes (green); highlight missed ones (orange)
        for i, (box, label) in enumerate(zip(r['gt_boxes'], r['gt_labels'])):
            iou_val = r['per_gt_iou'][i] if len(r['per_gt_iou']) > i else 0.0
            missed = iou_val < args.iou_thr
            color = (255, 140, 0) if missed else (0, 200, 0)  # orange if missed, green if matched
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, args.thickness)
            name = class_names[label] if label < len(class_names) else str(label)
            tag = f'{name} iou={iou_val:.2f}' if not missed else f'{name} MISSED'
            cv2.putText(img_rgb, tag, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw predicted boxes (red)
        if len(r['pred_boxes']):
            img_rgb = draw_boxes(img_rgb, r['pred_boxes'], r['pred_labels'],
                                 class_names, color=(220, 30, 30), thickness=args.thickness,
                                 scores=r['pred_scores'])

        img_rgb = add_stats_banner(
            img_rgb, rank,
            os.path.basename(r['img_info']['file_name']),
            n_gt=len(r['gt_boxes']),
            n_pred=len(r['pred_boxes']),
            n_missed=r['n_missed'],
            n_fp=r['n_fp'],
            score=r['score'],
        )

        base = os.path.splitext(os.path.basename(r['img_info']['file_name']))[0]
        out_name = f'rank{rank:03d}_score{r["score"]:03d}_{base}.jpg'
        out_path = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, img_rgb[:, :, ::-1])

        print(f'  #{rank:3d}  score={r["score"]:3d}  '
              f'missed={r["n_missed"]}  fp={r["n_fp"]}  '
              f'{os.path.basename(r["img_info"]["file_name"])}')

    print(f'\nSaved {len(topk)} images to: {os.path.abspath(args.out_dir)}')
    print('Legend: GREEN = matched GT | ORANGE = missed GT | RED = prediction')


if __name__ == '__main__':
    main()
