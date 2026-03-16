"""
Run inference on N random test images and overlay both predicted boxes
and ground-truth boxes side-by-side for visual comparison.

Usage:
    python tools/infer_with_gt.py \
        --config  work_dirs/yolov8s_cn_coty_tiled/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
        --checkpoint work_dirs/yolov8s_cn_coty_tiled/best_coco_bbox_mAP_epoch_46.pth \
        --ann data/cn_coty_tiled/annotations/test.json \
        --img-dir data/cn_coty_tiled/images/test \
        --out-dir vis_gt_vs_pred \
        --n 5 \
        --score-thr 0.3
"""
import argparse
import json
import os
import random

import numpy as np
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference + GT overlay for visual verification')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann', required=True, help='COCO annotation JSON')
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--out-dir', default='vis_gt_vs_pred')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of random images to process')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def draw_boxes(img_rgb, bboxes, labels, class_names, color, thickness=2,
               show_label=True, scores=None):
    """Draw bounding boxes on a copy of img_rgb (H,W,3 uint8)."""
    import cv2
    img = img_rgb.copy()
    for i, (box, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if show_label:
            name = class_names[label] if label < len(class_names) else str(label)
            score_str = f' {scores[i]:.2f}' if scores is not None else ''
            text = f'{name}{score_str}'
            cv2.putText(img, text, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load COCO annotations
    with open(args.ann) as f:
        coco = json.load(f)

    # Build lookup: image_id -> list of annotations
    gt_by_image = {}
    for ann in coco['annotations']:
        gt_by_image.setdefault(ann['image_id'], []).append(ann)

    # Build lookup: category_id -> name
    cat_names = {c['id']: c['name'] for c in coco['categories']}
    class_names = list(cat_names.values())

    # Pick N random images that have at least one annotation
    images_with_gt = [img for img in coco['images']
                      if img['id'] in gt_by_image]
    selected = random.sample(images_with_gt, min(args.n, len(images_with_gt)))

    # Load model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    import cv2

    for img_info in selected:
        img_path = os.path.join(args.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f'[SKIP] not found: {img_path}')
            continue

        # --- Inference ---
        result = inference_detector(model, img_path)
        pred = result.pred_instances
        keep = pred.scores > args.score_thr
        pred_boxes  = pred.bboxes[keep].cpu().numpy()
        pred_scores = pred.scores[keep].cpu().numpy()
        pred_labels = pred.labels[keep].cpu().numpy()

        # --- Ground truth ---
        anns = gt_by_image.get(img_info['id'], [])
        # COCO bbox: [x, y, w, h] -> [x1, y1, x2, y2]
        gt_boxes  = np.array([[a['bbox'][0], a['bbox'][1],
                                a['bbox'][0] + a['bbox'][2],
                                a['bbox'][1] + a['bbox'][3]]
                               for a in anns], dtype=np.float32)
        gt_labels = np.array(
            [list(cat_names.keys()).index(a['category_id']) for a in anns],
            dtype=np.int64)

        # --- Draw both GT and predictions on the same image ---
        img_bgr = cv2.imread(img_path)
        img_rgb = img_bgr[:, :, ::-1].copy()

        if len(gt_boxes):
            img_rgb = draw_boxes(img_rgb, gt_boxes, gt_labels, class_names,
                                 color=(0, 255, 0), thickness=2, scores=None)
        if len(pred_boxes):
            img_rgb = draw_boxes(img_rgb, pred_boxes, pred_labels, class_names,
                                 color=(255, 50, 50), thickness=2,
                                 scores=pred_scores)

        combined = img_rgb

        out_name = os.path.basename(img_info['file_name'])
        out_path = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, combined[:, :, ::-1])  # back to BGR for imwrite
        print(f'Saved: {out_path}  |  GT={len(gt_boxes)}  Pred={len(pred_boxes)}')

    print(f'\nAll done. Results in: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
