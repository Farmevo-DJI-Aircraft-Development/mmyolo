"""
Visualize COCO annotations by overlaying bounding boxes on images.

Usage:
    python tools/visualize_annotations.py \
        --ann data/cn_coty/annotations/train.json \
        --img-dir data/cn_coty/images/train \
        --out-dir vis_annotations \
        --num-images 20        # number of images to visualize (0 = all)
        --scale 0.25           # scale factor for output (default 0.25 for large DJI images)
"""

import argparse
import json
import os
import random
from collections import defaultdict

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COCO annotations")
    parser.add_argument(
        "--ann",
        default="data/cn_coty_tiled/annotations/train.json",
        help="Path to COCO annotation JSON",
    )
    parser.add_argument(
        "--img-dir",
        default="data/cn_coty_tiled/images/train",
        help="Directory containing images",
    )
    parser.add_argument(
        "--out-dir",
        default="vis_annotations",
        help="Output directory for visualized images",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of images to visualize (0 = all)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Scale factor applied to output images (default 0.25 for 8K DJI images)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image sampling",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.ann) as f:
        coco = json.load(f)

    # Build category id → name map
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Build image_id → annotations map
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    images = coco["images"]

    # Optionally sample a subset
    if args.num_images > 0 and args.num_images < len(images):
        random.seed(args.seed)
        images = random.sample(images, args.num_images)

    os.makedirs(args.out_dir, exist_ok=True)

    # Pick a fixed color per category
    color_map = {}
    rng = random.Random(0)
    for cat_id in cat_names:
        color_map[cat_id] = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))

    print(f"Visualizing {len(images)} images → {args.out_dir}/")

    for img_info in images:
        img_path = os.path.join(args.img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            print(f"  [SKIP] not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] could not read: {img_path}")
            continue

        anns = anns_by_image.get(img_info["id"], [])

        # Compute thickness/font scale relative to image size so they look
        # reasonable regardless of the original resolution.
        h, w = img.shape[:2]
        thickness = max(2, int(min(h, w) / 1000))
        font_scale = max(0.5, min(h, w) / 2000)

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            cat_id = ann["category_id"]
            color = color_map.get(cat_id, (0, 255, 0))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = cat_names.get(cat_id, str(cat_id))
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                img, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        # Scale down for output
        if args.scale != 1.0:
            new_w = int(w * args.scale)
            new_h = int(h * args.scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out_name = os.path.splitext(img_info["file_name"])[0] + "_vis.jpg"
        out_path = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  [{len(anns):3d} boxes] {out_name}")

    print("Done.")


if __name__ == "__main__":
    main()
