"""
Slice a COCO-format dataset of large images into fixed-size tiles.

Useful for drone imagery where objects are small relative to the full image.
Each large image is cut into overlapping tiles; annotations are clipped to
each tile and re-expressed in tile-local coordinates.

Usage:
    python tools/tile_coco_dataset.py \
        --ann   data/cn_coty/annotations/train.json \
        --img-dir  data/cn_coty/images/train \
        --out-dir  data/cn_coty_tiled \
        --tile-size 1024 \
        --overlap   0.2 \
        --min-bbox-area 16

Output layout:
    data/cn_coty_tiled/
        images/train/   ← tile images  (*.jpg)
        annotations/train.json  ← COCO JSON for the tiles
"""

import argparse
import json
import math
import os
from collections import defaultdict

import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann",      required=True)
    p.add_argument("--img-dir",  required=True)
    p.add_argument("--out-dir",  required=True)
    p.add_argument("--split",    default="train",
                   help="Split name used for sub-directory and output JSON key")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--overlap",   type=float, default=0.2,
                   help="Fractional overlap between adjacent tiles (0–1)")
    p.add_argument("--min-bbox-area", type=float, default=16,
                   help="Drop clipped boxes whose area (px²) is below this")
    p.add_argument("--jpeg-quality", type=int, default=92)
    return p.parse_args()


def compute_tile_starts(img_size: int, tile: int, overlap: float):
    """Return the list of top-left pixel offsets for tiles along one axis."""
    stride = int(tile * (1 - overlap))
    starts = list(range(0, img_size - tile + 1, stride))
    # Always include a tile that ends exactly at the image edge
    last = img_size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def clip_bbox(bbox, tile_x, tile_y, tile_size):
    """
    Clip a COCO bbox [x, y, w, h] to a tile and return tile-local coordinates.
    Returns None if the box does not overlap the tile.
    """
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    tx2, ty2 = tile_x + tile_size, tile_y + tile_size

    cx1 = max(x,  tile_x)
    cy1 = max(y,  tile_y)
    cx2 = min(x2, tx2)
    cy2 = min(y2, ty2)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    return [cx1 - tile_x, cy1 - tile_y, cx2 - cx1, cy2 - cy1]


def main():
    args = parse_args()

    with open(args.ann) as f:
        coco = json.load(f)

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    out_img_dir = os.path.join(args.out_dir, "images", args.split)
    out_ann_dir = os.path.join(args.out_dir, "annotations")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    new_images = []
    new_anns = []
    img_id = 0
    ann_id = 0

    total = len(coco["images"])
    for idx, img_info in enumerate(coco["images"]):
        src_path = os.path.join(args.img_dir, img_info["file_name"])
        if not os.path.exists(src_path):
            print(f"[{idx+1}/{total}] SKIP (not found): {src_path}")
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"[{idx+1}/{total}] SKIP (unreadable): {src_path}")
            continue

        H, W = img.shape[:2]
        src_anns = anns_by_image[img_info["id"]]

        xs = compute_tile_starts(W, args.tile_size, args.overlap)
        ys = compute_tile_starts(H, args.tile_size, args.overlap)
        n_tiles = len(xs) * len(ys)

        kept_tiles = 0
        for ty in ys:
            for tx in xs:
                tile_anns = []
                for ann in src_anns:
                    clipped = clip_bbox(ann["bbox"], tx, ty, args.tile_size)
                    if clipped is None:
                        continue
                    cw, ch = clipped[2], clipped[3]
                    if cw * ch < args.min_bbox_area:
                        continue
                    tile_anns.append((ann, clipped))

                # Skip empty tiles (no annotations)
                if not tile_anns:
                    continue

                img_id += 1
                base = os.path.splitext(img_info["file_name"])[0]
                tile_name = f"{base}_tile_{tx}_{ty}.jpg"
                tile_path = os.path.join(out_img_dir, tile_name)

                tile_img = img[ty:ty + args.tile_size, tx:tx + args.tile_size]
                cv2.imwrite(tile_path, tile_img,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

                new_images.append({
                    "id": img_id,
                    "file_name": tile_name,
                    "width":  args.tile_size,
                    "height": args.tile_size,
                    "source_image": img_info["file_name"],
                    "tile_x": tx,
                    "tile_y": ty,
                })

                for ann, clipped in tile_anns:
                    ann_id += 1
                    cx, cy, cw, ch = clipped
                    new_anns.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": ann["category_id"],
                        "bbox": [cx, cy, cw, ch],
                        "area": cw * ch,
                        "iscrowd": ann.get("iscrowd", 0),
                    })

                kept_tiles += 1

        print(f"[{idx+1}/{total}] {img_info['file_name']}  "
              f"{kept_tiles}/{n_tiles} non-empty tiles  "
              f"({len(src_anns)} src anns)")

    out_coco = {
        "images":      new_images,
        "annotations": new_anns,
        "categories":  coco["categories"],
    }
    out_json = os.path.join(out_ann_dir, f"{args.split}.json")
    with open(out_json, "w") as f:
        json.dump(out_coco, f)

    print(f"\nDone.")
    print(f"  Tiles written : {len(new_images)}")
    print(f"  Annotations   : {len(new_anns)}")
    print(f"  Output JSON   : {out_json}")
    print(f"  Output images : {out_img_dir}/")


if __name__ == "__main__":
    main()
