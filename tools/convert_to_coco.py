"""Convert LabelMe-format annotations to COCO format for mmYOLOv8 training.

The source dataset has images and LabelMe JSON annotations co-located:
    /home/ubuntu/dev/data/dataset/train/  *.JPG + *.json
    /home/ubuntu/dev/data/dataset/test/   *.JPG + *.json

Output structure (ready for mmYOLO training):
    <output_root>/
    ├── images/
    │   ├── train/   (symlinks to source images)
    │   └── test/    (symlinks to source images)
    └── annotations/
        ├── train.json
        └── test.json

Usage:
    python convert_to_coco.py [--data-root /path/to/dataset]
                              [--output-root /path/to/mmyolo/data/cn_coty]
                              [--copy]   # copy images instead of symlinking
"""

import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Default class mapping: LabelMe label name -> COCO category_id (1-indexed)
# Override at runtime via --classes
CLASS_MAP = {'cn_coty': 1}


def convert_split(src_dir: Path, output_root: Path, split: str, copy_images: bool) -> dict:
    """Convert one dataset split (train or test) to COCO JSON.

    Args:
        src_dir:      Directory containing .JPG images and .json annotations.
        output_root:  Root of the output directory tree.
        split:        'train' or 'test'
        copy_images:  If True, copy images; otherwise create symlinks.

    Returns:
        COCO-format dict with 'images', 'annotations', 'categories'.
    """
    img_out_dir = output_root / 'images' / split
    img_out_dir.mkdir(parents=True, exist_ok=True)

    ann_out_dir = output_root / 'annotations'
    ann_out_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': v, 'name': k} for k, v in CLASS_MAP.items()],
    }

    image_id = 0
    ann_id = 0
    skipped_images = 0
    skipped_shapes = 0

    # Collect all image files in the source directory
    img_files = sorted(
        p for p in src_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not img_files:
        print(f'  WARNING: No images found in {src_dir}')
        return coco

    for img_path in img_files:
        label_path = src_dir / img_path.with_suffix('.json').name

        if not label_path.exists():
            print(f'  WARNING: No annotation for {img_path.name}, skipping.')
            skipped_images += 1
            continue

        with open(label_path, encoding='utf-8') as f:
            labelme = json.load(f)

        image_id += 1

        coco['images'].append({
            'id': image_id,
            'file_name': img_path.name,
            'width': labelme['imageWidth'],
            'height': labelme['imageHeight'],
        })

        # Link or copy image into the output tree
        dst = img_out_dir / img_path.name
        if not dst.exists():
            if copy_images:
                shutil.copy2(img_path, dst)
            else:
                os.symlink(img_path.resolve(), dst)

        # Convert each annotation shape
        for shape in labelme.get('shapes', []):
            if shape.get('shape_type') != 'rectangle':
                print(f'  Skipping non-rectangle shape ({shape.get("shape_type")}) '
                      f'in {label_path.name}')
                skipped_shapes += 1
                continue

            label = shape['label']
            if label not in CLASS_MAP:
                print(f'  WARNING: Unknown label "{label}" in {label_path.name}, skipping.')
                skipped_shapes += 1
                continue

            pts = shape['points']  # [[x1,y1], [x2,y2]]
            if len(pts) < 2:
                print(f'  WARNING: Rectangle with <2 points in {label_path.name}, skipping.')
                skipped_shapes += 1
                continue
            x1 = min(pts[0][0], pts[1][0])
            y1 = min(pts[0][1], pts[1][1])
            x2 = max(pts[0][0], pts[1][0])
            y2 = max(pts[0][1], pts[1][1])
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                print(f'  WARNING: Degenerate bbox in {label_path.name}, skipping.')
                skipped_shapes += 1
                continue

            ann_id += 1
            coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': CLASS_MAP[label],
                'bbox': [x1, y1, w, h],   # COCO: [x, y, width, height]
                'area': w * h,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                'iscrowd': 0,
            })

    print(f'  [{split}] images={image_id}  annotations={ann_id}  '
          f'skipped_images={skipped_images}  skipped_shapes={skipped_shapes}')
    return coco


def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe dataset to COCO format for mmYOLO.')
    parser.add_argument(
        '--data-root',
        default='/home/ubuntu/dev/data/dataset',
        help='Root of the source dataset (contains train/ and test/ subdirs)',
    )
    parser.add_argument(
        '--output-root',
        default='/home/ubuntu/dev/mmyolo/data/cn_coty',
        help='Root of the output directory (will be created if absent)',
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy images to output dir instead of creating symlinks',
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        default=None,
        help='Class names in order (1-indexed). Overrides the default CLASS_MAP. '
             'Example: --classes canola',
    )
    args = parser.parse_args()

    if args.classes:
        global CLASS_MAP
        CLASS_MAP = {name: idx + 1 for idx, name in enumerate(args.classes)}

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    splits = {
        'train': data_root / 'train',
        'test': data_root / 'test',
    }

    for split, src_dir in splits.items():
        if not src_dir.exists():
            print(f'Source directory not found, skipping: {src_dir}')
            continue

        print(f'Converting {split} split: {src_dir}')
        coco = convert_split(src_dir, output_root, split, copy_images=args.copy)

        out_json = output_root / 'annotations' / f'{split}.json'
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2)
        print(f'  Saved: {out_json}')

    print('\nDone. Output structure:')
    for p in sorted(output_root.rglob('*')):
        if p.is_dir():
            print(f'  {p.relative_to(output_root)}/')
        elif p.suffix == '.json':
            print(f'  {p.relative_to(output_root)}')

    print(f'\nTo train mmYOLOv8, create a config that sets:')
    print(f'  data_root = "{output_root}/"')
    print(f'  ann_file (train) = "annotations/train.json"')
    print(f'  ann_file (val)   = "annotations/test.json"')
    print(f'  data_prefix      = dict(img="images/train/")  # or images/test/')


if __name__ == '__main__':
    main()
