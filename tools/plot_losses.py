"""
Plot training/validation losses from mmyolo scalars.json files.

Usage:
    python tools/plot_losses.py work_dirs/yolov8s_cn_coty/20260317_055818/vis_data/scalars.json
    python tools/plot_losses.py work_dirs/  # auto-find all scalars.json
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_scalars(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_train_val(records):
    train = [r for r in records if 'loss' in r and 'coco' not in str(r)]
    val = [r for r in records if 'coco/bbox_mAP' in r or 'pascal_voc/AP50' in r]
    # val records use epoch as x-axis
    return train, val


def plot_run(scalars_path, ax_loss, ax_lr=None):
    records = load_scalars(scalars_path)
    train, val = split_train_val(records)

    if not train:
        print(f"No training records found in {scalars_path}")
        return

    steps = [r['step'] for r in train]
    loss = [r['loss'] for r in train]
    loss_cls = [r.get('loss_cls') for r in train]
    loss_bbox = [r.get('loss_bbox') for r in train]
    loss_dfl = [r.get('loss_dfl') for r in train]

    label = Path(scalars_path).parts[-3]  # run name
    ax_loss.plot(steps, loss, label=f'{label} total', linewidth=1.5)
    if all(v is not None for v in loss_cls):
        ax_loss.plot(steps, loss_cls, label=f'{label} cls', linestyle='--', linewidth=1)
    if all(v is not None for v in loss_bbox):
        ax_loss.plot(steps, loss_bbox, label=f'{label} bbox', linestyle='--', linewidth=1)
    if all(v is not None for v in loss_dfl):
        ax_loss.plot(steps, loss_dfl, label=f'{label} dfl', linestyle='--', linewidth=1)

    if ax_lr is not None and train:
        lr = [r.get('lr', r.get('base_lr')) for r in train]
        ax_lr.plot(steps, lr, label=label, linewidth=1)

    # val mAP
    if val:
        val_epochs = [r.get('step', r.get('epoch')) for r in val]
        map50 = [r.get('coco/bbox_mAP_50', r.get('pascal_voc/AP50')) for r in val]
        if any(v is not None for v in map50):
            ax_loss.twinx().plot(
                val_epochs, map50,
                color='red', linestyle=':', linewidth=1.5,
                label='mAP@50 (right axis)'
            )

    print(f"Plotted {len(train)} train steps from {scalars_path}")


def find_scalars(root):
    return sorted(Path(root).rglob('scalars.json'))


def main():
    parser = argparse.ArgumentParser(description='Plot mmyolo training losses')
    parser.add_argument('path', help='Path to scalars.json or work_dirs/ root')
    parser.add_argument('--output', '-o', default='loss_plot.png', help='Output image path')
    parser.add_argument('--no-lr', action='store_true', help='Skip LR subplot')
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        paths = find_scalars(p)
        if not paths:
            print(f"No scalars.json found under {p}")
            return
        print(f"Found {len(paths)} run(s):")
        for sp in paths:
            print(f"  {sp}")
    else:
        paths = [p]

    n_rows = 1 if args.no_lr else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    ax_loss = axes[0]
    ax_lr = axes[1] if not args.no_lr else None

    for sp in paths:
        try:
            plot_run(sp, ax_loss, ax_lr)
        except Exception as e:
            print(f"Error processing {sp}: {e}")

    ax_loss.set_title('Training Loss')
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right', fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    if ax_lr is not None:
        ax_lr.set_title('Learning Rate')
        ax_lr.set_xlabel('Step')
        ax_lr.set_ylabel('LR')
        ax_lr.legend(loc='upper right', fontsize=8)
        ax_lr.grid(True, alpha=0.3)
        ax_lr.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150)
    print(f"\nSaved to {out.resolve()}")
    plt.show()


if __name__ == '__main__':
    main()
