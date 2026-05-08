#!/usr/bin/env python3
"""
Prepare TinyImageNet for class-aware experiments.

1. Downloads TinyImageNet-200 if not present.
2. Reorganizes train (removes extra images/ level) and val (sorts into class dirs).
3. Generates nested class subsets K=5,10,20,50 and saves to
   configs/tinyimagenet_class_subsets.json.

Usage:
    python scripts/prepare_tinyimagenet.py --data-dir data/tiny-imagenet-200
"""

import os
import json
import shutil
import argparse
import zipfile
import urllib.request
from pathlib import Path

import numpy as np

TINYIMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def download(url, dest_zip):
    print(f'Downloading TinyImageNet (~237 MB)...')
    def _progress(count, block, total):
        pct = min(count * block / total * 100, 100)
        print(f'\r  {pct:.1f}%', end='', flush=True)
    urllib.request.urlretrieve(url, dest_zip, reporthook=_progress)
    print()


def reorganize_train(train_dir):
    """Flatten train/CLASS/images/*.JPEG -> train/CLASS/*.JPEG."""
    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))
    moved = 0
    for cls in classes:
        img_dir = os.path.join(train_dir, cls, 'images')
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            src = os.path.join(img_dir, fname)
            dst = os.path.join(train_dir, cls, fname)
            shutil.move(src, dst)
            moved += 1
        os.rmdir(img_dir)
    print(f'  Reorganized train: moved {moved} files from images/ subdirs.')


def reorganize_val(val_dir):
    """Sort val/images/*.JPEG into val/CLASS/*.JPEG using val_annotations.txt."""
    ann_file = os.path.join(val_dir, 'val_annotations.txt')
    flat_dir = os.path.join(val_dir, 'images')

    if not os.path.isfile(ann_file) or not os.path.isdir(flat_dir):
        print('  Val already reorganized, skipping.')
        return

    img2cls = {}
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            img2cls[parts[0]] = parts[1]

    moved = 0
    for fname, cls in img2cls.items():
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(flat_dir, fname)
        dst = os.path.join(cls_dir, fname)
        if os.path.isfile(src):
            shutil.move(src, dst)
            moved += 1

    # Clean up flat dir if empty
    try:
        os.rmdir(flat_dir)
    except OSError:
        pass

    print(f'  Reorganized val: moved {moved} files into class subdirs.')


def load_class_info(data_dir):
    """Return sorted list of (class_index, synset_id, class_name)."""
    words_file = os.path.join(data_dir, 'words.txt')
    wnids_file = os.path.join(data_dir, 'wnids.txt')

    # Read human-readable names
    words = {}
    if os.path.isfile(words_file):
        with open(words_file) as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    words[parts[0]] = parts[1].split(',')[0].strip()

    # Read the 200 synset IDs used in TinyImageNet
    if os.path.isfile(wnids_file):
        with open(wnids_file) as f:
            wnids = sorted(l.strip() for l in f if l.strip())
    else:
        # Fall back: read from train directory
        train_dir = os.path.join(data_dir, 'train')
        wnids = sorted(d for d in os.listdir(train_dir)
                       if os.path.isdir(os.path.join(train_dir, d)))

    classes = []
    for idx, wnid in enumerate(wnids):
        name = words.get(wnid, wnid)
        classes.append({'index': idx, 'wnid': wnid, 'name': name})

    return classes


def generate_subsets(classes, ks=(5, 10, 20, 50), seed=42):
    """Generate nested class subsets using a fixed seed."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(classes)).tolist()

    # Largest K first, then slice for smaller K
    max_k = max(ks)
    selected = indices[:max_k]

    subsets = {}
    ks_sorted = sorted(ks)
    for k in ks_sorted:
        sel_k = selected[:k]
        subsets[str(k)] = {
            'class_indices': sel_k,
            'class_names':   [classes[i]['name'] for i in sel_k],
            'wnids':         [classes[i]['wnid']  for i in sel_k],
        }

    return subsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    type=str, default='data/tiny-imagenet-200')
    parser.add_argument('--config-out',  type=str, default='configs/tinyimagenet_class_subsets.json')
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs('data', exist_ok=True)

    # ── Download ───────────────────────────────────────────────────────────────
    if not os.path.isdir(data_dir):
        if args.no_download:
            raise FileNotFoundError(f'{data_dir} not found and --no-download set.')
        zip_path = 'data/tiny-imagenet-200.zip'
        if not os.path.isfile(zip_path):
            download(TINYIMAGENET_URL, zip_path)
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall('data')
        # Extracted as data/tiny-imagenet-200/
        print(f'Extracted to {data_dir}')

    # ── Reorganize ─────────────────────────────────────────────────────────────
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')

    print('Reorganizing train...')
    reorganize_train(train_dir)

    print('Reorganizing val...')
    reorganize_val(val_dir)

    # ── Class info ─────────────────────────────────────────────────────────────
    classes = load_class_info(data_dir)
    print(f'\nFound {len(classes)} classes.')
    print('First 5:', [(c['index'], c['name']) for c in classes[:5]])

    # ── Subsets ────────────────────────────────────────────────────────────────
    subsets = generate_subsets(classes, ks=(5, 10, 20, 50))

    config = {
        'dataset': 'tinyimagenet',
        'num_total_classes': len(classes),
        'all_classes': classes,
        'subsets': subsets,
    }

    os.makedirs(os.path.dirname(args.config_out), exist_ok=True)
    with open(args.config_out, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'\nSaved class subsets to {args.config_out}')

    # Print subset preview
    for k in (5, 10, 20, 50):
        s = subsets[str(k)]
        names = s['class_names']
        print(f'  K={k:2d}: {", ".join(names[:5])}{"..." if k > 5 else ""}')

    # ── Verify ─────────────────────────────────────────────────────────────────
    n_train = sum(
        len(os.listdir(os.path.join(train_dir, cls['wnid'])))
        for cls in classes
        if os.path.isdir(os.path.join(train_dir, cls['wnid']))
    )
    n_val = sum(
        len(os.listdir(os.path.join(val_dir, cls['wnid'])))
        for cls in classes
        if os.path.isdir(os.path.join(val_dir, cls['wnid']))
    )
    print(f'\nVerification:')
    print(f'  Train images: {n_train} (expected 100,000)')
    print(f'  Val images:   {n_val} (expected 10,000)')
    print('\nDone.')


if __name__ == '__main__':
    main()
