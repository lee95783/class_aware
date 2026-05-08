#!/usr/bin/env python3
"""
Generate multiple random class subsets per K for statistical evaluation.

Produces configs/class_subsets_multi.json with 3 independent random subsets
per K value, using seeds 42, 123, 456.

Usage:
    python scripts/generate_multi_subsets.py
"""

import json
import numpy as np
from pathlib import Path

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickle_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
    'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CIFAR100_CLASSES)}

K_VALUES = [5, 10, 20, 50]
SEEDS = [42, 123, 456]


def main():
    output = {'subsets': {}}

    for k in K_VALUES:
        output['subsets'][str(k)] = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            names = rng.choice(CIFAR100_CLASSES, size=k, replace=False).tolist()
            indices = sorted([CLASS_TO_IDX[n] for n in names])
            # reorder names to match sorted indices
            names_sorted = [CIFAR100_CLASSES[i] for i in indices]
            output['subsets'][str(k)].append({
                'seed': seed,
                'class_names': names_sorted,
                'class_indices': indices,
            })
        print(f'K={k}: generated {len(SEEDS)} subsets')
        for i, s in enumerate(output['subsets'][str(k)]):
            print(f'  subset {i} (seed={s["seed"]}): {s["class_names"]}')

    path = Path('configs/class_subsets_multi.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n✓ Saved: {path}')


if __name__ == '__main__':
    main()
