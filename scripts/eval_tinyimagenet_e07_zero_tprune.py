#!/usr/bin/env python3
"""
E7 baseline (TinyImageNet): Zero-TPrune — zero-shot WPR token pruning.

Same algorithm as eval_e07_zero_tprune.py, adapted for:
  - Model:   deit_small_patch16_224  (D=384, 6 heads/layer, 72 heads total)
  - Dataset: TinyImageNet-200 (64×64 → resize 224)
  - Checkpoint: weights/deit_small_patch16_224_tinyimagenet_best.pth

Usage:
    python scripts/eval_tinyimagenet_e07_zero_tprune.py --device 0
"""

import os, sys, json, argparse
import torch
import timm
import torchvision.transforms as T
import torchvision.datasets as datasets
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.eval_e07_zero_tprune import _wpr, forward_zero_tprune, evaluate, evaluate_zero_tprune

CHECKPOINT  = 'weights/deit_small_patch16_224_tinyimagenet_best.pth'
MODEL_NAME  = 'deit_small_patch16_224'
NUM_CLASSES = 200


# ── Data ──────────────────────────────────────────────────────────────────────

def filter_loader(base_loader, class_indices, batch_size=128, shuffle=False):
    dataset    = base_loader.dataset
    target_set = set(class_indices)
    labels     = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    indices    = [i for i, l in enumerate(labels) if l in target_set]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=4, pin_memory=True)


def load_class_subset(path, num_classes):
    with open(path) as f:
        d = json.load(f)
    s = d['subsets'][str(num_classes)]
    return s['class_indices'], s['class_names']


def get_val_loader(data_dir, batch_size=128, num_workers=8):
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device):
    model = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, model, full_val_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)
    val_loader = filter_loader(full_val_loader, target_classes, batch_size=128)

    n_patches   = 196
    keep_ratios = [0.7, 0.5]

    print(f'\n{"="*70}')
    print(f'Zero-TPrune (TinyImageNet)  |  K={num_classes}  |  layer={args.layer_idx}  |  zero-shot')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    unpruned_acc = evaluate(model, val_loader, device, target_classes)
    print(f'Unpruned baseline: {unpruned_acc:.2f}%\n')

    results = []
    for keep_ratio in keep_ratios:
        k   = max(1, int(n_patches * keep_ratio))
        acc = evaluate_zero_tprune(model, val_loader, device, target_classes,
                                   args.layer_idx, k, n_iter=args.n_iter)
        drop = unpruned_acc - acc
        print(f'  keep={keep_ratio}  k={k}: {acc:.2f}%  (drop={drop:+.2f}%)', flush=True)
        results.append({
            'layer_idx': args.layer_idx,
            'keep_ratio': keep_ratio,
            'k': k,
            'accuracy': acc,
            'drop': drop,
        })

    print(f'\n{"="*70}')
    print(f'SUMMARY  K={num_classes}  (unpruned={unpruned_acc:.2f}%):')
    for r in results:
        print(f'  keep={r["keep_ratio"]}  {r["accuracy"]:.2f}%  (drop={r["drop"]:+.2f}%)')

    out = {
        'method': 'Zero-TPrune',
        'dataset': 'tinyimagenet',
        'model': MODEL_NAME,
        'num_classes': num_classes,
        'layer_idx': args.layer_idx,
        'n_iter': args.n_iter,
        'unpruned_acc': unpruned_acc,
        'results': results,
    }
    save_path = os.path.join(args.output_dir, f'tinyimagenet_e07_zero_tprune_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--layer-idx',   type=int,  default=6)
    parser.add_argument('--n-iter',      type=int,  default=3)
    parser.add_argument('--device',      type=int,  default=0)
    parser.add_argument('--data-dir',    type=str,  default='./data/tiny-imagenet-200')
    parser.add_argument('--subset-file', type=str,  default='configs/tinyimagenet_class_subsets.json')
    parser.add_argument('--output-dir',  type=str,  default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    model           = load_model(device)
    full_val_loader = get_val_loader(args.data_dir, batch_size=128)

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, model, full_val_loader)


if __name__ == '__main__':
    main()
