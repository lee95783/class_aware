#!/usr/bin/env python3
"""
E2: Class-Aware MLP Pruning Sweep

Compares five scoring methods across prune ratios for 10-class deployment.

Methods:
  random        — random neuron selection
  magnitude     — L1 norm of fc1 weight rows
  global_taylor — Taylor |grad × act| on full 100-class training data
  class_taylor  — Taylor |grad × act| on K-class training data (ours)

Protocol for each (method, prune_ratio):
  1. Load fresh pretrained model
  2. Compute scores (cached per method, reused across ratios)
  3. Hard-prune bottom neurons (zero fc1 rows + fc2 cols)
  4. Fine-tune N epochs on K-class data with masks held fixed
  5. Evaluate on K-class test set (restricted eval)

Usage:
    python scripts/train_e02_mlp_sweep.py --num-classes 10 --epochs 5 --device 0
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders


# ── Data ──────────────────────────────────────────────────────────────────────

def filter_loader(base_loader, class_indices, batch_size=64, shuffle=False):
    dataset = base_loader.dataset
    target_set = set(class_indices)
    labels = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    indices = [i for i, l in enumerate(labels) if l in target_set]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=4, pin_memory=True)


def load_class_subset(path, num_classes, subset_id=None):
    with open(path) as f:
        d = json.load(f)
    entry = d['subsets'][str(num_classes)]
    # Multi-subset format: list of dicts with seed/class_names/class_indices
    if isinstance(entry, list):
        if subset_id is None:
            subset_id = 0
        s = entry[subset_id]
    else:
        s = entry
    return s['class_indices'], s['class_names']


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device) -> nn.Module:
    model = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    state = torch.load('weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth',
                       map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    return model


# ── Scoring methods ───────────────────────────────────────────────────────────

def scores_random(model):
    """Uniform random scores per neuron per layer."""
    return {l: torch.rand(blk.mlp.fc1.out_features)
            for l, blk in enumerate(model.blocks)}


def scores_magnitude(model):
    """L1 norm of fc1 weight rows — class-agnostic static score."""
    scores = {}
    for l, blk in enumerate(model.blocks):
        scores[l] = blk.mlp.fc1.weight.data.abs().sum(dim=1).cpu()
    return scores


def scores_taylor(model, loader, device, num_batches=50):
    """
    Taylor importance: mean |grad × act| at fc1 output over (batch, tokens).
    Works for both global (full loader) and class-aware (K-class loader).
    """
    num_layers = len(model.blocks)
    accum  = {l: None for l in range(num_layers)}
    counts = {l: 0    for l in range(num_layers)}
    acts   = {}

    handles = []
    for l, blk in enumerate(model.blocks):
        def fwd(mod, inp, out, i=l): acts[i] = out
        def bwd(mod, gin, gout, i=l):
            if i not in acts: return
            score = (acts[i] * gout[0]).abs().mean(dim=(0, 1)).detach().cpu()
            accum[i] = score if accum[i] is None else accum[i] + score
            counts[i] += 1
        handles.append(blk.mlp.fc1.register_forward_hook(fwd))
        handles.append(blk.mlp.fc1.register_full_backward_hook(bwd))

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    for i, (images, labels) in enumerate(loader):
        if i >= num_batches: break
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        F.cross_entropy(model(images), labels).backward()

    for h in handles:
        h.remove()
    for p in model.parameters():
        p.requires_grad_(False)

    return {l: accum[l] / max(counts[l], 1) for l in range(num_layers)}


# ── Pruning ───────────────────────────────────────────────────────────────────

def build_masks(scores, prune_ratio):
    masks = {}
    for l, s in scores.items():
        H = s.shape[0]
        k = max(1, int(H * (1.0 - prune_ratio)))
        mask = torch.zeros(H, dtype=torch.bool)
        mask[s.topk(k).indices] = True
        masks[l] = mask
    return masks


def apply_masks(model, masks):
    for l, blk in enumerate(model.blocks):
        if l not in masks: continue
        mask = masks[l].to(next(blk.mlp.fc1.parameters()).device)
        with torch.no_grad():
            blk.mlp.fc1.weight.data[~mask, :] = 0.0
            blk.mlp.fc1.bias.data[~mask]      = 0.0
            blk.mlp.fc2.weight.data[:, ~mask] = 0.0


def neuron_keep_rate(masks):
    total = kept = 0
    for m in masks.values():
        total += m.numel()
        kept  += m.sum().item()
    return kept / total if total > 0 else 1.0


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune(model, masks, train_loader, device, epochs, lr=1e-4):
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader))
    model.train()
    for epoch in range(epochs):
        correct = total = loss_sum = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            apply_masks(model, masks)
            loss_sum += loss.item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)
        print(f'    ep{epoch+1}: loss={loss_sum/len(train_loader):.4f}'
              f'  train={100*correct/total:.1f}%', flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, target_classes):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = class_idx[logits[:, class_idx].argmax(1)]
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int,   default=10)
    parser.add_argument('--epochs',       type=int,   default=5)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--device',       type=int,   default=0)
    parser.add_argument('--num-batches',  type=int,   default=50)
    parser.add_argument('--subset-file',  type=str,   default='configs/class_subsets.json')
    parser.add_argument('--subset-id',    type=int,   default=None,
                        help='Index into multi-subset list (for configs/class_subsets_multi.json)')
    parser.add_argument('--output-dir',   type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    prune_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    method_names = ['random', 'magnitude', 'global_taylor', 'class_taylor']

    target_classes, class_names = load_class_subset(args.subset_file, args.num_classes, args.subset_id)

    print(f'\n{"="*70}')
    print(f'E2: MLP Pruning Sweep  |  K={args.num_classes}  |  epochs={args.epochs}')
    print(f'Classes: {", ".join(class_names)}')
    print(f'{"="*70}\n')

    print('Loading CIFAR-100...')
    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')
    class_train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    class_test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)

    # ── Unpruned baseline ──────────────────────────────────────────────────────
    base_model = load_model(device)
    unpruned_acc = evaluate(base_model, class_test_loader, device, target_classes)
    print(f'Unpruned baseline: {unpruned_acc:.2f}%\n')

    # ── Pre-compute scores (reused across prune ratios) ────────────────────────
    print('Pre-computing scores...')

    print('  [1/4] random scores')
    random_scores = scores_random(base_model)

    print('  [2/4] magnitude scores')
    magnitude_scores = scores_magnitude(base_model)

    print('  [3/4] global Taylor scores (100 classes)...')
    model_gt = load_model(device)
    global_taylor_scores = scores_taylor(model_gt, full_train_loader, device, args.num_batches)
    del model_gt

    print(f'  [4/4] class Taylor scores ({args.num_classes} classes)...')
    model_ct = load_model(device)
    class_taylor_scores = scores_taylor(model_ct, class_train_loader, device, args.num_batches)
    del model_ct

    all_scores = {
        'random':        random_scores,
        'magnitude':     magnitude_scores,
        'global_taylor': global_taylor_scores,
        'class_taylor':  class_taylor_scores,
    }
    print('Done.\n')

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results = {m: [] for m in method_names}

    header = f'{"prune%":>7}  ' + '  '.join(f'{m:>14}' for m in method_names)
    print(header)
    print('-' * len(header))

    for pr in prune_ratios:
        row = []
        for method in method_names:
            print(f'\n  [{method}  pr={pr}]')
            model = load_model(device)
            masks = build_masks(all_scores[method], pr)
            apply_masks(model, masks)
            fine_tune(model, masks, class_train_loader, device, args.epochs, args.lr)
            acc = evaluate(model, class_test_loader, device, target_classes)
            keep = neuron_keep_rate(masks)
            results[method].append({'prune_ratio': pr, 'neuron_keep': keep, 'accuracy': acc})
            row.append(acc)
            del model

        print(f'\n{pr*100:>6.0f}%  ' + '  '.join(f'{a:>13.2f}%' for a in row))
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'SUMMARY  (unpruned: {unpruned_acc:.2f}%)')
    print(f'{"="*70}')
    print(f'{"prune%":>7}  ' + '  '.join(f'{m:>14}' for m in method_names))
    print('-' * (9 + 16 * len(method_names)))
    for i, pr in enumerate(prune_ratios):
        row_str = f'{pr*100:>6.0f}%  '
        for method in method_names:
            acc = results[method][i]['accuracy']
            row_str += f'{acc:>13.2f}%  '
        print(row_str)

    # ── Save ──────────────────────────────────────────────────────────────────
    save = {
        'experiment': 'E2',
        'title': 'Class-Aware MLP Pruning Sweep',
        'num_classes': args.num_classes,
        'subset_id': args.subset_id,
        'class_names': class_names,
        'epochs': args.epochs,
        'lr': args.lr,
        'prune_ratios': prune_ratios,
        'unpruned_acc': unpruned_acc,
        'methods': method_names,
        'results': results,
    }
    suffix = f'_s{args.subset_id}' if args.subset_id is not None else ''
    save_path = os.path.join(args.output_dir, f'e02_mlp_sweep_{args.num_classes}cls_{args.epochs}ep{suffix}.json')
    with open(save_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f'\n✓ Saved: {save_path}')


if __name__ == '__main__':
    main()
