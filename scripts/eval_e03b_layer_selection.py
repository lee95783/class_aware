#!/usr/bin/env python3
"""
E3b: Automatic Layer and Keep-Ratio Selection for CGTS

Objective: prune as early and as much as possible while retaining accuracy.

This experiment:
  1. Runs a fine-grained (layer, keep_ratio) grid to find the empirical
     Pareto frontier of FLOPs saved vs accuracy drop.
  2. Computes a prototype discriminability score D(L) at each layer
     purely from training data (no test labels needed).
  3. Validates that D(L) predicts the Pareto frontier — enabling
     automatic (L, k) selection without test-set sweeping.

D(L) definition — Fisher-like token discriminability:
  At layer L, for each image from the K target classes:
    s_i = x_patch_i · e_c   (dot product of each patch with class prototype)
  D(L) = mean over images of:
           (mean of top-50% scores − mean of bottom-50% scores) / std(scores)
  Higher D(L) → prototype clearly separates relevant from irrelevant tokens
             → can prune aggressively at this layer without accuracy loss

Automatic layer selection rule:
  L* = earliest L where L >= L_min AND D(L) > tau
  L_min = 4 (empirically: features below L=4 lack spatial semantics for pruning)
  tau   = midpoint of [min D(L), max D(L)] across candidate layers

Usage:
    python scripts/eval_e03b_layer_selection.py --num-classes 10 --device 0
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders


# ── Data ──────────────────────────────────────────────────────────────────────

def filter_loader(base_loader, class_indices, batch_size=128, shuffle=False):
    dataset = base_loader.dataset
    target_set = set(class_indices)
    labels = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    indices = [i for i, l in enumerate(labels) if l in target_set]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=4, pin_memory=True)


def load_class_subset(path, num_classes):
    with open(path) as f:
        d = json.load(f)
    s = d['subsets'][str(num_classes)]
    return s['class_indices'], s['class_names']


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device) -> nn.Module:
    model = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    state = torch.load('weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth',
                       map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Prototype ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_prototype(model, loader, device, layer_idx, num_batches=50):
    """Mean CLS feature at block[layer_idx].norm1 input."""
    captured = {}
    def hook(mod, inp, out): captured['f'] = inp[0].detach()
    handle = model.blocks[layer_idx].norm1.register_forward_hook(hook)
    all_cls = []
    for i, (imgs, _) in enumerate(loader):
        if i >= num_batches: break
        model(imgs.to(device))
        all_cls.append(captured['f'][:, 0, :].cpu())
    handle.remove()
    return torch.cat(all_cls, dim=0).mean(dim=0)  # [D]


# ── D(L): Prototype Discriminability Score ────────────────────────────────────

@torch.no_grad()
def compute_discriminability(model, loader, device, layer_idx, prototype, num_batches=30):
    """
    D(L) = mean over images of:
        (mean_top_half_score − mean_bottom_half_score) / std(scores)

    Measures how well e_c separates high-relevance from low-relevance tokens.
    Computed purely from training data — no test labels needed.
    """
    captured = {}
    def hook(mod, inp, out): captured['f'] = inp[0].detach()
    handle = model.blocks[layer_idx].norm1.register_forward_hook(hook)

    e_c = prototype.to(device)
    snr_vals = []

    for i, (imgs, _) in enumerate(loader):
        if i >= num_batches: break
        model(imgs.to(device))
        patch_feats = captured['f'][:, 1:, :]          # [B, N, D]
        scores = (patch_feats * e_c).sum(dim=-1)        # [B, N]

        N = scores.shape[1]
        half = N // 2
        sorted_s, _ = scores.sort(dim=1, descending=True)
        top_mean  = sorted_s[:, :half].mean(dim=1)     # [B]
        bot_mean  = sorted_s[:, half:].mean(dim=1)     # [B]
        std       = scores.std(dim=1).clamp(min=1e-6)  # [B]
        snr = (top_mean - bot_mean) / std              # [B]
        snr_vals.append(snr.cpu())

    handle.remove()
    return torch.cat(snr_vals).mean().item()


# ── CGTS forward ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cgts(model, loader, device, target_classes, prototype, layer_idx, keep_ratio):
    class_idx = torch.tensor(target_classes, device=device)
    n_patches = model.patch_embed.num_patches
    k = max(1, int(n_patches * keep_ratio))
    e_c = prototype.to(device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        B = imgs.size(0)
        x = model.patch_embed(imgs)
        x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)
        for l, blk in enumerate(model.blocks):
            if l == layer_idx:
                patch = x[:, 1:, :]
                top_idx = (patch * e_c).sum(-1).topk(k, dim=1).indices.sort(dim=1).values
                b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
                x = torch.cat([x[:, :1, :], patch[b_idx, top_idx]], dim=1)
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        logits = model.head(model.norm(x)[:, 0])
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, target_classes):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int, default=10)
    parser.add_argument('--device',       type=int, default=0)
    parser.add_argument('--num-batches',  type=int, default=50)
    parser.add_argument('--subset-file',  type=str, default='configs/class_subsets.json')
    parser.add_argument('--output-dir',   type=str, default='results/paper')
    parser.add_argument('--l-min',        type=int, default=4,
                        help='Minimum layer for auto-selection (default: 4)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Fine-grained sweep: more layers, more keep ratios
    layers      = [0, 2, 4, 6, 7, 8, 9, 10]
    keep_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    acc_budget  = 1.0   # max allowed accuracy drop from unpruned

    target_classes, class_names = load_class_subset(args.subset_file, args.num_classes)

    print(f'\n{"="*70}')
    print(f'E3b: Layer & Keep-Ratio Selection  |  K={args.num_classes}')
    print(f'Layers: {layers}')
    print(f'Keep ratios: {keep_ratios}')
    print(f'Accuracy budget: ≤{acc_budget}% drop from unpruned')
    print(f'Auto-selection: L >= {args.l_min} AND D(L) > tau')
    print(f'{"="*70}\n')

    print('Loading CIFAR-100...')
    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')
    class_train_loader = filter_loader(full_train_loader, target_classes, shuffle=True)
    class_test_loader  = filter_loader(full_test_loader,  target_classes, shuffle=False)

    model = load_model(device)
    n_patches   = model.patch_embed.num_patches
    num_blocks  = len(model.blocks)

    unpruned_acc = evaluate(model, class_test_loader, device, target_classes)
    threshold    = unpruned_acc - acc_budget
    print(f'Unpruned: {unpruned_acc:.2f}%  |  Accuracy threshold: {threshold:.2f}%\n')

    # ── Step 1: Compute D(L) and prototypes at each layer ─────────────────────
    print('Computing D(L) and prototypes at each layer...')
    print(f'  {"Layer":>6}  {"D(L)":>8}  {"proto_norm":>11}  {"blocks_after":>13}')
    print(f'  ' + '-' * 44)

    discriminability = {}
    prototypes = {}
    for l in layers:
        proto = compute_prototype(model, class_train_loader, device, l, args.num_batches)
        D     = compute_discriminability(model, class_train_loader, device, l, proto,
                                         num_batches=30)
        discriminability[l] = D
        prototypes[l] = proto
        blocks_after = num_blocks - l
        print(f'  {l:>6}  {D:>8.4f}  {proto.norm():>11.3f}  {blocks_after:>13}')
    print()

    # ── Step 2: Fine-grained accuracy sweep ───────────────────────────────────
    print('Running (layer, keep_ratio) accuracy sweep...')
    acc_grid = {}   # acc_grid[l][kr] = accuracy

    for l in layers:
        acc_grid[l] = {}
        proto = prototypes[l]
        row_str = f'  L{l:>2}  '
        for kr in keep_ratios:
            acc = evaluate_cgts(model, class_test_loader, device, target_classes,
                                proto, l, kr)
            acc_grid[l][kr] = acc
            marker = '✓' if acc >= threshold else '✗'
            row_str += f'  {acc:.1f}{marker}'
        print(row_str)
    print()

    # ── Step 3: Pareto frontier — maximize (blocks_after × tokens_dropped) ────
    print(f'{"="*70}')
    print(f'Pareto Frontier  (accuracy ≥ {threshold:.2f}%, maximize FLOPs saved)')
    print(f'FLOPs proxy = (num_blocks_after_L) × (1 − keep_ratio)')
    print(f'{"="*70}')
    print(f'  {"Layer":>6}  {"keep":>6}  {"accuracy":>10}  {"drop":>6}  '
          f'{"blocks_after":>13}  {"flops_proxy":>12}  {"D(L)":>8}')
    print(f'  ' + '-' * 72)

    pareto = []
    for l in layers:
        for kr in keep_ratios:
            acc = acc_grid[l][kr]
            if acc >= threshold:
                drop   = unpruned_acc - acc
                blocks = num_blocks - l
                flops  = blocks * (1.0 - kr)
                pareto.append({'layer': l, 'keep_ratio': kr, 'accuracy': acc,
                               'drop': drop, 'blocks_after': blocks,
                               'flops_proxy': flops, 'D': discriminability[l]})

    # Sort by flops_proxy descending
    pareto.sort(key=lambda x: x['flops_proxy'], reverse=True)
    for p in pareto:
        print(f'  {p["layer"]:>6}  {p["keep_ratio"]:>6.1f}  '
              f'{p["accuracy"]:>9.2f}%  {p["drop"]:>+5.2f}%  '
              f'{p["blocks_after"]:>13}  {p["flops_proxy"]:>12.2f}  '
              f'{p["D"]:>8.4f}')

    if pareto:
        best = pareto[0]
        print(f'\n  ★ Best: Layer {best["layer"]}, keep={best["keep_ratio"]}'
              f'  →  {best["accuracy"]:.2f}% acc, {best["flops_proxy"]:.2f} FLOPs proxy')

    # ── Step 4: Automatic layer selection via D(L) + L_min ────────────────────
    print(f'\n{"="*70}')
    print(f'Automatic Layer Selection: L >= {args.l_min} AND D(L) > tau')
    print(f'(tau = midpoint of D(L) range across candidate layers)')
    print(f'{"="*70}')

    candidate_layers = [l for l in layers if l >= args.l_min]
    d_vals = [discriminability[l] for l in candidate_layers]
    tau = (min(d_vals) + max(d_vals)) / 2

    print(f'  {"Layer":>6}  {"D(L)":>8}  {"pass":>6}  {"max_drop%":>10}  {"max_flops_proxy":>16}')
    print(f'  ' + '-' * 52)

    for l in layers:
        feasible_krs = [kr for kr in keep_ratios if acc_grid[l][kr] >= threshold]
        if feasible_krs:
            min_kr    = min(feasible_krs)
            max_drop  = 1.0 - min_kr
            max_flops = (num_blocks - l) * max_drop
        else:
            max_drop  = 0.0
            max_flops = 0.0
        passes = (l >= args.l_min) and (discriminability[l] > tau)
        flag = '✓' if passes else '✗'
        print(f'  {l:>6}  {discriminability[l]:>8.4f}  {flag:>6}  '
              f'{max_drop*100:>9.0f}%  {max_flops:>16.2f}')

    passing = [l for l in layers if l >= args.l_min and discriminability[l] > tau]
    auto_layer = min(passing) if passing else None
    print(f'\n  tau = {tau:.4f}  (L_min = {args.l_min})')
    if auto_layer is not None:
        print(f'  Auto-selected layer: {auto_layer}')

    # ── Save ──────────────────────────────────────────────────────────────────
    save = {
        'experiment': 'E3b',
        'title': 'Automatic Layer and Keep-Ratio Selection',
        'num_classes': args.num_classes,
        'class_names': class_names,
        'layers': layers,
        'keep_ratios': keep_ratios,
        'n_patches': n_patches,
        'num_blocks': num_blocks,
        'unpruned_acc': unpruned_acc,
        'acc_budget': acc_budget,
        'l_min': args.l_min,
        'tau': tau,
        'auto_layer': auto_layer,
        'discriminability': {str(l): discriminability[l] for l in layers},
        'prototype_norms': {str(l): prototypes[l].norm().item() for l in layers},
        'acc_grid': {str(l): {str(kr): acc_grid[l][kr] for kr in keep_ratios}
                    for l in layers},
        'pareto': pareto,
    }
    save_path = os.path.join(args.output_dir, f'e03b_layer_selection_{args.num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f'\n✓ Saved: {save_path}')


if __name__ == '__main__':
    main()
