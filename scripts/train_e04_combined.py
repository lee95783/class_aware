#!/usr/bin/env python3
"""
E4: Combined MLP Pruning + CGTS (Main Result)

Tests whether MLP pruning and CGTS compose — i.e., their benefits add up.

Protocol:
  1. Load pretrained model
  2. class_taylor MLP pruning at prune_ratio pr
  3. Fine-tune 5 epochs on K-class data
  4. Apply CGTS (zero-shot) at layer 9 with keep_ratio kr

Grid:
  MLP prune ratios:  0.0, 0.3, 0.5
  CGTS keep ratios:  1.0, 0.7, 0.5

Usage:
    python scripts/train_e04_combined.py --num-classes 10 --epochs 5 --device 0
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
    return model


# ── MLP pruning ───────────────────────────────────────────────────────────────

def compute_class_taylor(model, loader, device, num_batches=50):
    num_layers = len(model.blocks)
    accum = {l: None for l in range(num_layers)}
    counts = {l: 0 for l in range(num_layers)}
    acts = {}
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
    for p in model.parameters(): p.requires_grad_(True)
    for i, (imgs, lbls) in enumerate(loader):
        if i >= num_batches: break
        imgs, lbls = imgs.to(device), lbls.to(device)
        model.zero_grad()
        F.cross_entropy(model(imgs), lbls).backward()
    for h in handles: h.remove()
    for p in model.parameters(): p.requires_grad_(False)
    return {l: accum[l] / max(counts[l], 1) for l in range(num_layers)}


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
        total += m.numel(); kept += m.sum().item()
    return kept / total if total > 0 else 1.0


def fine_tune(model, masks, train_loader, device, epochs, lr=1e-4):
    for p in model.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader))
    model.train()
    for epoch in range(epochs):
        correct = total = loss_sum = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, lbls)
            loss.backward(); optimizer.step(); scheduler.step()
            apply_masks(model, masks)
            loss_sum += loss.item()
            correct += (logits.argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
        print(f'    ep{epoch+1}: loss={loss_sum/len(train_loader):.4f}'
              f'  train={100*correct/total:.1f}%', flush=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)


# ── CGTS ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_prototype(model, loader, device, layer_idx, num_batches=50):
    captured = {}
    def hook(mod, inp, out): captured['f'] = inp[0].detach()
    handle = model.blocks[layer_idx].norm1.register_forward_hook(hook)
    all_cls = []
    for i, (imgs, _) in enumerate(loader):
        if i >= num_batches: break
        model(imgs.to(device))
        all_cls.append(captured['f'][:, 0, :].cpu())
    handle.remove()
    return torch.cat(all_cls, dim=0).mean(dim=0)


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


# ── Evaluation ────────────────────────────────────────────────────────────────

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
    parser.add_argument('--num-classes',  type=int,   default=10)
    parser.add_argument('--epochs',       type=int,   default=5)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--cgts-layer',   type=int,   default=4)
    parser.add_argument('--device',       type=int,   default=0)
    parser.add_argument('--num-batches',  type=int,   default=50)
    parser.add_argument('--subset-file',  type=str,   default='configs/class_subsets.json')
    parser.add_argument('--output-dir',   type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    mlp_prune_ratios = [0.0, 0.3, 0.5]
    cgts_keep_ratios = [1.0, 0.6]

    target_classes, class_names = load_class_subset(args.subset_file, args.num_classes)

    print(f'\n{"="*70}')
    print(f'E4: Combined MLP + CGTS  |  K={args.num_classes}  |  CGTS layer={args.cgts_layer}')
    print(f'Classes: {", ".join(class_names)}')
    print(f'MLP prune ratios: {mlp_prune_ratios}  |  CGTS keep ratios: {cgts_keep_ratios}')
    print(f'{"="*70}\n')

    print('Loading CIFAR-100...')
    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')
    class_train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    class_test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)

    # Pre-compute class Taylor scores (reused across MLP prune ratios)
    print(f'Computing class Taylor scores ({args.num_batches} batches)...')
    score_model = load_model(device)
    taylor_scores = compute_class_taylor(score_model, class_train_loader, device, args.num_batches)
    del score_model
    print('  Done.\n')

    results = {}

    for pr in mlp_prune_ratios:
        results[pr] = {}

        print(f'─── MLP prune_ratio = {pr} ───')
        model = load_model(device)

        if pr == 0.0:
            masks = {l: torch.ones(blk.mlp.fc1.out_features, dtype=torch.bool)
                     for l, blk in enumerate(model.blocks)}
            neuron_keep = 1.0
        else:
            masks = build_masks(taylor_scores, pr)
            apply_masks(model, masks)
            neuron_keep = neuron_keep_rate(masks)

        print(f'  Neuron keep rate: {neuron_keep:.3f}')

        # Fine-tune (even for pr=0.0 to match E2 protocol)
        print(f'  Fine-tuning {args.epochs} epochs...')
        fine_tune(model, masks, class_train_loader, device, args.epochs, args.lr)

        # Accuracy after MLP pruning + fine-tune, no token pruning
        mlp_acc = evaluate(model, class_test_loader, device, target_classes)
        print(f'  MLP-only acc: {mlp_acc:.2f}%')

        # Compute prototype for CGTS
        proto = compute_prototype(model, class_train_loader, device,
                                  args.cgts_layer, args.num_batches)

        # CGTS sweep
        cgts_row = {}
        for kr in cgts_keep_ratios:
            if kr == 1.0:
                acc = mlp_acc
            else:
                acc = evaluate_cgts(model, class_test_loader, device, target_classes,
                                    proto, args.cgts_layer, kr)
            cgts_row[kr] = acc
            label = 'no CGTS' if kr == 1.0 else f'CGTS keep={kr}'
            print(f'    {label}: {acc:.2f}%  (vs mlp_only: {acc-mlp_acc:+.2f}%)')

        results[pr] = {'neuron_keep': neuron_keep, 'mlp_acc': mlp_acc, 'cgts': cgts_row}
        del model
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'{"="*70}')
    print(f'SUMMARY — MLP prune ratio × CGTS keep ratio  (K={args.num_classes})')
    print(f'{"="*70}')
    header = f'{"MLP pr":>8}  {"neuron_keep":>12}  ' + \
             '  '.join(f'{"CGTS "+str(kr):>10}' for kr in cgts_keep_ratios)
    print(header)
    print('-' * len(header))
    for pr in mlp_prune_ratios:
        r = results[pr]
        row = f'{pr:>8.1f}  {r["neuron_keep"]:>12.3f}  '
        row += '  '.join(f'{r["cgts"][kr]:>9.2f}%' for kr in cgts_keep_ratios)
        print(row)

    # ── Save ──────────────────────────────────────────────────────────────────
    save = {
        'experiment': 'E4',
        'title': 'Combined MLP Pruning + CGTS',
        'num_classes': args.num_classes,
        'class_names': class_names,
        'epochs': args.epochs,
        'cgts_layer': args.cgts_layer,
        'mlp_prune_ratios': mlp_prune_ratios,
        'cgts_keep_ratios': cgts_keep_ratios,
        'results': {str(pr): {
            'neuron_keep': results[pr]['neuron_keep'],
            'mlp_acc': results[pr]['mlp_acc'],
            'cgts': {str(kr): results[pr]['cgts'][kr] for kr in cgts_keep_ratios},
        } for pr in mlp_prune_ratios},
    }
    save_path = os.path.join(args.output_dir,
                             f'e04_combined_{args.num_classes}cls_{args.epochs}ep.json')
    with open(save_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f'\n✓ Saved: {save_path}')


if __name__ == '__main__':
    main()
