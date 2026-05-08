#!/usr/bin/env python3
"""
E3: CGTS Token Pruning Sweep (zero-shot, no MLP pruning)

Evaluates CGTS on the unpruned model across:
  - Pruning layers: 3, 6, 9
  - Token keep ratios: 0.9, 0.7, 0.5

Comparison methods:
  unpruned       — all tokens, no pruning
  random_drop    — drop random tokens at layer L (static per call)
  global_taylor  — drop lowest global Taylor-scored tokens (static, layer 0)
  cgts           — class prototype dot-product, per-image dynamic drop at layer L

Usage:
    python scripts/eval_e03_cgts_sweep.py --num-classes 10 --device 0
"""

import os
import sys
import json
import torch
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

def load_model(device):
    model = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    state = torch.load('weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth',
                       map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Token scoring ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_prototype(model, loader, device, layer_idx, num_batches=50):
    """Mean CLS feature at block[layer_idx].norm1 input over loader."""
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


def compute_global_taylor(model, loader, device, num_batches=50):
    """Mean |grad × act| at norm1 input, averaged over all layers → [N_patches]."""
    num_layers = len(model.blocks)
    accum = {l: None for l in range(num_layers)}
    counts = {l: 0 for l in range(num_layers)}
    acts, grads = {}, {}
    handles = []
    for l, blk in enumerate(model.blocks):
        def fwd(mod, inp, out, i=l): acts[i] = inp[0]
        def bwd(mod, gin, gout, i=l): grads[i] = gin[0]
        handles.append(blk.norm1.register_forward_hook(fwd))
        handles.append(blk.norm1.register_full_backward_hook(bwd))

    for p in model.parameters(): p.requires_grad_(True)
    for i, (imgs, lbls) in enumerate(loader):
        if i >= num_batches: break
        imgs, lbls = imgs.to(device), lbls.to(device)
        model.zero_grad()
        F.cross_entropy(model(imgs), lbls).backward()
        for l in range(num_layers):
            if l in acts and l in grads:
                s = (acts[l] * grads[l]).abs()[:, 1:, :].mean(dim=(0, 2)).detach().cpu()
                accum[l] = s if accum[l] is None else accum[l] + s
                counts[l] += 1
    for h in handles: h.remove()
    for p in model.parameters(): p.requires_grad_(False)

    per_layer = [accum[l] / max(counts[l], 1) for l in range(num_layers)]
    return torch.stack(per_layer).mean(dim=0)  # [N_patches]


# ── Forward passes ────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_cgts(model, images, prototype, layer_idx, k, device):
    """Per-image dynamic token selection at layer_idx."""
    B = images.size(0)
    e_c = prototype.to(device)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            patch = x[:, 1:, :]
            scores = (patch * e_c).sum(dim=-1)          # [B, N]
            top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x = torch.cat([x[:, :1, :], patch[b_idx, top_idx]], dim=1)
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


@torch.no_grad()
def forward_static_drop(model, images, keep_patch_idx, device):
    """Static token drop at layer 0 (same subset of patches for all images)."""
    B = images.size(0)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    seq_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                         keep_patch_idx.to(device) + 1])
    x = x[:, seq_idx, :]
    for blk in model.blocks:
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


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


@torch.no_grad()
def evaluate_cgts(model, loader, device, target_classes, prototype, layer_idx, k):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = forward_cgts(model, imgs, prototype, layer_idx, k, device)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_static(model, loader, device, target_classes, keep_idx):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = forward_static_drop(model, imgs, keep_idx, device)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, model, n_patches,
              full_test_loader, full_train_loader, taylor_sorted):
    layers      = [3, 6, 9]
    keep_ratios = [0.9, 0.7, 0.5]

    target_classes, class_names = load_class_subset(args.subset_file, num_classes)

    print(f'\n{"="*70}')
    print(f'E3: CGTS Token Pruning Sweep  |  K={num_classes}  |  zero-shot')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    class_train_loader = filter_loader(full_train_loader, target_classes, shuffle=True)
    class_test_loader  = filter_loader(full_test_loader,  target_classes, shuffle=False)

    unpruned_acc = evaluate(model, class_test_loader, device, target_classes)
    print(f'Unpruned: {unpruned_acc:.2f}%\n')

    print('Computing class prototypes at each layer...')
    prototypes = {}
    for l in layers:
        proto = compute_prototype(model, class_train_loader, device, l, args.num_batches)
        prototypes[l] = proto
        print(f'  Layer {l}: norm={proto.norm():.3f}')
    print()

    results = {}
    for layer_idx in layers:
        results[layer_idx] = {}
        proto = prototypes[layer_idx]

        print(f'Layer {layer_idx}:')
        print(f'  {"keep%":>6}  {"random":>8}  {"global_taylor":>14}  {"cgts":>8}  {"cgts-gt":>8}')
        print(f'  ' + '-' * 54)

        for kr in keep_ratios:
            k = max(1, int(n_patches * kr))
            drop_pct = int((1 - kr) * 100)

            rand_idx = torch.randperm(n_patches)[:k].sort().values
            rand_acc = evaluate_static(model, class_test_loader, device, target_classes, rand_idx)

            gt_idx   = taylor_sorted[:k].sort().values
            gt_acc   = evaluate_static(model, class_test_loader, device, target_classes, gt_idx)

            cgts_acc = evaluate_cgts(model, class_test_loader, device, target_classes,
                                     proto, layer_idx, k)

            results[layer_idx][kr] = {
                'keep_ratio': kr, 'drop_pct': drop_pct, 'k': k,
                'random': rand_acc, 'global_taylor': gt_acc, 'cgts': cgts_acc,
            }
            print(f'  {drop_pct:>5}%  {rand_acc:>7.2f}%  {gt_acc:>13.2f}%'
                  f'  {cgts_acc:>7.2f}%  {cgts_acc-gt_acc:>+7.2f}%')
        print()

    print(f'{"="*70}')
    print(f'SUMMARY  (unpruned: {unpruned_acc:.2f}%)')
    print(f'{"="*70}')
    print(f'\nCGTS accuracy by layer and token keep ratio:')
    print(f'  {"drop%":>6}  {"layer 3":>9}  {"layer 6":>9}  {"layer 9":>9}  {"best":>14}')
    print(f'  ' + '-' * 55)
    for kr in keep_ratios:
        drop_pct = int((1 - kr) * 100)
        accs = {l: results[l][kr]['cgts'] for l in layers}
        best_l = max(accs, key=accs.get)
        print(f'  {drop_pct:>5}%  ' +
              '  '.join(f'{accs[l]:>8.2f}%' for l in layers) +
              f'  L{best_l} ({accs[best_l]:.2f}%)')

    save = {
        'experiment': 'E3',
        'title': 'CGTS Token Pruning Sweep',
        'num_classes': num_classes,
        'class_names': class_names,
        'layers': layers,
        'keep_ratios': keep_ratios,
        'n_patches': n_patches,
        'unpruned_acc': unpruned_acc,
        'results': {str(l): {str(kr): v for kr, v in d.items()}
                    for l, d in results.items()},
    }
    save_path = os.path.join(args.output_dir, f'e03_cgts_sweep_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f'\n✓ Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--device',       type=int, default=0)
    parser.add_argument('--num-batches',  type=int, default=50)
    parser.add_argument('--subset-file',  type=str, default='configs/class_subsets.json')
    parser.add_argument('--output-dir',   type=str, default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading CIFAR-100...')
    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')

    model     = load_model(device)
    n_patches = model.patch_embed.num_patches

    # Global Taylor scores are class-agnostic: compute once, reuse across all K
    print(f'Computing global Taylor token scores ({args.num_batches} batches)...')
    taylor_sorted = compute_global_taylor(
        model, full_train_loader, device, args.num_batches).argsort(descending=True)
    print('  Done.\n')

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, model, n_patches,
                  full_test_loader, full_train_loader, taylor_sorted)


if __name__ == '__main__':
    main()
