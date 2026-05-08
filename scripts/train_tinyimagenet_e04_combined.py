#!/usr/bin/env python3
"""
E4 (TinyImageNet): Combined class-aware MLP pruning + CGTS.

Adapted from train_e04_combined.py for:
  - Model:   deit_small_patch16_224  (D=384, 200 classes)
  - Dataset: TinyImageNet-200
  - Checkpoint: weights/deit_small_patch16_224_tinyimagenet_best.pth

Protocol:
  1. Load fine-tuned DeiT-Small checkpoint
  2. Compute class-aware Taylor MLP importance on K-class training data
  3. Prune at ratio pr; fine-tune 5 epochs on K-class data
  4. Apply CGTS (zero-shot) at layer 6 with keep_ratio kr
  Grid: pr ∈ {0.0, 0.3, 0.5}  ×  kr ∈ {1.0, 0.6}

Usage:
    python scripts/train_tinyimagenet_e04_combined.py --num-classes 10 --device 0
    python scripts/train_tinyimagenet_e04_combined.py --num-classes 5 10 20 50 --device 0
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as T
import torchvision.datasets as datasets
from pathlib import Path
from torch.utils.data import DataLoader, Subset

CHECKPOINT  = 'weights/deit_small_patch16_224_tinyimagenet_best.pth'
NUM_CLASSES = 200


# ── Data ──────────────────────────────────────────────────────────────────────

def get_full_loaders(data_dir, batch_size=128, num_workers=8):
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def filter_loader(base_loader, class_indices, batch_size=64, shuffle=False):
    dataset = base_loader.dataset
    target_set = set(class_indices)
    indices = [i for i, l in enumerate(dataset.targets) if l in target_set]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=4, pin_memory=True)


def load_class_subset(path, num_classes):
    with open(path) as f:
        d = json.load(f)
    s = d['subsets'][str(num_classes)]
    return s['class_indices'], s['class_names']


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device):
    model = timm.create_model('deit_small_patch16_224', num_classes=NUM_CLASSES,
                               pretrained=False).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    return model


# ── Class-aware Taylor MLP pruning ────────────────────────────────────────────

def compute_class_taylor(model, loader, device, num_batches=50):
    num_layers = len(model.blocks)
    accum  = {l: None for l in range(num_layers)}
    counts = {l: 0    for l in range(num_layers)}
    acts, handles = {}, []
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
            loss = F.cross_entropy(model(imgs), lbls)
            loss.backward(); optimizer.step(); scheduler.step()
            apply_masks(model, masks)
            loss_sum += loss.item()
            correct  += (model(imgs).argmax(1) == lbls).sum().item() if False else 0
            total    += lbls.size(0)
        print(f'    ep{epoch+1}: loss={loss_sum/len(train_loader):.4f}', flush=True)
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
        x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1) + model.pos_embed
        x = model.pos_drop(x)
        for l, blk in enumerate(model.blocks):
            if l == layer_idx:
                patch   = x[:, 1:, :]
                top_idx = (patch * e_c).sum(-1).topk(k, dim=1).indices.sort(dim=1).values
                b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
                x = torch.cat([x[:, :1], patch[b_idx, top_idx]], dim=1)
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        logits = model.head(model.norm(x)[:, 0])
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total   += lbls.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, target_classes):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total   += lbls.size(0)
    return 100.0 * correct / total


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, full_train_loader, full_test_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)

    print(f'\n{"="*70}')
    print(f'E4 Combined (TinyImageNet/DeiT-Small)  |  K={num_classes}  |  CGTS layer={args.cgts_layer}')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    class_train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    class_test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)

    mlp_prune_ratios = [0.0, 0.3, 0.5]
    cgts_keep_ratios = [1.0, 0.6]

    print(f'Computing class Taylor scores...')
    score_model = load_model(device)
    taylor_scores = compute_class_taylor(score_model, class_train_loader, device, args.num_batches)
    del score_model
    print('  Done.\n')

    results = {}
    for pr in mlp_prune_ratios:
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
        print(f'  Fine-tuning {args.epochs} epochs...')
        fine_tune(model, masks, class_train_loader, device, args.epochs, args.lr)

        mlp_acc = evaluate(model, class_test_loader, device, target_classes)
        print(f'  MLP-only acc: {mlp_acc:.2f}%')

        proto = compute_prototype(model, class_train_loader, device,
                                  args.cgts_layer, args.num_batches)
        cgts_row = {}
        for kr in cgts_keep_ratios:
            acc = mlp_acc if kr == 1.0 else evaluate_cgts(
                model, class_test_loader, device, target_classes, proto, args.cgts_layer, kr)
            cgts_row[kr] = acc
            label = 'no CGTS' if kr == 1.0 else f'CGTS keep={kr}'
            print(f'    {label}: {acc:.2f}%')

        results[pr] = {'neuron_keep': neuron_keep, 'mlp_acc': mlp_acc, 'cgts': cgts_row}
        del model
        print()

    # Summary
    print(f'SUMMARY  K={num_classes}:')
    print(f'  {"pr":>5}  {"no CGTS":>10}  {"CGTS 0.6":>10}')
    for pr in mlp_prune_ratios:
        r = results[pr]
        print(f'  {pr:>5.1f}  {r["cgts"][1.0]:>9.2f}%  {r["cgts"][0.6]:>9.2f}%')

    save = {
        'experiment': 'E4',
        'dataset': 'tinyimagenet',
        'model': 'deit_small_patch16_224',
        'num_classes': num_classes,
        'class_names': class_names,
        'epochs': args.epochs,
        'cgts_layer': args.cgts_layer,
        'mlp_prune_ratios': mlp_prune_ratios,
        'cgts_keep_ratios': cgts_keep_ratios,
        'results': {str(pr): {
            'neuron_keep': results[pr]['neuron_keep'],
            'mlp_acc':     results[pr]['mlp_acc'],
            'cgts':        {str(kr): results[pr]['cgts'][kr] for kr in cgts_keep_ratios},
        } for pr in mlp_prune_ratios},
    }
    save_path = os.path.join(args.output_dir,
                             f'tinyimagenet_e04_combined_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--epochs',       type=int,   default=5)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--cgts-layer',   type=int,   default=6)
    parser.add_argument('--device',       type=int,   default=0)
    parser.add_argument('--num-batches',  type=int,   default=50)
    parser.add_argument('--data-dir',     type=str,   default='data/tiny-imagenet-200')
    parser.add_argument('--subset-file',  type=str,   default='configs/tinyimagenet_class_subsets.json')
    parser.add_argument('--output-dir',   type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT}')

    full_train_loader, full_test_loader = get_full_loaders(args.data_dir)
    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, full_train_loader, full_test_loader)


if __name__ == '__main__':
    main()
