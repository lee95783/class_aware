#!/usr/bin/env python3
"""
E7 baseline: SViTE — joint structured head pruning + static token pruning.

Head importance: L2 norm of attention projection weights per head (class-agnostic).
Token importance: mean L2 norm of patch features at `layer_idx` over K-class
training data (class-aware static mask — same positions kept for all images).

Pipeline:
  1. Compute head and token importance scores
  2. Zero out lowest-importance heads (hard pruning)
  3. Build static boolean token mask (top-k positions by token importance)
  4. Fine-tune with head masks re-applied and token mask fixed
  5. Evaluate with both masks applied

Usage:
    python scripts/train_e07_svite.py --num-classes 10 --device 0
"""

import os, sys, json, argparse
import torch
import torch.nn.functional as F
import timm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders
from src.pruning import prune_vit_attention_heads


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

def load_base_model(device):
    model = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    state = torch.load('weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth',
                       map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    return model


# ── Importance scoring ────────────────────────────────────────────────────────

def compute_head_importance(model):
    """L2 norm of proj weight columns per head — class-agnostic magnitude score."""
    num_heads = model.blocks[0].attn.num_heads
    head_dim  = model.blocks[0].attn.head_dim
    scores = {}
    for l, blk in enumerate(model.blocks):
        proj_w = blk.attn.proj.weight.data        # [D, D]
        scores[l] = torch.tensor([
            proj_w[:, h * head_dim:(h + 1) * head_dim].norm().item()
            for h in range(num_heads)
        ])
    return scores   # {layer: tensor[num_heads]}


@torch.no_grad()
def compute_token_importance(model, loader, device, layer_idx, num_batches=50):
    """Mean L2 norm of patch features at output of layer_idx — class-aware."""
    importance = None
    count = 0
    captured = {}

    handle = model.blocks[layer_idx].register_forward_hook(
        lambda m, inp, out: captured.__setitem__('x', out.detach())
    )
    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(images.to(device))
        imp = captured['x'][:, 1:, :].norm(dim=-1).mean(dim=0).cpu()  # [N]
        importance = imp if importance is None else importance + imp
        count += 1
    handle.remove()
    return importance / max(count, 1)   # [N]


# ── Pruning ───────────────────────────────────────────────────────────────────

def build_heads_to_prune(head_scores, prune_ratio):
    """Return (layer, head) pairs for the lowest-importance fraction."""
    all_scores = [(v.item(), l, h)
                  for l, s in head_scores.items()
                  for h, v in enumerate(s)]
    all_scores.sort()
    n_prune = max(0, int(len(all_scores) * prune_ratio))
    return [(l, h) for _, l, h in all_scores[:n_prune]]


def build_token_mask(token_importance, keep_ratio):
    """Boolean mask [N_patches]: True = keep."""
    N = token_importance.size(0)
    k = max(1, int(N * keep_ratio))
    mask = torch.zeros(N, dtype=torch.bool)
    mask[token_importance.topk(k).indices] = True
    return mask


# ── Forward with static token mask ───────────────────────────────────────────

def forward_with_mask(model, images, token_mask, layer_idx):
    """Forward pass applying a static boolean token mask after layer_idx.
    layer_idx < 0 disables masking (used for unpruned baseline)."""
    bb = model
    B = images.size(0)
    x = bb.patch_embed(images)
    x = torch.cat([bb.cls_token.expand(B, -1, -1), x], dim=1) + bb.pos_embed
    x = bb.pos_drop(x)

    for i, blk in enumerate(bb.blocks):
        x = blk(x)
        if layer_idx >= 0 and i == layer_idx:
            kept = x[:, 1:][:, token_mask]
            x = torch.cat([x[:, :1], kept], dim=1)

    x = bb.norm(x)
    return bb.head(x[:, 0])


# ── Training / evaluation ─────────────────────────────────────────────────────

def fine_tune(model, heads_to_prune, token_mask, layer_idx,
              train_loader, device, epochs, lr=1e-4):
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader))
    model.train()
    token_mask = token_mask.to(device)
    for epoch in range(epochs):
        correct = total = loss_sum = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = forward_with_mask(model, images, token_mask, layer_idx)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if heads_to_prune:
                prune_vit_attention_heads(model, heads_to_prune)
            loss_sum += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f'    ep{epoch+1}: loss={loss_sum/len(train_loader):.4f}'
              f'  train={100*correct/total:.1f}%', flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def evaluate(model, loader, device, target_classes, token_mask, layer_idx):
    class_idx = torch.tensor(target_classes, device=device)
    token_mask = token_mask.to(device)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = forward_with_mask(model, images, token_mask, layer_idx)
        preds = class_idx[logits[:, class_idx].argmax(1)]
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, full_train_loader, full_test_loader, head_scores):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)

    print(f'\n{"="*70}')
    print(f'E7 SViTE  |  K={num_classes}  |  layer={args.layer_idx}  |  epochs={args.epochs}')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    class_train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    class_test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)

    base_model = load_base_model(device)
    N_patches = base_model.patch_embed.num_patches

    # Unpruned baseline (layer_idx=-1 disables masking in forward_with_mask)
    unpruned_acc = evaluate(base_model, class_test_loader, device, target_classes,
                            token_mask=torch.ones(N_patches, dtype=torch.bool), layer_idx=-1)
    print(f'Unpruned baseline: {unpruned_acc:.2f}%\n')

    # Token importance is K-specific (class-aware); head importance is reused from caller
    print(f'Computing token importance at layer {args.layer_idx}...')
    token_importance = compute_token_importance(
        base_model, class_train_loader, device, args.layer_idx, args.num_batches)
    del base_model

    sweep = [
        (0.0, 1.0), (0.0, 0.7), (0.0, 0.5),
        (0.3, 1.0), (0.3, 0.7), (0.3, 0.5),
        (0.5, 1.0), (0.5, 0.7), (0.5, 0.5),
    ]

    print(f'\n{"head_prune":>12}  {"token_keep":>12}  {"accuracy":>10}  {"drop":>8}')
    print('-' * 50)

    results = []
    for head_prune, token_keep in sweep:
        label = f'h{int(head_prune*100)}_t{int(token_keep*100)}'
        print(f'\n[{label}] Training...', flush=True)

        model = load_base_model(device)
        heads_to_prune = build_heads_to_prune(head_scores, head_prune)
        if heads_to_prune:
            prune_vit_attention_heads(model, heads_to_prune)

        token_mask = build_token_mask(token_importance, token_keep)
        layer_idx = args.layer_idx if token_keep < 1.0 else -1

        fine_tune(model, heads_to_prune, token_mask, layer_idx,
                  class_train_loader, device, args.epochs, args.lr)
        acc = evaluate(model, class_test_loader, device, target_classes,
                       token_mask, layer_idx)
        drop = unpruned_acc - acc

        print(f'{head_prune:>12.1f}  {token_keep:>12.1f}  {acc:>10.2f}%  {drop:>+8.2f}%')
        results.append({
            'head_prune_ratio': head_prune,
            'head_keep_ratio': 1.0 - head_prune,
            'token_keep_ratio': token_keep,
            'layer_idx': layer_idx,
            'accuracy': acc,
            'drop': drop,
        })
        del model
        torch.cuda.empty_cache()

    out = {
        'method': 'SViTE',
        'num_classes': num_classes,
        'layer_idx': args.layer_idx,
        'epochs': args.epochs,
        'unpruned_acc': unpruned_acc,
        'results': results,
    }
    save_path = os.path.join(args.output_dir, f'e07_svite_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int,   nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--layer-idx',   type=int,   default=6)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--device',      type=int,   default=0)
    parser.add_argument('--num-batches', type=int,   default=50)
    parser.add_argument('--subset-file', type=str,   default='configs/class_subsets.json')
    parser.add_argument('--output-dir',  type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')

    # Head importance is class-agnostic: compute once and reuse across all K
    print('Computing head importance (magnitude, class-agnostic)...')
    base_model = load_base_model(device)
    head_scores = compute_head_importance(base_model)
    del base_model

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, full_train_loader, full_test_loader, head_scores)


if __name__ == '__main__':
    main()
