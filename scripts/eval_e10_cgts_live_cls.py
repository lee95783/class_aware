#!/usr/bin/env python3
"""
E10: Live-CLS CGTS (runtime image-adaptive prototype)

Instead of an offline mean prototype (p_K), use the CLS token of the
current image at the pruning layer as the scoring vector:

    score(patch_j) = cls_current · patch_j

cls_current is the CLS token at block[layer_idx] input — it has already
aggregated information from all patches via self-attention in blocks 0..layer_idx-1.
It encodes what THIS specific image looks like, not the class average.

Cost: identical to standard CGTS — one [B,N,D]·[B,D,1] batched dot product.
No offline computation, no calibration data needed. Fully zero-shot.

Compares:
  unpruned        — baseline
  cgts_mean       — standard offline mean prototype
  cgts_live_cls   — runtime CLS token as prototype  [this work]
  zero_tprune     — WPR-based zero-shot (reference)

Usage:
    python scripts/eval_e10_cgts_live_cls.py --num-classes 5 10 20 50 --device 0
"""

import os, sys, json, argparse
import torch
import timm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders
from scripts.eval_e07_zero_tprune import _wpr, _attn_weights_from_qkv

CHECKPOINT  = 'weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth'
MODEL_NAME  = 'deit_tiny_patch16_224'
NUM_CLASSES = 100
N_PATCHES   = 196


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


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device):
    model = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Prototype computation (for mean baseline) ─────────────────────────────────

@torch.no_grad()
def compute_mean_cls(model, loader, device, layer_idx, num_batches=50):
    captured = {}
    handle = model.blocks[layer_idx].norm1.register_forward_hook(
        lambda mod, inp, out: captured.update({'f': inp[0].detach()})
    )
    all_cls = []
    for i, (imgs, _) in enumerate(loader):
        if i >= num_batches: break
        model(imgs.to(device))
        all_cls.append(captured['f'][:, 0].cpu())
    handle.remove()
    return torch.cat(all_cls).mean(0)   # [D]


# ── Forward passes ────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_cgts_mean(model, images, prototype, layer_idx, k, device):
    """Standard CGTS: fixed offline prototype."""
    B = images.size(0)
    p = prototype.to(device)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            patch   = x[:, 1:]
            scores  = (patch * p).sum(-1)
            top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x       = torch.cat([x[:, :1], patch[b_idx, top_idx]], dim=1)
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


@torch.no_grad()
def forward_cgts_live_cls(model, images, layer_idx, k, device):
    """
    Live-CLS CGTS: use the CLS token of the current image at block[layer_idx]
    as the scoring prototype. Fully image-adaptive, zero extra cost.

    score(patch_j) = cls_current · patch_j

    cls_current is captured via a hook on block[layer_idx].norm1 input,
    which gives the pre-norm features — same hook point as mean prototype.
    """
    B = images.size(0)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            # CLS token at this layer input — [B, D]
            cls_current = x[:, 0]                             # already in sequence
            patch       = x[:, 1:]                            # [B, N, D]
            scores      = (patch * cls_current.unsqueeze(1)).sum(-1)  # [B, N]
            top_idx     = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx       = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x           = torch.cat([x[:, :1], patch[b_idx, top_idx]], dim=1)
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


@torch.no_grad()
def forward_zero_tprune(model, images, layer_idx, k, device, n_iter=3):
    B = images.size(0)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            captured = {}
            handle = blk.attn.qkv.register_forward_hook(
                lambda mod, inp, out: captured.update({'qkv': out})
            )
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            handle.remove()
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            attn_w  = _attn_weights_from_qkv(blk.attn, captured['qkv'])
            scores  = _wpr(attn_w, n_iter)[:, 1:]
            top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x       = torch.cat([x[:, :1], x[:, 1:][b_idx, top_idx]], dim=1)
        else:
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
def evaluate_fn(forward_fn, loader, device, target_classes):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = forward_fn(imgs)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, model,
              full_train_loader, full_test_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)
    train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)
    keep_ratios  = [0.7, 0.5]

    print(f'\n{"="*70}')
    print(f'E10 Live-CLS CGTS  |  K={num_classes}  |  layer={args.layer_idx}  |  zero-shot')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    unpruned_acc = evaluate(model, test_loader, device, target_classes)
    print(f'Unpruned: {unpruned_acc:.2f}%\n')

    mean_proto = compute_mean_cls(model, train_loader, device, args.layer_idx, args.num_batches)
    print(f'Mean prototype computed: shape={list(mean_proto.shape)}\n')

    results = []
    for keep in keep_ratios:
        k = max(1, int(N_PATCHES * keep))

        acc_mean = evaluate_fn(
            lambda imgs, _k=k: forward_cgts_mean(
                model, imgs, mean_proto, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        acc_live = evaluate_fn(
            lambda imgs, _k=k: forward_cgts_live_cls(
                model, imgs, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        acc_ztp = evaluate_fn(
            lambda imgs, _k=k: forward_zero_tprune(
                model, imgs, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        gain = acc_live - acc_mean
        print(f'  keep={keep}  k={k}:')
        print(f'    mean      {acc_mean:>7.2f}%  (drop={unpruned_acc-acc_mean:+.2f}%)')
        print(f'    live-cls  {acc_live:>7.2f}%  (drop={unpruned_acc-acc_live:+.2f}%)  '
              f'gain vs mean: {gain:+.2f}%')
        print(f'    zero-tpru {acc_ztp:>7.2f}%  (drop={unpruned_acc-acc_ztp:+.2f}%)')

        results.append({
            'keep_ratio':   keep,
            'k':            k,
            'acc_unpruned': unpruned_acc,
            'acc_mean':     acc_mean,
            'acc_live_cls': acc_live,
            'acc_ztp':      acc_ztp,
            'gain_vs_mean': gain,
        })

    print(f'\n{"="*70}')
    print(f'SUMMARY  K={num_classes}')
    print(f'  {"keep":>5}  {"mean":>8}  {"live-cls":>9}  {"gain":>6}  {"zero-tpr":>9}')
    print('-' * 52)
    for r in results:
        print(f'  {r["keep_ratio"]:>5.1f}  {r["acc_mean"]:>7.2f}%  '
              f'{r["acc_live_cls"]:>8.2f}%  {r["gain_vs_mean"]:>+5.2f}%  '
              f'{r["acc_ztp"]:>8.2f}%')

    out = {
        'method':      'CGTS-LiveCLS',
        'num_classes':  num_classes,
        'layer_idx':    args.layer_idx,
        'num_batches':  args.num_batches,
        'unpruned_acc': unpruned_acc,
        'results':      results,
    }
    save_path = os.path.join(args.output_dir, f'e10_cgts_live_cls_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--layer-idx',   type=int,  default=6)
    parser.add_argument('--num-batches', type=int,  default=50)
    parser.add_argument('--device',      type=int,  default=0)
    parser.add_argument('--subset-file', type=str,  default='configs/class_subsets.json')
    parser.add_argument('--output-dir',  type=str,  default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')

    model = load_model(device)

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, model,
                  full_train_loader, full_test_loader)


if __name__ == '__main__':
    main()
