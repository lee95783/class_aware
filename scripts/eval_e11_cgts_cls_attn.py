#!/usr/bin/env python3
"""
E11: CLS-Attention CGTS (image-adaptive attention-based scoring)

Instead of a dot product with a static prototype or raw CLS features,
score patches using the CLS→patch attention weight at the pruning layer:

    score(patch_j) = mean_heads( softmax(q_cls · k_j / sqrt(d)) )

This is the actual attention mass the CLS token puts on each patch —
the signal that determines how much each patch contributes to the
CLS output at that layer.

Compared to Zero-TPrune (WPR):
  - Zero-TPrune: full [B,H,N,N] matrix + 3 power iterations → O(N²) memory
  - CLS-Attn:    only CLS row [B,H,1,N] → O(B·H·N·d) memory (linear in N)
    No power iterations, no materialising the full attention matrix.

Compared to Live-CLS (Option A):
  - Live-CLS:  raw dot product x[:, 0] · x[:, j] (pre-norm, no scaling)
  - CLS-Attn:  q_cls · k_j / sqrt(d) after q/k projections + norms + softmax
    Uses the actual routing signal the network computes internally.

Compares:
  unpruned       — baseline
  cgts_mean      — offline mean prototype
  cgts_live_cls  — runtime CLS token dot product (E10)
  cgts_cls_attn  — CLS attention row (this work)
  zero_tprune    — WPR (reference)

Usage:
    python scripts/eval_e11_cgts_cls_attn.py --num-classes 5 10 20 50 --device 0
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
    """Standard CGTS: fixed offline prototype, pruning before the layer."""
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
    """Live-CLS CGTS (E10): raw CLS dot product before the layer."""
    B = images.size(0)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            cls_current = x[:, 0]
            patch       = x[:, 1:]
            scores      = (patch * cls_current.unsqueeze(1)).sum(-1)
            top_idx     = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx       = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x           = torch.cat([x[:, :1], patch[b_idx, top_idx]], dim=1)
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


@torch.no_grad()
def forward_cgts_cls_attn(model, images, layer_idx, k, device):
    """
    CLS-Attention CGTS (E11): score patches by the softmax attention the CLS
    token places on each patch at block[layer_idx].

        score(patch_j) = mean_heads( softmax(q_cls · k_j / sqrt(d_head)) )

    Implementation:
      - Hook blk.attn.qkv to capture Q, K projections.
      - Run the full block at layer_idx (attention + MLP) on all 197 tokens.
      - Extract only the CLS row of the attention matrix: [B, H, N] — no full
        N×N materialisation.
      - Average over heads → [B, N] scores.
      - Keep top-k patches; prune before subsequent blocks.

    Memory cost: O(B·H·N·d_head) vs O(B·H·N²) for Zero-TPrune.
    Compute cost: one extra [B,H,1,d]×[B,H,d,N] → [B,H,1,N] matmul per image
    (negligible vs the full block forward pass).
    """
    B = images.size(0)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            captured = {}
            handle = blk.attn.qkv.register_forward_hook(
                lambda mod, inp, out: captured.update({'qkv': out.detach()})
            )
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            handle.remove()
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

            # Extract Q, K from captured QKV — [B, N+1, 3*D]
            qkv_out = captured['qkv']
            N_tok = qkv_out.shape[1]
            qkv_r = qkv_out.reshape(
                B, N_tok, 3, blk.attn.num_heads, blk.attn.head_dim
            ).permute(2, 0, 3, 1, 4)          # [3, B, H, N+1, d_head]
            q_proj, k_proj = qkv_r[0], qkv_r[1]  # [B, H, N+1, d_head]
            q_proj = blk.attn.q_norm(q_proj)
            k_proj = blk.attn.k_norm(k_proj)

            # CLS→patch attention (only the CLS row, no full N×N matrix)
            q_cls     = q_proj[:, :, 0:1, :]   # [B, H, 1, d_head]
            k_patches = k_proj[:, :, 1:, :]    # [B, H, N, d_head]
            # [B, H, 1, N] → squeeze → [B, H, N]
            attn_cls = (q_cls @ k_patches.transpose(-2, -1) * blk.attn.scale
                        ).squeeze(2).softmax(dim=-1)
            scores   = attn_cls.mean(1)        # [B, N] — mean over heads

            top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x       = torch.cat([x[:, :1], x[:, 1:][b_idx, top_idx]], dim=1)
        else:
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
    print(f'E11 CLS-Attention CGTS  |  K={num_classes}  |  layer={args.layer_idx}  |  zero-shot')
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

        acc_cls_attn = evaluate_fn(
            lambda imgs, _k=k: forward_cgts_cls_attn(
                model, imgs, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        acc_ztp = evaluate_fn(
            lambda imgs, _k=k: forward_zero_tprune(
                model, imgs, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        gain_vs_mean = acc_cls_attn - acc_mean
        gain_vs_ztp  = acc_cls_attn - acc_ztp
        print(f'  keep={keep}  k={k}:')
        print(f'    mean       {acc_mean:>7.2f}%  (drop={unpruned_acc-acc_mean:+.2f}%)')
        print(f'    live-cls   {acc_live:>7.2f}%  (drop={unpruned_acc-acc_live:+.2f}%)')
        print(f'    cls-attn   {acc_cls_attn:>7.2f}%  (drop={unpruned_acc-acc_cls_attn:+.2f}%)  '
              f'gain vs mean: {gain_vs_mean:+.2f}%  gain vs ZTP: {gain_vs_ztp:+.2f}%')
        print(f'    zero-tpru  {acc_ztp:>7.2f}%  (drop={unpruned_acc-acc_ztp:+.2f}%)')

        results.append({
            'keep_ratio':      keep,
            'k':               k,
            'acc_unpruned':    unpruned_acc,
            'acc_mean':        acc_mean,
            'acc_live_cls':    acc_live,
            'acc_cls_attn':    acc_cls_attn,
            'acc_ztp':         acc_ztp,
            'gain_vs_mean':    gain_vs_mean,
            'gain_vs_ztp':     gain_vs_ztp,
        })

    print(f'\n{"="*70}')
    print(f'SUMMARY  K={num_classes}')
    print(f'  {"keep":>5}  {"mean":>8}  {"live":>8}  {"cls-attn":>9}  {"gain":>6}  {"vs ZTP":>7}  {"zero-tpr":>9}')
    print('-' * 68)
    for r in results:
        print(f'  {r["keep_ratio"]:>5.1f}  {r["acc_mean"]:>7.2f}%  {r["acc_live_cls"]:>7.2f}%  '
              f'{r["acc_cls_attn"]:>8.2f}%  {r["gain_vs_mean"]:>+5.2f}%  '
              f'{r["gain_vs_ztp"]:>+6.2f}%  {r["acc_ztp"]:>8.2f}%')

    out = {
        'method':      'CGTS-CLSAttn',
        'num_classes':  num_classes,
        'layer_idx':    args.layer_idx,
        'num_batches':  args.num_batches,
        'unpruned_acc': unpruned_acc,
        'results':      results,
    }
    save_path = os.path.join(args.output_dir, f'e11_cgts_cls_attn_{num_classes}cls.json')
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
