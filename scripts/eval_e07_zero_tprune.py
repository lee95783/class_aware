#!/usr/bin/env python3
"""
E7 baseline: Zero-TPrune — zero-shot token pruning via Weighted PageRank.

Zero-TPrune (Wang et al., CVPR 2024) scores patch tokens using a Weighted
PageRank (WPR) algorithm on the attention graph produced by a chosen block.
Tokens that are highly attended to by other important tokens receive high
scores; the bottom (1-keep_ratio) tokens are physically dropped after that
block. No training or class labels required.

Algorithm:
  1. Run blocks 0..layer_idx normally.
  2. At block layer_idx: capture the multi-head attention weights [B,H,N,N].
  3. Average over heads → A [B,N,N] (row-stochastic by softmax).
  4. WPR: r_{t+1} = A^T r_t / ||A^T r_t||_1, initialised to 1/N, n_iter steps.
  5. Keep top-k patch tokens by WPR score; CLS always kept.
  6. Run blocks layer_idx+1..11 on the shortened sequence.

Reference: Wang et al., "Zero-TPrune", CVPR 2024. arXiv:2305.17328.

Usage:
    python scripts/eval_e07_zero_tprune.py --num-classes 5 10 20 50 --device 0
"""

import os, sys, json, argparse
import torch
import timm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders


CHECKPOINT  = 'weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth'
MODEL_NAME  = 'deit_tiny_patch16_224'
NUM_CLASSES = 100


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


# ── WPR scoring ───────────────────────────────────────────────────────────────

def _wpr(attn_weights, n_iter):
    """
    Weighted PageRank on the attention graph.

    attn_weights: [B, H, N, N] — softmax attention from one block.
    Returns importance scores [B, N].

    A[b,i,j] = how much token i attends to token j.
    A is row-stochastic.  High-scoring tokens are those that many
    other important tokens attend to: r = A^T r (power iteration).
    """
    B, H, N, _ = attn_weights.shape
    A = attn_weights.mean(dim=1)                          # [B, N, N]
    r = torch.full((B, N), 1.0 / N, device=A.device)     # uniform init
    for _ in range(n_iter):
        r = torch.bmm(A.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)
        r = r / r.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return r                                              # [B, N]


# ── Forward pass ──────────────────────────────────────────────────────────────

def _attn_weights_from_qkv(attn_mod, qkv_out):
    """
    Compute [B, H, N, N] attention weight matrix from a captured QKV projection.

    timm may use fused_attn (F.scaled_dot_product_attention), which gives no
    intermediate attention matrix.  We recompute it here from the QKV values
    captured via a hook on attn_mod.qkv — one extra matmul, no wasted forward.
    """
    B, N, _ = qkv_out.shape
    qkv = qkv_out.reshape(B, N, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q = attn_mod.q_norm(q)
    k = attn_mod.k_norm(k)
    attn = (q * attn_mod.scale) @ k.transpose(-2, -1)   # [B, H, N, N]
    return attn.softmax(dim=-1)


@torch.no_grad()
def forward_zero_tprune(model, images, layer_idx, k, device, n_iter=3):
    """
    Run Zero-TPrune forward: prune k patch tokens after block[layer_idx]
    using WPR scores derived from that block's attention weights.
    """
    B = images.size(0)

    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            # Hook qkv to capture projected values; the actual forward may use
            # fused attention which produces no intermediate attention matrix.
            captured = {}
            handle = blk.attn.qkv.register_forward_hook(
                lambda mod, inp, out: captured.update({'qkv': out})
            )
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            handle.remove()
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

            attn_weights = _attn_weights_from_qkv(blk.attn, captured['qkv'])
            scores   = _wpr(attn_weights, n_iter)[:, 1:]   # [B, N_patches]
            top_idx  = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx    = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x        = torch.cat([x[:, :1], x[:, 1:][b_idx, top_idx]], dim=1)
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
def evaluate_zero_tprune(model, loader, device, target_classes, layer_idx, k, n_iter=3):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = forward_zero_tprune(model, imgs, layer_idx, k, device, n_iter)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, model, full_test_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)
    test_loader = filter_loader(full_test_loader, target_classes, batch_size=128)

    n_patches   = 196   # 14×14 patches for 224×224 images
    keep_ratios = [0.7, 0.5]
    layers      = [args.layer_idx]

    print(f'\n{"="*70}')
    print(f'Zero-TPrune  |  K={num_classes}  |  layer={args.layer_idx}  |  zero-shot')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    unpruned_acc = evaluate(model, test_loader, device, target_classes)
    print(f'Unpruned baseline: {unpruned_acc:.2f}%\n')

    results = []
    for layer_idx in layers:
        for keep_ratio in keep_ratios:
            k   = max(1, int(n_patches * keep_ratio))
            acc = evaluate_zero_tprune(model, test_loader, device, target_classes,
                                       layer_idx, k, n_iter=args.n_iter)
            drop = unpruned_acc - acc
            print(f'  layer={layer_idx}  keep={keep_ratio}  k={k}: {acc:.2f}%  (drop={drop:+.2f}%)',
                  flush=True)
            results.append({
                'layer_idx': layer_idx,
                'keep_ratio': keep_ratio,
                'k': k,
                'accuracy': acc,
                'drop': drop,
            })

    print(f'\n{"="*70}')
    print(f'SUMMARY  K={num_classes}  (unpruned={unpruned_acc:.2f}%):')
    print(f'  {"layer":>6}  {"keep":>6}  {"acc":>8}  {"drop":>8}')
    print('-' * 40)
    for r in results:
        print(f'  {r["layer_idx"]:>6}  {r["keep_ratio"]:>6.1f}  {r["accuracy"]:>7.2f}%  {r["drop"]:>+7.2f}%')

    out = {
        'method': 'Zero-TPrune',
        'num_classes': num_classes,
        'layer_idx': args.layer_idx,
        'n_iter': args.n_iter,
        'unpruned_acc': unpruned_acc,
        'results': results,
    }
    save_path = os.path.join(args.output_dir, f'e07_zero_tprune_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--layer-idx',  type=int,   default=6)
    parser.add_argument('--n-iter',     type=int,   default=3,
                        help='WPR power-iteration steps (3 is sufficient for convergence)')
    parser.add_argument('--device',     type=int,   default=0)
    parser.add_argument('--subset-file', type=str,  default='configs/class_subsets.json')
    parser.add_argument('--output-dir', type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    _, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=False, split='test')

    model = load_model(device)

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, model, full_test_loader)


if __name__ == '__main__':
    main()
