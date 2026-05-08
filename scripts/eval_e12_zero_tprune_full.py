#!/usr/bin/env python3
"""
E12: Complete Zero-TPrune — WPR r-stage + similarity s-stage, multi-layer.

The original Zero-TPrune (Wang et al., CVPR 2024) has two complementary
pruning stages applied at each pruning point:

  r-stage  (importance):  score each token by Weighted PageRank (WPR) on the
                           self-attention graph.  High score = many important
                           tokens attend to this token = keep.

  s-stage  (redundancy):  score each token by its maximum cosine similarity
                           to any other remaining token.  High score = this
                           token is a near-duplicate of another = remove.

Combined pruning score (higher = should prune):

    prune_i = (1 - r_i) + sim_weight * s_i

Setting sim_weight=0 recovers the r-stage-only version (our E7 baseline).
Setting sim_weight>0 penalises redundant tokens on top of unimportant ones.

Multi-layer scheduling:
  Given an overall keep ratio K_total and n pruning stages at layers
  [l_1, l_2, ..., l_n], distribute the budget geometrically:

    per_stage_keep = K_total ^ (1/n)

  so that the product of all per-stage keep ratios equals K_total.
  Pruning is applied after the full forward pass of each scheduled block.

Compares (all zero-shot, CIFAR-100, DeiT-Tiny):
  unpruned            — baseline
  ztp_partial         — r-stage only, single layer (our E7)
  ztp_full_single     — r+s stages, single layer
  ztp_full_multi      — r+s stages, 3-layer schedule [3, 6, 9]

Usage:
    python scripts/eval_e12_zero_tprune_full.py --num-classes 5 10 20 50 --device 0
    python scripts/eval_e12_zero_tprune_full.py --num-classes 10 --sim-weight 1.0 --n-iter 5
"""

import os, sys, json, argparse, math
import torch
import torch.nn.functional as F
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


# ── Scoring primitives ────────────────────────────────────────────────────────

def _max_cosine_similarity(features):
    """
    s-stage similarity scores.

    For each token, compute its maximum cosine similarity to any OTHER token.
    High score  →  this token is near-duplicate of another  →  candidate for removal.

    features : [B, N, D]  patch token features  (no CLS token)
    returns  : [B, N]     scores in (−1, 1]; self-similarity excluded
    """
    f_norm = F.normalize(features, dim=-1)                  # [B, N, D]
    sim    = torch.bmm(f_norm, f_norm.transpose(1, 2))      # [B, N, N]
    # Mask out diagonal (self-similarity = 1.0 always)
    B, N, _ = sim.shape
    diag = torch.eye(N, device=features.device, dtype=torch.bool).unsqueeze(0)
    sim  = sim.masked_fill(diag, -1.0)
    return sim.max(dim=-1).values                           # [B, N]


def _prune_tokens(x, r_scores, sim_weight, k, device):
    """
    Combined r+s pruning at one layer.

    x         : [B, N+1, D]   current sequence (CLS + N patches)
    r_scores  : [B, N]        WPR importance scores (high = keep)
    sim_weight: float          weight for s-stage penalty
    k         : int            number of patch tokens to keep
    returns   : [B, k+1, D]   pruned sequence
    """
    B = x.shape[0]
    patches = x[:, 1:]                              # [B, N, D]

    if sim_weight > 0.0:
        s_scores    = _max_cosine_similarity(patches)   # [B, N]
        prune_score = (1.0 - r_scores) + sim_weight * s_scores
    else:
        prune_score = (1.0 - r_scores)

    # Keep the k tokens with the LOWEST prune score
    top_idx = (-prune_score).topk(k, dim=1).indices.sort(dim=1).values
    b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
    return torch.cat([x[:, :1], patches[b_idx, top_idx]], dim=1)


# ── Forward passes ────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_ztp_partial(model, images, layer_idx, k, device, n_iter=3):
    """r-stage only, single layer (our E7 baseline)."""
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


@torch.no_grad()
def forward_ztp_full(model, images, prune_schedule, device, n_iter=5, sim_weight=0.5):
    """
    Complete Zero-TPrune: r-stage + s-stage at each scheduled pruning layer.

    prune_schedule : list of (layer_idx, per_layer_keep_ratio) tuples
                     applied in ascending layer_idx order.
    n_iter         : WPR power iterations (paper recommends 5+; 3 for speed).
    sim_weight     : weight for s-stage penalty (0 = r-stage only).
    """
    B      = images.size(0)
    device = images.device
    sched  = {l: r for l, r in prune_schedule}

    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    for l, blk in enumerate(model.blocks):
        if l in sched:
            # Run the full block first; hook QKV for WPR
            captured = {}
            handle = blk.attn.qkv.register_forward_hook(
                lambda mod, inp, out: captured.update({'qkv': out})
            )
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            handle.remove()
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

            # r-stage: WPR importance from this block's attention
            N_cur  = x.shape[1] - 1                            # current patch count
            k      = max(1, int(N_cur * sched[l]))
            attn_w = _attn_weights_from_qkv(blk.attn, captured['qkv'])
            r_scores = _wpr(attn_w, n_iter)[:, 1:]             # [B, N_cur]

            # s-stage: max cosine similarity from block output features
            x = _prune_tokens(x, r_scores, sim_weight, k, device)
        else:
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

    return model.head(model.norm(x)[:, 0])


def make_single_schedule(overall_keep, layer_idx=6):
    """Single pruning point."""
    return [(layer_idx, overall_keep)]


def make_multi_schedule(overall_keep, layers=(3, 6, 9)):
    """
    Geometric distribution of budget over n layers.
    per_stage_keep = overall_keep ^ (1/n)
    """
    n             = len(layers)
    per_keep      = overall_keep ** (1.0 / n)
    return [(l, per_keep) for l in layers]


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

def run_one_k(num_classes, args, device, model, full_test_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)
    test_loader = filter_loader(full_test_loader, target_classes, batch_size=64)
    keep_ratios = [0.7, 0.5]

    print(f'\n{"="*72}')
    print(f'E12 Full Zero-TPrune  |  K={num_classes}  |  sim_weight={args.sim_weight}'
          f'  |  n_iter={args.n_iter}')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*72}\n')

    unpruned_acc = evaluate(model, test_loader, device, target_classes)
    print(f'Unpruned: {unpruned_acc:.2f}%\n')

    results = []
    for keep in keep_ratios:
        k_single = max(1, int(N_PATCHES * keep))

        # 1. Partial (r-only, single layer) — E7 baseline
        acc_partial = evaluate_fn(
            lambda imgs, _k=k_single: forward_ztp_partial(
                model, imgs, args.layer_idx, _k, device, n_iter=args.n_iter),
            test_loader, device, target_classes)

        # 2. Full (r+s), single layer
        sched_single = make_single_schedule(keep, args.layer_idx)
        acc_full_single = evaluate_fn(
            lambda imgs, _s=sched_single: forward_ztp_full(
                model, imgs, _s, device, n_iter=args.n_iter,
                sim_weight=args.sim_weight),
            test_loader, device, target_classes)

        # 3. Full (r+s), multi-layer [3, 6, 9]
        sched_multi = make_multi_schedule(keep, layers=tuple(args.multi_layers))
        per_keep    = keep ** (1.0 / len(args.multi_layers))
        acc_full_multi = evaluate_fn(
            lambda imgs, _s=sched_multi: forward_ztp_full(
                model, imgs, _s, device, n_iter=args.n_iter,
                sim_weight=args.sim_weight),
            test_loader, device, target_classes)

        gain_s      = acc_full_single - acc_partial
        gain_multi  = acc_full_multi  - acc_partial

        # Effective token count for multi-layer
        n_eff = int(N_PATCHES * keep)

        print(f'  keep={keep}  (target k≈{n_eff}):')
        print(f'    partial   (r-only, layer {args.layer_idx}):  {acc_partial:>7.2f}%'
              f'  (drop={unpruned_acc-acc_partial:+.2f}%)')
        print(f'    full/1-layer (r+s, layer {args.layer_idx}): {acc_full_single:>7.2f}%'
              f'  (drop={unpruned_acc-acc_full_single:+.2f}%)  gain vs partial: {gain_s:+.2f}%')
        ml_str = '+'.join(map(str, args.multi_layers))
        print(f'    full/multi   (r+s, layers {ml_str}): {acc_full_multi:>7.2f}%'
              f'  (drop={unpruned_acc-acc_full_multi:+.2f}%)  gain vs partial: {gain_multi:+.2f}%'
              f'  [per-stage keep={per_keep:.3f}]')

        results.append({
            'keep_ratio':       keep,
            'k_effective':      n_eff,
            'acc_unpruned':     unpruned_acc,
            'acc_partial':      acc_partial,
            'acc_full_single':  acc_full_single,
            'acc_full_multi':   acc_full_multi,
            'gain_s':           gain_s,
            'gain_multi':       gain_multi,
            'multi_layers':     args.multi_layers,
            'per_stage_keep':   per_keep,
        })

    print(f'\n{"="*72}')
    print(f'SUMMARY  K={num_classes}  sim_weight={args.sim_weight}')
    print(f'  {"keep":>5}  {"partial":>9}  {"full/1L":>9}  {"Δ1L":>6}  {"full/ML":>9}  {"ΔML":>6}')
    print('-' * 57)
    for r in results:
        print(f'  {r["keep_ratio"]:>5.1f}  {r["acc_partial"]:>8.2f}%  '
              f'{r["acc_full_single"]:>8.2f}%  {r["gain_s"]:>+5.2f}%  '
              f'{r["acc_full_multi"]:>8.2f}%  {r["gain_multi"]:>+5.2f}%')

    out = {
        'method':       'ZeroTPrune-Full',
        'num_classes':  num_classes,
        'layer_idx':    args.layer_idx,
        'multi_layers': args.multi_layers,
        'n_iter':       args.n_iter,
        'sim_weight':   args.sim_weight,
        'unpruned_acc': unpruned_acc,
        'results':      results,
    }
    save_path = os.path.join(args.output_dir, f'e12_ztp_full_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',   type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--layer-idx',    type=int,  default=6,
                        help='Single-layer pruning point for partial/full-single comparisons')
    parser.add_argument('--multi-layers', type=int, nargs='+', default=[3, 6, 9],
                        help='Layer indices for multi-layer schedule')
    parser.add_argument('--n-iter',       type=int,  default=5,
                        help='WPR power iterations (paper recommends 5+)')
    parser.add_argument('--sim-weight',   type=float, default=0.5,
                        help='Weight for s-stage similarity penalty (0 = r-only)')
    parser.add_argument('--device',       type=int,  default=0)
    parser.add_argument('--subset-file',  type=str,  default='configs/class_subsets.json')
    parser.add_argument('--output-dir',   type=str,  default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    _, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')

    model = load_model(device)

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, model, full_test_loader)


if __name__ == '__main__':
    main()
