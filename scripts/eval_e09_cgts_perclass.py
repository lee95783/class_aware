#!/usr/bin/env python3
"""
E9: Per-Class Prototype CGTS (Option 1 of improved prototype quality)

Problem: single-prototype CGTS averages CLS features across all K classes.
At large K with visually heterogeneous classes the average prototype is noisy,
and token selection degrades.

Solution: compute one prototype per class.  At inference, score each patch
token against ALL K class prototypes and keep the max:

    score(token_j) = max_c  <prototype_c, token_j>

This selects tokens that are discriminative for ANY class in the deployment
subset — exactly what we want.  Storage: K × D bytes (K=50, D=192: 37.5 KB).
Inference: one matrix multiply [B, N, D] × [D, K] → [B, N, K], max over K.
Overhead vs single-prototype CGTS: one extra matmul (negligible vs blocks).

Compares:
  unpruned          — baseline
  cgts_single       — current method (mean of all K-class CLS features)
  cgts_perclass     — one prototype per class, max scoring  [this work]
  zero_tprune       — WPR-based zero-shot (from E7, for reference)

Usage:
    python scripts/eval_e09_cgts_perclass.py --num-classes 5 10 20 50 --device 0
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
N_PATCHES   = 196   # 14×14 for 224×224 images


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


# ── Prototype computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_single_prototype(model, loader, device, layer_idx, num_batches=50):
    """Mean CLS feature at block[layer_idx] over all images in loader."""
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
    return torch.cat(all_cls).mean(0)  # [D]


@torch.no_grad()
def compute_per_class_prototypes(model, loader, device, layer_idx,
                                  class_indices, num_batches=50):
    """
    Per-class mean CLS feature at block[layer_idx].
    Returns a matrix [K, D] where row c is the prototype for class_indices[c].
    """
    captured = {}
    handle = model.blocks[layer_idx].norm1.register_forward_hook(
        lambda mod, inp, out: captured.update({'f': inp[0].detach()})
    )

    # Accumulate CLS features per class
    cls_idx_to_pos = {c: i for i, c in enumerate(class_indices)}
    K = len(class_indices)
    D = model.embed_dim
    accum  = torch.zeros(K, D)
    counts = torch.zeros(K)

    for i, (imgs, lbls) in enumerate(loader):
        if i >= num_batches: break
        model(imgs.to(device))
        feats = captured['f'][:, 0].cpu()   # [B, D]
        for j, lbl in enumerate(lbls.tolist()):
            if lbl in cls_idx_to_pos:
                pos = cls_idx_to_pos[lbl]
                accum[pos]  += feats[j]
                counts[pos] += 1

    handle.remove()
    counts = counts.clamp(min=1)
    return accum / counts.unsqueeze(1)   # [K, D]


# ── Forward passes ────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_cgts_single(model, images, prototype, layer_idx, k, device):
    """Standard CGTS: one prototype, dot-product scoring, top-k keep."""
    B = images.size(0)
    p = prototype.to(device)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            patch   = x[:, 1:]
            scores  = (patch * p).sum(-1)                          # [B, N]
            top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x       = torch.cat([x[:, :1], patch[b_idx, top_idx]], dim=1)
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


@torch.no_grad()
def forward_cgts_perclass(model, images, proto_matrix, layer_idx, k, device):
    """
    Per-class CGTS: proto_matrix is [K, D].
    score(token_j) = max_c ( proto_matrix[c] · token_j )
    Single matmul [B, N, D] × [D, K] → [B, N, K], then max over K.
    """
    B = images.size(0)
    P = proto_matrix.to(device)   # [K, D]
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            patch   = x[:, 1:]                                      # [B, N, D]
            scores  = (patch @ P.T).max(dim=-1).values              # [B, N]
            top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x       = torch.cat([x[:, :1], patch[b_idx, top_idx]], dim=1)
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


@torch.no_grad()
def forward_zero_tprune(model, images, layer_idx, k, device, n_iter=3):
    """Zero-TPrune WPR scoring (imported logic, duplicated here for self-containment)."""
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
    print(f'E9 Per-Class CGTS  |  K={num_classes}  |  layer={args.layer_idx}  |  zero-shot')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    unpruned_acc = evaluate(model, test_loader, device, target_classes)
    print(f'Unpruned: {unpruned_acc:.2f}%\n')

    print('Computing prototypes...')
    single_proto = compute_single_prototype(
        model, train_loader, device, args.layer_idx, args.num_batches)
    perclass_proto = compute_per_class_prototypes(
        model, train_loader, device, args.layer_idx, target_classes, args.num_batches)
    print(f'  Single prototype:    shape={list(single_proto.shape)}')
    print(f'  Per-class prototype: shape={list(perclass_proto.shape)}  '
          f'({perclass_proto.numel()*4/1024:.1f} KB)\n')

    results = []
    for keep in keep_ratios:
        k = max(1, int(N_PATCHES * keep))

        acc_single = evaluate_fn(
            lambda imgs, _k=k: forward_cgts_single(
                model, imgs, single_proto, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        acc_perclass = evaluate_fn(
            lambda imgs, _k=k: forward_cgts_perclass(
                model, imgs, perclass_proto, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        acc_ztp = evaluate_fn(
            lambda imgs, _k=k: forward_zero_tprune(
                model, imgs, args.layer_idx, _k, device),
            test_loader, device, target_classes)

        gain = acc_perclass - acc_single
        print(f'  keep={keep}  k={k}:')
        print(f'    single    {acc_single:>7.2f}%  (drop={unpruned_acc-acc_single:+.2f}%)')
        print(f'    per-class {acc_perclass:>7.2f}%  (drop={unpruned_acc-acc_perclass:+.2f}%)  '
              f'gain vs single: {gain:+.2f}%')
        print(f'    zero-tpru {acc_ztp:>7.2f}%  (drop={unpruned_acc-acc_ztp:+.2f}%)')

        results.append({
            'keep_ratio':   keep,
            'k':            k,
            'acc_unpruned': unpruned_acc,
            'acc_single':   acc_single,
            'acc_perclass': acc_perclass,
            'acc_ztp':      acc_ztp,
            'gain_vs_single': acc_perclass - acc_single,
        })

    print(f'\n{"="*70}')
    print(f'SUMMARY  K={num_classes}')
    print(f'  {"keep":>5}  {"single":>8}  {"per-class":>10}  {"gain":>6}  {"zero-tpr":>9}')
    print('-' * 50)
    for r in results:
        print(f'  {r["keep_ratio"]:>5.1f}  {r["acc_single"]:>7.2f}%  '
              f'{r["acc_perclass"]:>9.2f}%  {r["gain_vs_single"]:>+5.2f}%  '
              f'{r["acc_ztp"]:>8.2f}%')

    out = {
        'method':     'CGTS-PerClass',
        'num_classes': num_classes,
        'layer_idx':   args.layer_idx,
        'num_batches': args.num_batches,
        'unpruned_acc': unpruned_acc,
        'proto_storage_kb': round(perclass_proto.numel() * 4 / 1024, 2),
        'results':     results,
    }
    save_path = os.path.join(args.output_dir, f'e09_cgts_perclass_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--layer-idx',   type=int,  default=6)
    parser.add_argument('--num-batches', type=int,  default=50,
                        help='Training batches used for prototype computation')
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
        run_one_k(num_classes, args, device, model, full_train_loader, full_test_loader)


if __name__ == '__main__':
    main()
