#!/usr/bin/env python3
"""
E8: Backbone Sharing Validation

Demonstrates that one shared backbone serves N different K-class deployment
subsets via N lightweight prototype vectors — no per-subset model checkpoints.

For each K in {5, 10, 20, 50}, we run CGTS on 3 random seeds (from
class_subsets_multi.json). All seeds share the same backbone weights; only
the prototype vector (~1.5 KB per subset) differs.

Key metrics reported:
  - Unpruned accuracy per subset
  - CGTS accuracy at keep=0.7 and keep=0.5 (best layer from E3: layer 6)
  - Prototype size in bytes
  - Memory comparison: 1 backbone + N prototypes vs N DynamicViT checkpoints

Usage:
    python scripts/eval_e08_backbone_sharing.py --device 0
"""

import os, sys, json, argparse
import torch
import timm
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


def load_subsets(path):
    with open(path) as f:
        return json.load(f)['subsets']


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device):
    model = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    state = torch.load('weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth',
                       map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Prototype computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_prototype(model, loader, device, layer_idx, num_batches=50):
    """Mean CLS feature at block[layer_idx].norm1 input over loader."""
    captured = {}
    handle = model.blocks[layer_idx].norm1.register_forward_hook(
        lambda m, inp, out: captured.__setitem__('f', inp[0].detach()))
    all_cls = []
    for i, (imgs, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(imgs.to(device))
        all_cls.append(captured['f'][:, 0, :].cpu())
    handle.remove()
    return torch.cat(all_cls, dim=0).mean(dim=0)  # [D]


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_cgts(model, images, prototype, layer_idx, k, device):
    B = images.size(0)
    e_c = prototype.to(device)
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for l, blk in enumerate(model.blocks):
        if l == layer_idx:
            patch = x[:, 1:, :]
            scores = (patch * e_c).sum(dim=-1)
            top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            x = torch.cat([x[:, :1, :], patch[b_idx, top_idx]], dim=1)
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return model.head(model.norm(x)[:, 0])


@torch.no_grad()
def evaluate(model, loader, device, target_classes):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        correct += (class_idx[model(imgs)[:, class_idx].argmax(1)] == lbls).sum().item()
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',  type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--layer-idx',    type=int, default=6)
    parser.add_argument('--keep-ratios',  type=float, nargs='+', default=[0.7, 0.5])
    parser.add_argument('--device',       type=int, default=0)
    parser.add_argument('--num-batches',  type=int, default=50)
    parser.add_argument('--subset-file',  type=str, default='configs/class_subsets_multi.json')
    parser.add_argument('--output-dir',   type=str, default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading CIFAR-100...')
    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')

    print('Loading shared backbone (loaded ONCE for all subsets)...')
    model = load_model(device)
    n_patches = model.patch_embed.num_patches
    embed_dim = model.embed_dim

    # Rough model size
    model_params = sum(p.numel() * p.element_size() for p in model.parameters())
    proto_bytes = embed_dim * 4  # float32
    print(f'  Backbone size: {model_params / 1e6:.2f} MB')
    print(f'  Prototype size: {proto_bytes} bytes ({proto_bytes / 1024:.2f} KB)\n')

    all_subsets = load_subsets(args.subset_file)

    all_results = {}

    for num_classes in args.num_classes:
        subsets = all_subsets[str(num_classes)]
        n_seeds = len(subsets)

        print(f'\n{"="*70}')
        print(f'E8: Backbone Sharing  |  K={num_classes}  |  layer={args.layer_idx}  |  {n_seeds} seeds')
        print(f'{"="*70}')
        print(f'\nOne backbone ({model_params/1e6:.2f} MB) → {n_seeds} subsets × {proto_bytes} B prototype\n')

        seed_results = []

        for s_idx, subset in enumerate(subsets):
            seed = subset['seed']
            class_indices = subset['class_indices']
            class_names = subset['class_names']

            print(f'  Subset {s_idx} (seed={seed}): {", ".join(class_names[:4])}{"..." if num_classes > 4 else ""}')

            train_loader = filter_loader(full_train_loader, class_indices, shuffle=True)
            test_loader  = filter_loader(full_test_loader,  class_indices, shuffle=False)

            unpruned = evaluate(model, test_loader, device, class_indices)

            proto = compute_prototype(model, train_loader, device, args.layer_idx, args.num_batches)

            row = {'seed': seed, 'class_indices': class_indices, 'class_names': class_names,
                   'unpruned': unpruned, 'cgts': {}}

            cgts_strs = []
            for kr in args.keep_ratios:
                k = max(1, int(n_patches * kr))
                acc = evaluate_cgts(model, test_loader, device, class_indices,
                                    proto, args.layer_idx, k)
                row['cgts'][kr] = acc
                cgts_strs.append(f'keep={kr}: {acc:.2f}% ({acc-unpruned:+.2f}%)')

            print(f'    unpruned={unpruned:.2f}%  |  ' + '  |  '.join(cgts_strs))
            seed_results.append(row)

        # Summary table
        print(f'\n  {"seed":>6}  {"unpruned":>10}', end='')
        for kr in args.keep_ratios:
            print(f'  {"cgts@"+str(int(kr*100))+"%":>12}', end='')
        print()
        print('  ' + '-' * (10 + 12 * len(args.keep_ratios) + 8))

        for row in seed_results:
            print(f'  {row["seed"]:>6}  {row["unpruned"]:>9.2f}%', end='')
            for kr in args.keep_ratios:
                acc = row['cgts'][kr]
                print(f'  {acc:>9.2f}% ({acc-row["unpruned"]:>+5.2f}%)', end='')
            print()

        # Mean ± std across seeds
        print(f'\n  Mean ± Std:')
        for kr in args.keep_ratios:
            accs = [row['cgts'][kr] for row in seed_results]
            mean = sum(accs) / len(accs)
            std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
            print(f'    keep={kr}: {mean:.2f}% ± {std:.2f}%')

        # Memory comparison
        dynamicvit_mb = model_params / 1e6  # per-subset checkpoint ≈ same size as backbone
        cgts_kb = proto_bytes / 1024
        print(f'\n  Memory comparison ({n_seeds} subsets):')
        print(f'    CGTS:       1 backbone ({model_params/1e6:.1f} MB) + {n_seeds} × {cgts_kb:.1f} KB prototype'
              f' = {model_params/1e6 + n_seeds * proto_bytes/1e6:.2f} MB total')
        print(f'    DynamicViT: {n_seeds} × {dynamicvit_mb:.1f} MB checkpoints'
              f' = {n_seeds * dynamicvit_mb:.1f} MB total')

        all_results[num_classes] = {
            'layer_idx': args.layer_idx,
            'keep_ratios': args.keep_ratios,
            'n_patches': n_patches,
            'backbone_mb': round(model_params / 1e6, 3),
            'prototype_bytes': proto_bytes,
            'subsets': seed_results,
        }

    # Save
    save_path = os.path.join(args.output_dir, 'e08_backbone_sharing.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n\n✓ Saved: {save_path}')


if __name__ == '__main__':
    main()
