#!/usr/bin/env python3
"""
E7 baseline (TinyImageNet): X-Pruner — class-conditional attention head gating (ALM).

Same method as train_e07_xpruner.py, adapted for:
  - Model:   deit_small_patch16_224  (D=384, 6 heads, 22M params)
  - Dataset: TinyImageNet-200 (200 classes)
  - Checkpoint: weights/deit_small_patch16_224_tinyimagenet_best.pth

Usage:
    python scripts/train_tinyimagenet_e07_xpruner.py --device 0
"""

import os, sys, json, argparse, math
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as T
import torchvision.datasets as datasets
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.x_pruner import XPrunerDeiT, evaluate_hard_pruned

CHECKPOINT = 'weights/deit_small_patch16_224_tinyimagenet_best.pth'
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
        T.Resize(256),
        T.CenterCrop(224),
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

def load_xpruner(device, target_ratio, k=10.0):
    model = XPrunerDeiT(
        model_name='deit_small_patch16_224',
        num_classes=NUM_CLASSES,
        pretrained=False,
        k=k,
        enable_mlp_pruning=False,
        enable_token_pruning=False,
    ).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.backbone.load_state_dict(state, strict=False)
    _init_gates(model, target_ratio, k)
    return model


def _init_gates(model, target_ratio, k=10.0):
    """Initialize head gates to start at target_ratio (same logic as CIFAR-100 version)."""
    with torch.no_grad():
        for blk in model.backbone.blocks:
            theta = blk.attn.theta.item()
            if target_ratio >= 1.0:
                init_val = 5.0
            elif target_ratio <= 0.0:
                init_val = -5.0
            else:
                logit_r = math.log(target_ratio / (1.0 - target_ratio))
                sg = theta + logit_r / k
                sg = max(0.01, min(0.99, sg))
                init_val = math.log(sg / (1.0 - sg))
            blk.attn.gate.data.fill_(init_val)


@torch.no_grad()
def evaluate_unpruned(loader, device, target_classes):
    model = timm.create_model('deit_small_patch16_224', num_classes=NUM_CLASSES,
                               pretrained=False).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)
    del model
    return 100.0 * correct / total


# ── Training (ALM dual optimization) ─────────────────────────────────────────

def train(model, loader, device, epochs, lr, target_ratio, beta, lr_dual):
    gamma = torch.tensor(0.0, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader))

    model.train()
    for epoch in range(epochs):
        correct = total = ce_sum = keep_sum = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            logits, keep_all = model(imgs, y=lbls, use_labels=True)
            ce = F.cross_entropy(logits, lbls)
            keep_ratio = torch.stack([k.mean() for k in keep_all]).mean()
            diff = target_ratio - keep_ratio
            alm = beta * diff ** 2 + gamma.detach() * diff
            (ce + alm).backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                gamma.add_(lr_dual * diff.detach())

            ce_sum += ce.item()
            keep_sum += keep_ratio.item()
            correct += (logits.argmax(1) == lbls).sum().item()
            total += lbls.size(0)

        n = len(loader)
        print(f'    ep{epoch+1}: ce={ce_sum/n:.4f}'
              f'  keep={keep_sum/n:.3f}  gamma={gamma.item():.3f}'
              f'  train={100*correct/total:.1f}%', flush=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_subset_mode(model, loader, device, target_classes):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits, _ = model(imgs)
        preds = class_idx[logits[:, class_idx].argmax(1)]
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, full_train_loader, full_test_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)

    print(f'\n{"="*70}')
    print(f'E7 X-Pruner (TinyImageNet/DeiT-Small)  |  K={num_classes}')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)

    unpruned_acc = evaluate_unpruned(test_loader, device, target_classes)
    print(f'Unpruned baseline: {unpruned_acc:.2f}%\n')

    results = []
    for target_ratio in [1.0, 0.7, 0.5]:
        print(f'[keep={target_ratio}] Training...', flush=True)
        model = load_xpruner(device, target_ratio)
        train(model, train_loader, device, args.epochs, args.lr,
              target_ratio=target_ratio, beta=args.beta, lr_dual=args.lr_dual)

        model.prepare_subset_inference(target_classes)
        acc_soft = evaluate_subset_mode(model, test_loader, device, target_classes)
        drop_soft = unpruned_acc - acc_soft
        print(f'  keep={target_ratio} (soft): {acc_soft:.2f}%  (drop={drop_soft:+.2f}%)', flush=True)

        acc_hard, n_pruned, n_total = evaluate_hard_pruned(
            model, test_loader, device, target_classes,
            model_name='deit_small_patch16_224', num_classes_total=NUM_CLASSES)
        drop_hard = unpruned_acc - acc_hard
        actual_keep = (n_total - n_pruned) / n_total if n_total > 0 else 1.0
        print(f'  keep={target_ratio} (hard): {acc_hard:.2f}%  (drop={drop_hard:+.2f}%)'
              f'  pruned {n_pruned}/{n_total} heads (actual keep={actual_keep:.2f})', flush=True)

        results.append({
            'target_ratio': target_ratio,
            'accuracy_soft': acc_soft, 'drop_soft': drop_soft,
            'accuracy_hard': acc_hard, 'drop_hard': drop_hard,
            'heads_pruned': n_pruned, 'heads_total': n_total,
        })
        del model
        torch.cuda.empty_cache()

    print(f'\nSUMMARY  K={num_classes}  (unpruned={unpruned_acc:.2f}%):')
    print(f'  {"keep":>6}  {"soft acc":>9}  {"hard acc":>9}  {"heads pruned":>14}')
    for r in results:
        print(f'  {r["target_ratio"]:>6.1f}  {r["accuracy_soft"]:>8.2f}%  '
              f'{r["accuracy_hard"]:>8.2f}%  '
              f'{r["heads_pruned"]:>5}/{r["heads_total"]} heads')

    out = {
        'method': 'XPruner-ALM',
        'dataset': 'tinyimagenet',
        'model': 'deit_small_patch16_224',
        'num_classes': num_classes,
        'epochs': args.epochs,
        'beta': args.beta,
        'lr_dual': args.lr_dual,
        'unpruned_acc': unpruned_acc,
        'results': results,
    }
    save_path = os.path.join(args.output_dir, f'tinyimagenet_e07_xpruner_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--beta',        type=float, default=1.0)
    parser.add_argument('--lr-dual',     type=float, default=0.1)
    parser.add_argument('--device',      type=int,   default=0)
    parser.add_argument('--data-dir',    type=str,   default='data/tiny-imagenet-200')
    parser.add_argument('--subset-file', type=str,   default='configs/tinyimagenet_class_subsets.json')
    parser.add_argument('--output-dir',  type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f'Checkpoint not found: {CHECKPOINT}\n'
            f'Run finetune_deit_small_tinyimagenet.py first.')

    full_train_loader, full_test_loader = get_full_loaders(args.data_dir)

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, full_train_loader, full_test_loader)


if __name__ == '__main__':
    main()
