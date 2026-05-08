#!/usr/bin/env python3
"""
E7 baseline: X-Pruner — class-conditional attention head pruning.

Uses the ALM (Augmented Lagrangian Method) loss as proposed in the XPruner
paper: L = CE + beta*(target - keep)^2 + gamma*(target - keep), where gamma
is a learnable Lagrange multiplier updated via gradient ascent (dual step).

Pipeline for each K:
  1. Load base weights into XPrunerDeiT backbone; init gates to target ratio
  2. Train with ALM dual optimization: primal Adam (minimize) + dual SGD (maximize gamma)
  3. Collapse gates to K-class subset via prepare_subset_inference()
  4. Evaluate with restricted argmax (K target classes only)

Usage:
    python scripts/train_e07_xpruner.py --num-classes 5 10 20 50 --device 0
"""

import os, sys, json, argparse, math
import torch
import torch.nn.functional as F
import timm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders
from scripts.x_pruner import XPrunerDeiT, evaluate_hard_pruned


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

def load_xpruner(device, target_ratio, k=10.0):
    model = XPrunerDeiT(
        model_name='deit_tiny_patch16_224',
        num_classes=100,
        pretrained=False,
        k=k,
        enable_mlp_pruning=False,
        enable_token_pruning=False,
    ).to(device)
    state = torch.load('weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth',
                       map_location=device, weights_only=False)
    model.backbone.load_state_dict(state, strict=False)
    _init_gates(model, target_ratio, k)
    return model


def _init_gates(model, target_ratio, k=10.0):
    """Initialize head gates so keep ratio starts at target_ratio.

    Solves: sigmoid(k*(sigmoid(gate) - theta)) = target_ratio
    => gate = logit(theta + logit(target_ratio)/k)
    """
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


# ── Training (ALM dual optimization) ─────────────────────────────────────────

def train(model, loader, device, epochs, lr, target_ratio, beta, lr_dual):
    """ALM dual optimization as in the XPruner paper.

    Primal step (minimize): Adam on all model parameters.
      L = CE + beta*(diff^2) + gamma*diff,  diff = target - keep_ratio

    Dual step (maximize gamma): SGD gradient ascent.
      gamma <- gamma + lr_dual * dL/d_gamma = gamma + lr_dual * diff
    """
    gamma = torch.tensor(0.0, device=device, requires_grad=True)

    primal_params = list(model.parameters())
    optimizer = torch.optim.AdamW(primal_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader))

    model.train()
    for epoch in range(epochs):
        correct = total = ce_sum = keep_sum = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            # ── Primal step ──
            optimizer.zero_grad()
            logits, keep_all = model(imgs, y=lbls, use_labels=True)
            ce = F.cross_entropy(logits, lbls)
            keep_ratio = torch.stack([k.mean() for k in keep_all]).mean()
            diff = target_ratio - keep_ratio
            alm = beta * diff ** 2 + gamma.detach() * diff
            (ce + alm).backward()
            optimizer.step()
            scheduler.step()

            # ── Dual step: gradient ascent on gamma ──
            # dL/d_gamma = diff  =>  gamma += lr_dual * diff
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
def evaluate(model, loader, device, target_classes):
    """Restricted evaluation: argmax over K target class logits."""
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits, _ = model(imgs, y=None, use_labels=False)
        preds = class_idx[logits[:, class_idx].argmax(1)]
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_subset_mode(model, loader, device, target_classes):
    """Restricted evaluation using subset-collapsed gates (deployment mode)."""
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits, _ = model(imgs)
        preds = class_idx[logits[:, class_idx].argmax(1)]
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
    return 100.0 * correct / total


# ── Unpruned baseline using plain timm model ──────────────────────────────────

@torch.no_grad()
def evaluate_unpruned(loader, device, target_classes):
    model = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    state = torch.load('weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth',
                       map_location=device, weights_only=False)
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


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, full_train_loader, full_test_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)

    print(f'\n{"="*70}')
    print(f'E7 X-Pruner  |  K={num_classes}  |  epochs={args.epochs}')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)

    unpruned_acc = evaluate_unpruned(test_loader, device, target_classes)
    print(f'Unpruned baseline: {unpruned_acc:.2f}%\n')

    sweep = [1.0, 0.7, 0.5]
    results = []

    for target_ratio in sweep:
        label = f'keep={target_ratio}'
        print(f'[{label}] Training...', flush=True)

        model = load_xpruner(device, target_ratio)
        train(model, train_loader, device, args.epochs, args.lr,
              target_ratio=target_ratio,
              beta=args.beta, lr_dual=args.lr_dual)

        # Collapse gates to the K-class subset for deployment-style evaluation
        model.prepare_subset_inference(target_classes)
        acc_soft = evaluate_subset_mode(model, test_loader, device, target_classes)
        drop_soft = unpruned_acc - acc_soft
        print(f'  keep={target_ratio} (soft): {acc_soft:.2f}%  (drop={drop_soft:+.2f}%)', flush=True)

        acc_hard, n_pruned, n_total = evaluate_hard_pruned(
            model, test_loader, device, target_classes,
            model_name='deit_tiny_patch16_224', num_classes_total=100)
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

    print(f'\n{"="*70}')
    print(f'SUMMARY  K={num_classes}  (unpruned={unpruned_acc:.2f}%):')
    print(f'  {"keep":>6}  {"soft acc":>9}  {"hard acc":>9}  {"heads pruned":>14}')
    print('-' * 50)
    for r in results:
        print(f'  {r["target_ratio"]:>6.1f}  {r["accuracy_soft"]:>8.2f}%  '
              f'{r["accuracy_hard"]:>8.2f}%  '
              f'{r["heads_pruned"]:>5}/{r["heads_total"]} heads')

    out = {
        'method': 'XPruner-ALM',
        'num_classes': num_classes,
        'epochs': args.epochs,
        'beta': args.beta,
        'lr_dual': args.lr_dual,
        'unpruned_acc': unpruned_acc,
        'results': results,
    }
    save_path = os.path.join(args.output_dir, f'e07_xpruner_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--beta',        type=float, default=1.0,
                        help='ALM quadratic penalty weight (beta in paper)')
    parser.add_argument('--lr-dual',     type=float, default=0.1,
                        help='Dual step size for Lagrange multiplier gamma')
    parser.add_argument('--device',      type=int,   default=0)
    parser.add_argument('--subset-file', type=str,   default='configs/class_subsets.json')
    parser.add_argument('--output-dir',  type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    full_train_loader, full_test_loader = get_dataloaders(
        data_dir='./data', dataset_name='cifar100',
        batch_size=128, image_size=224, num_workers=4, train=True, split='test')

    for num_classes in args.num_classes:
        run_one_k(num_classes, args, device, full_train_loader, full_test_loader)


if __name__ == '__main__':
    main()
