#!/usr/bin/env python3
"""
E7 baseline: DynamicViT — learned token predictor for class-aware deployment.

Adds a small token predictor MLP after `layer_idx`. During training, tokens
are soft-gated via sigmoid(scores) with a sparsity loss; at inference, hard
top-k selection physically drops low-scoring tokens.

Loss = cross-entropy + sparsity_weight * (mean_keep - target_keep)^2

Usage:
    python scripts/train_e07_dynamicvit.py --num-classes 10 --device 0
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders


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


# ── DynamicViT ────────────────────────────────────────────────────────────────

class DynamicViT(nn.Module):
    """DeiT-tiny with a learned token predictor inserted after `layer_idx`.

    Training: tokens are soft-weighted by sigmoid(score); a sparsity loss
    penalises deviation from the target keep ratio.
    Inference: hard top-k selection — physically drops low-scoring tokens.
    """

    def __init__(self, backbone, layer_idx=6, keep_ratio=0.7):
        super().__init__()
        self.backbone = backbone
        self.layer_idx = layer_idx
        self.keep_ratio = keep_ratio
        dim = backbone.embed_dim
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, images):
        bb = self.backbone
        B = images.size(0)

        x = bb.patch_embed(images)
        x = torch.cat([bb.cls_token.expand(B, -1, -1), x], dim=1) + bb.pos_embed
        x = bb.pos_drop(x)

        sparsity_loss = torch.zeros(1, device=images.device)

        for i, blk in enumerate(bb.blocks):
            x = blk(x)

            if i == self.layer_idx:
                patch = x[:, 1:, :]                          # [B, N, D]
                scores = self.predictor(patch).squeeze(-1)   # [B, N]
                N = patch.size(1)
                k = max(1, int(N * self.keep_ratio))

                if self.training:
                    keep_w = torch.sigmoid(scores)            # [B, N]
                    sparsity_loss = (keep_w.mean() - self.keep_ratio) ** 2
                    x = torch.cat([x[:, :1], patch * keep_w.unsqueeze(-1)], dim=1)
                else:
                    top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
                    b_idx = torch.arange(B, device=images.device).unsqueeze(1).expand(B, k)
                    x = torch.cat([x[:, :1], patch[b_idx, top_idx]], dim=1)

        x = bb.norm(x)
        logits = bb.head(x[:, 0])

        if self.training:
            return logits, sparsity_loss.squeeze()
        return logits


# ── Training / evaluation ─────────────────────────────────────────────────────

def fine_tune(model, train_loader, device, epochs, lr=1e-4, sparsity_weight=2.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader))
    model.train()
    for epoch in range(epochs):
        correct = total = ce_sum = sp_sum = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, sp = model(images)
            ce = F.cross_entropy(logits, labels)
            (ce + sparsity_weight * sp).backward()
            optimizer.step()
            scheduler.step()
            ce_sum += ce.item(); sp_sum += sp.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f'    ep{epoch+1}: ce={ce_sum/len(train_loader):.4f}'
              f'  sp={sp_sum/len(train_loader):.4f}'
              f'  train={100*correct/total:.1f}%', flush=True)
    model.eval()


@torch.no_grad()
def evaluate(model, loader, device, target_classes):
    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = class_idx[model(images)[:, class_idx].argmax(1)]
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ── Per-K run ─────────────────────────────────────────────────────────────────

def run_one_k(num_classes, args, device, full_train_loader, full_test_loader):
    target_classes, class_names = load_class_subset(args.subset_file, num_classes)

    print(f'\n{"="*70}')
    print(f'E7 DynamicViT  |  K={num_classes}  |  layer={args.layer_idx}  |  epochs={args.epochs}')
    print(f'Classes: {", ".join(class_names[:5])}{"..." if num_classes > 5 else ""}')
    print(f'{"="*70}\n')

    class_train_loader = filter_loader(full_train_loader, target_classes, batch_size=64, shuffle=True)
    class_test_loader  = filter_loader(full_test_loader,  target_classes, batch_size=128)

    base_model = load_base_model(device)
    unpruned_acc = evaluate(base_model, class_test_loader, device, target_classes)
    del base_model
    print(f'Unpruned baseline: {unpruned_acc:.2f}%\n')

    keep_ratios = [0.7, 0.5]
    results = []

    for keep_ratio in keep_ratios:
        print(f'Training DynamicViT  keep_ratio={keep_ratio}...')
        model = DynamicViT(load_base_model(device), layer_idx=args.layer_idx,
                           keep_ratio=keep_ratio).to(device)
        fine_tune(model, class_train_loader, device, args.epochs,
                  args.lr, args.sparsity_weight)
        acc = evaluate(model, class_test_loader, device, target_classes)
        drop = unpruned_acc - acc
        print(f'  keep={keep_ratio}: {acc:.2f}%  (drop={drop:+.2f}%)\n')
        results.append({'keep_ratio': keep_ratio, 'accuracy': acc, 'drop': drop})
        del model
        torch.cuda.empty_cache()

    print(f'{"="*70}')
    print(f'SUMMARY  K={num_classes}  (unpruned={unpruned_acc:.2f}%):')
    print(f'{"keep_ratio":>12}  {"accuracy":>10}  {"drop":>8}')
    print('-' * 35)
    for r in results:
        print(f'{r["keep_ratio"]:>12.1f}  {r["accuracy"]:>10.2f}%  {r["drop"]:>+8.2f}%')

    out = {
        'method': 'DynamicViT',
        'num_classes': num_classes,
        'layer_idx': args.layer_idx,
        'epochs': args.epochs,
        'sparsity_weight': args.sparsity_weight,
        'unpruned_acc': unpruned_acc,
        'results': results,
    }
    save_path = os.path.join(args.output_dir, f'e07_dynamicvit_{num_classes}cls.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes',     type=int,   nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--epochs',          type=int,   default=5)
    parser.add_argument('--layer-idx',       type=int,   default=6)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--sparsity-weight', type=float, default=2.0)
    parser.add_argument('--device',          type=int,   default=0)
    parser.add_argument('--subset-file',     type=str,   default='configs/class_subsets.json')
    parser.add_argument('--output-dir',      type=str,   default='results/paper')
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
