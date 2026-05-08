#!/usr/bin/env python3
"""
Fine-tune DeiT-Small (ImageNet pretrained) on TinyImageNet-200.

Produces the base checkpoint used by all TinyImageNet E7 experiments:
    weights/deit_small_patch16_224_tinyimagenet_best.pth

Usage:
    python scripts/finetune_deit_small_tinyimagenet.py --device 0
    python scripts/finetune_deit_small_tinyimagenet.py --device 0 --epochs 30 --batch-size 128
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

WEIGHTS_OUT = 'weights/deit_small_patch16_224_tinyimagenet_best.pth'


def get_loaders(data_dir, batch_size, num_workers=8):
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f'Train: {len(train_ds):,} images across {len(train_ds.classes)} classes')
    print(f'Val:   {len(val_ds):,} images')
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train(model, train_loader, val_loader, device, epochs, lr, weight_decay, warmup_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup then cosine decay
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    os.makedirs('weights', exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = loss_sum = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item()
            correct += (logits.detach().argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total
        val_acc   = evaluate(model, val_loader, device)
        lr_now    = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch:3d}/{epochs}  '
              f'loss={loss_sum/len(train_loader):.4f}  '
              f'train={train_acc:.2f}%  val={val_acc:.2f}%  '
              f'lr={lr_now:.2e}', flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f'  ✓ Saved best ({best_acc:.2f}%)')

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      type=str,   default='data/tiny-imagenet-200')
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--batch-size',    type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=5e-5)
    parser.add_argument('--weight-decay',  type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int,   default=3)
    parser.add_argument('--num-workers',   type=int,   default=8)
    parser.add_argument('--device',        type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load DeiT-Small with ImageNet pretrained weights, replace head for 200 classes
    print('Loading deit_small_patch16_224 (ImageNet pretrained)...')
    model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=200)
    model = model.to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    train_loader, val_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    print(f'\nFine-tuning for {args.epochs} epochs...')
    best_acc = train(model, train_loader, val_loader, device,
                     args.epochs, args.lr, args.weight_decay, args.warmup_epochs)

    print(f'\nBest val accuracy: {best_acc:.2f}%')
    print(f'Checkpoint saved to: {WEIGHTS_OUT}')


if __name__ == '__main__':
    main()
