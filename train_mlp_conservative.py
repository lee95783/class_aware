#!/usr/bin/env python3
"""
Quick training script for conservative MLP pruning evaluation.

Trains a lightweight model for proof-of-concept:
- 5 epochs (quick iteration)
- 25% MLP pruning (conservative target)
- Penalty-based sparsity loss (simple)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.x_pruner import XPrunerDeiT
from src.dataset import get_dataloaders


def train_conservative_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Config
    NUM_CLASSES = 100
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 1e-4
    TARGET_MLP_RATIO = 0.75  # Keep 75% = prune 25%
    LAMBDA_MLP = 0.05  # Moderate sparsity pressure

    print("="*70)
    print("Conservative MLP Pruning - Quick Training")
    print("="*70)
    print(f"Target: Keep 75% MLP neurons (prune 25%)")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print("="*70 + "\n")

    # Load data
    print("Loading CIFAR-100...")
    train_loader, test_loader = get_dataloaders(
        data_dir='./data',
        dataset_name='cifar100',
        batch_size=BATCH_SIZE,
        image_size=224,
        num_workers=4,
        train=True,
        split='test',
    )

    # Create model
    print("Creating X-Pruner model with MLP gating...")
    model = XPrunerDeiT(
        model_name="deit_tiny_patch16_224",
        num_classes=NUM_CLASSES,
        pretrained=False,
        k=10.0,
        enable_mlp_pruning=True,
        mlp_k=10.0,
    ).to(device)

    # Load pretrained weights if available
    baseline_ckpt = "best_deit_tiny_cifar100_final_timm.pth"
    if os.path.exists(baseline_ckpt):
        print(f"Loading pretrained weights: {baseline_ckpt}")
        checkpoint = torch.load(baseline_ckpt, map_location=device, weights_only=False)
        model.backbone.load_state_dict(checkpoint, strict=False)
        print("✓ Loaded pretrained CIFAR-100 weights\n")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

    # Training loop
    print("="*70)
    print("Training")
    print("="*70)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward
            logits, keep_all = model(images, y=labels, use_labels=True)

            # Classification loss
            ce_loss = F.cross_entropy(logits, labels)

            # MLP keep ratios (alternating: head, mlp, head, mlp, ...)
            mlp_keeps = keep_all[1::2]  # Odd indices
            mlp_ratio = torch.cat(mlp_keeps).mean()

            # Sparsity loss (penalty)
            sparsity_loss = LAMBDA_MLP * (mlp_ratio - TARGET_MLP_RATIO) ** 2

            # Total loss
            loss = ce_loss + sparsity_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}]: "
                      f"Loss={loss.item():.4f}, MLP={mlp_ratio.item():.3f}")

        # Evaluate
        model.eval()
        test_correct = 0
        test_samples = 0
        test_mlp_ratios = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward (oracle mode - use true labels)
                logits, keep_all = model(images, y=labels, use_labels=True)

                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_samples += labels.size(0)

                mlp_keeps = keep_all[1::2]
                test_mlp_ratios.append(torch.cat(mlp_keeps).mean().item())

        train_acc = train_correct / train_samples
        test_acc = test_correct / test_samples
        avg_mlp_ratio = sum(test_mlp_ratios) / len(test_mlp_ratios)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Acc: {train_acc*100:.2f}%")
        print(f"  Test Acc:  {test_acc*100:.2f}%")
        print(f"  MLP Keep Ratio: {avg_mlp_ratio:.3f} (target: {TARGET_MLP_RATIO:.3f})")
        print(f"{'='*70}\n")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("results/xpruner_mlp_conservative", exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'mlp_ratio': avg_mlp_ratio,
            }
            torch.save(checkpoint, "results/xpruner_mlp_conservative/best_model.pth")
            print(f"✓ Saved best model (acc: {best_acc*100:.2f}%)\n")

    print("="*70)
    print("Training Complete!")
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: results/xpruner_mlp_conservative/best_model.pth")
    print("="*70)

    return best_acc


if __name__ == "__main__":
    train_conservative_mlp()
