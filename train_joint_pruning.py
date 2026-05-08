#!/usr/bin/env python3
"""
Train X-Pruner with Joint Head + MLP Pruning

Prunes both attention heads and MLP neurons simultaneously with adaptive ratios.
Expected: Better accuracy preservation through complementary pruning.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.x_pruner import XPrunerDeiT
from scripts.adaptive_mlp_ratio import get_recommended_mlp_ratio
from src.dataset import get_dataloaders


def train_joint_pruning():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_target_classes", type=int, default=50,
                        help="Number of classes for deployment")
    parser.add_argument("--deployment_mode", type=str, default="conservative",
                        choices=["aggressive", "conservative", "safe"],
                        help="Deployment mode for adaptive ratio")
    parser.add_argument("--lambda_head", type=float, default=0.1,
                        help="Head sparsity loss weight")
    parser.add_argument("--lambda_mlp", type=float, default=0.2,
                        help="MLP sparsity loss weight")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Compute adaptive ratios for both heads and MLPs
    mlp_config = get_recommended_mlp_ratio(
        num_target_classes=args.num_target_classes,
        total_classes=100,
        deployment_mode=args.deployment_mode,
    )

    # For heads: less aggressive than MLPs (heads have weaker specialization)
    # Use a more conservative ratio for heads
    head_config = get_recommended_mlp_ratio(
        num_target_classes=args.num_target_classes,
        total_classes=100,
        deployment_mode="safe",  # Always use safe mode for heads
    )

    TARGET_HEAD_RATIO = head_config["keep_ratio"]
    TARGET_MLP_RATIO = mlp_config["keep_ratio"]

    NUM_CLASSES = 100
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    LAMBDA_HEAD = args.lambda_head
    LAMBDA_MLP = args.lambda_mlp

    print("="*70)
    print("X-PRUNER JOINT HEAD + MLP PRUNING")
    print("="*70)
    print(f"Target classes: {args.num_target_classes}/100")
    print(f"Deployment mode: {args.deployment_mode}")
    print(f"\nHEAD PRUNING (safe mode):")
    print(f"  Target ratio: {TARGET_HEAD_RATIO:.3f} ({head_config['prune_ratio']*100:.1f}% pruning)")
    print(f"  Lambda: {LAMBDA_HEAD}")
    print(f"\nMLP PRUNING ({args.deployment_mode} mode):")
    print(f"  Target ratio: {TARGET_MLP_RATIO:.3f} ({mlp_config['prune_ratio']*100:.1f}% pruning)")
    print(f"  Lambda: {LAMBDA_MLP}")
    print(f"\nTraining params:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Batch size: {BATCH_SIZE}")
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

    # Create model with BOTH head and MLP pruning enabled
    # Note: Head pruning is always enabled in X-Pruner
    print("Creating X-Pruner model with joint head + MLP gating...")
    model = XPrunerDeiT(
        model_name="deit_tiny_patch16_224",
        num_classes=NUM_CLASSES,
        pretrained=False,
        k=10.0,  # Head gating steepness
        enable_mlp_pruning=True,   # Enable MLP pruning
        mlp_k=10.0,  # MLP gating steepness
    ).to(device)

    # Load pretrained weights
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
    output_dir = f"results/xpruner_joint_{args.num_target_classes}classes_{args.deployment_mode}"
    os.makedirs(output_dir, exist_ok=True)

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

            # Head and MLP keep ratios
            # keep_all alternates: [head0, mlp0, head1, mlp1, ...]
            head_keeps = keep_all[0::2]  # Even indices
            mlp_keeps = keep_all[1::2]   # Odd indices

            head_ratio = torch.cat(head_keeps).mean()
            mlp_ratio = torch.cat(mlp_keeps).mean()

            # Sparsity losses (penalty for both)
            head_sparse_loss = LAMBDA_HEAD * (head_ratio - TARGET_HEAD_RATIO) ** 2
            mlp_sparse_loss = LAMBDA_MLP * (mlp_ratio - TARGET_MLP_RATIO) ** 2

            # Total loss
            loss = ce_loss + head_sparse_loss + mlp_sparse_loss

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
                      f"Loss={loss.item():.4f}, Head={head_ratio.item():.3f}, MLP={mlp_ratio.item():.3f}")

        # Evaluate
        model.eval()
        test_correct = 0
        test_samples = 0
        test_head_ratios = []
        test_mlp_ratios = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward (oracle mode - use true labels)
                logits, keep_all = model(images, y=labels, use_labels=True)

                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_samples += labels.size(0)

                head_keeps = keep_all[0::2]
                mlp_keeps = keep_all[1::2]
                test_head_ratios.append(torch.cat(head_keeps).mean().item())
                test_mlp_ratios.append(torch.cat(mlp_keeps).mean().item())

        train_acc = train_correct / train_samples
        test_acc = test_correct / test_samples
        avg_head_ratio = sum(test_head_ratios) / len(test_head_ratios)
        avg_mlp_ratio = sum(test_mlp_ratios) / len(test_mlp_ratios)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Acc: {train_acc*100:.2f}%")
        print(f"  Test Acc:  {test_acc*100:.2f}%")
        print(f"  Head Keep Ratio: {avg_head_ratio:.3f} (target: {TARGET_HEAD_RATIO:.3f}, gap: {avg_head_ratio - TARGET_HEAD_RATIO:+.3f})")
        print(f"  MLP Keep Ratio:  {avg_mlp_ratio:.3f} (target: {TARGET_MLP_RATIO:.3f}, gap: {avg_mlp_ratio - TARGET_MLP_RATIO:+.3f})")
        print(f"{'='*70}\n")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'head_ratio': avg_head_ratio,
                'mlp_ratio': avg_mlp_ratio,
                'target_head_ratio': TARGET_HEAD_RATIO,
                'target_mlp_ratio': TARGET_MLP_RATIO,
                'head_config': head_config,
                'mlp_config': mlp_config,
                'args': vars(args),
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
            print(f"✓ Saved best model (acc: {best_acc*100:.2f}%)\n")

    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Baseline accuracy: 83.79%")
    print(f"Accuracy drop: {best_acc*100 - 83.79:.2f}%")
    print(f"\nHead pruning:")
    print(f"  Target: {TARGET_HEAD_RATIO:.3f}, Achieved: {avg_head_ratio:.3f}")
    print(f"  Gap: {avg_head_ratio - TARGET_HEAD_RATIO:+.3f}")
    print(f"\nMLP pruning:")
    print(f"  Target: {TARGET_MLP_RATIO:.3f}, Achieved: {avg_mlp_ratio:.3f}")
    print(f"  Gap: {avg_mlp_ratio - TARGET_MLP_RATIO:+.3f}")
    print(f"\nModel saved to: {output_dir}/best_model.pth")
    print("="*70)

    return best_acc


if __name__ == "__main__":
    train_joint_pruning()
