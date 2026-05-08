#!/usr/bin/env python3
"""
Train E1: Class-Aware vs Class-Agnostic Pruning Comparison

This script trains baseline pruning methods for E1 experiment:
- Random pruning
- Magnitude pruning (L1-norm)
- Gradient pruning
- Ours (Class-aware MLP pruning)

All methods use the same:
- 50 target classes (from configs/class_subsets.json)
- 46% keep ratio (54% pruning)
- 100 epochs
- Batch size 256

Usage:
    python scripts/train_e01_baselines.py \
        --method random \
        --device 0
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.x_pruner import XPrunerDeiT, XPrunerALMLoss
from src.dataset import get_dataloaders
from torch.utils.data import DataLoader, Subset


def filter_classes(data_loader: DataLoader, target_classes: List[int], batch_size: int) -> DataLoader:
    """Filter dataset to only include target classes."""
    dataset = data_loader.dataset
    target_set = set(target_classes)

    # Read labels directly from dataset attributes (no image decoding)
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        # Fallback: iterate (slow)
        labels = [dataset[idx][1] for idx in range(len(dataset))]

    indices = [idx for idx, label in enumerate(labels) if label in target_set]

    # Create filtered subset
    filtered_dataset = Subset(dataset, indices)

    # Create new dataloader
    filtered_loader = DataLoader(
        filtered_dataset,
        batch_size=batch_size,
        shuffle=isinstance(data_loader.sampler, torch.utils.data.RandomSampler),
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
    )

    print(f"  Filtered {len(indices):,} samples from {len(dataset):,} total")

    return filtered_loader


def load_class_subset(subset_file: str, num_classes: int) -> List[int]:
    """Load class indices from subset file."""
    with open(subset_file, 'r') as f:
        subsets = json.load(f)

    if str(num_classes) not in subsets['subsets']:
        raise ValueError(f"No subset for {num_classes} classes in {subset_file}")

    class_indices = subsets['subsets'][str(num_classes)]['class_indices']
    class_names = subsets['subsets'][str(num_classes)]['class_names']

    print(f"\nUsing {num_classes} classes from subset file:")
    print(f"  Classes: {', '.join(class_names[:5])}... ({len(class_names)} total)")
    print(f"  Indices: {class_indices[:10]}... ({len(class_indices)} total)\n")

    return class_indices


class RandomMaskPruner:
    """Random pruning baseline - randomly select neurons to keep."""

    def __init__(self, keep_ratio: float):
        self.keep_ratio = keep_ratio
        self.masks = {}

    def initialize_masks(self, model: nn.Module):
        """Initialize random masks for MLP layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name and 'fc1' in name:
                # Random mask for each MLP intermediate dimension
                hidden_dim = module.out_features
                num_keep = int(hidden_dim * self.keep_ratio)

                mask = torch.zeros(hidden_dim, device=module.weight.device)
                keep_indices = torch.randperm(hidden_dim)[:num_keep]
                mask[keep_indices] = 1.0

                self.masks[name] = mask
                print(f"  {name}: {num_keep}/{hidden_dim} neurons ({self.keep_ratio:.1%})")

    def apply_masks(self, model: nn.Module):
        """Apply masks to model outputs during forward pass."""
        # Register forward hooks to apply masks
        for name, module in model.named_modules():
            if name in self.masks:
                mask = self.masks[name]
                def hook(module, input, output, mask=mask):
                    return output * mask
                module.register_forward_hook(hook)


class MagnitudePruner:
    """Magnitude pruning baseline - keep neurons with highest L1-norm."""

    def __init__(self, keep_ratio: float):
        self.keep_ratio = keep_ratio
        self.masks = {}

    def compute_masks(self, model: nn.Module):
        """Compute masks based on L1-norm of weights."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name and 'fc1' in name:
                # Compute L1-norm for each output neuron
                weights = module.weight.data  # [out_features, in_features]
                l1_norms = weights.abs().sum(dim=1)  # [out_features]

                hidden_dim = module.out_features
                num_keep = int(hidden_dim * self.keep_ratio)

                # Keep neurons with highest L1-norm
                _, keep_indices = torch.topk(l1_norms, num_keep)

                mask = torch.zeros(hidden_dim, device=module.weight.device)
                mask[keep_indices] = 1.0

                self.masks[name] = mask
                print(f"  {name}: {num_keep}/{hidden_dim} neurons (L1-norm)")

    def apply_masks(self, model: nn.Module):
        """Apply masks to model outputs during forward pass."""
        for name, module in model.named_modules():
            if name in self.masks:
                mask = self.masks[name]
                def hook(module, input, output, mask=mask):
                    return output * mask
                module.register_forward_hook(hook)


class GradientPruner:
    """Gradient pruning baseline - keep neurons with highest gradient magnitude."""

    def __init__(self, keep_ratio: float):
        self.keep_ratio = keep_ratio
        self.masks = {}
        self.gradient_accum = {}

    def accumulate_gradients(self, model: nn.Module, num_batches: int = 100):
        """Accumulate gradients over multiple batches to estimate importance."""
        print(f"  Accumulating gradients over {num_batches} batches...")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name and 'fc1' in name:
                hidden_dim = module.out_features
                self.gradient_accum[name] = torch.zeros(hidden_dim, device=module.weight.device)

        # Hook to accumulate gradients
        def grad_hook(name):
            def hook(grad):
                # grad shape: [out_features, in_features]
                grad_norms = grad.abs().sum(dim=1)  # [out_features]
                self.gradient_accum[name] += grad_norms
            return hook

        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if name in self.gradient_accum:
                handle = module.weight.register_hook(grad_hook(name))
                handles.append(handle)

        return handles

    def compute_masks(self, model: nn.Module):
        """Compute masks based on accumulated gradients."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name and 'fc1' in name:
                if name not in self.gradient_accum:
                    continue

                grad_importance = self.gradient_accum[name]

                hidden_dim = module.out_features
                num_keep = int(hidden_dim * self.keep_ratio)

                # Keep neurons with highest gradient magnitude
                _, keep_indices = torch.topk(grad_importance, num_keep)

                mask = torch.zeros(hidden_dim, device=module.weight.device)
                mask[keep_indices] = 1.0

                self.masks[name] = mask
                print(f"  {name}: {num_keep}/{hidden_dim} neurons (gradient)")

    def apply_masks(self, model: nn.Module):
        """Apply masks to model outputs during forward pass."""
        for name, module in model.named_modules():
            if name in self.masks:
                mask = self.masks[name]
                def hook(module, input, output, mask=mask):
                    return output * mask
                module.register_forward_hook(hook)


class ActivationMagnitudePruner:
    """Subset-aware pruning - keep neurons with highest mean activation on the target subset."""

    def __init__(self, keep_ratio: float):
        self.keep_ratio = keep_ratio
        self.masks = {}
        self.activation_accum = {}
        self.sample_counts = {}

    def accumulate_activations(self, model: nn.Module, data_loader, num_batches: int = 100):
        """Accumulate mean activation magnitudes over subset data."""
        print(f"  Accumulating activations over {num_batches} batches...")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name and 'fc1' in name:
                hidden_dim = module.out_features
                self.activation_accum[name] = torch.zeros(hidden_dim, device=module.weight.device)
                self.sample_counts[name] = 0

        def act_hook(name):
            def hook(module, input, output):
                # output: [B, hidden_dim] (after fc1, before activation)
                # output: [B, N_tokens, hidden_dim] — average over batch and tokens
                self.activation_accum[name] += output.detach().abs().mean(dim=(0, 1))
                self.sample_counts[name] += 1
            return hook

        handles = []
        for name, module in model.named_modules():
            if name in self.activation_accum:
                handles.append(module.register_forward_hook(act_hook(name)))

        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                images = images.to(next(model.parameters()).device)
                model(images, y=None, use_labels=False)
                if (batch_idx + 1) % 20 == 0:
                    print(f"    Batch {batch_idx+1}/{num_batches}")

        for handle in handles:
            handle.remove()

        # Normalize by number of batches
        for name in self.activation_accum:
            if self.sample_counts[name] > 0:
                self.activation_accum[name] /= self.sample_counts[name]

    def compute_masks(self, model: nn.Module):
        """Compute masks based on mean activation magnitude on the subset."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name and 'fc1' in name:
                if name not in self.activation_accum:
                    continue

                importance = self.activation_accum[name]
                hidden_dim = module.out_features
                num_keep = int(hidden_dim * self.keep_ratio)

                _, keep_indices = torch.topk(importance, num_keep)

                mask = torch.zeros(hidden_dim, device=module.weight.device)
                mask[keep_indices] = 1.0

                self.masks[name] = mask
                print(f"  {name}: {num_keep}/{hidden_dim} neurons (activation magnitude)")

    def apply_masks(self, model: nn.Module):
        """Apply masks to model outputs during forward pass."""
        for name, module in model.named_modules():
            if name in self.masks:
                mask = self.masks[name]
                def hook(module, input, output, mask=mask):
                    return output * mask
                module.register_forward_hook(hook)


def train_baseline(
    method: str,
    keep_ratio: float,
    num_classes: int,
    class_subset_file: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: int,
    checkpoint_dir: str,
):
    """Train baseline pruning method."""

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print(f"E1 EXPERIMENT: {method.upper()} PRUNING")
    print("="*80)
    print(f"Method: {method}")
    print(f"Keep ratio: {keep_ratio:.1%} (prune {1-keep_ratio:.1%})")
    print(f"Target classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*80 + "\n")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load class subset
    target_classes = load_class_subset(class_subset_file, num_classes)

    # Load data
    print("Loading CIFAR-100...")
    train_loader, test_loader = get_dataloaders(
        data_dir='./data',
        dataset_name='cifar100',
        batch_size=batch_size,
        image_size=224,
        num_workers=4,
        train=True,
        split='test',
    )

    # Filter to target classes
    print(f"Filtering to {len(target_classes)} target classes...")
    train_loader = filter_classes(train_loader, target_classes, batch_size)
    test_loader = filter_classes(test_loader, target_classes, batch_size)

    # Create model
    print(f"Creating DeiT-Tiny model...")
    model = XPrunerDeiT(
        model_name="deit_tiny_patch16_224",
        num_classes=100,  # Full 100 classes
        pretrained=False,
        k=10.0,
        enable_mlp_pruning=False,  # Disable X-Pruner gating for baselines
        mlp_k=10.0,
    ).to(device)

    # Load pretrained weights
    baseline_ckpt = "weights/deit_tiny_patch16_224_cifar100_finetuned_best.pth"
    if os.path.exists(baseline_ckpt):
        print(f"Loading pretrained weights: {baseline_ckpt}")
        checkpoint = torch.load(baseline_ckpt, map_location=device, weights_only=False)
        model.backbone.load_state_dict(checkpoint, strict=False)
        print("✓ Loaded pretrained CIFAR-100 weights\n")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Initialize pruner
    print(f"Initializing {method} pruner...")
    if method == "random":
        pruner = RandomMaskPruner(keep_ratio)
        pruner.initialize_masks(model)
        pruner.apply_masks(model)

    elif method == "magnitude":
        pruner = MagnitudePruner(keep_ratio)
        pruner.compute_masks(model)
        pruner.apply_masks(model)

    elif method == "gradient":
        pruner = GradientPruner(keep_ratio)
        # Accumulate gradients first
        handles = pruner.accumulate_gradients(model, num_batches=100)

        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 100:
                break

            images, labels = images.to(device), labels.to(device)

            # Forward
            logits, _ = model(images, y=labels, use_labels=False)
            loss = F.cross_entropy(logits, labels)

            # Backward
            loss.backward()

            if (batch_idx + 1) % 20 == 0:
                print(f"    Batch {batch_idx+1}/100: Loss={loss.item():.4f}")

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Compute masks
        pruner.compute_masks(model)
        pruner.apply_masks(model)

        # Reset model
        model.zero_grad()

    elif method == "activation":
        pruner = ActivationMagnitudePruner(keep_ratio)
        pruner.accumulate_activations(model, train_loader, num_batches=100)
        pruner.compute_masks(model)
        pruner.apply_masks(model)

    elif method == "ours":
        # Class-aware method uses X-Pruner with MLP gating
        print("Using class-aware MLP pruning with gating...")
        # Recreate model with MLP gating enabled
        model = XPrunerDeiT(
            model_name="deit_tiny_patch16_224",
            num_classes=100,
            pretrained=False,
            k=10.0,
            enable_mlp_pruning=True,
            mlp_k=10.0,
        ).to(device)

        # Load pretrained weights
        if os.path.exists(baseline_ckpt):
            checkpoint = torch.load(baseline_ckpt, map_location=device, weights_only=False)
            model.backbone.load_state_dict(checkpoint, strict=False)

        pruner = None  # No separate pruner needed

    else:
        raise ValueError(f"Unknown method: {method}")

    print("✓ Pruner initialized\n")

    # Loss function for ours
    if method == "ours":
        alm_loss_fn = XPrunerALMLoss(target_ratio=keep_ratio, lambda_sp=1.0, mode='joint').to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    if method == "ours":
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(alm_loss_fn.parameters()),
            lr=lr, weight_decay=0.05
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print("="*80)
    print("TRAINING")
    print("="*80)

    best_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward
            if method == "ours":
                logits, keep_all = model(images, y=labels, use_labels=True)

                # MLP keep ratios (alternating: head, mlp, head, mlp, ...)
                mlp_keeps = keep_all[1::2]  # Odd indices

                loss, loss_info = alm_loss_fn(logits, labels, mlp_keeps)
                mlp_ratio = torch.tensor(loss_info["keep_ratio"])
            else:
                logits, _ = model(images, y=labels, use_labels=False)
                loss = F.cross_entropy(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                if method == "ours":
                    print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}]: "
                          f"Loss={loss.item():.4f}, MLP Ratio={mlp_ratio.item():.3f}")
                else:
                    print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}]: "
                          f"Loss={loss.item():.4f}")

        # Step scheduler
        scheduler.step()

        # Evaluate
        model.eval()
        test_correct = 0
        test_samples = 0
        test_mlp_ratios = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward
                if method == "ours":
                    logits, keep_all = model(images, y=None, use_labels=False)
                    mlp_keeps = keep_all[1::2]
                    test_mlp_ratios.append(torch.cat(mlp_keeps).mean().item())
                else:
                    logits, _ = model(images, y=labels, use_labels=False)

                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_samples += labels.size(0)

        train_acc = train_correct / train_samples
        test_acc = test_correct / test_samples

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Acc: {train_acc*100:.2f}%")
        print(f"  Test Acc:  {test_acc*100:.2f}%")
        if method == "ours":
            avg_mlp_ratio = sum(test_mlp_ratios) / len(test_mlp_ratios)
            print(f"  MLP Keep Ratio: {avg_mlp_ratio:.3f} (target: {keep_ratio:.3f})")
            print(f"  Ratio Gap: {avg_mlp_ratio - keep_ratio:+.3f}")
        print(f"{'='*80}\n")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'train_acc': train_acc,
                'method': method,
                'keep_ratio': keep_ratio,
                'num_classes': num_classes,
                'target_classes': target_classes,
            }
            if method == "ours":
                checkpoint['mlp_ratio'] = avg_mlp_ratio

            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (acc: {best_acc*100:.2f}%)\n")

    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Method: {method}")
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Keep ratio: {keep_ratio:.1%}")
    if method == "ours":
        print(f"Achieved MLP ratio: {avg_mlp_ratio:.3f}")
        print(f"Ratio gap: {avg_mlp_ratio - keep_ratio:+.3f}")
    print(f"Model saved to: {checkpoint_dir}/checkpoint_best.pth")
    print("="*80)

    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description='Train E1 baselines: Class-aware vs Class-agnostic'
    )
    parser.add_argument('--method', type=str, required=True,
                       choices=['random', 'magnitude', 'gradient', 'activation', 'ours'],
                       help='Pruning method to train')
    parser.add_argument('--keep-ratio', type=float, default=0.46,
                       help='Fraction of neurons to keep')
    parser.add_argument('--num-classes', type=int, default=50,
                       help='Number of target classes')
    parser.add_argument('--class-subset-file', type=str,
                       default='configs/class_subsets.json',
                       help='Path to class subset file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device ID')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Checkpoint directory (default: experiments/e01_class_aware_vs_agnostic/{method})')

    args = parser.parse_args()

    # Set default checkpoint dir
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"experiments/e01_class_aware_vs_agnostic/{args.method}"

    # Train
    best_acc = train_baseline(
        method=args.method,
        keep_ratio=args.keep_ratio,
        num_classes=args.num_classes,
        class_subset_file=args.class_subset_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\n✓ Final accuracy: {best_acc*100:.2f}%\n")


if __name__ == "__main__":
    main()
