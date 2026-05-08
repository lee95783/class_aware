#!/usr/bin/env python3
"""
Comprehensive evaluation of MLP pruning methods.

Compares:
1. Baseline (no pruning)
2. Magnitude-based MLP pruning
3. Class-aware MLP pruning (X-Pruner)
4. Joint head+MLP pruning

Metrics:
- Accuracy
- Latency (B=1 and B=128)
- FLOPs reduction
- Parameter reduction
"""

import os
import sys
import time
import torch
import torch.nn as nn
import timm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.x_pruner import XPrunerDeiT
from src.dataset import get_dataloaders


def measure_latency(model, input_size, device, num_iters=50):
    """Measure model latency."""
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy) if not hasattr(model, 'backbone') else model(dummy, use_labels=False)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy) if not hasattr(model, 'backbone') else model(dummy, use_labels=False)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) / num_iters * 1000  # ms


def evaluate_accuracy(model, test_loader, device, is_xpruner=False):
    """Evaluate model accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if is_xpruner:
                # X-Pruner: two-pass inference
                # Pass 1: Bootstrap (get prediction without labels)
                logits_boot, _ = model(images, use_labels=False)
                preds_boot = logits_boot.argmax(dim=1)

                # Pass 2: Class-conditional (use predicted labels)
                logits, _ = model(images, y=preds_boot, use_labels=True)
            else:
                logits = model(images)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples


def prune_mlp_magnitude(model, prune_ratio=0.5):
    """
    Magnitude-based MLP pruning (baseline comparison).
    Returns a new model with structurally reduced MLPs.
    """
    device = next(model.parameters()).device

    for layer_idx, blk in enumerate(model.blocks):
        d_model = blk.mlp.fc1.in_features
        d_ff_orig = blk.mlp.fc1.out_features
        d_ff_new = int(d_ff_orig * (1 - prune_ratio))

        with torch.no_grad():
            # Compute neuron importance (L1 norm)
            fc1_importance = blk.mlp.fc1.weight.abs().sum(dim=1)
            keep_indices = torch.topk(fc1_importance, k=d_ff_new, largest=True).indices
            keep_indices = keep_indices.sort()[0]

            # Create new layers
            new_fc1 = nn.Linear(d_model, d_ff_new, bias=True).to(device)
            new_fc1.weight.data = blk.mlp.fc1.weight[keep_indices]
            new_fc1.bias.data = blk.mlp.fc1.bias[keep_indices]

            new_fc2 = nn.Linear(d_ff_new, d_model, bias=True).to(device)
            new_fc2.weight.data = blk.mlp.fc2.weight[:, keep_indices]
            new_fc2.bias.data = blk.mlp.fc2.bias

            # Replace
            blk.mlp.fc1 = new_fc1
            blk.mlp.fc2 = new_fc2

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model, input_size=(1, 3, 224, 224)):
    """Rough FLOPs estimation for ViT."""
    # For DeiT-Tiny: ~1.3 GFLOPs baseline
    # This is a simplified estimation
    d_model = 192
    d_ff = model.blocks[0].mlp.fc1.out_features
    num_heads = 3
    num_layers = len(model.blocks)
    num_tokens = 197  # 196 patches + 1 CLS

    # Attention FLOPs per layer
    qkv_flops = 3 * num_tokens * d_model * d_model
    attn_flops = 2 * num_heads * num_tokens * num_tokens * (d_model // num_heads)
    proj_flops = num_tokens * d_model * d_model
    attention_flops = qkv_flops + attn_flops + proj_flops

    # MLP FLOPs per layer
    mlp_flops = 2 * num_tokens * d_model * d_ff

    # Total
    total_flops = num_layers * (attention_flops + mlp_flops)

    return total_flops / 1e9  # GFLOPs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load test data
    print("Loading CIFAR-100 test data...")
    _, test_loader = get_dataloaders(
        data_dir='./data',
        dataset_name='cifar100',
        batch_size=128,
        image_size=224,
        num_workers=4,
        train=False,
        split='test',
    )

    results = []

    # ====================================================================
    # 1. Baseline (no pruning)
    # ====================================================================
    print("\n" + "="*70)
    print("1. Baseline (No Pruning)")
    print("="*70)

    baseline = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100).to(device)

    # Load fine-tuned weights if available
    baseline_ckpt = "best_deit_tiny_cifar100_final_timm.pth"
    if os.path.exists(baseline_ckpt):
        print(f"Loading fine-tuned checkpoint: {baseline_ckpt}")
        baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device))

    baseline.eval()

    # Metrics
    params_base = count_parameters(baseline)
    flops_base = estimate_flops(baseline)
    acc_base = evaluate_accuracy(baseline, test_loader, device, is_xpruner=False)
    lat_b1_base = measure_latency(baseline, (1, 3, 224, 224), device)
    lat_b128_base = measure_latency(baseline, (128, 3, 224, 224), device)

    print(f"Parameters: {params_base:,}")
    print(f"FLOPs: {flops_base:.2f} G")
    print(f"Accuracy: {acc_base*100:.2f}%")
    print(f"Latency (B=1): {lat_b1_base:.2f} ms")
    print(f"Latency (B=128): {lat_b128_base:.2f} ms")

    results.append({
        'method': 'Baseline',
        'params': params_base,
        'flops': flops_base,
        'accuracy': acc_base,
        'latency_b1': lat_b1_base,
        'latency_b128': lat_b128_base,
    })

    # ====================================================================
    # 2. Magnitude-based MLP pruning (50%)
    # ====================================================================
    print("\n" + "="*70)
    print("2. Magnitude-based MLP Pruning (50%)")
    print("="*70)

    mag_model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100).to(device)
    if os.path.exists(baseline_ckpt):
        mag_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))

    mag_model = prune_mlp_magnitude(mag_model, prune_ratio=0.5)
    mag_model.eval()

    params_mag = count_parameters(mag_model)
    flops_mag = estimate_flops(mag_model)
    acc_mag = evaluate_accuracy(mag_model, test_loader, device, is_xpruner=False)
    lat_b1_mag = measure_latency(mag_model, (1, 3, 224, 224), device)
    lat_b128_mag = measure_latency(mag_model, (128, 3, 224, 224), device)

    print(f"Parameters: {params_mag:,} ({(1-params_mag/params_base)*100:.1f}% reduction)")
    print(f"FLOPs: {flops_mag:.2f} G ({(1-flops_mag/flops_base)*100:.1f}% reduction)")
    print(f"Accuracy: {acc_mag*100:.2f}% ({(acc_mag-acc_base)*100:+.2f}%)")
    print(f"Latency (B=1): {lat_b1_mag:.2f} ms ({(lat_b1_mag/lat_b1_base-1)*100:+.1f}%)")
    print(f"Latency (B=128): {lat_b128_mag:.2f} ms ({(lat_b128_mag/lat_b128_base-1)*100:+.1f}%)")

    results.append({
        'method': 'Magnitude MLP (50%)',
        'params': params_mag,
        'flops': flops_mag,
        'accuracy': acc_mag,
        'latency_b1': lat_b1_mag,
        'latency_b128': lat_b128_mag,
    })

    # ====================================================================
    # 3. Class-aware MLP pruning (X-Pruner)
    # ====================================================================
    print("\n" + "="*70)
    print("3. Class-aware MLP Pruning (X-Pruner)")
    print("="*70)

    xpruner_ckpt = "results/xpruner_mlp/best_model.pth"
    if os.path.exists(xpruner_ckpt):
        print(f"Loading X-Pruner checkpoint: {xpruner_ckpt}")

        xpruner_model = XPrunerDeiT(
            model_name="deit_tiny_patch16_224",
            num_classes=100,
            pretrained=False,
            k=10.0,
            enable_mlp_pruning=True,
            mlp_k=10.0,
        ).to(device)

        checkpoint = torch.load(xpruner_ckpt, map_location=device)
        xpruner_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        xpruner_model.eval()

        # Get keep ratios from checkpoint
        eval_stats = checkpoint.get('eval_stats', {})
        head_ratio = eval_stats.get('head_ratio', 0.0)
        mlp_ratio = eval_stats.get('mlp_ratio', 0.0)

        print(f"Learned head keep ratio: {head_ratio:.3f}")
        print(f"Learned MLP keep ratio: {mlp_ratio:.3f}")

        # Evaluate
        acc_xpruner = evaluate_accuracy(xpruner_model, test_loader, device, is_xpruner=True)

        # Note: Latency measurement for X-Pruner includes gating overhead
        # For fair comparison, should export to compact model first
        print(f"Accuracy: {acc_xpruner*100:.2f}% ({(acc_xpruner-acc_base)*100:+.2f}%)")
        print("\nNote: For latency measurement, export to compact model using:")
        print("  python scripts/export_mlp_pruned.py --checkpoint results/xpruner_mlp/best_model.pth")

        results.append({
            'method': f'X-Pruner MLP ({mlp_ratio:.1%} keep)',
            'params': count_parameters(xpruner_model),
            'flops': '(with gating)',
            'accuracy': acc_xpruner,
            'latency_b1': '(export first)',
            'latency_b128': '(export first)',
        })
    else:
        print(f"X-Pruner checkpoint not found: {xpruner_ckpt}")
        print("Train first using:")
        print("  python scripts/train_xpruner_mlp.py --enable_mlp_pruning --epochs 20")

    # ====================================================================
    # Summary Table
    # ====================================================================
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    print(f"\n{'Method':<30} {'Accuracy':<12} {'Params':<15} {'Latency B=1':<15}")
    print("-"*70)
    for r in results:
        acc_str = f"{r['accuracy']*100:.2f}%"
        params_str = f"{r['params']:,}" if isinstance(r['params'], int) else str(r['params'])
        lat_str = f"{r['latency_b1']:.2f} ms" if isinstance(r['latency_b1'], float) else str(r['latency_b1'])

        print(f"{r['method']:<30} {acc_str:<12} {params_str:<15} {lat_str:<15}")

    print("\n" + "="*70)
    print("Key Findings:")
    print("="*70)
    print(f"1. Baseline accuracy: {acc_base*100:.2f}%")
    print(f"2. Magnitude pruning (50% MLP): {acc_mag*100:.2f}% ({(acc_mag-acc_base)*100:+.2f}%)")
    print(f"3. Magnitude pruning speedup (B=1): {(lat_b1_base/lat_b1_mag-1)*100:.1f}%")

    if len(results) > 2:
        print(f"4. Class-aware pruning: See exported model results")

    print("\nNext steps:")
    print("1. If X-Pruner trained: Export compact model and benchmark")
    print("2. Compare class-aware vs magnitude at same pruning ratio")
    print("3. Test on class subsets to validate class-conditional benefit")


if __name__ == "__main__":
    main()
