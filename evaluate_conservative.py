#!/usr/bin/env python3
"""
Comprehensive evaluation of conservative MLP pruning (25%).

Compares:
1. Baseline (no pruning)
2. Magnitude pruning (25%, no fine-tuning)
3. Class-aware MLP pruning (25%, X-Pruner trained)

Metrics:
- Accuracy (oracle mode with true labels)
- Latency at B=1 and B=128
- Parameter count
- Actual keep ratios
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
            if hasattr(model, 'backbone'):
                _ = model(dummy, use_labels=False)[0]
            else:
                _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            if hasattr(model, 'backbone'):
                _ = model(dummy, use_labels=False)[0]
            else:
                _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) / num_iters * 1000


def evaluate_accuracy(model, test_loader, device, is_xpruner=False):
    """Evaluate model accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if is_xpruner:
                # Oracle mode: use true labels for gates
                logits, _ = model(images, y=labels, use_labels=True)
            else:
                logits = model(images)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples


def prune_mlp_magnitude(model, prune_ratio=0.25):
    """Magnitude-based MLP pruning."""
    device = next(model.parameters()).device

    for layer_idx, blk in enumerate(model.blocks):
        d_model = blk.mlp.fc1.in_features
        d_ff_orig = blk.mlp.fc1.out_features
        d_ff_new = int(d_ff_orig * (1 - prune_ratio))

        with torch.no_grad():
            fc1_importance = blk.mlp.fc1.weight.abs().sum(dim=1)
            keep_indices = torch.topk(fc1_importance, k=d_ff_new, largest=True).indices
            keep_indices = keep_indices.sort()[0]

            new_fc1 = nn.Linear(d_model, d_ff_new, bias=True).to(device)
            new_fc1.weight.data = blk.mlp.fc1.weight[keep_indices]
            new_fc1.bias.data = blk.mlp.fc1.bias[keep_indices]

            new_fc2 = nn.Linear(d_ff_new, d_model, bias=True).to(device)
            new_fc2.weight.data = blk.mlp.fc2.weight[:, keep_indices]
            new_fc2.bias.data = blk.mlp.fc2.bias

            blk.mlp.fc1 = new_fc1
            blk.mlp.fc2 = new_fc2

    return model


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
    print(f"Test batches: {len(test_loader)}\n")

    results = []

    # =================================================================
    # 1. Baseline
    # =================================================================
    print("="*70)
    print("1. Baseline (No Pruning)")
    print("="*70)

    baseline = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100).to(device)
    baseline_ckpt = "best_deit_tiny_cifar100_final_timm.pth"

    if os.path.exists(baseline_ckpt):
        baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=False))

    baseline.eval()

    params_base = sum(p.numel() for p in baseline.parameters())
    acc_base = evaluate_accuracy(baseline, test_loader, device, is_xpruner=False)
    lat_b1_base = measure_latency(baseline, (1, 3, 224, 224), device)
    lat_b128_base = measure_latency(baseline, (128, 3, 224, 224), device)

    print(f"Parameters: {params_base:,}")
    print(f"Accuracy: {acc_base*100:.2f}%")
    print(f"Latency (B=1): {lat_b1_base:.2f} ms")
    print(f"Latency (B=128): {lat_b128_base:.2f} ms")

    results.append({
        'method': 'Baseline',
        'params': params_base,
        'accuracy': acc_base,
        'lat_b1': lat_b1_base,
        'lat_b128': lat_b128_base,
    })

    # =================================================================
    # 2. Magnitude Pruning (25%)
    # =================================================================
    print("\n" + "="*70)
    print("2. Magnitude-based MLP Pruning (25%)")
    print("="*70)

    mag_model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100).to(device)
    if os.path.exists(baseline_ckpt):
        mag_model.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=False))

    mag_model = prune_mlp_magnitude(mag_model, prune_ratio=0.25)
    mag_model.eval()

    params_mag = sum(p.numel() for p in mag_model.parameters())
    acc_mag = evaluate_accuracy(mag_model, test_loader, device, is_xpruner=False)
    lat_b1_mag = measure_latency(mag_model, (1, 3, 224, 224), device)
    lat_b128_mag = measure_latency(mag_model, (128, 3, 224, 224), device)

    print(f"Parameters: {params_mag:,} ({(1-params_mag/params_base)*100:.1f}% reduction)")
    print(f"Accuracy: {acc_mag*100:.2f}% ({(acc_mag-acc_base)*100:+.2f}%)")
    print(f"Latency (B=1): {lat_b1_mag:.2f} ms ({(lat_b1_mag/lat_b1_base-1)*100:+.1f}%)")
    print(f"Latency (B=128): {lat_b128_mag:.2f} ms ({(lat_b128_mag/lat_b128_base-1)*100:+.1f}%)")

    results.append({
        'method': 'Magnitude (25%, no FT)',
        'params': params_mag,
        'accuracy': acc_mag,
        'lat_b1': lat_b1_mag,
        'lat_b128': lat_b128_mag,
    })

    # =================================================================
    # 3. Class-aware MLP Pruning (X-Pruner)
    # =================================================================
    print("\n" + "="*70)
    print("3. Class-aware MLP Pruning (X-Pruner, 25%)")
    print("="*70)

    xpruner_ckpt = "results/xpruner_mlp_conservative/best_model.pth"

    if os.path.exists(xpruner_ckpt):
        xpruner_model = XPrunerDeiT(
            model_name="deit_tiny_patch16_224",
            num_classes=100,
            pretrained=False,
            k=10.0,
            enable_mlp_pruning=True,
            mlp_k=10.0,
        ).to(device)

        checkpoint = torch.load(xpruner_ckpt, map_location=device, weights_only=False)
        xpruner_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        xpruner_model.eval()

        mlp_ratio = checkpoint.get('mlp_ratio', 0.75)
        print(f"Trained MLP keep ratio: {mlp_ratio:.3f} (target: 0.75)")

        # Evaluate with gating (not exported yet)
        acc_xpruner = evaluate_accuracy(xpruner_model, test_loader, device, is_xpruner=True)
        params_xpruner = sum(p.numel() for p in xpruner_model.parameters())

        print(f"Parameters: {params_xpruner:,} (includes gates)")
        print(f"Accuracy: {acc_xpruner*100:.2f}% ({(acc_xpruner-acc_base)*100:+.2f}%)")
        print(f"\nNote: For latency, export to compact model first:")
        print(f"  python scripts/export_mlp_pruned.py \\")
        print(f"    --checkpoint {xpruner_ckpt} \\")
        print(f"    --output_dir results/exported_conservative \\")
        print(f"    --export_type compact")

        results.append({
            'method': f'X-Pruner (keep {mlp_ratio:.1%})',
            'params': params_xpruner,
            'accuracy': acc_xpruner,
            'lat_b1': 'export first',
            'lat_b128': 'export first',
        })
    else:
        print(f"X-Pruner checkpoint not found: {xpruner_ckpt}")
        print("Train first using: python train_mlp_conservative.py")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*70)
    print("CONSERVATIVE PRUNING EVALUATION SUMMARY")
    print("="*70)

    print(f"\n{'Method':<30} {'Accuracy':<15} {'Latency B=1':<15} {'vs Baseline'}")
    print("-"*80)

    for r in results:
        acc_str = f"{r['accuracy']*100:.2f}%"
        lat_str = f"{r['lat_b1']:.2f} ms" if isinstance(r['lat_b1'], float) else r['lat_b1']

        if isinstance(r['lat_b1'], float) and isinstance(results[0]['lat_b1'], float):
            vs_base = f"{(r['lat_b1']/results[0]['lat_b1']-1)*100:+.1f}%"
        else:
            vs_base = "-"

        print(f"{r['method']:<30} {acc_str:<15} {lat_str:<15} {vs_base}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print(f"1. Baseline: {acc_base*100:.2f}% accuracy, {lat_b1_base:.2f} ms @ B=1")
    print(f"2. Magnitude (25%): {acc_mag*100:.2f}% ({(acc_mag-acc_base)*100:+.2f}%), "
          f"{lat_b1_mag:.2f} ms ({(lat_b1_mag/lat_b1_base-1)*100:+.1f}%)")

    if len(results) > 2:
        print(f"3. Class-aware (25%): {results[2]['accuracy']*100:.2f}% "
              f"({(results[2]['accuracy']-acc_base)*100:+.2f}%)")
        print(f"   Advantage over magnitude: {(results[2]['accuracy']-acc_mag)*100:+.2f}%")

    print("\nConservative pruning (25%) results:")
    print(f"- Parameter reduction: {(1-params_mag/params_base)*100:.1f}%")
    print(f"- Magnitude accuracy drop: {(acc_mag-acc_base)*100:.2f}%")
    print(f"- Magnitude latency change: {(lat_b1_mag/lat_b1_base-1)*100:+.1f}% @ B=1")

    if len(results) > 2:
        print(f"- Class-aware accuracy advantage: {(results[2]['accuracy']-acc_mag)*100:+.2f}%")


if __name__ == "__main__":
    main()
