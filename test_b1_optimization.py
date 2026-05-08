#!/usr/bin/env python3
"""
Quick test of B=1 optimization - streamlined version for fast evaluation.
"""

import os
import sys
import torch
import time
import timm

sys.path.append(os.path.dirname(__file__))

from scripts.x_pruner import XPrunerDeiT
from scripts.x_pruner_optimized import convert_xpruner_to_optimized


def measure_latency(model, input_size, device, num_iters=50):
    """Quick latency measurement."""
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) / num_iters * 1000


def main():
    print("="*70)
    print("X-Pruner B=1 Optimization - Quick Test")
    print("="*70)

    MODEL_NAME = "deit_tiny_patch16_224"
    NUM_CLASSES = 100
    CHECKPOINT = "./results/xpruner_50_classes_lam5/xpruner_model_finetuned.pth"
    TARGET_CLASSES = list(range(10))  # First 10 classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Step 1: Baseline
    print("\n" + "-"*70)
    print("Step 1: Baseline timm model")
    print("-"*70)

    baseline = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(device)
    baseline.eval()

    param_count = sum(p.numel() for p in baseline.parameters())
    print(f"  Parameters: {param_count:,}")

    lat_baseline = measure_latency(baseline, (1, 3, 224, 224), device)
    print(f"  B=1 Latency: {lat_baseline:.3f} ms")

    del baseline
    torch.cuda.empty_cache()

    # Step 2: X-Pruner (subset mode)
    print("\n" + "-"*70)
    print("Step 2: X-Pruner (subset mode)")
    print("-"*70)

    xpruner = XPrunerDeiT(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
        enable_token_pruning=False
    )

    if os.path.exists(CHECKPOINT):
        print(f"  Loading: {CHECKPOINT}")
        state_dict = torch.load(CHECKPOINT, map_location="cpu")
        xpruner.load_state_dict(state_dict, strict=False)
    else:
        print(f"  Warning: Checkpoint not found, using fresh model")

    xpruner = xpruner.to(device)
    xpruner.eval()
    xpruner.prepare_subset_inference(TARGET_CLASSES, num_patch_tokens=196)

    lat_xpruner = measure_latency(xpruner, (1, 3, 224, 224), device)
    print(f"  B=1 Latency: {lat_xpruner:.3f} ms")
    print(f"  vs Baseline: {lat_xpruner/lat_baseline:.3f}x ({(lat_xpruner/lat_baseline - 1)*100:+.1f}%)")

    # Step 3: Optimized static
    print("\n" + "-"*70)
    print("Step 3: Optimized static model")
    print("-"*70)
    print("  Converting (this may take 10-20 seconds)...")

    optimized = convert_xpruner_to_optimized(
        xpruner,
        class_indices=TARGET_CLASSES,
        policy='mean',
        threshold=0.5,
        use_flash_attention=True,
        device=device,
        model_name=MODEL_NAME
    )
    optimized.eval()

    param_count_opt = sum(p.numel() for p in optimized.parameters())
    param_reduction = (1 - param_count_opt / param_count) * 100
    print(f"  Parameters: {param_count_opt:,} ({param_reduction:.1f}% reduction)")

    lat_optimized = measure_latency(optimized, (1, 3, 224, 224), device)
    print(f"  B=1 Latency: {lat_optimized:.3f} ms")
    print(f"  vs Baseline: {lat_optimized/lat_baseline:.3f}x ({(lat_optimized/lat_baseline - 1)*100:+.1f}%)")
    print(f"  vs X-Pruner: {lat_optimized/lat_xpruner:.3f}x ({(lat_optimized/lat_xpruner - 1)*100:+.1f}%)")

    # Step 4: torch.compile (if available)
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version >= (2, 0):
        print("\n" + "-"*70)
        print("Step 4: With torch.compile")
        print("-"*70)
        try:
            print("  Compiling...")
            optimized_compiled = torch.compile(optimized, mode='reduce-overhead')

            lat_compiled = measure_latency(optimized_compiled, (1, 3, 224, 224), device)
            print(f"  B=1 Latency: {lat_compiled:.3f} ms")
            print(f"  vs Baseline: {lat_compiled/lat_baseline:.3f}x ({(lat_compiled/lat_baseline - 1)*100:+.1f}%)")
            print(f"  vs Optimized: {lat_compiled/lat_optimized:.3f}x ({(lat_compiled/lat_optimized - 1)*100:+.1f}%)")
        except Exception as e:
            print(f"  Compilation failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nParameter Reduction: {param_reduction:.1f}%")
    print(f"\nLatency (Batch Size 1):")
    print(f"  Baseline:          {lat_baseline:.3f} ms (1.00x)")
    print(f"  X-Pruner (subset): {lat_xpruner:.3f} ms ({lat_xpruner/lat_baseline:.2f}x)")
    print(f"  Optimized static:  {lat_optimized:.3f} ms ({lat_optimized/lat_baseline:.2f}x)")

    if lat_optimized < lat_baseline:
        speedup_pct = (1 - lat_optimized/lat_baseline) * 100
        print(f"\n✓ SUCCESS: Optimized is {speedup_pct:.1f}% FASTER at B=1!")
    else:
        slowdown_pct = (lat_optimized/lat_baseline - 1) * 100
        print(f"\n⚠ Still {slowdown_pct:.1f}% slower (need more aggressive pruning)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
