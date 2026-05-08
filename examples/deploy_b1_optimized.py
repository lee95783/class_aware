#!/usr/bin/env python3
"""
Quick-start example: Deploy X-Pruner for Batch Size 1 with optimizations.

This script demonstrates the complete workflow:
1. Load trained X-Pruner checkpoint
2. Convert to optimized static model
3. Benchmark against baseline
4. Export for production deployment

Usage:
    python examples/deploy_b1_optimized.py
"""

import os
import sys
import torch
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.x_pruner import XPrunerDeiT
from scripts.x_pruner_optimized import convert_xpruner_to_optimized
import timm


def measure_latency(model, input_size, device, num_iters=100):
    """Quick latency measurement."""
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
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
    print("X-Pruner Batch Size 1 Optimization - Quick Start Example")
    print("="*70)

    # Configuration
    MODEL_NAME = "deit_tiny_patch16_224"
    NUM_CLASSES = 100
    CHECKPOINT_PATH = "./results/xpruner_50_classes_lam5/xpruner_model_finetuned.pth"
    TARGET_CLASSES = list(range(10))  # First 10 classes as deployment subset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Num Classes: {NUM_CLASSES}")
    print(f"  Deployment Subset: {len(TARGET_CLASSES)} classes")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # Step 1: Load baseline model
    print("\n" + "-"*70)
    print("Step 1: Load baseline timm model")
    print("-"*70)

    baseline = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(device)
    baseline.eval()

    param_count = sum(p.numel() for p in baseline.parameters())
    print(f"  Parameters: {param_count:,}")

    # Measure baseline latency
    lat_baseline = measure_latency(baseline, (1, 3, 224, 224), device)
    print(f"  B=1 Latency: {lat_baseline:.3f} ms")

    # Step 2: Load trained X-Pruner
    print("\n" + "-"*70)
    print("Step 2: Load trained X-Pruner model")
    print("-"*70)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  ⚠ Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"  Creating a fresh X-Pruner model for demonstration...")
        xpruner = XPrunerDeiT(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            pretrained=True,
            enable_token_pruning=False
        )
    else:
        xpruner = XPrunerDeiT(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            pretrained=False,
            enable_token_pruning=False
        )
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        xpruner.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded checkpoint: {CHECKPOINT_PATH}")

    xpruner = xpruner.to(device)
    xpruner.eval()

    # Prepare subset mode
    xpruner.prepare_subset_inference(TARGET_CLASSES, num_patch_tokens=196)

    lat_xpruner = measure_latency(xpruner, (1, 3, 224, 224), device)
    print(f"  B=1 Latency (subset mode): {lat_xpruner:.3f} ms")
    print(f"  vs Baseline: {lat_xpruner/lat_baseline:.3f}x ({(lat_xpruner/lat_baseline - 1)*100:+.1f}%)")

    # Step 3: Convert to optimized static model
    print("\n" + "-"*70)
    print("Step 3: Convert to optimized static model")
    print("-"*70)

    print("  Converting... (this may take a few seconds)")
    optimized = convert_xpruner_to_optimized(
        xpruner,
        class_indices=TARGET_CLASSES,
        policy='mean',
        threshold=0.5,
        use_flash_attention=True,
        device=device
    )
    optimized.eval()

    param_count_opt = sum(p.numel() for p in optimized.parameters())
    param_reduction = (1 - param_count_opt / param_count) * 100
    print(f"  ✓ Conversion complete")
    print(f"  Parameters: {param_count_opt:,} ({param_reduction:.1f}% reduction)")

    lat_optimized = measure_latency(optimized, (1, 3, 224, 224), device)
    print(f"  B=1 Latency: {lat_optimized:.3f} ms")
    print(f"  vs Baseline: {lat_optimized/lat_baseline:.3f}x ({(lat_optimized/lat_baseline - 1)*100:+.1f}%)")
    print(f"  vs X-Pruner subset: {lat_optimized/lat_xpruner:.3f}x ({(lat_optimized/lat_xpruner - 1)*100:+.1f}%)")

    # Step 4: Try compilation (optional)
    print("\n" + "-"*70)
    print("Step 4: Advanced optimization (torch.compile)")
    print("-"*70)

    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version >= (2, 0):
        try:
            print("  Compiling with torch.compile...")
            optimized_compiled = torch.compile(optimized, mode='reduce-overhead')

            lat_compiled = measure_latency(optimized_compiled, (1, 3, 224, 224), device)
            print(f"  ✓ Compilation successful")
            print(f"  B=1 Latency: {lat_compiled:.3f} ms")
            print(f"  vs Baseline: {lat_compiled/lat_baseline:.3f}x ({(lat_compiled/lat_baseline - 1)*100:+.1f}%)")
            print(f"  vs Optimized static: {lat_compiled/lat_optimized:.3f}x ({(lat_compiled/lat_optimized - 1)*100:+.1f}%)")
        except Exception as e:
            print(f"  ⚠ Compilation failed: {e}")
    else:
        print(f"  ⚠ torch.compile requires PyTorch 2.0+ (you have {torch.__version__})")

    # Step 5: Save for deployment
    print("\n" + "-"*70)
    print("Step 5: Export for production deployment")
    print("-"*70)

    output_dir = "./results/optimized_deployment"
    os.makedirs(output_dir, exist_ok=True)

    # Save state dict
    model_path = os.path.join(output_dir, "optimized_b1_model.pth")
    torch.save(optimized.state_dict(), model_path)
    print(f"  ✓ Model saved: {model_path}")

    # Save metadata
    metadata = {
        'model_name': MODEL_NAME,
        'num_classes': NUM_CLASSES,
        'target_classes': TARGET_CLASSES,
        'policy': 'mean',
        'threshold': 0.5,
        'param_count': param_count_opt,
        'param_reduction_pct': param_reduction,
        'latency_ms': lat_optimized,
        'speedup_vs_baseline': lat_baseline / lat_optimized
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {metadata_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nParameter Reduction: {param_reduction:.1f}%")
    print(f"\nLatency Comparison (Batch Size 1):")
    print(f"  Baseline:          {lat_baseline:.3f} ms")
    print(f"  X-Pruner (subset): {lat_xpruner:.3f} ms ({lat_xpruner/lat_baseline:.3f}x)")
    print(f"  Optimized static:  {lat_optimized:.3f} ms ({lat_optimized/lat_baseline:.3f}x)")

    if lat_optimized < lat_baseline:
        speedup_pct = (1 - lat_optimized/lat_baseline) * 100
        print(f"\n✓ SUCCESS: Optimized model is {speedup_pct:.1f}% FASTER at B=1!")
    else:
        slowdown_pct = (lat_optimized/lat_baseline - 1) * 100
        print(f"\n⚠ Optimized model is still {slowdown_pct:.1f}% slower at B=1")
        print(f"  Suggestion: Increase pruning ratio during training for better B=1 performance")

    print(f"\nDeployment artifacts saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Test accuracy on validation set")
    print("  2. Benchmark on target hardware")
    print("  3. Consider quantization for additional speedup")
    print("  4. Export to ONNX/TensorRT for production")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
