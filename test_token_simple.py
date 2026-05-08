#!/usr/bin/env python3
"""Simple token pruning test - just 50% token pruning."""

import os
import sys
import torch
import time
import timm

sys.path.append(os.path.dirname(__file__))
from scripts.convert_to_deployment import create_fully_optimized_model


def measure_latency(model, input_size, device, num_iters=50):
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) / num_iters * 1000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Baseline
print("Loading baseline...")
baseline = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100).to(device)
baseline.eval()

lat_base = measure_latency(baseline, (1, 3, 224, 224), device)
print(f"Baseline B=1: {lat_base:.2f} ms\n")

# Optimized with 50% token pruning
print("Creating optimized model (50% token pruning + head pruning)...")
optimized = create_fully_optimized_model(
    xpruner_checkpoint_path="./results/xpruner_50_classes_lam5/xpruner_model_finetuned.pth",
    class_indices=list(range(10)),
    model_name="deit_tiny_patch16_224",
    num_classes=100,
    head_policy='mean',
    head_threshold=0.5,
    enable_token_pruning=True,
    token_keep_ratio=0.5,
    use_flash_attention=True,
    device=device
)
optimized.eval()

lat_opt_b1 = measure_latency(optimized, (1, 3, 224, 224), device)
lat_opt_b128 = measure_latency(optimized, (128, 3, 224, 224), device)

print(f"\nOptimized B=1:   {lat_opt_b1:.2f} ms ({lat_base/lat_opt_b1:.2f}x, {(lat_base/lat_opt_b1-1)*100:+.1f}%)")
print(f"Optimized B=128: {lat_opt_b128:.2f} ms")

if lat_opt_b1 < lat_base:
    print(f"\n✓ SUCCESS: Token pruning achieved {(1-lat_opt_b1/lat_base)*100:.1f}% speedup at B=1!")
else:
    print(f"\n⚠ Still {(lat_opt_b1/lat_base-1)*100:.1f}% slower at B=1")
