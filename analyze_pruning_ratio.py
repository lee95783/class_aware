#!/usr/bin/env python3
"""Analyze actual pruning ratio in the checkpoint."""

import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from scripts.x_pruner import XPrunerDeiT

CHECKPOINT = "./results/xpruner_50_classes_lam5/xpruner_model_finetuned.pth"
MODEL_NAME = "deit_tiny_patch16_224"
NUM_CLASSES = 100
TARGET_CLASSES = list(range(10))

print("="*70)
print("Pruning Ratio Analysis")
print("="*70)

# Load model
xpruner = XPrunerDeiT(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    pretrained=False,
    enable_token_pruning=False
)

if os.path.exists(CHECKPOINT):
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    xpruner.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {CHECKPOINT}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xpruner = xpruner.to(device)
xpruner.eval()

# Analyze per-layer pruning
print("Per-Layer Head Pruning Analysis (mean policy, threshold=0.5):")
print("-"*70)

total_heads = 0
total_pruned = 0
total_kept = 0

for layer_idx, blk in enumerate(xpruner.backbone.blocks):
    if hasattr(blk.attn, 'gate'):
        gate = blk.attn.gate.cpu()  # [H, C]
        theta = blk.attn.theta.cpu()
        k = blk.attn.k
        num_heads = blk.attn.num_heads

        # Extract gates for target classes
        class_gates = gate[:, TARGET_CLASSES]  # [H, |subset|]

        # Compute keep scores (mean policy)
        g = torch.sigmoid(class_gates).mean(dim=1)  # [H]
        keep_score = torch.sigmoid(k * (g - theta))

        # Determine kept heads (threshold 0.5)
        kept_mask = keep_score > 0.5
        num_kept = kept_mask.sum().item()
        num_pruned = num_heads - num_kept

        total_heads += num_heads
        total_kept += num_kept
        total_pruned += num_pruned

        print(f"  Layer {layer_idx:2d}: {num_kept}/{num_heads} kept "
              f"({num_pruned} pruned, {num_pruned/num_heads*100:.1f}%)")
        print(f"             keep_scores: {keep_score.tolist()}")

print("-"*70)
print(f"\nTotal: {total_kept}/{total_heads} heads kept")
print(f"Pruned: {total_pruned}/{total_heads} ({total_pruned/total_heads*100:.1f}%)")
print(f"Keep ratio: {total_kept/total_heads:.3f}")

# Estimate FLOPs reduction
attn_flops_fraction = 0.33  # Attention is ~33% of ViT FLOPs
estimated_flop_reduction = (total_pruned / total_heads) * attn_flops_fraction
print(f"\nEstimated FLOPs reduction: {estimated_flop_reduction*100:.1f}%")

# Recommendations
print("\n" + "="*70)
print("Recommendations")
print("="*70)

if total_pruned / total_heads < 0.3:
    print("⚠ Pruning ratio is LOW (<30%)")
    print("  For measurable B=1 speedup, aim for 40-50% pruning")
    print("  Retrain with higher target_sparsity (e.g., 0.5 or 0.4)")
elif total_pruned / total_heads < 0.4:
    print("⚠ Pruning ratio is MODERATE (30-40%)")
    print("  Should see small B=1 speedup on modern GPUs (A100/H100)")
    print("  For stronger speedup, increase target_sparsity")
else:
    print("✓ Pruning ratio is GOOD (≥40%)")
    print("  Should see clear B=1 speedup!")

print("\n" + "="*70)
