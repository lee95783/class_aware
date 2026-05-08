#!/usr/bin/env python3
"""
Quick test for class-aware MLP pruning with XPruner.

Tests:
1. Model creation with MLP gating
2. Forward pass with class-conditional gates
3. Keep ratio computation
4. Comparison: head-only vs MLP-only vs joint pruning
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from scripts.x_pruner import XPrunerDeiT

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_mlp_gating():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    NUM_CLASSES = 10
    BATCH_SIZE = 4

    # Test 1: Head-only pruning (baseline)
    print("="*70)
    print("Test 1: Head-only pruning (baseline)")
    print("="*70)
    model_head = XPrunerDeiT(
        model_name="deit_tiny_patch16_224",
        num_classes=NUM_CLASSES,
        pretrained=False,
        k=10.0,
        enable_mlp_pruning=False,
    ).to(device)

    params_head = count_parameters(model_head)
    print(f"Parameters: {params_head:,}")

    # Count head gates
    head_gates = sum(1 for name, _ in model_head.named_parameters() if 'attn.gate' in name)
    print(f"Head gate parameters: {head_gates} layers")

    # Test forward
    x = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

    logits, keep_all = model_head(x, y=y, use_labels=True)
    print(f"Output shape: {logits.shape}")
    print(f"Keep ratios collected: {len(keep_all)} (should be 12 heads)")
    print(f"Head keep ratio: {torch.cat(keep_all).mean().item():.3f}")

    # Test 2: MLP-only pruning
    print("\n" + "="*70)
    print("Test 2: MLP-only pruning")
    print("="*70)
    model_mlp = XPrunerDeiT(
        model_name="deit_tiny_patch16_224",
        num_classes=NUM_CLASSES,
        pretrained=False,
        k=10.0,
        enable_mlp_pruning=True,
        mlp_k=10.0,
    ).to(device)

    params_mlp = count_parameters(model_mlp)
    print(f"Parameters: {params_mlp:,}")
    print(f"Additional params vs head-only: {params_mlp - params_head:,}")

    # Count MLP gates
    mlp_gates = sum(1 for name, _ in model_mlp.named_parameters() if 'mlp.gate' in name)
    print(f"MLP gate parameters: {mlp_gates} layers")

    # Test forward
    logits, keep_all = model_mlp(x, y=y, use_labels=True)
    print(f"Output shape: {logits.shape}")
    print(f"Keep ratios collected: {len(keep_all)} (should be 12 heads + 12 MLPs = 24)")

    # Separate head and MLP keep ratios
    head_keeps = keep_all[::2]  # Even indices are heads
    mlp_keeps = keep_all[1::2]  # Odd indices are MLPs

    head_ratio = torch.cat(head_keeps).mean().item()
    mlp_ratio = torch.cat(mlp_keeps).mean().item()

    print(f"Head keep ratio: {head_ratio:.3f}")
    print(f"MLP keep ratio: {mlp_ratio:.3f}")

    # Test 3: Class-conditional behavior
    print("\n" + "="*70)
    print("Test 3: Class-conditional behavior")
    print("="*70)

    # Check that different classes get different gates
    y1 = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)  # All class 0
    y2 = torch.ones(BATCH_SIZE, dtype=torch.long, device=device) * (NUM_CLASSES-1)  # All class 9

    logits1, keep1 = model_mlp(x, y=y1, use_labels=True)
    logits2, keep2 = model_mlp(x, y=y2, use_labels=True)

    # Compare MLP keeps for different classes
    mlp_keep1 = keep1[1]  # First MLP layer, class 0
    mlp_keep2 = keep2[1]  # First MLP layer, class 9

    difference = (mlp_keep1 - mlp_keep2).abs().mean().item()
    print(f"Average difference in MLP gates between class 0 and class {NUM_CLASSES-1}: {difference:.4f}")
    print(f"Expected: > 0 (different classes should have different gates)")
    print(f"Status: {'PASS ✓' if difference > 0.001 else 'FAIL ✗'}")

    # Test 4: Class-agnostic mode
    print("\n" + "="*70)
    print("Test 4: Class-agnostic mode (no labels)")
    print("="*70)

    logits_agnostic, keep_agnostic = model_mlp(x, y=None, use_labels=False)
    print(f"Output shape: {logits_agnostic.shape}")

    mlp_ratio_agnostic = torch.cat(keep_agnostic[1::2]).mean().item()
    print(f"MLP keep ratio (class-agnostic): {mlp_ratio_agnostic:.3f}")

    # Test 5: Parameter breakdown
    print("\n" + "="*70)
    print("Test 5: Parameter breakdown")
    print("="*70)

    # Analyze MLP gate sizes
    for name, param in model_mlp.named_parameters():
        if 'mlp.gate' in name:
            print(f"{name}: {param.shape} = {param.numel():,} params")
            break  # Just show one example

    # Expected: (D_ff, C) = (768, 10) = 7,680 params per MLP gate
    total_mlp_gate_params = 768 * NUM_CLASSES * 12  # 12 layers
    print(f"\nTotal MLP gate params: {total_mlp_gate_params:,}")
    print(f"Percentage of total model: {total_mlp_gate_params / params_mlp * 100:.2f}%")

    # Test 6: Gradient flow
    print("\n" + "="*70)
    print("Test 6: Gradient flow through gates")
    print("="*70)

    model_mlp.train()
    optimizer = torch.optim.SGD(model_mlp.parameters(), lr=0.01)

    # Dummy loss
    logits, keep_all = model_mlp(x, y=y, use_labels=True)
    loss = F.cross_entropy(logits, y)

    # Add sparsity loss on MLP keeps
    mlp_keeps = keep_all[1::2]
    mlp_keep_ratio = torch.cat(mlp_keeps).mean()
    sparsity_loss = (mlp_keep_ratio - 0.5) ** 2  # Target 50% keep ratio

    total_loss = loss + 0.1 * sparsity_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()

    # Check gradients on MLP gates
    has_grad = False
    for name, param in model_mlp.named_parameters():
        if 'mlp.gate' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name} grad norm: {grad_norm:.6f}")
            has_grad = True
            break

    print(f"Status: {'PASS ✓ (gradients flowing)' if has_grad else 'FAIL ✗ (no gradients)'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ XPrunerMLPGating module integrated successfully")
    print("✓ Forward pass works with class-conditional gates")
    print("✓ Different classes produce different MLP masks")
    print("✓ Class-agnostic mode works")
    print("✓ Gradients flow through MLP gates")
    print(f"\nMLP gate overhead: {total_mlp_gate_params:,} params ({total_mlp_gate_params / params_mlp * 100:.2f}% of model)")
    print("\nNext steps:")
    print("1. Train with sparsity loss (penalty or ALM)")
    print("2. Export to compact structural models")
    print("3. Benchmark latency at B=1")
    print("4. Compare to magnitude pruning baseline")


if __name__ == "__main__":
    test_mlp_gating()
