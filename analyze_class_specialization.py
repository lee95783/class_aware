"""
Analyze class-specific specialization patterns in heads, tokens, and MLPs.

This script investigates:
1. Do different classes activate different attention heads?
2. Do different classes select different tokens?
3. Do different classes activate different MLP neurons?
4. Which component shows the strongest class-conditional specialization?
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import get_dataloaders
from src.models import get_model


def analyze_head_specialization(model, loader, device, num_classes=50):
    """
    Analyze: Do different classes preferentially use different attention heads?

    Returns:
        head_class_importance: [num_layers, num_heads, num_classes]
            Average attention output magnitude per head per class
    """
    model.eval()
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads

    # Track attention output magnitude per head per class
    head_activations = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            x = model.patch_embed(images)
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)

            for layer_idx, blk in enumerate(model.blocks):
                # Hook to capture attention outputs before residual
                B, N, C = x.shape
                qkv = blk.attn.qkv(blk.norm1(x)).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                attn = (q @ k.transpose(-2, -1)) * (C // num_heads) ** -0.5
                attn = attn.softmax(dim=-1)
                attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]

                # Measure head contribution: split by heads
                head_out = attn_out.reshape(B, N, num_heads, C // num_heads)  # [B, N, H, D_h]
                head_magnitude = head_out.norm(dim=-1).mean(dim=1)  # [B, H] - average over tokens

                # Store per class
                for b in range(B):
                    cls = labels[b].item()
                    for h in range(num_heads):
                        head_activations[layer_idx][cls].append(head_magnitude[b, h].item())

                # Continue forward pass
                x = x + blk.drop_path1(blk.ls1(blk.attn.proj(blk.attn.proj_drop(attn_out))))
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

    # Compute average activation per head per class
    head_class_importance = np.zeros((num_layers, num_heads, num_classes))
    for layer_idx in range(num_layers):
        for cls in range(num_classes):
            if cls in head_activations[layer_idx]:
                for h in range(num_heads):
                    activations = [head_activations[layer_idx][cls][i*num_heads + h]
                                   for i in range(len(head_activations[layer_idx][cls]) // num_heads)]
                    head_class_importance[layer_idx, h, cls] = np.mean(activations)

    return head_class_importance


def analyze_token_specialization(model, loader, device, num_classes=50):
    """
    Analyze: Do different classes preferentially attend to different spatial tokens?

    Returns:
        token_class_importance: [num_layers, num_tokens, num_classes]
            Average attention weight received by each token position per class
    """
    model.eval()
    num_layers = len(model.blocks)
    num_tokens = 197  # 196 patches + 1 CLS

    # Track attention to each token position per class
    token_attention = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            x = model.patch_embed(images)
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)

            for layer_idx, blk in enumerate(model.blocks):
                B, N, C = x.shape
                qkv = blk.attn.qkv(blk.norm1(x)).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                attn = (q @ k.transpose(-2, -1)) * (C // blk.attn.num_heads) ** -0.5
                attn = attn.softmax(dim=-1)  # [B, H, N, N]

                # Average attention weight TO each token (averaged over all source tokens and heads)
                token_importance = attn.mean(dim=1).mean(dim=1)  # [B, N] - how much each token is attended to

                # Store per class
                for b in range(B):
                    cls = labels[b].item()
                    token_attention[layer_idx][cls].append(token_importance[b].cpu().numpy())

                # Continue forward pass
                attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = x + blk.drop_path1(blk.ls1(blk.attn.proj(blk.attn.proj_drop(attn_out))))
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

    # Compute average attention per token per class
    token_class_importance = np.zeros((num_layers, num_tokens, num_classes))
    for layer_idx in range(num_layers):
        for cls in range(num_classes):
            if cls in token_attention[layer_idx]:
                token_class_importance[layer_idx, :, cls] = np.mean(token_attention[layer_idx][cls], axis=0)

    return token_class_importance


def analyze_mlp_specialization(model, loader, device, num_classes=50):
    """
    Analyze: Do different classes activate different MLP neurons?

    Returns:
        mlp_class_importance: [num_layers, hidden_dim, num_classes]
            Average activation magnitude per MLP neuron per class
    """
    model.eval()
    num_layers = len(model.blocks)
    hidden_dim = model.blocks[0].mlp.fc1.out_features

    # Track MLP neuron activations per class
    mlp_activations = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            x = model.patch_embed(images)
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)

            for layer_idx, blk in enumerate(model.blocks):
                # Forward through attention
                B, N, C = x.shape
                qkv = blk.attn.qkv(blk.norm1(x)).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * (C // blk.attn.num_heads) ** -0.5
                attn = attn.softmax(dim=-1)
                attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = x + blk.drop_path1(blk.ls1(blk.attn.proj(blk.attn.proj_drop(attn_out))))

                # Capture MLP activations
                x_norm = blk.norm2(x)
                mlp_hidden = blk.mlp.fc1(x_norm)  # [B, N, hidden_dim]
                mlp_hidden = blk.mlp.act(mlp_hidden)  # After GELU activation

                # Measure neuron importance: average over tokens
                neuron_magnitude = mlp_hidden.mean(dim=1)  # [B, hidden_dim]

                # Store per class
                for b in range(B):
                    cls = labels[b].item()
                    mlp_activations[layer_idx][cls].append(neuron_magnitude[b].cpu().numpy())

                # Complete MLP forward (use drop1 instead of drop)
                mlp_out = blk.mlp.fc2(blk.mlp.drop1(mlp_hidden))
                x = x + blk.drop_path2(blk.ls2(mlp_out))

    # Compute average activation per neuron per class
    mlp_class_importance = np.zeros((num_layers, hidden_dim, num_classes))
    for layer_idx in range(num_layers):
        for cls in range(num_classes):
            if cls in mlp_activations[layer_idx]:
                mlp_class_importance[layer_idx, :, cls] = np.mean(mlp_activations[layer_idx][cls], axis=0)

    return mlp_class_importance


def compute_specialization_metrics(importance_array):
    """
    Compute specialization metrics from importance array [components, classes].

    Returns:
        - variance_ratio: Variance across classes / total variance (higher = more specialized)
        - gini_coefficient: Gini coefficient across classes (0 = uniform, 1 = highly specialized)
        - top1_concentration: Average fraction of importance in top-1 class
    """
    num_components, num_classes = importance_array.shape

    # Normalize each component to sum to 1 across classes
    importance_norm = importance_array / (importance_array.sum(axis=1, keepdims=True) + 1e-8)

    # Variance ratio
    variance_ratio = np.var(importance_norm, axis=1).mean()

    # Gini coefficient (averaged across components)
    gini_coeffs = []
    for i in range(num_components):
        sorted_imp = np.sort(importance_norm[i])
        n = len(sorted_imp)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_imp)) / (n * np.sum(sorted_imp)) - (n + 1) / n
        gini_coeffs.append(gini)
    gini_coefficient = np.mean(gini_coeffs)

    # Top-1 concentration
    top1_concentration = importance_norm.max(axis=1).mean()

    return {
        'variance_ratio': variance_ratio,
        'gini_coefficient': gini_coefficient,
        'top1_concentration': top1_concentration
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = get_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)

    # Load fine-tuned checkpoint
    ckpt_path = Path("best_deit_tiny_cifar100_final_timm.pth")
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Warning: No checkpoint found, using random initialization")

    # Load data - use validation set for analysis
    _, test_loader = get_dataloaders(
        data_dir='./data',
        dataset_name='cifar100',
        batch_size=64,
        image_size=224,
        num_workers=4,
        train=False,
        split='val'
    )

    # For 50-class subset analysis
    num_classes = 100  # Using full CIFAR-100 for now

    print("\n" + "="*80)
    print("ANALYZING CLASS-SPECIFIC SPECIALIZATION")
    print("="*80)

    # Analyze head specialization
    print("\n[1/3] Analyzing attention head specialization...")
    head_importance = analyze_head_specialization(model, test_loader, device, num_classes=num_classes)
    print(f"    Shape: {head_importance.shape} (layers, heads, classes)")

    # Flatten to [all_heads, classes] for metric computation
    head_flat = head_importance.reshape(-1, num_classes)
    head_metrics = compute_specialization_metrics(head_flat)
    print(f"    Variance Ratio: {head_metrics['variance_ratio']:.4f}")
    print(f"    Gini Coefficient: {head_metrics['gini_coefficient']:.4f}")
    print(f"    Top-1 Concentration: {head_metrics['top1_concentration']:.4f}")

    # Analyze token specialization
    print("\n[2/3] Analyzing token position specialization...")
    token_importance = analyze_token_specialization(model, test_loader, device, num_classes=num_classes)
    print(f"    Shape: {token_importance.shape} (layers, tokens, classes)")

    # Use only patch tokens (exclude CLS)
    token_flat = token_importance[:, 1:, :].reshape(-1, num_classes)
    token_metrics = compute_specialization_metrics(token_flat)
    print(f"    Variance Ratio: {token_metrics['variance_ratio']:.4f}")
    print(f"    Gini Coefficient: {token_metrics['gini_coefficient']:.4f}")
    print(f"    Top-1 Concentration: {token_metrics['top1_concentration']:.4f}")

    # Analyze MLP specialization
    print("\n[3/3] Analyzing MLP neuron specialization...")
    mlp_importance = analyze_mlp_specialization(model, test_loader, device, num_classes=num_classes)
    print(f"    Shape: {mlp_importance.shape} (layers, neurons, classes)")

    mlp_flat = mlp_importance.reshape(-1, num_classes)
    mlp_metrics = compute_specialization_metrics(mlp_flat)
    print(f"    Variance Ratio: {mlp_metrics['variance_ratio']:.4f}")
    print(f"    Gini Coefficient: {mlp_metrics['gini_coefficient']:.4f}")
    print(f"    Top-1 Concentration: {mlp_metrics['top1_concentration']:.4f}")

    # Summary comparison
    print("\n" + "="*80)
    print("SPECIALIZATION COMPARISON")
    print("="*80)
    print(f"{'Component':<20} {'Variance Ratio':<18} {'Gini Coeff':<15} {'Top-1 Conc':<15}")
    print("-"*80)
    print(f"{'Attention Heads':<20} {head_metrics['variance_ratio']:<18.4f} "
          f"{head_metrics['gini_coefficient']:<15.4f} {head_metrics['top1_concentration']:<15.4f}")
    print(f"{'Token Positions':<20} {token_metrics['variance_ratio']:<18.4f} "
          f"{token_metrics['gini_coefficient']:<15.4f} {token_metrics['top1_concentration']:<15.4f}")
    print(f"{'MLP Neurons':<20} {mlp_metrics['variance_ratio']:<18.4f} "
          f"{mlp_metrics['gini_coefficient']:<15.4f} {mlp_metrics['top1_concentration']:<15.4f}")
    print("="*80)

    print("\nInterpretation:")
    print("- Higher variance ratio = more class-specific specialization")
    print("- Higher Gini coefficient = more unequal distribution (some classes dominate)")
    print("- Higher Top-1 concentration = each component strongly prefers certain classes")

    # Determine which component is most class-specialized
    specialization_scores = {
        'Heads': head_metrics['variance_ratio'] + head_metrics['gini_coefficient'],
        'Tokens': token_metrics['variance_ratio'] + token_metrics['gini_coefficient'],
        'MLPs': mlp_metrics['variance_ratio'] + mlp_metrics['gini_coefficient']
    }

    most_specialized = max(specialization_scores, key=specialization_scores.get)
    print(f"\n>>> Most class-specialized component: {most_specialized}")

    # Save results
    results = {
        'head_metrics': head_metrics,
        'token_metrics': token_metrics,
        'mlp_metrics': mlp_metrics,
        'specialization_scores': specialization_scores,
        'most_specialized': most_specialized
    }

    output_path = Path("docs/class_specialization_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                json_results[k] = {kk: float(vv) if isinstance(vv, np.floating) else vv
                                   for kk, vv in v.items()}
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
