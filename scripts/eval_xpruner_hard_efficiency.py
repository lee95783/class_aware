#!/usr/bin/env python3
"""
Efficiency Evaluation: X-Pruner Hard-Pruned Models

Benchmarks memory and latency for X-Pruner with structural head removal,
using the actual heads_pruned counts from the e07 experiment results.

Since trained checkpoints were not saved, we create structurally equivalent
models with random weights (structure/latency is weight-independent).
Head distribution across layers is approximated as uniform.

Results from e07_xpruner_*cls.json at ratio=0.5:
  K=5:  8/36 heads pruned
  K=10: 2/36 heads pruned
  K=20: 6/36 heads pruned
  K=50: 16/36 heads pruned

At ratio=0.7, 0 heads are pruned for all K (same as soft/unpruned).

Usage:
    python scripts/eval_xpruner_hard_efficiency.py --device 0
"""

import os, sys, json, argparse, time
import torch
import torch.nn as nn
import timm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pruning import compact_vit_attention_heads

# DeiT-Tiny: 3 heads per layer, 12 layers = 36 heads total
N_HEADS_PER_LAYER = 3
N_LAYERS = 12
N_HEADS_TOTAL = N_HEADS_PER_LAYER * N_LAYERS  # 36


def heads_to_prune_uniform(n_prune, n_layers=N_LAYERS, heads_per_layer=N_HEADS_PER_LAYER):
    """Distribute n_prune heads across layers from last to first, keeping min 1 per layer."""
    if n_prune == 0:
        return []
    max_per_layer = heads_per_layer - 1
    heads_to_prune = []
    remaining = n_prune
    for layer_idx in reversed(range(n_layers)):
        if remaining <= 0:
            break
        prune_here = min(remaining, max_per_layer)
        for h in range(prune_here):
            heads_to_prune.append((layer_idx, h))
        remaining -= prune_here
    return heads_to_prune


def make_hard_pruned_model(device, heads_to_prune_list):
    """Create a structurally compacted DeiT-Tiny with specified heads removed."""
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False)
    if heads_to_prune_list:
        compact_vit_attention_heads(m, heads_to_prune_list)
    m.to(device)
    m.eval()
    return m


def param_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def benchmark(model, batch_size, device, n_iter=200, n_warmup=20):
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        if device.type == 'cuda':
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                model(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    trimmed = times[n_iter // 10: -n_iter // 10]
    return sum(trimmed) / len(trimmed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',    type=int, default=0)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 64])
    parser.add_argument('--n-iter',    type=int, default=200)
    parser.add_argument('--output-dir', type=str, default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Configs: (label, n_pruned, n_total, target_ratio, num_classes)
    configs = [
        ('Unpruned',                    0,  36, None, None),
        ('X-Pruner hard (keep=0.7)',     0,  36, 0.7,  'all K'),  # 0 heads pruned
        ('X-Pruner hard K=5  (keep=0.5)',  8,  36, 0.5,  5),
        ('X-Pruner hard K=10 (keep=0.5)',  2,  36, 0.5,  10),
        ('X-Pruner hard K=20 (keep=0.5)',  6,  36, 0.5,  20),
        ('X-Pruner hard K=50 (keep=0.5)', 16,  36, 0.5,  50),
    ]

    print('=' * 80)
    print('X-Pruner Hard Pruning — Memory + Latency')
    print(f'Device: {device}')
    print('Heads distributed uniformly from last layers (approx. structural equivalent).')
    print('=' * 80)

    bs_header = '  '.join(f'B={bs:>2} (ms)' for bs in args.batch_sizes)
    print(f'\n  {"Method":<38}  {"MB":>7}  {bs_header}  {"B=64 img/s":>10}')
    print('  ' + '-' * 90)

    results = {}
    baseline_lat = {}

    for label, n_pruned, n_total, ratio, k in configs:
        prune_list = heads_to_prune_uniform(n_pruned)
        model = make_hard_pruned_model(device, prune_list)

        mb = param_mb(model)
        heads_kept = n_total - n_pruned
        row_lats = []
        for bs in args.batch_sizes:
            lat = benchmark(model, bs, device, n_iter=args.n_iter)
            row_lats.append(lat)

        if label == 'Unpruned':
            baseline_lat = {bs: row_lats[i] for i, bs in enumerate(args.batch_sizes)}

        # img/s at B=64
        lat_b64 = row_lats[args.batch_sizes.index(64)] if 64 in args.batch_sizes else None
        imgs_per_sec = int(64 / (lat_b64 / 1000)) if lat_b64 else None

        lat_str = '  '.join(f'{l:>9.2f}' for l in row_lats)
        imgs_str = f'{imgs_per_sec:>10d}' if imgs_per_sec else ' ' * 10
        print(f'  {label:<38}  {mb:>7.2f}  {lat_str}  {imgs_str}  '
              f'(kept {heads_kept}/{n_total} heads)', flush=True)

        results[label] = {
            'n_pruned': n_pruned,
            'n_total': n_total,
            'heads_kept': heads_kept,
            'model_mb': round(mb, 3),
            'latency_ms': {bs: round(row_lats[i], 3) for i, bs in enumerate(args.batch_sizes)},
            'imgs_per_sec_b64': imgs_per_sec,
        }

        del model
        torch.cuda.empty_cache()

    save_path = os.path.join(args.output_dir, 'xpruner_hard_efficiency.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'device': str(device), 'batch_sizes': args.batch_sizes, 'results': results}, f, indent=2)
    print(f'\nSaved: {save_path}')

    # Summary vs unpruned
    print('\n── vs Unpruned ──')
    unp_mb = results['Unpruned']['model_mb']
    unp_lat64 = results['Unpruned']['latency_ms'].get(64)
    for label, r in results.items():
        if label == 'Unpruned':
            continue
        mb_delta = r['model_mb'] - unp_mb
        lat64 = r['latency_ms'].get(64)
        speedup = unp_lat64 / lat64 if lat64 else None
        print(f'  {label:<38}  ΔMB={mb_delta:+.2f}  '
              f'B=64 speedup={speedup:.3f}x  '
              f'({r["n_pruned"]} heads pruned)')


if __name__ == '__main__':
    main()
