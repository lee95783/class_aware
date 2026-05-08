#!/usr/bin/env python3
"""
Efficiency Evaluation: Memory and Latency for All Methods

Measures for each compression method:
  1. Parameter memory (MB) — backbone checkpoint size
  2. Per-subset overhead (KB) — additional storage per K-class deployment
  3. Theoretical MACs per image (GMACs) — using sequence-length-aware calculation
  4. Actual inference latency (ms) at batch_size=1 and batch_size=64

Methods compared:
  - Unpruned baseline (DeiT-Tiny)
  - CGTS only (token keep=0.7, layer 6, zero-shot)
  - MLP 30% + CGTS 0.7 (our full method)
  - MLP 50% + CGTS 0.5 (our aggressive setting)
  - DynamicViT keep=0.7 (learned token predictor, layer 6)
  - X-Pruner keep=0.7 (soft head gating, ALM)
  - SViTE keep=0.7 (static token mask, layer 6)

For latency, each method is timed with fresh random weights — structure matters,
not accuracy. MLP pruning is applied as structural compaction (remove zero rows)
so it gives real latency reduction.

Usage:
    python scripts/eval_efficiency.py --device 0
"""

import os, sys, time, json, argparse
import torch
import torch.nn as nn
import timm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.x_pruner import XPrunerDeiT
from scripts.eval_e07_zero_tprune import _wpr, _attn_weights_from_qkv


# ── Model parameters ──────────────────────────────────────────────────────────
# DeiT-Tiny: D=192, H=3, n_heads=3, n_layers=12, patch=16, img=224
D       = 192
N_FULL  = 197        # 196 patches + 1 CLS
N_HEADS = 3
N_LAYERS= 12
D_FF    = D * 4      # 768 MLP hidden dim


# ── MAC computation ───────────────────────────────────────────────────────────

def attention_macs(n_tokens, d=D, n_heads=N_HEADS):
    """MACs for one attention block with n_tokens."""
    # QKV proj: 3 × N × D²
    qkv = 3 * n_tokens * d * d
    # Attention scores: N × N × D (per head, summed)
    attn = 2 * n_tokens * n_tokens * d
    # Output proj: N × D²
    proj = n_tokens * d * d
    return qkv + attn + proj


def mlp_macs(n_tokens, d=D, d_ff=D_FF, keep_ratio=1.0):
    """MACs for one MLP block. keep_ratio accounts for structural neuron pruning."""
    effective_ff = max(1, int(d_ff * keep_ratio))
    fc1 = n_tokens * d * effective_ff
    fc2 = n_tokens * effective_ff * d
    return fc1 + fc2


def total_macs(n_tokens_schedule, mlp_keep=1.0):
    """
    n_tokens_schedule: list of n_tokens per layer (length = N_LAYERS)
    mlp_keep: fraction of MLP neurons kept (structural pruning)
    Returns total MACs in GMACs.
    """
    total = 0
    for n in n_tokens_schedule:
        total += attention_macs(n) + mlp_macs(n, keep_ratio=mlp_keep)
    # Patch embed: img_size² × 3 × D × (patch_size²×3) ≈ small, skip
    return total / 1e9


def token_schedule(cgts_layer=None, keep_ratio=1.0):
    """Build token count per layer for a given pruning strategy."""
    n = N_FULL
    schedule = []
    for l in range(N_LAYERS):
        if cgts_layer is not None and l == cgts_layer:
            n = max(1, int((n - 1) * keep_ratio) + 1)  # keep CLS + top-k patches
        schedule.append(n)
    return schedule


# ── Latency helpers ───────────────────────────────────────────────────────────

def benchmark(model_fn, batch_size, device, n_iter=200, n_warmup=20):
    """Time a callable (batch of images → logits). Returns mean ms."""
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model_fn(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # Measure
    times = []
    for _ in range(n_iter):
        if device.type == 'cuda':
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                model_fn(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                model_fn(x)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    trimmed = times[n_iter//10 : -n_iter//10]  # trim 10% each end
    return sum(trimmed) / len(trimmed)


# ── Model builders ────────────────────────────────────────────────────────────

def make_baseline(device):
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    m.eval()
    return m


def make_cgts(device, layer_idx=6, keep_ratio=0.7):
    """Baseline model with CGTS forward (physically shorter token sequence)."""
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    m.eval()
    k = max(1, int(196 * keep_ratio))
    proto = torch.randn(D, device=device)  # dummy prototype (structure only)

    def forward_cgts(x):
        B = x.size(0)
        feat = m.patch_embed(x)
        feat = torch.cat((m.cls_token.expand(B, -1, -1), feat), dim=1)
        feat = feat + m.pos_embed
        feat = m.pos_drop(feat)
        for l, blk in enumerate(m.blocks):
            if l == layer_idx:
                patch = feat[:, 1:]
                scores = (patch * proto).sum(-1)
                top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
                b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
                feat = torch.cat([feat[:, :1], patch[b_idx, top_idx]], dim=1)
            feat = feat + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(feat))))
            feat = feat + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(feat))))
        return m.head(m.norm(feat)[:, 0])

    return forward_cgts


def make_mlp_pruned_cgts(device, mlp_keep=0.7, layer_idx=6, token_keep=0.7):
    """Structurally compact MLP (reduced d_ff) + CGTS token pruning."""
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    # Structurally reduce each MLP block
    d_ff_new = max(1, int(D_FF * mlp_keep))
    for blk in m.blocks:
        old_fc1 = blk.mlp.fc1
        old_fc2 = blk.mlp.fc2
        new_fc1 = nn.Linear(D, d_ff_new, bias=True).to(device)
        new_fc2 = nn.Linear(d_ff_new, D, bias=True).to(device)
        blk.mlp.fc1 = new_fc1
        blk.mlp.fc2 = new_fc2
    m.eval()

    k = max(1, int(196 * token_keep))
    proto = torch.randn(D, device=device)

    def forward_fn(x):
        B = x.size(0)
        feat = m.patch_embed(x)
        feat = torch.cat((m.cls_token.expand(B, -1, -1), feat), dim=1)
        feat = feat + m.pos_embed
        feat = m.pos_drop(feat)
        for l, blk in enumerate(m.blocks):
            if l == layer_idx:
                patch = feat[:, 1:]
                scores = (patch * proto).sum(-1)
                top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
                b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
                feat = torch.cat([feat[:, :1], patch[b_idx, top_idx]], dim=1)
            feat = feat + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(feat))))
            feat = feat + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(feat))))
        return m.head(m.norm(feat)[:, 0])

    return forward_fn, m


def make_dynamicvit(device, layer_idx=6, keep_ratio=0.7):
    """DynamicViT: backbone + token predictor MLP (layer_idx insertion)."""
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    predictor = nn.Sequential(
        nn.LayerNorm(D),
        nn.Linear(D, D // 4),
        nn.GELU(),
        nn.Linear(D // 4, 1),
    ).to(device)
    k = max(1, int(196 * keep_ratio))
    m.eval(); predictor.eval()

    def forward_fn(x):
        B = x.size(0)
        feat = m.patch_embed(x)
        feat = torch.cat((m.cls_token.expand(B, -1, -1), feat), dim=1)
        feat = feat + m.pos_embed
        feat = m.pos_drop(feat)
        for l, blk in enumerate(m.blocks):
            feat = feat + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(feat))))
            feat = feat + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(feat))))
            if l == layer_idx:
                patch = feat[:, 1:]
                scores = predictor(patch).squeeze(-1)
                top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
                b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
                feat = torch.cat([feat[:, :1], patch[b_idx, top_idx]], dim=1)
        return m.head(m.norm(feat)[:, 0])

    return forward_fn


def make_svite(device, layer_idx=6, keep_ratio=0.7):
    """SViTE: static token mask (same top-k indices for all images in batch)."""
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    k = max(1, int(196 * keep_ratio))
    # Fixed random mask (simulates pre-computed class-aware token mask)
    fixed_idx = torch.randperm(196, device=device)[:k].sort().values
    m.eval()

    def forward_fn(x):
        B = x.size(0)
        feat = m.patch_embed(x)
        feat = torch.cat((m.cls_token.expand(B, -1, -1), feat), dim=1)
        feat = feat + m.pos_embed
        feat = m.pos_drop(feat)
        for l, blk in enumerate(m.blocks):
            if l == layer_idx:
                patch = feat[:, 1:]
                kept = patch[:, fixed_idx, :]
                feat = torch.cat([feat[:, :1], kept], dim=1)
            feat = feat + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(feat))))
            feat = feat + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(feat))))
        return m.head(m.norm(feat)[:, 0])

    return forward_fn


def make_zero_tprune(device, layer_idx=6, keep_ratio=0.7, n_iter=3):
    """Zero-TPrune: WPR scoring on live attention graph, no prototype needed."""
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100, pretrained=False).to(device)
    m.eval()
    k = max(1, int(196 * keep_ratio))

    def forward_fn(x):
        B = x.size(0)
        feat = m.patch_embed(x)
        feat = torch.cat((m.cls_token.expand(B, -1, -1), feat), dim=1)
        feat = feat + m.pos_embed
        feat = m.pos_drop(feat)
        for l, blk in enumerate(m.blocks):
            if l == layer_idx:
                captured = {}
                handle = blk.attn.qkv.register_forward_hook(
                    lambda mod, inp, out: captured.update({'qkv': out})
                )
                feat = feat + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(feat))))
                handle.remove()
                feat = feat + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(feat))))
                attn_w  = _attn_weights_from_qkv(blk.attn, captured['qkv'])
                scores  = _wpr(attn_w, n_iter)[:, 1:]
                top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
                b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
                feat    = torch.cat([feat[:, :1], feat[:, 1:][b_idx, top_idx]], dim=1)
            else:
                feat = feat + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(feat))))
                feat = feat + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(feat))))
        return m.head(m.norm(feat)[:, 0])

    return forward_fn, m


def make_xpruner(device, keep_ratio=0.7):
    """X-Pruner: soft head gating — same wall-clock as unpruned (no structural change)."""
    m = XPrunerDeiT(
        model_name='deit_tiny_patch16_224',
        num_classes=100, pretrained=False, k=10.0,
        enable_mlp_pruning=False, enable_token_pruning=False,
    ).to(device)
    m.eval()
    # Use subset mode (pre-collapsed gates) for fair single-pass timing
    fake_classes = list(range(10))
    m.prepare_subset_inference(fake_classes)

    def forward_fn(x):
        logits, _ = m(x)
        return logits

    return forward_fn


# ── Memory accounting ─────────────────────────────────────────────────────────

def param_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',     type=int,   default=0)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 64])
    parser.add_argument('--n-iter',     type=int,   default=200)
    parser.add_argument('--output-dir', type=str,   default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 80)
    print('Efficiency Evaluation: Memory + Latency')
    print(f'Device: {device}')
    print('=' * 80)

    # ── 1. Theoretical MACs ───────────────────────────────────────────────────

    print('\n── Theoretical MACs (GMACs per image) ──')

    mac_configs = {
        'Unpruned':              (token_schedule(),                       1.0),
        'CGTS only (keep=0.7)':  (token_schedule(cgts_layer=6, keep_ratio=0.7), 1.0),
        'CGTS only (keep=0.5)':  (token_schedule(cgts_layer=6, keep_ratio=0.5), 1.0),
        'MLP30%+CGTS0.7 (ours)': (token_schedule(cgts_layer=6, keep_ratio=0.7), 0.7),
        'MLP50%+CGTS0.5 (ours)': (token_schedule(cgts_layer=6, keep_ratio=0.5), 0.5),
        'DynamicViT (keep=0.7)': (token_schedule(cgts_layer=6, keep_ratio=0.7), 1.0),
        'SViTE (keep=0.7)':      (token_schedule(cgts_layer=6, keep_ratio=0.7), 1.0),
        'X-Pruner (keep=0.7)':   (token_schedule(),                       1.0),  # soft only
    }

    mac_results = {}
    unpruned_macs = total_macs(token_schedule(), 1.0)

    print(f'\n  {"Method":<30}  {"GMACs":>8}  {"vs unpruned":>12}')
    print('  ' + '-' * 56)
    for name, (sched, mlp_keep) in mac_configs.items():
        macs = total_macs(sched, mlp_keep)
        ratio = macs / unpruned_macs
        print(f'  {name:<30}  {macs:>8.3f}  {ratio:>11.1%}')
        mac_results[name] = {'gmacs': round(macs, 4), 'ratio': round(ratio, 4)}

    # ── 2. Parameter memory ───────────────────────────────────────────────────

    print('\n── Parameter Memory ──')
    baseline_model = make_baseline(device)
    baseline_mb = param_mb(baseline_model)
    baseline_params = count_params(baseline_model)
    del baseline_model

    _, mlp30_model = make_mlp_pruned_cgts(device, mlp_keep=0.7)
    mlp30_mb = param_mb(mlp30_model)
    mlp30_params = count_params(mlp30_model)
    del mlp30_model

    _, mlp50_model = make_mlp_pruned_cgts(device, mlp_keep=0.5, token_keep=0.5)
    mlp50_mb = param_mb(mlp50_model)
    mlp50_params = count_params(mlp50_model)
    del mlp50_model

    # Per-subset overhead
    proto_bytes     = D * 4                                  # 768 B
    xpruner_bytes   = N_HEADS * 100 * 4 * N_LAYERS          # 14,400 B
    dynamicvit_mb   = baseline_mb                            # full checkpoint per subset

    mem_table = [
        ('Unpruned',               baseline_mb, baseline_params, 0, 'N/A (no sharing)'),
        ('CGTS (ours)',             baseline_mb, baseline_params, proto_bytes, f'{proto_bytes} B → {baseline_mb:.2f} MB + {proto_bytes/1024:.2f} KB/subset'),
        ('MLP30%+CGTS0.7 (ours)',  mlp30_mb,    mlp30_params,    proto_bytes, f'{proto_bytes} B → {mlp30_mb:.2f} MB + {proto_bytes/1024:.2f} KB/subset'),
        ('MLP50%+CGTS0.5 (ours)',  mlp50_mb,    mlp50_params,    proto_bytes, f'{proto_bytes} B → {mlp50_mb:.2f} MB + {proto_bytes/1024:.2f} KB/subset'),
        ('DynamicViT',             baseline_mb, baseline_params, int(dynamicvit_mb*1e6), f'~{dynamicvit_mb:.1f} MB/subset (no sharing)'),
        ('SViTE',                  baseline_mb, baseline_params, int(dynamicvit_mb*1e6), f'~{dynamicvit_mb:.1f} MB/subset (no sharing)'),
        ('X-Pruner',               baseline_mb, baseline_params, xpruner_bytes, f'{xpruner_bytes/1024:.1f} KB → {baseline_mb:.2f} MB + {xpruner_bytes/1024:.1f} KB/subset'),
    ]

    print(f'\n  {"Method":<28}  {"Model (MB)":>10}  {"Params":>10}  {"Per-subset":>14}')
    print('  ' + '-' * 70)
    for name, mb, params, overhead, _ in mem_table:
        overhead_str = f'{overhead/1024:.1f} KB' if overhead < 1e6 else f'{overhead/1e6:.1f} MB'
        print(f'  {name:<28}  {mb:>10.2f}  {params:>10,}  {overhead_str:>14}')

    # ── 3. Actual latency ─────────────────────────────────────────────────────

    print('\n── Actual Inference Latency ──')
    latency_results = {}

    methods_latency = [
        ('Unpruned',               lambda: (make_baseline(device), None)),
        ('CGTS keep=0.7',          lambda: (make_cgts(device, layer_idx=6, keep_ratio=0.7), None)),
        ('CGTS keep=0.5',          lambda: (make_cgts(device, layer_idx=6, keep_ratio=0.5), None)),
        ('Zero-TPrune keep=0.7',   lambda: make_zero_tprune(device, layer_idx=6, keep_ratio=0.7)),
        ('Zero-TPrune keep=0.5',   lambda: make_zero_tprune(device, layer_idx=6, keep_ratio=0.5)),
        ('MLP30%+CGTS0.7 (ours)',  lambda: make_mlp_pruned_cgts(device, mlp_keep=0.7, layer_idx=6, token_keep=0.7)),
        ('MLP50%+CGTS0.5 (ours)',  lambda: make_mlp_pruned_cgts(device, mlp_keep=0.5, layer_idx=6, token_keep=0.5)),
        ('DynamicViT keep=0.7',    lambda: (make_dynamicvit(device, layer_idx=6, keep_ratio=0.7), None)),
        ('SViTE keep=0.7',         lambda: (make_svite(device, layer_idx=6, keep_ratio=0.7), None)),
        ('X-Pruner keep=0.7',      lambda: (make_xpruner(device, keep_ratio=0.7), None)),
    ]

    # Print header
    bs_header = '  '.join(f'B={bs:>3} (ms)' for bs in args.batch_sizes)
    print(f'\n  {"Method":<28}  {bs_header}  {"B=1 speedup":>12}')
    print('  ' + '-' * (30 + 16 * len(args.batch_sizes) + 14))

    baseline_lat = {}
    for name, builder in methods_latency:
        fn_or_model, _ = builder()
        if callable(fn_or_model) and not isinstance(fn_or_model, nn.Module):
            fn = fn_or_model
        else:
            model = fn_or_model
            fn = model

        row_lats = []
        for bs in args.batch_sizes:
            lat = benchmark(fn, bs, device, n_iter=args.n_iter)
            row_lats.append(lat)
            if bs == 1 and name == 'Unpruned':
                baseline_lat[1] = lat
            if bs == 64 and name == 'Unpruned':
                baseline_lat[64] = lat

        speedup_b1 = baseline_lat.get(1, row_lats[0]) / row_lats[0] if row_lats else 1.0
        lat_str = '  '.join(f'{l:>11.2f}' for l in row_lats)
        print(f'  {name:<28}  {lat_str}  {speedup_b1:>11.2f}x', flush=True)

        latency_results[name] = {bs: round(row_lats[i], 3)
                                 for i, bs in enumerate(args.batch_sizes)}
        latency_results[name]['speedup_b1'] = round(speedup_b1, 3)

        # Clean up
        if isinstance(fn_or_model, nn.Module):
            del fn_or_model
        torch.cuda.empty_cache()

    # ── 4. Save ───────────────────────────────────────────────────────────────
    results = {
        'macs': mac_results,
        'memory': {
            name: {'model_mb': round(mb, 3), 'params': params,
                   'per_subset_bytes': overhead}
            for name, mb, params, overhead, _ in mem_table
        },
        'latency_ms': latency_results,
        'batch_sizes': args.batch_sizes,
        'device': str(device),
    }
    save_path = os.path.join(args.output_dir, 'efficiency.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Saved: {save_path}')

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('SUMMARY TABLE (for paper)')
    print('=' * 80)
    print(f'\n  {"Method":<28}  {"GMACs":>7}  {"Model MB":>9}  {"Per-subset":>12}  '
          f'{"B=1 (ms)":>9}  {"B=1 speedup":>12}')
    print('  ' + '-' * 85)
    for name, (sched, mlp_keep) in mac_configs.items():
        mb  = next((m for n, m, _, _, _ in mem_table if n in name or name in n), baseline_mb)
        overhead = next((o for n, _, _, o, _ in mem_table if n in name or name in n), 0)
        macs = mac_results[name]['gmacs']
        lat_b1 = latency_results.get(name, {}).get(1, 0)
        speedup = latency_results.get(name, {}).get('speedup_b1', 1.0)
        oh_str = f'{overhead/1024:.1f} KB' if overhead < 1e6 else f'{overhead/1e6:.1f} MB'
        print(f'  {name:<28}  {macs:>7.3f}  {mb:>9.2f}  {oh_str:>12}  '
              f'{lat_b1:>9.2f}  {speedup:>11.2f}x')


if __name__ == '__main__':
    main()
