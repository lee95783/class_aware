#!/usr/bin/env python3
"""
Efficiency evaluation for Zero-TPrune vs CGTS vs unpruned.

Measures:
  - GMACs per image (theoretical, sequence-length-aware)
  - Parameter memory (MB) and per-subset storage
  - Actual inference latency at B=1 and B=64
  - Peak GPU memory during forward pass

Zero-TPrune overhead vs CGTS:
  - Same token count after pruning (same GMACs in layers 7-11)
  - Extra at layer 6: recompute QKV-based attention matrix [B,H,N,N] + n_iter WPR steps
  - WPR: n_iter × bmm([B,N,N], [B,N,1]) → [B,N] — each is N×N multiply-adds

Usage:
    python scripts/eval_zero_tprune_efficiency.py --device 0
"""

import os, sys, time, json, argparse
import torch
import torch.nn as nn
import timm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.eval_e07_zero_tprune import _wpr, _attn_weights_from_qkv

# DeiT-Tiny dimensions
D        = 192
N_FULL   = 197      # 196 patches + 1 CLS
N_HEADS  = 3
N_LAYERS = 12
D_FF     = D * 4


# ── MAC computation ───────────────────────────────────────────────────────────

def attention_macs(n_tokens, d=D, n_heads=N_HEADS):
    qkv  = 3 * n_tokens * d * d
    attn = 2 * n_tokens * n_tokens * d
    proj = n_tokens * d * d
    return qkv + attn + proj


def mlp_macs(n_tokens, d=D, d_ff=D_FF):
    return 2 * n_tokens * d * d_ff


def extra_zero_tprune_macs(n_tokens=196, d=D, n_heads=N_HEADS, n_iter=3):
    """
    Extra MACs at the pruning layer for Zero-TPrune:
      1. Recompute attention from captured QKV: Q @ K^T  [B,H,N,N]
         = n_heads * N * N * (d // n_heads)  =  N * N * d
      2. WPR: n_iter * bmm(A^T, r) where A=[B,N,N], r=[B,N,1]
         = n_iter * N * N   (each element: N multiply-adds)
    """
    recompute_attn = n_tokens * n_tokens * d          # A=Q@K^T (all heads together)
    wpr_iter       = n_iter * n_tokens * n_tokens     # each power iteration
    return recompute_attn + wpr_iter


def total_macs_cgts(keep_ratio=0.7):
    """GMACs for CGTS at given keep ratio (token drop at layer 6)."""
    total = 0
    n = N_FULL
    for l in range(N_LAYERS):
        if l == 6:
            n = max(1, int((n - 1) * keep_ratio) + 1)
        total += attention_macs(n) + mlp_macs(n)
    return total / 1e9


def total_macs_zero_tprune(keep_ratio=0.7, n_iter=3):
    """GMACs for Zero-TPrune: same sequence reduction + extra WPR overhead."""
    base = total_macs_cgts(keep_ratio)
    extra = extra_zero_tprune_macs(n_tokens=196, n_iter=n_iter) / 1e9
    return base + extra


# ── Latency benchmark ─────────────────────────────────────────────────────────

def benchmark(fn, batch_size, device, n_iter=200, n_warmup=20):
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(n_warmup):
        with torch.no_grad():
            fn(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        if device.type == 'cuda':
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                fn(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                fn(x)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    trimmed = times[n_iter // 10: -n_iter // 10]
    return sum(trimmed) / len(trimmed)


def peak_gpu_memory_mb(fn, batch_size, device):
    """Peak GPU memory (MB) during a single forward pass."""
    if device.type != 'cuda':
        return 0.0
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    with torch.no_grad():
        fn(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1e6


# ── Model builders ────────────────────────────────────────────────────────────

def make_unpruned(device):
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100,
                          pretrained=False).to(device).eval()
    return m


def make_cgts(device, layer_idx=6, keep_ratio=0.7):
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100,
                          pretrained=False).to(device).eval()
    k = max(1, int(196 * keep_ratio))
    proto = torch.randn(D, device=device)

    def forward_fn(x):
        B = x.size(0)
        feat = m.patch_embed(x)
        feat = torch.cat((m.cls_token.expand(B, -1, -1), feat), dim=1)
        feat = feat + m.pos_embed
        feat = m.pos_drop(feat)
        for l, blk in enumerate(m.blocks):
            if l == layer_idx:
                patch   = feat[:, 1:]
                scores  = (patch * proto).sum(-1)
                top_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
                b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
                feat    = torch.cat([feat[:, :1], patch[b_idx, top_idx]], dim=1)
            feat = feat + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(feat))))
            feat = feat + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(feat))))
        return m.head(m.norm(feat)[:, 0])

    return forward_fn, m


def make_zero_tprune(device, layer_idx=6, keep_ratio=0.7, n_iter=3):
    m = timm.create_model('deit_tiny_patch16_224', num_classes=100,
                          pretrained=False).to(device).eval()
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',     type=int, default=0)
    parser.add_argument('--n-iter',     type=int, default=200)
    parser.add_argument('--n-warmup',   type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='results/paper')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 75)
    print('Zero-TPrune vs CGTS vs Unpruned — Efficiency')
    print(f'Device: {device}')
    print('=' * 75)

    # ── Theoretical MACs ──────────────────────────────────────────────────────
    unpruned_macs  = total_macs_cgts(keep_ratio=1.0)      # no CGTS = all 197 tokens
    cgts07_macs    = total_macs_cgts(keep_ratio=0.7)
    cgts05_macs    = total_macs_cgts(keep_ratio=0.5)
    ztp07_macs     = total_macs_zero_tprune(keep_ratio=0.7)
    ztp05_macs     = total_macs_zero_tprune(keep_ratio=0.5)
    extra07        = extra_zero_tprune_macs(196) / 1e9
    extra05        = extra_zero_tprune_macs(196) / 1e9   # same (pruning happens after)

    print('\n── Theoretical GMACs per image ──')
    print(f'\n  {"Method":<35}  {"GMACs":>7}  {"vs unpruned":>12}  {"vs CGTS":>9}')
    print('  ' + '-' * 70)
    rows = [
        ('Unpruned',              unpruned_macs,  None,         None),
        ('CGTS (keep=0.7)',       cgts07_macs,    unpruned_macs, None),
        ('CGTS (keep=0.5)',       cgts05_macs,    unpruned_macs, None),
        ('Zero-TPrune (keep=0.7)', ztp07_macs,   unpruned_macs, cgts07_macs),
        ('Zero-TPrune (keep=0.5)', ztp05_macs,   unpruned_macs, cgts05_macs),
    ]
    for name, macs, base, cgts_ref in rows:
        vs_up  = f'{macs/base:.1%}'   if base     else  '  baseline'
        vs_cg  = f'+{(macs-cgts_ref)/cgts_ref:.1%} overhead' if cgts_ref else '      —'
        print(f'  {name:<35}  {macs:>7.3f}  {vs_up:>12}  {vs_cg:>20}')

    print(f'\n  Extra WPR overhead per image: {extra07*1e3:.2f} MMACs '
          f'({extra07/unpruned_macs:.2%} of unpruned)')

    # ── Parameter memory ──────────────────────────────────────────────────────
    print('\n── Memory ──')
    baseline_m = make_unpruned(device)
    model_mb   = sum(p.numel() * p.element_size() for p in baseline_m.parameters()) / 1e6
    del baseline_m

    proto_bytes = D * 4   # 192 floats × 4 B = 768 B (CGTS)

    print(f'\n  {"Method":<35}  {"Model (MB)":>10}  {"Per-subset":>14}')
    print('  ' + '-' * 65)
    print(f'  {"Unpruned":<35}  {model_mb:>10.2f}  {"N/A":>14}')
    print(f'  {"CGTS (any keep ratio)":<35}  {model_mb:>10.2f}  {"768 B":>14}  ← 0.8 KB prototype')
    print(f'  {"Zero-TPrune (any keep ratio)":<35}  {model_mb:>10.2f}  {"0 B":>14}  ← nothing (zero-shot)')

    # ── Latency ───────────────────────────────────────────────────────────────
    print('\n── Actual Inference Latency ──')
    print(f'\n  {"Method":<35}  {"B=1 (ms)":>9}  {"B=64 (ms)":>10}  '
          f'{"img/s":>7}  {"vs unpruned":>12}')
    print('  ' + '-' * 80)

    configs = [
        ('Unpruned',               lambda: make_unpruned(device)),
        ('CGTS (keep=0.7)',        lambda: make_cgts(device, keep_ratio=0.7)[0]),
        ('CGTS (keep=0.5)',        lambda: make_cgts(device, keep_ratio=0.5)[0]),
        ('Zero-TPrune (keep=0.7)', lambda: make_zero_tprune(device, keep_ratio=0.7)[0]),
        ('Zero-TPrune (keep=0.5)', lambda: make_zero_tprune(device, keep_ratio=0.5)[0]),
    ]

    results = {}
    unpruned_b64 = None

    for name, builder in configs:
        fn = builder()
        lat1  = benchmark(fn, 1,  device, n_iter=args.n_iter, n_warmup=args.n_warmup)
        lat64 = benchmark(fn, 64, device, n_iter=args.n_iter, n_warmup=args.n_warmup)
        mem   = peak_gpu_memory_mb(fn, 64, device)
        imgs  = round(64 / (lat64 / 1000))
        if unpruned_b64 is None:
            unpruned_b64 = lat64
        speedup = unpruned_b64 / lat64
        print(f'  {name:<35}  {lat1:>9.2f}  {lat64:>10.2f}  {imgs:>7}  {speedup:>11.2f}x',
              flush=True)
        results[name] = {
            'lat_b1_ms':  round(lat1, 2),
            'lat_b64_ms': round(lat64, 2),
            'imgs_per_sec': imgs,
            'speedup_vs_unpruned': round(speedup, 3),
            'peak_gpu_mem_mb': round(mem, 1),
        }
        torch.cuda.empty_cache()

    # ── Peak GPU memory ───────────────────────────────────────────────────────
    print('\n── Peak GPU Memory at B=64 ──')
    print(f'\n  {"Method":<35}  {"Peak mem (MB)":>14}')
    print('  ' + '-' * 55)
    for name, r in results.items():
        print(f'  {name:<35}  {r["peak_gpu_mem_mb"]:>14.1f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 75)
    print('SUMMARY')
    print('=' * 75)
    print(f'\n  {"Method":<35}  {"GMACs":>7}  {"MB":>6}  {"Per-sub":>8}  '
          f'{"B=64ms":>8}  {"img/s":>7}  {"Overhead vs CGTS":>18}')
    print('  ' + '-' * 105)

    mac_map = {
        'Unpruned':               unpruned_macs,
        'CGTS (keep=0.7)':        cgts07_macs,
        'CGTS (keep=0.5)':        cgts05_macs,
        'Zero-TPrune (keep=0.7)': ztp07_macs,
        'Zero-TPrune (keep=0.5)': ztp05_macs,
    }
    sub_map = {
        'Unpruned':               '—',
        'CGTS (keep=0.7)':        '0.8 KB',
        'CGTS (keep=0.5)':        '0.8 KB',
        'Zero-TPrune (keep=0.7)': '0 B',
        'Zero-TPrune (keep=0.5)': '0 B',
    }
    cgts_lat = {
        'CGTS (keep=0.7)':        results['CGTS (keep=0.7)']['lat_b64_ms'],
        'CGTS (keep=0.5)':        results['CGTS (keep=0.5)']['lat_b64_ms'],
        'Zero-TPrune (keep=0.7)': results['CGTS (keep=0.7)']['lat_b64_ms'],
        'Zero-TPrune (keep=0.5)': results['CGTS (keep=0.5)']['lat_b64_ms'],
    }

    for name, r in results.items():
        macs = mac_map[name]
        sub  = sub_map[name]
        if name in cgts_lat:
            ref = cgts_lat[name]
            overhead = (r['lat_b64_ms'] - ref) / ref
            oh_str = f'{overhead:+.1%}'
        else:
            oh_str = '—'
        print(f'  {name:<35}  {macs:>7.3f}  {model_mb:>6.2f}  {sub:>8}  '
              f'{r["lat_b64_ms"]:>8.2f}  {r["imgs_per_sec"]:>7}  {oh_str:>18}')

    # Save
    out = {
        'macs': {k: round(v, 4) for k, v in mac_map.items()},
        'model_mb': round(model_mb, 3),
        'per_subset': sub_map,
        'latency': results,
    }
    save_path = os.path.join(args.output_dir, 'zero_tprune_efficiency.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {save_path}')


if __name__ == '__main__':
    main()
