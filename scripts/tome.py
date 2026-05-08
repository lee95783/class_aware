"""
Token Merging (ToMe) for Vision Transformers.

Based on:
  "Token Merging: Your ViT But Faster" (Bolya et al., ICLR 2023)
  https://github.com/facebookresearch/ToMe

This module provides a training-free drop-in acceleration for any timm ViT/DeiT
model.  It works by merging redundant tokens inside every transformer block using
a lightweight bipartite soft-matching algorithm on the existing self-attention
keys – no extra parameters, no gating network, no two-pass inference.

Usage:
    import timm
    from scripts.tome import tome_patch, tome_unpatch

    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    tome_patch(model, r=8)        # merge 8 token-pairs per block
    out = model(images)           # inference is now faster
    tome_unpatch(model)           # restore original model if needed
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core matching / merging primitives
# ---------------------------------------------------------------------------

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Partitions tokens into two disjoint sets A and B, finds the top-r most
    similar (A, B) pairs by cosine similarity, and returns:
      merge  – a callable that merges src tokens into dst tokens
      unmerge – a callable that expands merged tokens back (for skip connections)

    Args:
        metric: (B, N, C) similarity keys (e.g. self-attention keys).
        r: number of token-pairs to merge.
        class_token: if True, the first token ([CLS]) is never merged.

    Returns:
        merge(x)  : (B, N, C) -> (B, N-r, C)
        unmerge(x): (B, N-r, C) -> (B, N, C)
    """
    protected = 1 if class_token else 0

    B, N, C = metric.shape

    with torch.no_grad():
        # --- guarantee we never try to merge more tokens than exist ----------
        t = N - protected          # mergeable tokens
        r = min(r, t // 2)         # can merge at most half of the tokens
        if r <= 0:
            return _identity_merge, _identity_merge

        # Separate protected (CLS) token from the rest
        if protected:
            metric_cls = metric[:, :protected, :]
            metric_patch = metric[:, protected:, :]
        else:
            metric_patch = metric

        # Alternate partition: A = even indices, B = odd indices
        # This is the simplest unbiased partition used in the original paper.
        a_idx = torch.arange(0, metric_patch.shape[1], 2, device=metric.device)
        b_idx = torch.arange(1, metric_patch.shape[1], 2, device=metric.device)

        a = metric_patch[:, a_idx]  # (B, Na, C)
        b = metric_patch[:, b_idx]  # (B, Nb, C)

        # Cosine similarity between every a and every b token
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
        scores = a_norm @ b_norm.transpose(-1, -2)  # (B, Na, Nb)

        # For each token in A, find its best match in B
        node_max, node_idx = scores.max(dim=-1)  # (B, Na)

        # Pick the top-r most similar (a, b) pairs
        edge_idx = node_max.argsort(dim=-1, descending=True)[:, :r]  # (B, r)

    # Build index maps -------------------------------------------------
    # unm_a: un-merged tokens from set A (those NOT in the top-r)
    # src_a: merged tokens from set A (top-r, will be merged INTO their B match)
    # dst_b: all tokens from set B (destinations of merging)

    Na = a_idx.shape[0]
    Nb = b_idx.shape[0]

    # Gather merged A->B pairs
    _batch = torch.arange(B, device=metric.device)[:, None]

    def _make_merge(edge_idx=edge_idx, node_idx=node_idx, Na=Na, Nb=Nb,
                    protected=protected, B=B, N=N, r=r, _batch=_batch,
                    a_idx=a_idx, b_idx=b_idx):
        """Returns merge and unmerge callables closed over the indices."""

        # Indices in the *original* token order (within patch tokens)
        merged_a_local = edge_idx  # (B, r) – which A-tokens are merged (local A idx)
        dst_b_local = torch.gather(node_idx, 1, edge_idx)  # (B, r) – their matched B (local B idx)

        # Mask over set-A for unmerged tokens
        unmerged_mask = torch.ones(B, Na, dtype=torch.bool, device=metric.device)
        unmerged_mask.scatter_(1, edge_idx, False)

        def merge(x: torch.Tensor) -> torch.Tensor:
            """(B, N, C) -> (B, N-r, C): merge r tokens."""
            if protected:
                cls = x[:, :protected, :]
                patch = x[:, protected:, :]
            else:
                patch = x

            a_tok = patch[:, a_idx]  # (B, Na, C)
            b_tok = patch[:, b_idx]  # (B, Nb, C)

            # Average merged A tokens into their matched B token
            src = a_tok[_batch, merged_a_local]  # (B, r, C)
            # Use scatter to average: first add src to matched dst, then divide by 2
            dst = b_tok.clone()
            dst.scatter_add_(
                1,
                dst_b_local.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
                src,
            )
            # Normalise by count (each matched dst now has weight 2)
            counts = b_tok.new_ones(B, Nb, 1)
            counts.scatter_add_(
                1,
                dst_b_local.unsqueeze(-1),
                b_tok.new_ones(B, r, 1),
            )
            dst = dst / counts

            # Unmerged A tokens
            unm = a_tok[unmerged_mask].view(B, Na - r, x.shape[-1])

            parts = [dst, unm]
            if protected:
                parts = [cls] + parts

            return torch.cat(parts, dim=1)  # (B, protected + Nb + Na - r, C)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            """(B, N-r, C) -> (B, N, C): expand merged tokens back for residuals."""
            C_feat = x.shape[-1]

            if protected:
                cls = x[:, :protected, :]
                rest = x[:, protected:, :]
            else:
                rest = x

            # rest = [dst (Nb), unm (Na - r)]
            dst = rest[:, :Nb, :]
            unm = rest[:, Nb:, :]

            # Reconstruct set B: dst already contains merged values
            b_out = dst

            # Reconstruct set A: unmerged stay as-is, merged get dst value
            a_out = x.new_zeros(B, Na, C_feat)
            a_out[unmerged_mask] = unm.reshape(-1, C_feat)
            # Merged A tokens get the value of their destination B token
            merged_dst_vals = dst[_batch, dst_b_local]  # (B, r, C)
            a_out[_batch, merged_a_local] = merged_dst_vals

            # Interleave back to original order: A at even, B at odd
            patch_out = x.new_zeros(B, Na + Nb, C_feat)
            patch_out[:, a_idx] = a_out
            patch_out[:, b_idx] = b_out

            if protected:
                return torch.cat([cls, patch_out], dim=1)
            else:
                return patch_out

        return merge, unmerge

    return _make_merge()


def _identity_merge(x: torch.Tensor) -> torch.Tensor:
    return x


# ---------------------------------------------------------------------------
# Model patching utilities
# ---------------------------------------------------------------------------

class ToMeBlock(nn.Module):
    """
    Drop-in replacement for a timm ViT Block that applies Token Merging.

    Wraps the original block, inserting merge/unmerge around the attention +
    MLP sub-blocks so that the residual stream stays compatible.
    """

    def __init__(self, block: nn.Module, r: int = 8, class_token: bool = True):
        super().__init__()
        self._block = block
        self.r = r
        self.class_token = class_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blk = self._block

        # ---------- Attention ----------
        x_norm = blk.norm1(x)

        # Obtain keys for similarity metric
        B, N, C = x_norm.shape
        H = blk.attn.num_heads
        Dh = C // H
        qkv = blk.attn.qkv(x_norm).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        metric = qkv[1].mean(dim=1)  # Average keys across heads: (B, N, C//H * H) -> (B, N, Dh)

        # Compute merge/unmerge maps
        merge, unmerge = bipartite_soft_matching(metric, self.r, class_token=self.class_token)

        # Standard attention path (through original block's attention)
        attn_out = blk.attn(x_norm)
        # Handle timm versions that return tuple
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        x = x + blk.drop_path1(attn_out)

        # Merge tokens AFTER attention, BEFORE MLP
        x = merge(x)

        # ---------- MLP ----------
        x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

        return x


class ToMeBlockCompact(nn.Module):
    """
    A more aggressive variant that merges tokens BEFORE attention, giving
    speedup on both the O(N²) attention and O(N) MLP.

    Trade-off: slightly less accurate metric (computed on fewer tokens in
    later layers), but gives maximum latency reduction.
    """

    def __init__(self, block: nn.Module, r: int = 8, class_token: bool = True):
        super().__init__()
        self._block = block
        self.r = r
        self.class_token = class_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blk = self._block

        # Compute similarity metric from current token embeddings
        x_norm = blk.norm1(x)
        B, N, C = x_norm.shape

        if N <= 2:
            # Too few tokens to merge, just pass through
            attn_out = blk.attn(x_norm)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = x + blk.drop_path1(attn_out)
            x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))
            return x

        # Use token embeddings themselves as the metric (fast, no extra computation)
        merge, _ = bipartite_soft_matching(x_norm, self.r, class_token=self.class_token)

        # Merge BEFORE attention → both attention (O(N²)) and MLP (O(N)) benefit
        x = merge(x)

        # Standard block forward on reduced tokens
        x_norm = blk.norm1(x)
        attn_out = blk.attn(x_norm)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + blk.drop_path1(attn_out)
        x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

        return x


def tome_patch(
    model: nn.Module,
    r: int = 8,
    class_token: bool = True,
    start_layer: int = 0,
    mode: str = "compact",
):
    """
    Patches a timm ViT/DeiT model in-place to use Token Merging.

    Args:
        model: timm ViT model with .blocks attribute.
        r: number of token-pairs to merge per layer.
        class_token: whether the model uses a CLS token (True for ViT/DeiT).
        start_layer: first block to apply merging (0 = all blocks).
        mode: "compact" (merge before attention, maximum speedup) or
              "standard" (merge between attention and MLP, slightly better accuracy).
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model must have .blocks attribute (timm ViT/DeiT).")

    BlockClass = ToMeBlockCompact if mode == "compact" else ToMeBlock

    new_blocks = []
    for i, blk in enumerate(model.blocks):
        if i >= start_layer:
            new_blocks.append(BlockClass(blk, r=r, class_token=class_token))
        else:
            new_blocks.append(blk)

    model.blocks = nn.Sequential(*new_blocks)
    model._tome_info = {"r": r, "mode": mode, "start_layer": start_layer}


def tome_unpatch(model: nn.Module):
    """
    Removes Token Merging from a patched model, restoring original blocks.
    """
    if not hasattr(model, "blocks"):
        return

    new_blocks = []
    for blk in model.blocks:
        if isinstance(blk, (ToMeBlock, ToMeBlockCompact)):
            new_blocks.append(blk._block)
        else:
            new_blocks.append(blk)

    model.blocks = nn.Sequential(*new_blocks)
    if hasattr(model, "_tome_info"):
        del model._tome_info
