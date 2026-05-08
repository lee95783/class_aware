import torch
import torch.nn as nn


def prune_vit_attention_heads(model, heads_to_prune):
    """
    Zeroes out attention heads in timm ViT/DeiT-style models.

    Args:
        model (nn.Module): Model with a .blocks list containing .attn modules.
        heads_to_prune (Iterable[Tuple[int, int]]): (layer_idx, head_idx) pairs.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not expose .blocks; pruning expects a ViT/DeiT-style model.")

    heads_by_layer = {}
    for layer_idx, head_idx in heads_to_prune:
        heads_by_layer.setdefault(layer_idx, []).append(head_idx)

    with torch.no_grad():
        for layer_idx, head_indices in heads_by_layer.items():
            if layer_idx < 0 or layer_idx >= len(model.blocks):
                raise ValueError(f"Layer index {layer_idx} out of range for model.blocks.")
            attn = model.blocks[layer_idx].attn
            num_heads = attn.num_heads
            embed_dim = attn.qkv.in_features
            head_dim = embed_dim // num_heads

            for head_idx in head_indices:
                if head_idx < 0 or head_idx >= num_heads:
                    raise ValueError(
                        f"Head index {head_idx} out of range for layer {layer_idx}."
                    )

                for block_offset in (0, embed_dim, 2 * embed_dim):
                    start = block_offset + head_idx * head_dim
                    end = block_offset + (head_idx + 1) * head_dim
                    attn.qkv.weight[start:end, :].zero_()
                    if attn.qkv.bias is not None:
                        attn.qkv.bias[start:end].zero_()

                proj_start = head_idx * head_dim
                proj_end = (head_idx + 1) * head_dim
                attn.proj.weight[:, proj_start:proj_end].zero_()


def structured_prune_vit_heads(model, keep_heads_per_layer):
    """
    Backward-compatible structured pruning entrypoint.

    Args:
        model (nn.Module): ViT/DeiT-style model exposing `model.blocks`.
        keep_heads_per_layer (dict[int, list[int]]): head indices to keep per layer.

    Notes:
        This implementation preserves tensor shapes and performs deterministic
        hard pruning by zeroing pruned heads. It is compatible with legacy
        scripts that previously called `structured_prune_vit_heads(...)`.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not expose .blocks; expected ViT/DeiT-style model.")

    heads_to_prune = []
    for layer_idx, blk in enumerate(model.blocks):
        attn = blk.attn
        num_heads = int(attn.num_heads)
        keep = keep_heads_per_layer.get(layer_idx, list(range(num_heads)))
        keep_set = {int(h) for h in keep}

        for head_idx in range(num_heads):
            if head_idx not in keep_set:
                heads_to_prune.append((layer_idx, head_idx))

    prune_vit_attention_heads(model, heads_to_prune)
    return heads_to_prune


class CompactedViTAttention(nn.Module):
    """
    Structural head-pruned attention for timm ViT/DeiT blocks.
    Keeps model embed dim unchanged while reducing attention internal width.
    """

    def __init__(self, attn: nn.Module, keep_head_indices):
        super().__init__()
        if len(keep_head_indices) == 0:
            raise ValueError("Cannot compact attention with zero heads kept.")

        self._orig_num_heads = int(attn.num_heads)
        embed_dim = int(attn.qkv.in_features)
        if embed_dim % self._orig_num_heads != 0:
            raise ValueError("Embed dim must be divisible by num_heads for ViT attention.")

        self.head_dim = embed_dim // self._orig_num_heads
        self.num_heads = len(keep_head_indices)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.attn_drop = attn.attn_drop
        self.proj_drop = attn.proj_drop

        self.qkv = nn.Linear(embed_dim, 3 * self.inner_dim, bias=attn.qkv.bias is not None)
        self.proj = nn.Linear(self.inner_dim, embed_dim, bias=attn.proj.bias is not None)

        keep_head_indices = sorted(int(h) for h in keep_head_indices)
        with torch.no_grad():
            old_qkv_w = attn.qkv.weight.detach()
            old_qkv_b = attn.qkv.bias.detach() if attn.qkv.bias is not None else None
            old_proj_w = attn.proj.weight.detach()
            old_proj_b = attn.proj.bias.detach() if attn.proj.bias is not None else None

            qkv_rows = []
            qkv_bias = []
            embed = embed_dim
            for part_offset in (0, embed, 2 * embed):
                for h in keep_head_indices:
                    start = part_offset + h * self.head_dim
                    end = start + self.head_dim
                    qkv_rows.append(old_qkv_w[start:end, :])
                    if old_qkv_b is not None:
                        qkv_bias.append(old_qkv_b[start:end])

            self.qkv.weight.copy_(torch.cat(qkv_rows, dim=0))
            if old_qkv_b is not None:
                self.qkv.bias.copy_(torch.cat(qkv_bias, dim=0))

            proj_cols = []
            for h in keep_head_indices:
                start = h * self.head_dim
                end = start + self.head_dim
                proj_cols.append(old_proj_w[:, start:end])
            self.proj.weight.copy_(torch.cat(proj_cols, dim=1))
            if old_proj_b is not None:
                self.proj.bias.copy_(old_proj_b)

    def forward(self, x, **kwargs):
        bsz, n_tokens, _ = x.shape
        # Optional mask implementation if attn_mask exists could be added here,
        # but timm default ViT often passes None or an unused attn_mask.
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, self.inner_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def compact_vit_attention_heads(model, heads_to_prune, min_heads_per_layer=1):
    """
    Structurally compacts ViT attention modules by removing pruned heads and
    shrinking qkv/proj dimensions.

    Args:
        model (nn.Module): Model with .blocks[i].attn.
        heads_to_prune (Iterable[Tuple[int, int]]): (layer_idx, head_idx).
        min_heads_per_layer (int): Minimum heads to keep per layer.

    Returns:
        dict[int, list[int]]: kept head indices by layer.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not expose .blocks; compaction expects ViT/DeiT-style model.")

    heads_by_layer = {}
    for layer_idx, head_idx in heads_to_prune:
        heads_by_layer.setdefault(int(layer_idx), set()).add(int(head_idx))

    kept_heads_by_layer = {}
    for layer_idx, blk in enumerate(model.blocks):
        attn = blk.attn
        num_heads = int(attn.num_heads)
        prune_heads = heads_by_layer.get(layer_idx, set())
        keep_heads = [h for h in range(num_heads) if h not in prune_heads]

        if len(keep_heads) < min_heads_per_layer:
            raise ValueError(
                f"Layer {layer_idx} keeps {len(keep_heads)} heads; requires >= {min_heads_per_layer}."
            )

        kept_heads_by_layer[layer_idx] = keep_heads
        if len(keep_heads) == num_heads:
            continue
        blk.attn = CompactedViTAttention(attn, keep_heads)

    return kept_heads_by_layer


def compute_head_importance(saliency_matrix, class_indices=None, reduction="mean", normalize_layers=False):
    """
    Computes per-head importance from a saliency matrix.

    Args:
        saliency_matrix (Tensor): [Layers, Heads, Classes]
        class_indices (list[int] | None): Subset of classes to average over.
        reduction (str): "mean" or "sum" over classes.
        normalize_layers (bool): If True, min-max normalize scores per layer to [0, 1].

    Returns:
        Tensor: [Layers, Heads] importance scores.
    """
    if saliency_matrix.dim() != 3:
        raise ValueError("Saliency matrix must have shape [Layers, Heads, Classes].")

    if class_indices is not None:
        saliency_matrix = saliency_matrix[:, :, class_indices]

    if reduction == "mean":
        scores = saliency_matrix.mean(dim=2)
    elif reduction == "sum":
        scores = saliency_matrix.sum(dim=2)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    if normalize_layers:
        # scores is [Layers, Heads]
        min_vals = scores.min(dim=1, keepdim=True)[0]
        max_vals = scores.max(dim=1, keepdim=True)[0]
        range_vals = max_vals - min_vals
        
        # Avoid division by zero
        range_vals[range_vals == 0] = 1.0
        
        scores = (scores - min_vals) / range_vals
        
    return scores


def apply_head_masks(model, heads_to_mask):
    """
    Applies non-destructive masks to attention heads using forward hooks.

    Args:
        model (nn.Module): Model with .blocks having .attn modules.
        heads_to_mask (Iterable[Tuple[int, int]]): (layer_idx, head_idx) to mask.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not expose .blocks.")

    heads_by_layer = {}
    for layer_idx, head_idx in heads_to_mask:
        heads_by_layer.setdefault(layer_idx, []).append(head_idx)

    # Store hooks to handle for later removal if desired
    if not hasattr(model, "_head_mask_hooks"):
        model._head_mask_hooks = []

    for layer_idx, head_indices in heads_by_layer.items():
        if layer_idx < 0 or layer_idx >= len(model.blocks):
            continue
        
        attn = model.blocks[layer_idx].attn
        
        # We need to capture these values for the hook
        # num_heads = attn.num_heads
        # embed_dim = attn.qkv.in_features
        # head_dim = embed_dim // num_heads

        # Define the hook
        def _get_mask_hook(head_indices_to_zero, attn_module):
            # attn_module is the 'attn' layer. 
            # In timm, the output of the attention mechanism *before* projection 
            # is not easily accessible via a simple hook on a submodule unless we hook 
            # the forward of the specific operations.
            # However, we can hook the input to the projection layer (attn.proj).
            # The input to attn.proj is (B, N, C), where C is embed_dim.
            # It is the concatenated output of all heads.
            # We can zero out the parts corresponding to the masked heads.
            
            def hook(module, args):
                # args[0] is the input tensor to the projection layer
                x = args[0] # Shape: [Batch, Tokens, EmbedDim]
                
                B, N, C = x.shape
                num_heads = attn_module.num_heads
                head_dim = C // num_heads
                
                # Check for compatibility
                if C % num_heads != 0:
                     return # Should not happen in standard ViTs
                
                # Create a mask if we wanted to be efficient, but modifying in place is easier
                # However, tuples are immutable, args is a tuple. 
                # We can modify the tensor content since it's a view or just the tensor object.
                # BUT, modifying input in-place in a pre-hook can be dangerous if it's used elsewhere.
                # Usually safely we can modify it in place if it's the direct result of previous op.
                
                # Let's be safe and clone if needed, but in-place is necessary to affect the module input.
                # Actually, Forward Pre-hook return value:
                # "The hook can return a tuple or a single value... that will be used as the modified input."
                
                # So we should return the modified tensor.
                
                # Since we want to mask heads:
                # The layout is [head_0_features | head_1_features | ... ]
                
                # We can construct a mask tensor once, but for now let's just zero out slices.
                # To do this cleanly without modifying the original tensor in place (which might cause backward issues),
                # we clone it.
                x_mod = x.clone() 
                
                for head_idx in head_indices_to_zero:
                    start = head_idx * head_dim
                    end = (head_idx + 1) * head_dim
                    x_mod[:, :, start:end] = 0.0
                
                return x_mod

            return hook

        # Register the hook on the projection layer
        # This assumes attn.proj exists and is the final linear projection.
        hook_handle = attn.proj.register_forward_pre_hook(_get_mask_hook(head_indices, attn))
        model._head_mask_hooks.append(hook_handle)


def clear_head_masks(model):
    """
    Removes all head mask hooks registered by apply_head_masks.
    """
    if hasattr(model, "_head_mask_hooks"):
        for handle in model._head_mask_hooks:
            handle.remove()
        model._head_mask_hooks = []


def select_heads_from_saliency(
    saliency_matrix,
    num_heads,
    per_layer=False,
    class_indices=None,
    reduction="mean",
    normalize_layers=False,
    layer_sensitive=False,
    candidate_layer_fraction=0.5,
):
    """
    Selects heads to prune based on saliency matrix.

    Args:
        saliency_matrix (Tensor): [Layers, Heads, Classes]
        num_heads (int): Number of heads to prune. 
                         If per_layer=True, this is the number of heads to prune PER LAYER.
                         If per_layer=False, this is the GLOBAL number of heads to prune.
        per_layer (bool): If True, prune num_heads from each layer.
        class_indices (list[int]): Classes to aggregate over.
        reduction (str): Aggregation method ("mean", "sum").
        normalize_layers (bool): Apply layer-wise normalization.
        layer_sensitive (bool): If True, restrict pruning to candidate layers (bottom X%).
                                Ignored if per_layer=True.
        candidate_layer_fraction (float): Fraction of layers to consider as candidates.

    Returns:
        list[tuple[int, int]]: List of (layer_idx, head_idx) to prune.
    """
    scores = compute_head_importance(
        saliency_matrix,
        class_indices=class_indices,
        reduction=reduction,
        normalize_layers=normalize_layers,
    )
    num_layers, num_heads_total = scores.shape
    selected = []

    if per_layer:
        for layer_idx in range(num_layers):
            k = min(num_heads, num_heads_total)
            values, indices = torch.topk(scores[layer_idx], k=k, largest=False)
            for head_idx in indices.tolist():
                selected.append((layer_idx, head_idx))
    elif layer_sensitive:
        # 1. Compute Layer Importance
        layer_importance = scores.mean(dim=1)  # [Layers]
        
        # 2. Identify Candidate Layers (Bottom X%)
        num_candidates = max(1, int(num_layers * candidate_layer_fraction))
        _, candidate_layer_indices = torch.topk(
            layer_importance, k=num_candidates, largest=False
        )
        candidate_indices_set = set(candidate_layer_indices.tolist())

        # 3. Collect heads from candidate layers
        candidate_heads = []
        for l in candidate_indices_set:
            for h in range(num_heads_total):
                candidate_heads.append((l, h))
        
        # 4. Filter scores for these heads
        # We need to prune 'num_heads' from this pool.
        cand_flat_scores = []
        for l, h in candidate_heads:
            cand_flat_scores.append(scores[l, h].item())
            
        cand_flat_scores = torch.tensor(cand_flat_scores, device=scores.device)
        
        # 5. Select lowest N from this pool
        k = min(num_heads, cand_flat_scores.numel())
        _, topk_indices = torch.topk(cand_flat_scores, k=k, largest=False)
        
        for idx in topk_indices.tolist():
            selected.append(candidate_heads[idx])

    else:
        flat_scores = scores.flatten()
        k = min(num_heads, flat_scores.numel())
        _, flat_indices = torch.topk(flat_scores, k=k, largest=False)
        for flat_idx in flat_indices.tolist():
            layer_idx = flat_idx // num_heads_total
            head_idx = flat_idx % num_heads_total
            selected.append((layer_idx, head_idx))

    return selected
