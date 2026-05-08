import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.pruning import compact_vit_attention_heads


class XPrunerHeadMaskedAttention(nn.Module):
    """
    Minimal X-Pruner-style attention:
    - keeps original timm Attention weights
    - adds class-conditional scalar gates per head: gate[layer, head, class]
    - applies gate for the current sample's label y (teacher-forced during training)
    - uses a differentiable pruning gate via sigmoid(k*(score - theta_layer))
    """
    def __init__(self, attn: nn.Module, num_classes: int, layer_idx: int, k: float = 10.0):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.num_classes = num_classes
        self.layer_idx = layer_idx
        self.k = k

        # Learnable class-conditional head gates: (H, C)
        # In the paper, masks are class-aware and differentiable. 
        self.gate = nn.Parameter(torch.zeros(self.num_heads, self.num_classes))

        # Learnable per-layer threshold theta (used to decide pruned vs kept)
        self.theta = nn.Parameter(torch.tensor(0.5))

        # Subset-mode inference: pre-collapsed gate for a class subset.
        # When active, forward() uses this directly (no labels needed).
        self.subset_mode = False
        self.register_buffer("_subset_keep", torch.empty(0), persistent=False)

    def forward(self, x, y=None, use_labels: bool = True):
        """
        x: (B, N, D)
        y: (B,) ground-truth labels (needed during mask learning)
        use_labels: if False, uses argmax prediction as a proxy label (optional)
        """
        B, N, C = x.shape
        Dh = C // self.num_heads

        # Timm's qkv linear layer output is (B, N, 3*C)
        # We need to reshape it to (B, N, 3, H, Dh) then permute to (3, B, H, N, Dh)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, Dh)

        # Select head gates:
        # - subset mode: use pre-collapsed gate (no labels needed)
        # - training / oracle: class-conditional gate with provided labels
        # - inference bootstrap: class-agnostic average gate
        if self.subset_mode and self._subset_keep.numel() > 0:
            # Pre-computed keep values, just broadcast to batch. (H,) -> (B, H)
            keep = self._subset_keep.unsqueeze(0).expand(B, -1)
        elif use_labels and y is not None:
            # Per-sample per-head gates: (B, H)
            g = self.gate[:, y].transpose(0, 1)
            g = torch.sigmoid(g)
            keep = torch.sigmoid(self.k * (g - self.theta))  # (B, H)
        else:
            # Class-agnostic fallback avoids GT-label leakage and class-0 bias.
            g = self.gate.mean(dim=1, keepdim=True).transpose(0, 1).expand(B, -1)
            g = torch.sigmoid(g)
            keep = torch.sigmoid(self.k * (g - self.theta))  # (B, H)

        out = out * keep.view(B, self.num_heads, 1, 1)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, keep  # keep used for regularization / reporting

    def binarize_mask(self):
        """
        Converts the soft mask to a hard binary mask (0 or 1) based on current theta.
        Freezes the gate and theta.
        """
        with torch.no_grad():
            # Current decision: keep if k*(g - theta) > 0  => g > theta
            # self.gate shape: (H, C)
            g = torch.sigmoid(self.gate)
            
            # Binary mask: 1 if g > theta else 0
            # We can just update self.gate to be very large or very small
            # Or better: replace self.gate with fixed parameter
            
            # The forward pass uses: keep = sigmoid(k * (g - theta))
            # If we want hard 0/1, we can set k to be very large (infinity)
            # OR we can just overwrite gate values with +inf / -inf ?
            
            # Let's compute the binary decision
            decision = (g > self.theta).float() # (H, C)
            
            # To make sigmoid(...) output 1 or 0 strongly:
            # If decision is 1, set gate to theta + large
            # If decision is 0, set gate to theta - large
            
            large_val = 100.0 / self.k # k*large = 100 -> sigmoid(100) ~ 1
            
            new_gate = torch.where(decision > 0.5, self.theta + large_val, self.theta - large_val)
            
            self.gate.data.copy_(new_gate)
            self.gate.requires_grad = False
            self.theta.requires_grad = False

    @torch.no_grad()
    def collapse_to_subset(self, class_indices):
        """
        Collapse per-class gates into a single fixed keep vector for a class
        subset.  After this call, forward() no longer needs labels.

        Args:
            class_indices (list[int] | Tensor): class indices in the target subset.
        """
        if isinstance(class_indices, (list, tuple)):
            class_indices = torch.tensor(class_indices, dtype=torch.long,
                                        device=self.gate.device)
        # Average gate across subset classes: (H, |S|) -> (H,)
        g = self.gate[:, class_indices].mean(dim=1)
        g = torch.sigmoid(g)
        keep = torch.sigmoid(self.k * (g - self.theta))  # (H,)
        self._subset_keep = keep
        self.subset_mode = True


class XPrunerMLPGating(nn.Module):
    """
    Class-aware MLP with learnable neuron gates.

    Wraps a timm MLP block and adds class-conditional gating per neuron.
    Similar to XPrunerHeadMaskedAttention but for MLP neurons instead of heads.
    """
    def __init__(self, mlp: nn.Module, num_classes: int, layer_idx: int, k: float = 10.0):
        super().__init__()
        self.fc1 = mlp.fc1  # D → D_ff
        self.act = mlp.act  # GELU
        self.fc2 = mlp.fc2  # D_ff → D
        # Handle both old (drop) and new (drop1) attribute names
        self.drop = getattr(mlp, 'drop', None) or getattr(mlp, 'drop1', None)

        self.d_ff = mlp.fc1.out_features
        self.num_classes = num_classes
        self.layer_idx = layer_idx
        self.k = k

        # Learnable class-conditional neuron gates: (D_ff, C)
        # gate[neuron_id, class_id] = importance of this neuron for this class
        self.gate = nn.Parameter(torch.zeros(self.d_ff, self.num_classes))

        # Learnable per-layer threshold
        self.theta = nn.Parameter(torch.tensor(0.5))

        # Subset-mode inference: pre-collapsed gate for a class subset
        self.subset_mode = False
        self.register_buffer("_subset_keep", torch.empty(0), persistent=False)

    def forward(self, x, y=None, use_labels: bool = True):
        """
        x: (B, N, D) - input features
        y: (B,) - ground-truth labels (for class-conditional gating)
        use_labels: if False, uses class-agnostic average gate

        Returns:
            out: (B, N, D) - output features
            keep: (B, D_ff) - keep probability per neuron (for regularization)
        """
        B, N, D = x.shape

        # Standard MLP forward up to activation
        h = self.fc1(x)  # (B, N, D_ff)
        h = self.act(h)   # (B, N, D_ff)

        # Compute class-conditional neuron gates
        if self.subset_mode and self._subset_keep.numel() > 0:
            # Pre-computed keep values for subset, just broadcast
            keep = self._subset_keep.unsqueeze(0).expand(B, -1)  # (B, D_ff)
        elif use_labels and y is not None:
            # Per-sample per-neuron gates: (B, D_ff)
            g = self.gate[:, y].transpose(0, 1)  # (D_ff, C) -> (B, D_ff)
            g = torch.sigmoid(g)
            keep = torch.sigmoid(self.k * (g - self.theta))
        else:
            # Class-agnostic fallback (average over all classes)
            g = self.gate.mean(dim=1, keepdim=True).transpose(0, 1).expand(B, -1)
            g = torch.sigmoid(g)
            keep = torch.sigmoid(self.k * (g - self.theta))

        # Apply neuron-wise gating: (B, N, D_ff) * (B, 1, D_ff)
        h = h * keep.unsqueeze(1)

        # Complete MLP forward
        if self.drop is not None:
            h = self.drop(h)
        out = self.fc2(h)  # (B, N, D)

        return out, keep

    def binarize_mask(self):
        """
        Converts soft gates to hard binary masks based on current theta.
        Freezes the gate and theta parameters.
        """
        with torch.no_grad():
            g = torch.sigmoid(self.gate)  # (D_ff, C)
            decision = (g > self.theta).float()

            large_val = 100.0 / self.k
            new_gate = torch.where(decision > 0.5, self.theta + large_val, self.theta - large_val)

            self.gate.data.copy_(new_gate)
            self.gate.requires_grad = False
            self.theta.requires_grad = False

    @torch.no_grad()
    def collapse_to_subset(self, class_indices):
        """
        Collapse per-class gates into a single fixed keep vector for a class subset.
        After this, forward() no longer needs labels.

        Args:
            class_indices (list[int] | Tensor): class indices in the target subset
        """
        if isinstance(class_indices, (list, tuple)):
            class_indices = torch.tensor(class_indices, dtype=torch.long, device=self.gate.device)

        # Average gate across subset classes: (D_ff, |S|) -> (D_ff,)
        g = self.gate[:, class_indices].mean(dim=1)
        g = torch.sigmoid(g)
        keep = torch.sigmoid(self.k * (g - self.theta))  # (D_ff,)
        self._subset_keep = keep
        self.subset_mode = True


class XPrunerDeiT(nn.Module):
    def __init__(
        self,
        model_name="deit_tiny_patch16_224",
        num_classes=10,
        pretrained=True,
        k=10.0,
        enable_token_pruning=False,
        token_k=10.0,
        token_min_keep_ratio=0.05,
        token_start_layer=0,
        physical_token_pruning=True,
        physical_token_pruning_train=False,
        token_min_keep_tokens=8,
        use_static_token_compaction=False,
        enable_mlp_pruning=False,
        mlp_k=10.0,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes
        self.enable_token_pruning = bool(enable_token_pruning)
        self.enable_mlp_pruning = bool(enable_mlp_pruning)
        self.token_start_layer = int(token_start_layer)
        self.use_static_token_compaction = bool(use_static_token_compaction)
        self.subset_mode = False

        # Replace each block's attn with masked version
        for i, blk in enumerate(self.backbone.blocks):
            blk.attn = XPrunerHeadMaskedAttention(blk.attn, num_classes=num_classes, layer_idx=i, k=k)

        # Replace each block's MLP with gated version if enabled
        if self.enable_mlp_pruning:
            for i, blk in enumerate(self.backbone.blocks):
                blk.mlp = XPrunerMLPGating(blk.mlp, num_classes=num_classes, layer_idx=i, k=mlp_k)

        self.token_gates = None
        if self.enable_token_pruning:
            embed_dim = int(self.backbone.embed_dim)
            self.token_gates = nn.ModuleList(
                [
                    XPrunerTokenGate(
                        embed_dim=embed_dim,
                        num_classes=num_classes,
                        layer_idx=i,
                        k=token_k,
                        min_keep_ratio=token_min_keep_ratio,
                        physical_pruning=physical_token_pruning,
                        physical_pruning_train=physical_token_pruning_train,
                        min_keep_tokens=token_min_keep_tokens,
                        use_static_compaction=self.use_static_token_compaction,
                    )
                    for i in range(len(self.backbone.blocks))
                ]
            )

    def _forward_impl(self, x, y=None, use_labels=True):
        """Single pass through backbone with either class-conditional or class-agnostic masks."""
        B = x.size(0)
        keep_all = []

        # Patch embed + pos embed etc are inside timm model's forward_features,
        # but we need to thread labels into attention, so we re-implement a minimal forward here.
        # This works for timm ViT/DeiT family.
        bb = self.backbone
        x = bb.patch_embed(x)
        cls_token = bb.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + bb.pos_embed
        x = bb.pos_drop(x)

        for layer_idx, blk in enumerate(bb.blocks):
            # blk.norm1, blk.attn, blk.drop_path1, blk.norm2, blk.mlp, blk.drop_path2
            x_norm = blk.norm1(x)
            attn_out, keep = blk.attn(x_norm, y=y, use_labels=use_labels)
            keep_all.append(keep)
            x = x + blk.drop_path1(attn_out)

            if self.enable_token_pruning and self.token_gates is not None:
                if layer_idx >= self.token_start_layer:
                    gate = self.token_gates[layer_idx]
                    # Static structured compaction is applied once at token_start_layer.
                    # Later layers keep dense shapes for index-safety and stable runtime.
                    if self.use_static_token_compaction and layer_idx > self.token_start_layer:
                        old_physical = gate.physical_pruning
                        gate.physical_pruning = False
                        x, token_keep = gate(x, y=y, use_labels=use_labels)
                        gate.physical_pruning = old_physical
                    else:
                        x, token_keep = gate(x, y=y, use_labels=use_labels)
                    keep_all.append(token_keep)

            # MLP forward - with or without gating
            if self.enable_mlp_pruning:
                mlp_out, mlp_keep = blk.mlp(blk.norm2(x), y=y, use_labels=use_labels)
                keep_all.append(mlp_keep)
                x = x + blk.drop_path2(mlp_out)
            else:
                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

        x = bb.norm(x)
        cls = x[:, 0]
        logits = bb.head(cls)
        return logits, keep_all

    def forward(self, x, y=None, use_labels=True):
        """
        Returns logits and keep matrices.
        - Subset mode: single pass with pre-collapsed gates (no labels needed).
        - If labels are provided and use_labels=True: class-conditional masking (training/oracle).
        - Otherwise: two-pass inference.
          Pass 1 uses class-agnostic gates to get provisional predictions.
          Pass 2 applies class-conditional masks using those predictions.
        """
        if self.subset_mode:
            # Single-pass inference: all gates use pre-collapsed subset values.
            return self._forward_impl(x, y=None, use_labels=False)

        if use_labels and y is not None:
            return self._forward_impl(x, y=y, use_labels=True)

        logits_bootstrap, _ = self._forward_impl(x, y=None, use_labels=False)
        pred = logits_bootstrap.argmax(dim=1)
        return self._forward_impl(x, y=pred, use_labels=True)

    def binarize_masks(self):
        for blk in self.backbone.blocks:
            blk.attn.binarize_mask()
        if self.enable_token_pruning and self.token_gates is not None:
            for gate in self.token_gates:
                gate.binarize_mask()

    @torch.no_grad()
    def prepare_subset_inference(self, class_indices, num_patch_tokens=196):
        """
        Collapse all per-class gates (head + token) for a class subset and
        switch to single-pass inference mode.

        After this call:
          - forward() runs a single pass (no two-pass bootstrap).
          - No class labels (y) are needed at inference time.
          - Token gates use a pre-computed fixed k_keep (no .item() CPU sync).

        Args:
            class_indices (list[int]): class indices in the target subset.
            num_patch_tokens (int): expected number of patch tokens (default
                196 for 224×224 / 16×16 patches).
        """
        # Collapse head attention gates.
        for blk in self.backbone.blocks:
            if isinstance(blk.attn, XPrunerHeadMaskedAttention):
                blk.attn.collapse_to_subset(class_indices)

        # Collapse token gates.
        if self.enable_token_pruning and self.token_gates is not None:
            for gate in self.token_gates:
                gate.collapse_to_subset(class_indices,
                                        num_patch_tokens=num_patch_tokens)

        self.subset_mode = True

    def set_static_token_compaction(self, enabled: bool = True):
        self.use_static_token_compaction = bool(enabled)
        if self.enable_token_pruning and self.token_gates is not None:
            for gate in self.token_gates:
                gate.use_static_compaction = self.use_static_token_compaction

    def set_fixed_token_indices(self, enabled: bool = True):
        if self.enable_token_pruning and self.token_gates is not None:
            for gate in self.token_gates:
                gate.enable_fixed_index_mode(enabled)

    @torch.no_grad()
    def prepare_fixed_token_inference(self, loader, device, max_batches=None):
        """
        Calibrate fixed global token indices once, then bypass token scoring at inference.
        """
        self.build_static_token_compaction(loader=loader, device=device, max_batches=max_batches)
        self.set_fixed_token_indices(True)

    @torch.no_grad()
    def prepare_fixed_subset_inference(self, class_indices, loader, device, max_batches=None, num_patch_tokens=196):
        """
        Deployment mode: collapse head gates to a class subset and use fixed token indices.
        """
        self.prepare_subset_inference(class_indices, num_patch_tokens=num_patch_tokens)
        self.build_static_token_compaction(loader=loader, device=device, max_batches=max_batches)
        self.set_fixed_token_indices(True)

    @torch.no_grad()
    def build_static_token_compaction(self, loader, device, max_batches=None):
        """
        Build class-aware fixed token indices for each gated layer.
        This calibrates per-class token importance from keep scores, then stores
        top-k token positions (same k for all classes in a layer).
        """
        if not self.enable_token_pruning or self.token_gates is None:
            raise RuntimeError("Token pruning must be enabled to build static token compaction.")

        self.eval()
        token_sums = {}
        token_counts = {}
        global_sums = {}
        global_counts = {}

        # Temporarily disable physical pruning during calibration to keep token length fixed.
        old_physical = []
        for gate in self.token_gates:
            old_physical.append(gate.physical_pruning)
            gate.physical_pruning = False

        target_layer = int(self.token_start_layer)
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            images = images.to(device)
            labels = labels.to(device)
            bsz = images.size(0)

            bb = self.backbone
            x = bb.patch_embed(images)
            cls_token = bb.cls_token.expand(bsz, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + bb.pos_embed
            x = bb.pos_drop(x)

            for layer_idx, blk in enumerate(bb.blocks):
                x_norm = blk.norm1(x)
                attn_out, _ = blk.attn(x_norm, y=labels, use_labels=True)
                x = x + blk.drop_path1(attn_out)

                if layer_idx == target_layer:
                    x, token_keep = self.token_gates[layer_idx](x, y=labels, use_labels=True)
                    t = token_keep.size(1)
                    if layer_idx not in token_sums:
                        token_sums[layer_idx] = x.new_zeros((self.num_classes, t))
                        token_counts[layer_idx] = x.new_zeros((self.num_classes,))
                        global_sums[layer_idx] = x.new_zeros((t,))
                        global_counts[layer_idx] = 0

                    token_sums[layer_idx].index_add_(0, labels, token_keep)
                    token_counts[layer_idx].index_add_(0, labels, x.new_ones((bsz,)))
                    global_sums[layer_idx] += token_keep.sum(dim=0)
                    global_counts[layer_idx] += bsz

                x = x + blk.drop_path2(blk.mlp(blk.norm2(x)))

        for layer_idx, gate in enumerate(self.token_gates):
            if layer_idx != target_layer or layer_idx not in token_sums:
                continue

            class_count = token_counts[layer_idx].clamp_min(1.0).unsqueeze(1)
            class_mean = token_sums[layer_idx] / class_count  # [C, T]
            global_mean = global_sums[layer_idx] / max(1, int(global_counts[layer_idx]))  # [T]
            t = class_mean.size(1)

            mean_keep = float(global_mean.mean().item())
            k_keep = int(round(mean_keep * t))
            k_keep = max(gate.min_keep_tokens, min(t, k_keep))

            class_idx = torch.topk(class_mean, k=k_keep, dim=1, largest=True, sorted=False).indices
            class_idx = torch.sort(class_idx, dim=1).values
            global_idx = torch.topk(global_mean, k=k_keep, dim=0, largest=True, sorted=False).indices
            global_idx = torch.sort(global_idx, dim=0).values
            global_keep = global_mean.index_select(0, global_idx)

            gate.set_static_indices(class_idx, global_idx, global_keep=global_keep)

        for gate, old in zip(self.token_gates, old_physical):
            gate.physical_pruning = old

        self.set_static_token_compaction(True)


class XPrunerTokenGate(nn.Module):
    """
    Class-aware differentiable token pruning gate.
    - keeps CLS token always active
    - predicts per-token keep probability from token features + class-conditional bias
    - applies smooth keep mask with optional minimum keep-ratio safeguard
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        layer_idx: int,
        k: float = 10.0,
        min_keep_ratio: float = 0.05,
        physical_pruning: bool = True,
        physical_pruning_train: bool = False,
        min_keep_tokens: int = 8,
        use_static_compaction: bool = False,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_classes = int(num_classes)
        self.k = float(k)
        self.min_keep_ratio = float(max(0.0, min(1.0, min_keep_ratio)))
        self.physical_pruning = bool(physical_pruning)
        self.physical_pruning_train = bool(physical_pruning_train)
        self.min_keep_tokens = int(max(1, min_keep_tokens))
        self.use_static_compaction = bool(use_static_compaction)

        self.token_score = nn.Linear(embed_dim, 1, bias=True)
        self.class_gate = nn.Parameter(torch.zeros(self.num_classes))
        self.theta = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("static_keep_indices", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("static_global_indices", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("static_global_keep", torch.empty(0), persistent=True)

        # Subset-mode inference: pre-collapsed class bias for a class subset.
        # When active, forward() uses _subset_bias directly (no labels needed).
        self.subset_mode = False
        self.register_buffer("_subset_bias", torch.tensor(0.0), persistent=False)
        # Fixed k_keep pre-computed during collapse (eliminates .item() CPU sync).
        self._fixed_k_keep = 0
        # Deployment mode: bypass token scoring and use fixed global indices directly.
        self.fixed_index_mode = False

    def set_static_indices(
        self,
        class_indices: torch.Tensor,
        global_indices: torch.Tensor,
        global_keep: torch.Tensor | None = None,
    ):
        if class_indices.ndim != 2:
            raise ValueError("class_indices must be [num_classes, k].")
        if global_indices.ndim != 1:
            raise ValueError("global_indices must be [k].")
        if class_indices.size(0) != self.num_classes:
            raise ValueError("class_indices first dimension must match num_classes.")
        if class_indices.size(1) != global_indices.size(0):
            raise ValueError("class_indices and global_indices must use the same k.")
        self.static_keep_indices = class_indices.to(dtype=torch.long, device=self.class_gate.device)
        self.static_global_indices = global_indices.to(dtype=torch.long, device=self.class_gate.device)
        if global_keep is None:
            global_keep = torch.ones(global_indices.size(0), device=self.class_gate.device)
        self.static_global_keep = global_keep.to(dtype=self.class_gate.dtype, device=self.class_gate.device)

    def has_static_indices(self):
        return (
            self.static_keep_indices.numel() > 0
            and self.static_global_indices.numel() > 0
            and self.static_global_keep.numel() == self.static_global_indices.numel()
        )

    def enable_fixed_index_mode(self, enabled: bool = True):
        self.fixed_index_mode = bool(enabled)

    def forward(self, x, y=None, use_labels: bool = True):
        # x: [B, N, D], token 0 is CLS and is never pruned
        bsz, n_tokens, _ = x.shape
        if n_tokens <= 1:
            keep = x.new_ones((bsz, 1))
            return x, keep

        cls_tok = x[:, :1, :]
        patch_tok = x[:, 1:, :]

        if self.fixed_index_mode and self.has_static_indices():
            topk_idx = self.static_global_indices
            if topk_idx.numel() > patch_tok.size(1):
                topk_idx = topk_idx[: patch_tok.size(1)]
            patch_tok = patch_tok.index_select(1, topk_idx)
            keep = self.static_global_keep[: topk_idx.numel()].view(1, -1).expand(bsz, -1)
            patch_tok = patch_tok * keep.unsqueeze(-1)
            out = torch.cat([cls_tok, patch_tok], dim=1)
            return out, keep

        # Content-aware per-token score.
        token_score = self.token_score(patch_tok).squeeze(-1)  # [B, T]

        # Class bias: subset-mode > class-conditional > class-agnostic fallback.
        if self.subset_mode:
            class_bias = self._subset_bias.view(1, 1).expand(bsz, 1)
        elif use_labels and y is not None:
            class_bias = self.class_gate[y].unsqueeze(1)  # [B, 1]
        else:
            class_bias = self.class_gate.mean().view(1, 1).expand(bsz, 1)  # [B, 1]

        g = torch.sigmoid(token_score + class_bias)  # [B, T]
        keep = torch.sigmoid(self.k * (g - self.theta))  # [B, T]

        # Keep at least a floor in expectation to avoid full-collapse.
        if self.min_keep_ratio > 0.0:
            current = keep.mean(dim=1, keepdim=True)  # [B, 1]
            scale = (self.min_keep_ratio / (current + 1e-6)).clamp(min=1.0)
            keep = (keep * scale).clamp(max=1.0)

        do_physical = self.physical_pruning and ((not self.training) or self.physical_pruning_train)
        if do_physical and patch_tok.size(1) > self.min_keep_tokens:
            if self.use_static_compaction and self.has_static_indices():
                k_keep = int(self.static_global_indices.size(0))
                if use_labels and y is not None:
                    cls = y.clamp(min=0, max=self.num_classes - 1)
                    topk_idx = self.static_keep_indices[cls]
                else:
                    topk_idx = self.static_global_indices.view(1, -1).expand(bsz, -1)
            else:
                # Use pre-computed fixed k if available (subset mode),
                # otherwise fall back to dynamic calculation (training).
                t = patch_tok.size(1)
                if self.subset_mode and self._fixed_k_keep > 0:
                    k_keep = max(self.min_keep_tokens, min(t, self._fixed_k_keep))
                else:
                    target_keep = float(keep.mean().item())
                    k_keep = int(round(target_keep * t))
                    k_keep = max(self.min_keep_tokens, min(t, k_keep))

                # Select top-k per sample, then restore token order by index sort.
                topk_idx = torch.topk(keep, k=k_keep, dim=1, largest=True, sorted=False).indices  # [B, k]
                topk_idx = torch.sort(topk_idx, dim=1).values
            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, patch_tok.size(-1))
            patch_tok = torch.gather(patch_tok, 1, gather_idx)
            keep = torch.gather(keep, 1, topk_idx)

        patch_tok = patch_tok * keep.unsqueeze(-1)
        out = torch.cat([cls_tok, patch_tok], dim=1)  # [B, 1+k, D]
        return out, keep

    def binarize_mask(self):
        # Freeze token-pruning parameters; forward still produces near-binary masks via high-k sigmoid.
        self.k = max(self.k, 100.0)
        self.class_gate.requires_grad = False
        self.theta.requires_grad = False
        self.token_score.weight.requires_grad = False
        if self.token_score.bias is not None:
            self.token_score.bias.requires_grad = False

    @torch.no_grad()
    def collapse_to_subset(self, class_indices, num_patch_tokens=196):
        """
        Collapse per-class gate into a single fixed bias for a class subset.
        Pre-computes fixed k_keep to avoid .item() CPU sync at inference.

        Args:
            class_indices (list[int] | Tensor): class indices in the target subset.
            num_patch_tokens (int): expected number of patch tokens (default
                196 for 224×224 / 16×16 patches).  Used to compute fixed k.
        """
        if isinstance(class_indices, (list, tuple)):
            class_indices = torch.tensor(class_indices, dtype=torch.long,
                                        device=self.class_gate.device)
        # Average bias across subset classes.
        bias = self.class_gate[class_indices].mean()
        self._subset_bias = bias

        # Estimate fixed k_keep from the averaged keep ratio.
        # token_score has zero mean at init; the keep ratio is driven by bias + theta.
        # A closed-form estimate: keep_ratio ≈ sigmoid(k * (sigmoid(bias) - theta))
        g = torch.sigmoid(bias)
        keep_ratio = float(torch.sigmoid(self.k * (g - self.theta)).item())
        k_keep = int(round(keep_ratio * num_patch_tokens))
        k_keep = max(self.min_keep_tokens, min(num_patch_tokens, k_keep))
        self._fixed_k_keep = k_keep

        self.subset_mode = True




class XPrunerPenaltyLoss(nn.Module):
    def __init__(self, target_ratio=0.7, lambda_sm=0.1, lambda_sp=1.0):
        super().__init__()
        self.target_ratio = target_ratio
        self.lambda_sm = float(lambda_sm)
        self.lambda_sp = float(lambda_sp)
        
    def forward(self, logits, labels, masks, inputs=None):
        # 1. Cross-Entropy Loss
        l_ce = F.cross_entropy(logits, labels)

        # 2. Smoothness Constraint (optional, based on prior context)
        # l_smooth logic if needed...
        l_smooth = torch.tensor(0.0, device=logits.device)

        # 3. Sparsity Penalty (Quadratic penalty to enforce target ratio)
        current_keep_ratio = torch.stack([m.mean() for m in masks]).mean()
        
        # Quadratic penalty: (current - target)^2
        l_penalty = (current_keep_ratio - self.target_ratio).pow(2)

        # Total Loss
        total_loss = l_ce + (self.lambda_sp * l_penalty)
        
        return total_loss, {
            "ce": l_ce.item(),
            "keep_ratio": current_keep_ratio.item(),
            "penalty": l_penalty.item(),
            "loss": total_loss.item()
        }



class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class XPrunerALMLoss(nn.Module):
    def __init__(self, target_ratio=0.7, lambda_sm=0.1, lambda_sp=0.1, mode='dual'):
        super().__init__()
        self.target_ratio = target_ratio
        self.lambda_sm = float(lambda_sm)
        self.lambda_sp = float(lambda_sp)
        self.mode = mode
        
        # Augmented Lagrangian parameters
        # Beta (quadratic penalty weight) should be fixed or manually updated, not learned via gradient descent/ascent implies learning the PENALTY weight.
        # We use lambda_sp as the fixed beta.
        self.register_buffer('beta', torch.tensor(float(lambda_sp)))
        
        # Gamma (Lagrange multiplier) is learnable (dual variable).
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits, labels, masks, inputs=None):
        # 1. Cross-Entropy Loss
        l_ce = F.cross_entropy(logits, labels)

        # 2. Sparsity Constraint (L2 norm of masks) - Ignored in ALM typically, or kept small
        l_sparse = torch.stack([m.pow(2).sum() for m in masks]).sum()

        # 3. Smoothness Constraint
        l_smooth = torch.tensor(0.0, device=logits.device)
        
        # 4. Augmented Lagrangian for Pruning Rate
        current_keep_ratio = torch.stack([m.mean() for m in masks]).mean()
        
        # diff = target - current.
        # If target=0.7, current=1.0 -> diff = -0.3.
        # We want current <= target usually? Or exactly target?
        # If exactly target, diff means deviation w.r.t target.
        # ALM minimizes L(x) + lambda*c(x) + rho/2 * c(x)^2
        # where c(x) = 0 is constraint.
        # Here c(x) = diff. 
        
        diff = self.target_ratio - current_keep_ratio
        
        # Apply Gradient Reversal on Gamma if Joint Optimization (Min-Min -> GDA)
        gamma_term = self.gamma
        if self.mode == 'joint':
            gamma_term = GradientReversal.apply(self.gamma)
            
        l_alm = self.beta * (diff ** 2) + gamma_term * diff
        
        total_loss = l_ce + l_alm
        
        return total_loss, {
            "ce": l_ce.item(),
            "keep_ratio": current_keep_ratio.item(),
            "l_sparse": l_sparse.item(),
            "l_alm": l_alm.item(),
            "beta": self.beta.item(),
            "gamma": self.gamma.item(),
            "loss": total_loss.item()
        }


@torch.no_grad()
def evaluate_hard_pruned(model, test_loader, device, target_classes, model_name, num_classes_total):
    """Structurally remove heads with gate < 0.5 and evaluate the compacted backbone."""
    heads_to_prune = []
    n_total = 0
    for l, blk in enumerate(model.backbone.blocks):
        attn = blk.attn
        if not isinstance(attn, XPrunerHeadMaskedAttention):
            continue
        n_total += attn.num_heads
        if attn._subset_keep.numel() == 0:
            continue
        prune_mask = attn._subset_keep < 0.5
        if prune_mask.all():
            prune_mask = prune_mask.clone()
            prune_mask[attn._subset_keep.argmax()] = False
        for h in range(attn.num_heads):
            if prune_mask[h].item():
                heads_to_prune.append((l, h))

    backbone = timm.create_model(model_name, num_classes=num_classes_total, pretrained=False)
    backbone.load_state_dict(model.backbone.state_dict(), strict=False)
    compact_vit_attention_heads(backbone, heads_to_prune)
    backbone.to(device).eval()

    class_idx = torch.tensor(target_classes, device=device)
    correct = total = 0
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = backbone(imgs)
        correct += (class_idx[logits[:, class_idx].argmax(1)] == lbls).sum().item()
        total += lbls.size(0)

    del backbone
    return 100.0 * correct / total, len(heads_to_prune), n_total
