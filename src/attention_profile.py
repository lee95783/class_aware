import torch
import torch.nn as nn


class ViTSaliencyManager:
    def __init__(self, model, num_classes=1000, metric='taylor'):
        self.model = model
        self.num_layers = len(model.blocks)
        self.num_heads = model.blocks[0].attn.num_heads
        self.num_classes = num_classes
        self.metric = metric

        # Matrix to store accumulated saliency: [Layers, Heads, Classes]
        self.saliency_matrix = torch.zeros(self.num_layers, self.num_heads, num_classes)
        self.class_counts = torch.zeros(num_classes)

        self.hooks = []
        self.activations = {}
        self.gradients = {}

    def _get_forward_hook(self, layer_idx):
        def hook(module, input, output):
            # Output shape is [Batch, Tokens, Dim]
            self.activations[layer_idx] = output

        return hook

    def _get_backward_hook(self, layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_output is a tuple; index 0 is [Batch, Tokens, Dim]
            self.gradients[layer_idx] = grad_output[0]

        return hook

    def attach(self):
        """Attaches hooks to the output projection of each attention layer."""
        for i, block in enumerate(self.model.blocks):
            # We hook the 'proj' layer because it's the point where heads are combined
            target_layer = block.attn.proj
            self.hooks.append(target_layer.register_forward_hook(self._get_forward_hook(i)))
            self.hooks.append(target_layer.register_full_backward_hook(self._get_backward_hook(i)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def update_saliency(self, images, labels):
        """Calculates Taylor-based saliency for the current batch."""
        self.model.zero_grad()
        output = self.model(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()

        batch_size = images.shape[0]
        dim_per_head = self.model.embed_dim // self.num_heads

        for l in range(self.num_layers):
            # act & grad shape: [Batch, Tokens, Dim]
            act = self.activations[l]
            grad = self.gradients[l]

            # 1. Compute Saliency
            dim_per_head = self.model.embed_dim // self.num_heads

            if self.metric == 'taylor':
                # Taylor saliency: |activation * gradient|
                # Reshape to separate the heads: [Batch, Tokens, Num_Heads, Dim_Per_Head]
                saliency = (act * grad).abs().view(batch_size, -1, self.num_heads, dim_per_head)

                # Sum over tokens and head-dimensions to get [Batch, Num_Heads]
                head_importance = saliency.sum(dim=(1, 3))

                # Aggregate into the global matrix based on class labels
                for i in range(batch_size):
                    target_class = labels[i].item()
                    self.saliency_matrix[l, :, target_class] += head_importance[i].cpu()
                    if l == 0:  # Only count once per image
                        self.class_counts[target_class] += 1

            elif self.metric == 'fisher':
                # Fisher saliency: (grad**2).mean(dim=0) * (act**2).mean(dim=0)
                # Expectation over the batch (assumed to be same class or we aggregate carefully)
                # Note: "mean(dim=0)" usually implies over batch dimension.

                # [Batch, Tokens, Dim] -> [Tokens, Dim]
                act_sq_mean = (act ** 2).mean(dim=0)
                grad_sq_mean = (grad ** 2).mean(dim=0)

                fisher = act_sq_mean * grad_sq_mean  # [Tokens, Dim]

                # Reshape to [Tokens, NumHeads, HeadDim]
                fisher = fisher.view(-1, self.num_heads, dim_per_head)

                # Debug prints
                if l == 0:
                    print(
                        f"DEBUG Fisher Layer {l}: ActSqMean Max: {act_sq_mean.max()}, GradSqMean Max: {grad_sq_mean.max()}")
                    print(f"DEBUG Fisher Layer {l}: Act Shape: {act.shape}, Grad Shape: {grad.shape}")

                # Sum over tokens and head-dimensions to get [NumHeads]
                head_importance = fisher.sum(dim=(0, 2))

                # Assign to class
                # Since we collapsed the batch dimension, we assume the batch is homogenous
                # OR we take the mode class OR we iterate if mixed (but that invalidates mean(dim=0) assumption for single class fisher)
                # We assume the caller provides a batch of a single class for accurate estimation.
                # Let's use the first label.
                if len(labels) > 0:
                    target_class = labels[0].item()
                    self.saliency_matrix[l, :, target_class] += head_importance.cpu()
                    if l == 0:
                        self.class_counts[target_class] += 1

            elif self.metric == 'activation':
                # Activation-based Saliency: |Activation| (L1 norm of output)
                # act: [Batch, Tokens, Dim]
                # We want to measure how "active" a head is.

                # Reshape: [Batch, Tokens, NumHeads, HeadDim]
                act_reshaped = act.view(batch_size, -1, self.num_heads, dim_per_head)

                # L1 Norm over tokens and head_dim: [Batch, NumHeads]
                head_importance = act_reshaped.abs().sum(dim=(1, 3))

                for i in range(batch_size):
                    target_class = labels[i].item()
                    self.saliency_matrix[l, :, target_class] += head_importance[i].cpu()
                    if l == 0:
                        self.class_counts[target_class] += 1

            else:
                raise ValueError(f"Unknown metric: {self.metric}")

    def get_normalized_matrix(self):
        # Avoid division by zero for classes not seen in the batch
        counts = self.class_counts.view(1, 1, -1).expand_as(self.saliency_matrix)
        return self.saliency_matrix / (counts + 1e-6)


def compute_magnitude_importance(model):
    """
    Computes magnitude-based importance for attention heads.
    Importance = L2 norm of Q, K, V weights for the head.

    Args:
        model: ViT/DeiT model.

    Returns:
        Tensor: [Layers, Heads] importance scores.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not expose .blocks")

    scores = []

    for i, block in enumerate(model.blocks):
        attn = block.attn
        qkv = attn.qkv  # Linear(in_features=dim, out_features=3*dim)
        weight = qkv.weight  # [3*Dim, Dim]

        num_heads = attn.num_heads
        head_dim = qkv.in_features // num_heads

        # Reshape to [3, NumHeads, HeadDim, Dim]
        # In timm, qkv weight is [3 * dim, dim].
        # We assume the layout is compatible with reshaping to [3, num_heads, head_dim, dim]
        # or [num_heads, 3, head_dim, dim].
        # Actually, timm's QKV is usually cat([q, k, v], dim=0).
        # q is [dim, dim], k is [dim, dim], v is [dim, dim] stacked.
        # So weight is [3*dim, dim].
        # Q part is [0:dim, :], etc.
        # We can just reshape weight to [3, num_heads, head_dim, -1]

        # weight shape [3*H*D, H*D]
        # view as [3, H, D, H*D] ? No.
        # It's [3 * (H * D), (H * D)]
        # We want to group by Head.
        # Rows correspond to output features.
        # Output features are ordered Q_h1, Q_h2... then K_h1... ?
        # ACTUALLY, timm implementation of Linear qkv:
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # The output is B, N, 3*C.
        # It is usually split: qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        # So output layout is [3, num_heads, head_dim] interleaved or blocked?
        # Reshape to (B, N, 3, num_heads, head_dim) implies 3 is the slow dim?
        # Wait, if reshape is (B, N, 3, num_heads, head_dim), then the linear output elements are varying fastest in head_dim, then num_heads, then 3?
        # No, reshape fills from last dim.
        # If output is [B, N, 3*C] and we reshape to [..., 3, H, D], then the last dimension (3*C) is split into 3, H, D.
        # So the weight rows (output features) are ordered: Q_h1_d1, Q_h1_d2... Q_h2..., K..., V...
        # Wait, 3*num_heads*head_dim.
        # If we reshape [3, H, D], then it means discrete blocks of D, then H, then 3.
        # So 3 is the SLOWEST index in the reshape of the last dimension.
        # This implies: All Qs, then all Ks, all Vs? No.
        # Reshape takes the flat array and chops it.
        # If we have [B, N, 3*H*D].
        # Reshape to [B, N, 3, H, D].
        # This means the 3*H*D elements are ordered as:
        # 0,0,0 (q, h0, d0), 0,0,1 ...
        # So it is: Q_h0, Q_h1... ? No.
        # It means chunks of size (H*D) belong to Q, then K, then V? NO.
        # If specific reshape is (B, N, 3, H, D), then 3 is outside H.
        # This implies weight rows are ordered 3 blocks: Block Q, Block K, Block V?
        # And inside each block, H blocks of D?
        # Let's assume standard timm usage:
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Yes, means output is [3, H, D].
        # So weight rows [0 : H*D] are Q. [H*D : 2*H*D] are K.
        # Inside Q, rows [0 : D] are Head 0.

        # So we can view weight as [3, num_heads, head_dim, input_dim].

        w = weight.view(3, num_heads, head_dim, -1)

        # L2 norm: sum squares
        head_imp = (w ** 2).sum(dim=(0, 2, 3)).sqrt()
        scores.append(head_imp)

    return torch.stack(scores)  # [Layers, Heads]
