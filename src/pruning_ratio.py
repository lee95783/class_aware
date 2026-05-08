import math


def adaptive_keep_ratio(
    num_remaining_classes: int,
    total_classes: int = 100,
    min_keep: float = 0.5,
    max_keep: float = 0.8,
    mode: str = "sqrt",
) -> float:
    """
    Compute an adaptive keep ratio based on remaining classes.

    Fewer classes -> lower keep ratio (more pruning).
    More classes  -> higher keep ratio (less pruning).
    """
    if total_classes <= 0:
        raise ValueError("total_classes must be > 0")
    if num_remaining_classes <= 0:
        raise ValueError("num_remaining_classes must be > 0")
    if min_keep <= 0 or max_keep > 1 or min_keep > max_keep:
        raise ValueError("Require 0 < min_keep <= max_keep <= 1")

    x = max(0.0, min(1.0, float(num_remaining_classes) / float(total_classes)))

    if mode == "linear":
        scale = x
    elif mode == "sqrt":
        scale = math.sqrt(x)
    elif mode == "log":
        # Normalized log curve in [0,1], steeper at low class counts.
        scale = math.log1p(9.0 * x) / math.log(10.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    keep = min_keep + (max_keep - min_keep) * scale
    return float(max(min_keep, min(max_keep, keep)))


def adaptive_prune_fraction(
    num_remaining_classes: int,
    total_classes: int = 100,
    min_keep: float = 0.5,
    max_keep: float = 0.8,
    mode: str = "sqrt",
) -> float:
    """Return pruning fraction = 1 - adaptive_keep_ratio(...)."""
    return 1.0 - adaptive_keep_ratio(
        num_remaining_classes=num_remaining_classes,
        total_classes=total_classes,
        min_keep=min_keep,
        max_keep=max_keep,
        mode=mode,
    )
