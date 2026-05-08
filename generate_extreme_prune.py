import json

heads_to_prune = []
# 12 layers, 3 heads each. We drop head 1 and 2 from each layer
for l in range(12):
    heads_to_prune.append([l, 1])
    heads_to_prune.append([l, 2])

out = {
    "heads_to_prune": heads_to_prune,
    "total_heads": 36,
    "pruned_heads": 24,
    "keep_heads": 12,
    "keep_ratio": 12.0 / 36.0
}

with open("./results/test_export_extreme_heads.json", "w") as f:
    json.dump(out, f)
