# class_aware

| 实验 | 内容 | 对应脚本（CIFAR‑100 / DeiT‑Tiny） | 对应脚本（TinyImageNet / DeiT‑Small） |
|------|------|-----------------------------------|----------------------------------------|
| **E1** | Class‑aware vs class‑agnostic 剪枝比较 | `train_e01_baselines.py` | — |
| **E2** | MLP 剪枝方法对比（random, magnitude, global_taylor, class_taylor） | `train_e02_mlp_sweep.py` | — |
| **E3** | CGTS 扫描（层、保留比例） | `eval_e03_cgts_sweep.py`<br>`eval_e03b_layer_selection.py` | — |
| **E4** | **本文方法**：MLP 剪枝 + CGTS 联合 | `train_e04_combined.py` | `train_tinyimagenet_e04_combined.py` |
| **E7** | **对比方法**：Zero‑TPrune, DynamicViT, SViTE, X‑Pruner | `eval_e07_zero_tprune.py`<br>`train_e07_dynamicvit.py`<br>`train_e07_svite.py`<br>`train_e07_xpruner.py` | `eval_tinyimagenet_e07_zero_tprune.py`<br>`train_tinyimagenet_e07_dynamicvit.py`<br>`train_tinyimagenet_e07_svite.py`<br>`train_tinyimagenet_e07_xpruner.py` |
| **E8** | Backbone 共享验证（单个 backbone + 多个 prototype） | `eval_e08_backbone_sharing.py` | — |
| **E9** | Per‑class prototype CGTS（多原型） | `eval_e09_cgts_perclass.py` | — |
| **E10** | Live‑CLS CGTS（运行时 CLS 作为原型） | `eval_e10_cgts_live_cls.py` | — |
| **E11** | CLS‑Attention CGTS（基于 CLS 注意力行） | `eval_e11_cgts_cls_attn.py` | — |
| **E12** | Full Zero‑TPrune（r‑stage + s‑stage, 多层） | `eval_e12_zero_tprune_full.py` | — |
| **效率** | 理论 MACs、参数内存、实际延迟（B=1, B=64） | `eval_efficiency.py`<br>`eval_zero_tprune_efficiency.py`<br>`eval_xpruner_hard_efficiency.py` | — |
