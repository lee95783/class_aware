# Joint MLP and Token Pruning for Personalizing Vision Transformers

## Environment Requirements
Install required packages:

```
pip install -r requirements.txt
```
Optional: for FLOPs calculation (thop), visualization (matplotlib, seaborn), or clustering experiments (scikit-learn), install them manually as needed.


## Dataset Preparation

**CIFAR-100** will be automatically downloaded to `./data/` when you run any training/evaluation script for the first time. No manual action needed.

**TinyImageNet-200** requires a one-time setup:

```
python scripts/prepare_tinyimagenet.py --data-dir data/tiny-imagenet-200
```

> During experiments, the target class subsets are automatically extracted based on the configuration file `configs/class_subsets.json` — no manual preparation is required.

## Pretrained Weight Preparation

Baseline models are trained on the **full dataset** and used for all subsequent experiments:

| Model         | Dataset           | Training Command                                            |
|---------------|-------------------|--------------------------------------------------------------|
| DeiT-Tiny     | CIFAR-100 | `python main.py --dataset cifar100 --epochs 30`            |
| DeiT-Small    | TinyImageNet-200  | `python scripts/finetune_deit_small_tinyimagenet.py`         |

> Weights are saved in the `weights/` directory by default. 

## Quick Start: MLP Pruning + CGTS

Run the following command to train and evaluate the joint pruning model (MLP + CGTS), which outputs accuracy results.

```
python scripts/train_e04_combined.py --num-classes 10 --epochs 5 --device 0
```

## Paper Experiments

| Experiment | Description | Scripts (CIFAR‑100 / DeiT‑Tiny) | Scripts (TinyImageNet / DeiT‑Small) |
|------|------|-----------------------------------|----------------------------------------|
| **E1** | Unpruned baseline accuracy for different K | `evaluate_base_model.py` | `finetune_deit_small_tinyimagenet.py` |
| **E3** | CGTS sweep (pruning layer, token keep ratio) | `eval_e03_cgts_sweep.py`<br>`eval_e03b_layer_selection.py` | — |
| **E4** | **Our method**:MLP pruning + CGTS joint | `train_e04_combined.py` | `train_tinyimagenet_e04_combined.py` |
| **E7** | **Comparison methods**:Zero‑TPrune, DynamicViT, SViTE, X‑Pruner | `eval_e07_zero_tprune.py`<br>`train_e07_dynamicvit.py`<br>`train_e07_svite.py`<br>`train_e07_xpruner.py` | `eval_tinyimagenet_e07_zero_tprune.py`<br>`train_tinyimagenet_e07_dynamicvit.py`<br>`train_tinyimagenet_e07_svite.py`<br>`train_tinyimagenet_e07_xpruner.py` |
| **Efficiency** | GMACs,model size (MB),latency(B=1, B=64),per-subset storage | `eval_efficiency.py`<br>`eval_zero_tprune_efficiency.py`<br>`eval_xpruner_hard_efficiency.py`<br>`eval_e08_backbone_sharing.py` | — |

## Repository Structure

```
class_aware/
├── configs/                 				# 配置文件目录
│   ├── class_subsets.json        			# 定义目标类别子集（CIFAR）
│   ├── class_subsets_multi.json  		# 多任务/多子集配置
│   └── tinyimagenet_class_subsets.json    # Tiny ImageNet 配置
├── examples/               				# 快速上手示例
│   └── deploy_b1_optimized.py
├── scripts/               			        # 各类实验脚本（扫描、消融研究等）
│   ├── train_e01_baselines.py
│   ├── train_e02_mlp_sweep.py
│   ├── eval_e03_cgts_sweep.py
│   ├── eval_e03b_layer_selection.py
│   ├── train_e04_combined.py
│   ├── train_tinyimagenet_e04_combined.py
│   ├── train_e07_dynamicvit.py
│   ├── train_e07_svite.py
│   ├── train_e07_xpruner.py
│   ├── train_tinyimagenet_e07_dynamicvit.py
│   ├── train_tinyimagenet_e07_svite.py
│   ├── train_tinyimagenet_e07_xpruner.py
│   └── eval_e07_zero_tprune.py
│   └── eval_e08_backbone_sharing.py
│   └── eval_e09_cgts_perclass.py
│   └── eval_e10_cgts_live_cls.py
│   └── eval_e11_cgts_cls_attn.py
│   └── eval_e12_zero_tprune_full.py
│   └── eval_efficiency.py
│   └── eval_tinyimagenet_e07_zero_tprune.py
│   └── eval_xpruner_hard_efficiency.py
│   └── eval_zero_tprune_efficiency.py
│   └── finetune_deit_small_tinyimagenet.py
│   └── generate_multi_subsets.py
│   └── prepare_tinyimagenet.py
│   └── tome.py
│   └── x_pruner.py
├── src/                    				# 核心源码
│   ├── __init__.py
│   ├── models.py                                    # 模型定义 (ResNet, VGG 等)
│   ├── pruning.py                                   # 核心剪枝算法实现
│   ├── pruning_ratio.py                          # 逐层剪枝率计算
│   ├── dataset.py                                   # 数据加载与预处理
│   ├── utils.py                                        # 工具函数（日志、 checkpoint 管理等）
│   └── attention_profile.py                     # 注意力/激活分布分析
└── README.md
└── requirements.txt
└── analyze_class_specialization.py
└── analyze_pruning_ratio.py
└── eval_hard_pruned.py
└── eval_subset_hard_pruned.py
└── evaluate_base_model.py
└── evaluate_conservative.py
└── evaluate_mlp_pruning.py
└── evaluate_subset.py
└── final_benchmark.py
└── generate_extreme_prune.py
└── main.py
└── test_b1_optimization.py
└── test_mlp_gating.py
└── test_mlp_pruning_quick.py
└── test_token_pruning_b1.py
└── test_token_simple.py
└── train_joint_pruning.py
└── train_mlp_adaptive.py
└── train_mlp_conservative.py
```
