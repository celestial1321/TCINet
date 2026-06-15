<div align="center">

# TCINet: Tissue-Conditioned Adaptive Cross-Branch Interaction Network for Nuclei Instance Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)

</div>

---

## Overview

**TCINet** is a single-stage regression model for nuclei instance segmentation in H&E stained whole-slide images. It introduces **task-conditional structured information routing** across prediction branches via two key innovations:

- **TC-TACBI** (Tissue-Conditioned Adaptive Cross-Branch Interaction): Dynamically routes information across three prediction branches (Nucleus Presence, HV Maps, Nucleus Type) through asymmetric paths, with routing strength conditioned on a global tissue-type signal.
- **TSFA** (Task-Specific Feature Adapter): Lightweight residual bottleneck at decoder skip connections that produces branch-specific feature representations before cross-branch interaction.

### Key Results on PanNuke (3-fold CV)

| Metric | Value | Rank |
|--------|-------|------|
| **mPQ** | 0.519 | #1 among all published methods |
| **bPQ** | 0.692 | #1 among all published methods |
| **Dead PQ** | 0.197 | Highest on record (14.5% improvement) |

Zero-shot tests on **MoNuSeg** (F1,d = 0.876) and **CoNSeP** confirm strong generalization.

---

## Architecture

<p align="center">
  <em>TCINet architecture with TC-TACBI cross-branch interaction module</em>
</p>

TCINet builds on the CellViT backbone with three decoder branches:
- **NP** (Nucleus Presence): Binary nuclei segmentation
- **HV** (Horizontal-Vertical): Distance map regression for instance separation
- **NT** (Nucleus Type): Multi-class nuclei classification

TC-TACBI introduces three **asymmetric cross-branch paths**:
1. HV -> NP: Spatial attention gating
2. NP -> HV: Spatial gating
3. (NP, HV) -> NT: Channel attention with pooled spatial context

Each path is dynamically scaled by a **tissue-type MLP** at inference time.

---

## Installation

### Requirements

- Python 3.9+
- PyTorch >= 2.0
- CUDA 11.8+ (recommended)

```bash
# Clone the repository
git clone https://github.com/celestial1321/TCINet.git
cd TCINet

# Create conda environment
conda env create -f environment.yml
conda activate tcinet

# Or install via pip
pip install -r requirements.txt
```

---

## Quick Start

### Data Preparation

TCINet uses the **PanNuke** dataset for training and supports **MoNuSeg** and **CoNSeP** for evaluation.

1. Download PanNuke from [here](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
2. Preprocess using the provided script:
```bash
python preprocessing/patch_extraction/main_extraction.py --config configs/examples/preprocessing/patch_extraction/patch_extraction.yaml
```

### Training

```bash
# Train TCINet on PanNuke (Fold 0)
python cell_segmentation/run_cellvit.py \
    --config "configs/Ablation with boundary fold0.yaml" \
    --model "models/D cellvit lk boundary tctacbi.py"
```

Key training settings:
- Backbone: SAM-H (ViT-H encoder)
- Input size: 256 x 256
- Batch size: 8 (with gradient accumulation x2)
- Epochs: 130
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Mixed precision: Yes

### Inference

```bash
# TCINet inference on PanNuke
python cell_segmentation/inference/inference_tcinet_pannuke.py \
    --model /path/to/checkpoint.pth \
    --gpu 0 \
    --dataset PanNuke
```

### Model Checkpoints

Pretrained TCINet checkpoints will be released upon paper acceptance.

---

## Repository Structure

```
TCINet/
├── models/                     # Model definitions
│   ├── D cellvit lk boundary tctacbi.py  # TCINet (full model)
│   ├── cellvit_tacnet_v2.py    # TACNet v2 variant
│   ├── cellvit-tacnet-v3.py    # TACNet v3 variant
│   └── segmentation/           # Base segmentation models
├── cell_segmentation/          # Training & inference
│   ├── experiments/            # Experiment runners
│   ├── inference/              # Inference scripts
│   │   └── inference_tcinet_pannuke.py
│   ├── datasets/               # Data loaders (PanNuke, CoNSeP, MoNuSeg)
│   └── utils/                  # Metrics, post-processing
├── base_ml/                    # Training utilities
├── preprocessing/              # Patch extraction & preprocessing
├── configs/                    # Training configurations
│   └── Ablation with boundary fold0.yaml  # TCINet config
├── tacbi_visualization/        # TACBI attention visualizations
├── docs/                       # Documentation & figures
└── requirements.txt            # Python dependencies
```

---

## Results

### PanNuke (3-fold cross-validation)

| Method | mPQ | bPQ | mF1,d |
|--------|-----|-----|-------|
| HoVer-Net | - | - | - |
| CellViT | 0.510 | 0.680 | 0.830 |
| CellViT++ | 0.513 | 0.683 | - |
| **TCINet** | **0.519** | **0.692** | - |

### Zero-shot Generalization

| Dataset | F1,d |
|---------|------|
| MoNuSeg | 0.876 |
| CoNSeP | - |

---

## Ablation Studies

| Variant | LKCellBlock | Boundary Loss | TC-TACBI | TSFA | mPQ |
|---------|:-----------:|:-------------:|:--------:|:----:|-----|
| A (Baseline) | | | | | - |
| B (LKCell) | ✓ | | | | - |
| C (Boundary) | ✓ | ✓ | | | - |
| **D (TCINet)** | ✓ | ✓ | ✓ | | **0.519** |
| E (Full TACNet) | ✓ | ✓ | ✓ | ✓ | - |

---

## Citation

If you find TCINet useful in your research, please cite our paper:

```bibtex
@article{TCINet2026,
    title   = {TCINet: Tissue-Conditioned Adaptive Cross-Branch Interaction Network for Nuclei Instance Segmentation},
    author  = {Qian Zhenghang},
    journal = {},
    year    = {2026},
    note    = {Preprint}
}
```

We also acknowledge the CellViT backbone:

```bibtex
@article{CellViT,
    title   = {CellViT: Vision Transformers for precise cell segmentation and classification},
    journal = {Medical Image Analysis},
    volume  = {94},
    pages   = {103143},
    year    = {2024},
    doi     = {10.1016/j.media.2024.103143},
    author  = {Fabian Hörst and Moritz Rempe and Lukas Heine and Constantin Seibold and Julius Keyl and Giulia Baldini and Selma Ugurel and Jens Siveke and Barbara Grünwald and Jan Egger and Jens Kleesiek},
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions and feedback, please contact: 1094581352@qq.com or open an issue.

