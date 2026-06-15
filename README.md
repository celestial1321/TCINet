<div align="center">

# TCINet: Tissue-Conditioned Adaptive Cross-Branch Interaction Network for Nuclei Instance Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

**TCINet** is a single-stage regression model for nuclei instance segmentation in H&E stained whole-slide images. TCINet introduces **task-conditional structured information routing** across prediction branches—a departure from conventional multi-branch architectures that share identical features across all tasks.

### Key Innovations

- **TC-TACBI** (Tissue-Conditioned Adaptive Cross-Branch Interaction): Three asymmetric cross-branch paths (HV→NP spatial attention, NP→HV spatial gating, (NP,HV)→NT channel attention) dynamically scaled by a tissue-type MLP. Unlike symmetric cross-stitch or NDDR-style sharing, TC-TACBI routes information conditionally based on tissue context.
- **TSFA** (Task-Specific Feature Adapter): Lightweight residual bottleneck at decoder skip connections that produces branch-specific representations *before* cross-branch interaction begins, ensuring each branch enters TC-TACBI with geometrically distinct signals.
- **LKCellBlock**: Large-kernel depthwise convolution blocks that expand the receptive field for improved boundary delineation.
- **Boundary-Weighted BCE Loss**: Dynamically weights the loss at nuclear boundaries for sharper instance separation.

### Architecture

TCINet uses a Vision Transformer encoder (SAM-H) with a multi-branch decoder:

| Branch | Task |
|--------|------|
| **NP** | Nucleus Presence — binary nuclei segmentation |
| **HV** | Horizontal-Vertical distance maps — instance separation |
| **NT** | Nucleus Type — 6-class nuclei classification |

TC-TACBI sits between the three decoder branches, routing information asymmetrically. A tissue-type classifier provides the global context signal that dynamically scales each interaction path.

---

## Installation

### Requirements

- Python 3.9+
- PyTorch >= 2.0
- CUDA 11.8+ (recommended)

```bash
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

Download and preprocess the PanNuke dataset:

```bash
# 1. Download PanNuke from https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
# 2. Preprocess patches
python preprocessing/patch_extraction/main_extraction.py \
    --config configs/examples/preprocessing/patch_extraction/patch_extraction.yaml
```

### Training

```bash
python cell_segmentation/run_tcinet.py \
    --config "configs/Ablation with boundary fold0.yaml"
```

Key training configuration:
- Backbone: SAM-H (ViT-H)
- Input size: 256 x 256
- Batch size: 8 (gradient accumulation x2)
- Epochs: 130
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)

### Inference

```bash
python cell_segmentation/inference/inference_tcinet_pannuke.py \
    --model /path/to/checkpoint.pth \
    --gpu 0
```

---

## Repository Structure

```
TCINet/
├── models/                          # Model definitions
│   ├── D tcinet lk boundary tctacbi.py   # TCINet (full model)
│   ├── tcinet_tacnet_v2.py         # TACNet v2 (with TSFA)
│   ├── tcinet-tacnet-v3.py         # TACNet v3
│   ├── A tcinet baseline.py        # Baseline (ablation)
│   ├── Bc tcinet lkcellblock.py    # + LKCellBlock (ablation)
│   ├── encoders/                    # SAM ViT encoder
│   ├── segmentation/                # Base network components
│   └── utils/                       # Attention, residual blocks
├── cell_segmentation/               # Training, inference, metrics
│   ├── experiments/                 # Experiment runners
│   ├── inference/                   # Inference scripts
│   ├── datasets/                    # PanNuke, CoNSeP, MoNuSeg
│   ├── trainer/                     # Training loop
│   └── utils/                       # Metrics, post-processing
├── base_ml/                         # ML utilities (loss, optim, etc.)
├── preprocessing/                   # WSI patch extraction
├── configs/                         # Experiment configurations
├── datamodel/                       # Data model definitions
└── utils/                           # Logging, file handling
```

---

## Model Variants

| Variant | LKCellBlock | Boundary Loss | TC-TACBI | TSFA |
|---------|:-----------:|:-------------:|:--------:|:----:|
| Baseline | | | | |
| + LKCell | ✓ | | | |
| + Boundary | ✓ | ✓ | | |
| **TCINet** | ✓ | ✓ | ✓ | |
| Full TACNet | ✓ | ✓ | ✓ | ✓ |

---

## Citation

```bibtex
@article{TCINet2026,
    title   = {TCINet: Tissue-Conditioned Adaptive Cross-Branch Interaction Network for Nuclei Instance Segmentation},
    author  = {Qian, Zhenghang},
    year    = {2026},
    note    = {Preprint}
}
```

## Acknowledgments

TCINet builds upon and extends the CellViT architecture. We thank the CellViT authors for their open-source contribution:

```bibtex
@article{CellViT,
    title   = {CellViT: Vision Transformers for precise cell segmentation and classification},
    journal = {Medical Image Analysis},
    volume  = {94},
    pages   = {103143},
    year    = {2024},
    author  = {Fabian Hörst and Moritz Rempe and Lukas Heine and Constantin Seibold and Julius Keyl and Giulia Baldini and Selma Ugurel and Jens Siveke and Barbara Grünwald and Jan Egger and Jens Kleesiek},
}
```

The SAM encoder is from Meta's Segment Anything Model (Apache 2.0 license).

---

## License

This project is licensed under the MIT License. SAM-derived components retain their original Apache 2.0 license. See [LICENSE](LICENSE) for details.

---

## Contact

For questions: 1094581352@qq.com  |  Open an issue on GitHub.