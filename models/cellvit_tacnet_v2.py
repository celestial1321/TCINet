# -*- coding: utf-8 -*-
# TCINet-TACNet v2: Task-Aware Cross-branch Interaction Network
#
# 1. TSFA（Task-Specific Feature Adapter）— skip connection 级创新
#    - 问题：shared decoder (decoder1/2/3) 的输出对三个任务一视同仁
#      但 NP 需要高频边界特征，HV 需要平滑梯度特征，NT 需要语义判别特征
#    - 方案：在 shared decoder 输出 cat 进 branch decoder 之前，加任务特定 adapter
#      每个分支在 b3 和 b2 两个 skip connection 处各有一个轻量 adapter
#    - 参考：Adapter 思想来自 NLP (Houlsby et al., 2019)，ViT-Adapter (Chen et al., ICLR 2023)
#
# 2. Tissue-Conditioned TACBI — decoder 中间层级创新
#    - 问题：TACBI 的三条交互路径对所有组织类型用相同强度，但不同组织的最优交互模式不同
#      密集组织（淋巴）需要更强的 HV→NP 边界引导
#      稀疏组织（结缔）需要更强的 NP→HV 前景约束
#    - 方案：用 tissue_logits (19维) → MLP → 3个动态 alpha，替代固定的 alpha_np/hv/nt
#    - 参考：TAND (arxiv 2511.13615) Spatial-FiLM; Co-Seg (MICCAI 2025) tissue-nuclei mutual guidance
#
# 3. LKCellBlock — decoder head 级创新（已验证有效）
#    - 13×13 depthwise 大卷积核捕获完整细胞核结构
#    - 参考：LKCell (Neurocomputing 2024)
#
# 4. BoundaryWeightedBCELoss — 训练策略级创新（已验证有效）
#    - 边界像素加权，直接优化 bPQ
#
# 整体架构叙事：
#    Encoder → Shared Decoder → TSFA（纵向任务适配）→ Branch Decoder → TACBI（横向跨分支交互）
#                                                                     ↑ tissue_logits 动态调控
#    → LKCellBlock Header → Output
#
# @ Fabian Hörst (原版 TCINet)
# TACNet v2 改进 by current user

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_segmentation.utils.post_proc_TCINet import DetectionCellPostProcessor
from models.segmentation.cell_segmentation.utils import Conv2DBlock, Deconv2DBlock, ViTTCINet, ViTTCINetDeit
from cell_segmentation.utils.post_proc_TCINet import (
    DetectionCellPostProcessor,
    get_tissue_seed_thresh,
)

# ============================================================================
# 模块 1：LKCellBlock（已验证有效，保持不变）
# ============================================================================
class LKCellBlock(nn.Module):
    """LKCell 风格的大卷积核块。"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 13):
        super().__init__()
        padding = kernel_size // 2
        self.large_kernel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=3, dilation=3, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        feat = self.large_kernel(x)
        feat = self.dilated_conv(feat)
        return self.act(feat + identity)


# ============================================================================
# 模块 2：TSFA（Task-Specific Feature Adapter）
#
# 在 shared decoder 输出和 branch decoder 的 skip connection 之间插入
# 每个任务有自己的 adapter，把通用特征变换为任务特定特征
#
# 结构：bottleneck 设计（降维 → 激活 → 升维 → 残差）
#   input(C) → Conv1x1(C→C//r) → BN → GELU → Conv1x1(C//r→C) → BN → + input
#   r=4 (reduction ratio)
#
# NP adapter 学习强化高频边界信息
# HV adapter 学习强化平滑梯度信息
# NT adapter 学习强化语义判别信息
# ============================================================================
class TSFA(nn.Module):
    """Task-Specific Feature Adapter（增强版）.

    两层 bottleneck + 3×3 depthwise 空间建模。
    结构：input → 1×1 降维(C→C//2) → BN → GELU → 3×3 depthwise → BN → GELU
          → 1×1 升维(C//2→C) → BN → + input

    Args:
        channels: 输入/输出通道数
        reduction: bottleneck 降维比（默认 2）
    """
    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        mid = max(channels // reduction, 32)
        self.adapter = nn.Sequential(
            # 降维
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # 空间建模（3×3 depthwise，捕获局部结构）
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # 升维
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # 恒等初始化：最后 BN weight=0 → 初始输出全 0 → 残差后等于原输入
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, x):
        return x + self.adapter(x)


# ============================================================================
# 模块 3：Tissue-Conditioned TACBI
#
# 改进 v1 TACBI：
# - 固定的 alpha_np/hv/nt → 由 tissue_logits 动态生成
# - 不同组织类型自动学习不同的交互强度
# ============================================================================
class TACBI(nn.Module):
    """Tissue-Conditioned Task-Aware Cross-Branch Interaction Module.

    在 decoder b2 层级（128通道）对 NP、HV、NT 三分支特征进行交互。
    交互强度由 tissue_logits 动态控制。

    Args:
        feat_channels: b2 特征通道数（默认 128）
        num_tissue_classes: 组织类型数（默认 19）
    """
    def __init__(self, feat_channels: int = 128, num_tissue_classes: int = 19):
        super().__init__()

        # Path 1: HV → NP（两层 spatial attention，更强的空间建模）
        self.hv_to_np_gate = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3,
                      padding=1, groups=feat_channels, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3,
                      padding=1, groups=feat_channels, bias=False),
            nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.hv_to_np_gate[-2].weight)
        nn.init.constant_(self.hv_to_np_gate[-2].bias, 0.0)

        # Path 2: NP → HV（两层 spatial gate）
        self.np_to_hv_gate = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3,
                      padding=1, groups=feat_channels, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3,
                      padding=1, groups=feat_channels, bias=False),
            nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.np_to_hv_gate[-2].weight)
        nn.init.constant_(self.np_to_hv_gate[-2].bias, 0.0)

        # Path 3: (NP+HV) → NT（加宽 MLP：256→256→128）
        self.np_hv_to_nt = nn.Sequential(
            nn.Linear(feat_channels * 2, feat_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels * 2, feat_channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.np_hv_to_nt[-2].weight)
        nn.init.constant_(self.np_hv_to_nt[-2].bias, 0.0)

        # Tissue-Conditioned 动态 alpha（加宽：19→64→3）
        self.tissue_to_alpha = nn.Sequential(
            nn.Linear(num_tissue_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.tissue_to_alpha[-1].weight)
        nn.init.zeros_(self.tissue_to_alpha[-1].bias)

    def forward(self, np_b2, hv_b2, nt_b2, tissue_logits):
        """
        Args:
            np_b2, hv_b2, nt_b2: [B, 128, H/2, W/2] 三分支 b2 特征
            tissue_logits: [B, 19] 组织类型预测 logits
        Returns:
            np_out, hv_out, nt_out: 交互后的 b2 特征
        """
        # 动态 alpha：由 tissue type 控制交互强度
        tissue_prob = F.softmax(tissue_logits, dim=-1)
        alphas = self.tissue_to_alpha(tissue_prob)  # [B, 3]
        alpha_np = alphas[:, 0].view(-1, 1, 1, 1)   # [B, 1, 1, 1]
        alpha_hv = alphas[:, 1].view(-1, 1, 1, 1)
        alpha_nt = alphas[:, 2].view(-1, 1, 1, 1)

        # Path 1: HV → NP
        hv_attn = self.hv_to_np_gate(hv_b2)
        np_out = np_b2 + alpha_np * (np_b2 * hv_attn)

        # Path 2: NP → HV
        np_gate = self.np_to_hv_gate(np_b2)
        hv_out = hv_b2 + alpha_hv * (hv_b2 * np_gate)

        # Path 3: (NP+HV) → NT
        np_vec = F.adaptive_avg_pool2d(np_b2, 1).flatten(1)
        hv_vec = F.adaptive_avg_pool2d(hv_b2, 1).flatten(1)
        ch_attn = self.np_hv_to_nt(torch.cat([np_vec, hv_vec], dim=1))
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)
        nt_out = nt_b2 + alpha_nt * (nt_b2 * ch_attn)

        return np_out, hv_out, nt_out


# ============================================================================
# 模块 4：Boundary-Aware Loss
# ============================================================================
class BoundaryWeightedBCELoss(nn.Module):
    """边界加权 BCE Loss，直接针对 bPQ 优化。"""
    def __init__(self, boundary_weight: float = 5.0, kernel_size: int = 3):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size

    def _extract_boundary(self, binary_mask):
        mask = binary_mask.unsqueeze(1).float()
        pad = self.kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-mask, kernel_size=self.kernel_size, stride=1, padding=pad)
        return dilated - eroded

    def forward(self, pred_logits, target):
        fg_target = target[:, 1, :, :]
        boundary = self._extract_boundary(fg_target)
        weight_map = torch.ones_like(fg_target).unsqueeze(1)
        weight_map = weight_map + boundary * (self.boundary_weight - 1.0)
        weight_map = weight_map.expand_as(pred_logits)
        log_probs = F.log_softmax(pred_logits, dim=1)
        loss = -target * log_probs * weight_map
        return loss.mean()


# ============================================================================
# 主模型：TCINet-TACNet v2
# ============================================================================
class TCINet(nn.Module):
    """TCINet-TACNet v2.

    架构创新：
    1. TSFA：skip connection 处的任务特定特征适配（纵向）
    2. Tissue-Conditioned TACBI：组织类型感知的跨分支交互（横向 + tissue 条件）
    3. LKCellBlock：大卷积核 decoder head
    4. BoundaryWeightedBCELoss：边界加权训练

    信息流：
    Encoder → z1,z2,z3,z4
                ↓
    Shared Decoder → b3_shared, b2_shared, b1_shared
                ↓ TSFA（每个分支各自适配）
    Branch Decoder (bottleneck → decoder3 → decoder2) → b2
                ↓ TACBI（三分支交互，tissue 控制强度）
    Branch Decoder (decoder1 → LKCellBlock header) → output
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ):
        super().__init__()
        assert len(extract_layers) == 4

        self.patch_size = 16
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.encoder = ViTTCINet(
            patch_size=self.patch_size,
            num_classes=self.num_tissue_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # Shared decoder backbone
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        self.regression_loss = regression_loss
        offset_branches = 2 if self.regression_loss else 0

        # ===== TSFA：每个分支在 b3 和 b2 skip connection 处各有一个 adapter =====
        # b3_shared 通道数 = bottleneck_dim, b2_shared 通道数 = 256
        # NP adapters
        self.tsfa_np_b3 = TSFA(self.bottleneck_dim)
        self.tsfa_np_b2 = TSFA(256)
        # HV adapters
        self.tsfa_hv_b3 = TSFA(self.bottleneck_dim)
        self.tsfa_hv_b2 = TSFA(256)
        # NT adapters
        self.tsfa_nt_b3 = TSFA(self.bottleneck_dim)
        self.tsfa_nt_b2 = TSFA(256)

        # ===== Tissue-Conditioned TACBI =====
        self.tacbi = TACBI(feat_channels=128, num_tissue_classes=num_tissue_classes)

        # ===== 三分支 decoder heads =====
        self.nuclei_binary_map_decoder = self.create_upsampling_branch_lk(2 + offset_branches)
        self.hv_map_decoder = self.create_upsampling_branch_lk(2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(self.num_nuclei_classes)

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        assert x.shape[-2] % self.patch_size == 0
        assert x.shape[-1] % self.patch_size == 0

        out_dict = {}
        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        # Shared decoder 中间结果（只计算一次）
        b3_shared = self.decoder3(z3)  # [B, bottleneck_dim, 32, 32]
        b2_shared = self.decoder2(z2)  # [B, 256, 64, 64]
        b1_shared = self.decoder1(z1)  # [B, 128, 128, 128]

        # Phase 1：三分支各自通过 TSFA 适配后计算到 b2
        np_b2 = self._compute_to_b2_with_tsfa(
            z4, b3_shared, b2_shared,
            self.nuclei_binary_map_decoder, self.tsfa_np_b3, self.tsfa_np_b2)
        hv_b2 = self._compute_to_b2_with_tsfa(
            z4, b3_shared, b2_shared,
            self.hv_map_decoder, self.tsfa_hv_b3, self.tsfa_hv_b2)
        nt_b2 = self._compute_to_b2_with_tsfa(
            z4, b3_shared, b2_shared,
            self.nuclei_type_maps_decoder, self.tsfa_nt_b3, self.tsfa_nt_b2)

        # Phase 2：Tissue-Conditioned TACBI 交互
        np_b2, hv_b2, nt_b2 = self.tacbi(np_b2, hv_b2, nt_b2, classifier_logits)

        # Phase 3：各分支从 b2 到输出
        b0 = self.decoder0(z0)

        if self.regression_loss:
            nb_map = self._b2_to_output(np_b2, b1_shared, b0, self.nuclei_binary_map_decoder)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._b2_to_output(
                np_b2, b1_shared, b0, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._b2_to_output(
            hv_b2, b1_shared, b0, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._b2_to_output(
            nt_b2, b1_shared, b0, self.nuclei_type_maps_decoder)

        if retrieve_tokens:
            out_dict["tokens"] = z4
        return out_dict

    def _compute_to_b2_with_tsfa(self, z4, b3_shared, b2_shared,
                                  branch_decoder, tsfa_b3, tsfa_b2):
        """通过 TSFA 适配后计算到 b2。

        TSFA 在 skip connection 处对 shared decoder 输出做任务特定变换：
        b3_shared → tsfa_b3 → cat with b4 → decoder3_upsampler
        b2_shared → tsfa_b2 → cat with b3 → decoder2_upsampler
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        # TSFA 适配 b3 skip connection
        b3_adapted = tsfa_b3(b3_shared)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3_adapted, b4], dim=1))
        # TSFA 适配 b2 skip connection
        b2_adapted = tsfa_b2(b2_shared)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2_adapted, b3], dim=1))
        return b2  # [B, 128, H/2, W/2]

    def _b2_to_output(self, b2, b1_shared, b0, branch_decoder):
        """b2 → decoder1 → decoder0_header = 最终输出"""
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1_shared, b2], dim=1))
        return branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

    def _forward_upsample(self, z0, z1, z2, z3, z4, branch_decoder):
        """原版完整上采样（保留兼容）"""
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        return branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """原版 decoder head（NT 分支）"""
        bottleneck_upsampler = nn.ConvTranspose2d(
            self.embed_dim, self.bottleneck_dim, kernel_size=2, stride=2)
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        return nn.Sequential(OrderedDict([
            ("bottleneck_upsampler", bottleneck_upsampler),
            ("decoder3_upsampler", decoder3_upsampler),
            ("decoder2_upsampler", decoder2_upsampler),
            ("decoder1_upsampler", decoder1_upsampler),
            ("decoder0_header", decoder0_header),
        ]))

    def create_upsampling_branch_lk(self, num_classes: int) -> nn.Module:
        """LKCellBlock decoder head（NP、HV 分支）"""
        bottleneck_upsampler = nn.ConvTranspose2d(
            self.embed_dim, self.bottleneck_dim, kernel_size=2, stride=2)
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )
        decoder0_header = nn.Sequential(
            LKCellBlock(in_channels=64 * 2, out_channels=64, kernel_size=13),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        return nn.Sequential(OrderedDict([
            ("bottleneck_upsampler", bottleneck_upsampler),
            ("decoder3_upsampler", decoder3_upsampler),
            ("decoder2_upsampler", decoder2_upsampler),
            ("decoder1_upsampler", decoder1_upsampler),
            ("decoder0_header", decoder0_header),
        ]))

    # 替换 TCINet_tacnet_v2.py 中的 calculate_instance_map 方法
# 在文件顶部导入处加上：
#   from cell_segmentation.utils.post_proc_TCINet import (
#       DetectionCellPostProcessor,
#       get_tissue_seed_thresh,
#   )

    def calculate_instance_map(self, predictions, magnification=40, tissue_aware=False):
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(0, 2, 3, 1)
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(0, 2, 3, 1)
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        instance_preds, type_preds = [], []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):

            # ── Tissue-Aware seed threshold ────────────────────────────────
            if tissue_aware:
                seed_thresh = get_tissue_seed_thresh(predictions_["tissue_types"][i])
            else:
                seed_thresh = 0.47   # 原版默认值
            # ──────────────────────────────────────────────────────────────

            cell_post_processor = DetectionCellPostProcessor(
                nr_types=self.num_nuclei_classes,
                magnification=magnification,
                gt=False,
                seed_thresh=seed_thresh,
            )

            pred_map = np.concatenate([
                torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach().cpu()[..., None],
                torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach().cpu()[..., None],
                predictions_["hv_map"][i].detach().cpu(),
            ], axis=-1)

            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(self, instance_maps, type_preds):
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros((batch_size, h, w, self.num_nuclei_classes))
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][instance_map == nuclei] = nuclei
            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map
        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)

    def freeze_encoder(self):
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":
                p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


class TCINet256(TCINet):
    def __init__(self, model256_path, num_nuclei_classes, num_tissue_classes,
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0, regression_loss=False):
        self.patch_size = 16
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.extract_layers = [3, 6, 9, 12]
        self.input_channels = 3
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        super().__init__(
            num_nuclei_classes=num_nuclei_classes, num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim, input_channels=self.input_channels,
            depth=self.depth, num_heads=self.num_heads, extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, regression_loss=regression_loss,
        )
        self.model256_path = model256_path

    def load_pretrained_encoder(self, model256_path):
        state_dict = torch.load(str(model256_path), map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")


class TCINetSAM(TCINet):
    def __init__(self, model_path, num_nuclei_classes, num_tissue_classes,
                 vit_structure, drop_rate=0, regression_loss=False):
        if vit_structure.upper() == "SAM-B":
            self.init_vit_b()
        elif vit_structure.upper() == "SAM-L":
            self.init_vit_l()
        elif vit_structure.upper() == "SAM-H":
            self.init_vit_h()
        else:
            raise NotImplementedError("Unknown ViT-SAM backbone structure")

        self.input_channels = 3
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.num_nuclei_classes = num_nuclei_classes
        self.model_path = model_path

        super().__init__(
            num_nuclei_classes=num_nuclei_classes, num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim, input_channels=self.input_channels,
            depth=self.depth, num_heads=self.num_heads, extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
            drop_rate=drop_rate, regression_loss=regression_loss,
        )

        self.prompt_embed_dim = 256
        self.encoder = ViTTCINetDeit(
            extract_layers=self.extract_layers, depth=self.depth,
            embed_dim=self.embed_dim, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14, out_chans=self.prompt_embed_dim,
        )
        self.classifier_head = (
            nn.Linear(self.prompt_embed_dim, num_tissue_classes)
            if num_tissue_classes > 0 else nn.Identity()
        )

    def load_pretrained_encoder(self, model_path):
        state_dict = torch.load(str(model_path), map_location="cpu")
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")

    def forward(self, x, retrieve_tokens=False):
        assert x.shape[-2] % self.patch_size == 0
        assert x.shape[-1] % self.patch_size == 0

        out_dict = {}
        classifier_logits, _, z = self.encoder(x)
        tissue_logits = self.classifier_head(classifier_logits)
        out_dict["tissue_types"] = tissue_logits

        z0, z1, z2, z3, z4 = x, *z
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        # Shared decoder 中间结果
        b3_shared = self.decoder3(z3)
        b2_shared = self.decoder2(z2)
        b1_shared = self.decoder1(z1)

        # Phase 1：TSFA 适配 + 计算到 b2
        np_b2 = self._compute_to_b2_with_tsfa(
            z4, b3_shared, b2_shared,
            self.nuclei_binary_map_decoder, self.tsfa_np_b3, self.tsfa_np_b2)
        hv_b2 = self._compute_to_b2_with_tsfa(
            z4, b3_shared, b2_shared,
            self.hv_map_decoder, self.tsfa_hv_b3, self.tsfa_hv_b2)
        nt_b2 = self._compute_to_b2_with_tsfa(
            z4, b3_shared, b2_shared,
            self.nuclei_type_maps_decoder, self.tsfa_nt_b3, self.tsfa_nt_b2)

        # Phase 2：Tissue-Conditioned TACBI
        np_b2, hv_b2, nt_b2 = self.tacbi(np_b2, hv_b2, nt_b2, tissue_logits)

        # Phase 3：各分支到输出
        b0 = self.decoder0(z0)

        if self.regression_loss:
            nb_map = self._b2_to_output(np_b2, b1_shared, b0, self.nuclei_binary_map_decoder)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._b2_to_output(
                np_b2, b1_shared, b0, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._b2_to_output(
            hv_b2, b1_shared, b0, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._b2_to_output(
            nt_b2, b1_shared, b0, self.nuclei_type_maps_decoder)

        if retrieve_tokens:
            out_dict["tokens"] = z4
        return out_dict

    def init_vit_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]

    def init_vit_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]

    def init_vit_h(self):
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32]


@dataclass
class DataclassHVStorage:
    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    tissue_types: torch.Tensor
    nuclei_type_map: torch.Tensor
    instance_map: torch.Tensor
    instance_types_nuclei: torch.Tensor
    batch_size: int
    instance_types: list = None
    regression_map: torch.Tensor = None
    regression_loss: bool = False
    h: int = 256
    w: int = 256
    num_tissue_classes: int = 19
    num_nuclei_classes: int = 6

    def get_dict(self) -> dict:
        property_dict = self.__dict__
        if not self.regression_loss and "regression_map" in property_dict.keys():
            property_dict.pop("regression_map")
        return property_dict