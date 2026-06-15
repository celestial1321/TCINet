# -*- coding: utf-8 -*-
# 消融实验 B/C 组：原版 TCINet + LKCellBlock（NP+HV 双分支）
# B 组：配套 ablation_no_boundary_fold0.yaml（无 BoundaryLoss）
# C 组：配套 ablation_with_boundary_fold0.yaml（有 BoundaryLoss）
# 两组模型结构完全相同，差异仅在 config 的 loss 配置

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_segmentation.utils.post_proc_TCINet import DetectionCellPostProcessor
from .utils import Conv2DBlock, Deconv2DBlock, ViTTCINet, ViTTCINetDeit


class LKCellBlock(nn.Module):
    """LKCell 风格大卷积核块，替换 NP/HV 分支的 decoder0_header。"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 13):
        super().__init__()
        padding = kernel_size // 2
        self.large_kernel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=3, dilation=3, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.shortcut = (nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        feat = self.large_kernel(x)
        feat = self.dilated_conv(feat)
        return self.act(feat + identity)


class BoundaryWeightedBCELoss(nn.Module):
    """边界加权 BCE Loss（C 组通过 config 启用）。"""
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


class TCINet(nn.Module):
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
            patch_size=self.patch_size, num_classes=self.num_tissue_classes,
            embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
        )

        if self.embed_dim < 512:
            self.skip_dim_11 = 256; self.skip_dim_12 = 128; self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512; self.skip_dim_12 = 256; self.bottleneck_dim = 512

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
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate))

        self.regression_loss = regression_loss
        offset_branches = 2 if self.regression_loss else 0

        # NP/HV 使用 LKCellBlock header，NT 使用原版
        self.nuclei_binary_map_decoder = self.create_upsampling_branch_lk(2 + offset_branches)
        self.hv_map_decoder = self.create_upsampling_branch_lk(2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(self.num_nuclei_classes)

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0
        out_dict = {}
        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits
        z0, z1, z2, z3, z4 = x, *z
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        if self.regression_loss:
            nb_map = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)
        if retrieve_tokens: out_dict["tokens"] = z4
        return out_dict

    def _forward_upsample(self, z0, z1, z2, z3, z4, branch_decoder):
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
        """原版 decoder head（NT 分支用）"""
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
        """LKCellBlock decoder head（NP/HV 分支用）"""
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

    def calculate_instance_map(self, predictions, magnification=40):
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(0, 2, 3, 1)
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(0, 2, 3, 1)
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)
        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False)
        instance_preds, type_preds = [], []
        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate([
                torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i].detach().cpu()[..., None],
                torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i].detach().cpu()[..., None],
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
        self.patch_size=16; self.embed_dim=384; self.depth=12; self.num_heads=6
        self.mlp_ratio=4; self.qkv_bias=True; self.extract_layers=[3,6,9,12]
        self.input_channels=3; self.num_tissue_classes=num_tissue_classes
        self.num_nuclei_classes=num_nuclei_classes
        super().__init__(num_nuclei_classes=num_nuclei_classes, num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim, input_channels=self.input_channels, depth=self.depth,
            num_heads=self.num_heads, extract_layers=self.extract_layers, mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, regression_loss=regression_loss)
        self.model256_path = model256_path

    def load_pretrained_encoder(self, model256_path):
        state_dict = torch.load(str(model256_path), map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        print(f"Loading checkpoint: {self.encoder.load_state_dict(state_dict, strict=False)}")


class TCINetSAM(TCINet):
    def __init__(self, model_path, num_nuclei_classes, num_tissue_classes,
                 vit_structure, drop_rate=0, regression_loss=False):
        if vit_structure.upper() == "SAM-B": self.init_vit_b()
        elif vit_structure.upper() == "SAM-L": self.init_vit_l()
        elif vit_structure.upper() == "SAM-H": self.init_vit_h()
        else: raise NotImplementedError("Unknown ViT-SAM backbone structure")
        self.input_channels=3; self.mlp_ratio=4; self.qkv_bias=True
        self.num_nuclei_classes=num_nuclei_classes; self.model_path=model_path
        super().__init__(num_nuclei_classes=num_nuclei_classes, num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim, input_channels=self.input_channels, depth=self.depth,
            num_heads=self.num_heads, extract_layers=self.extract_layers, mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias, drop_rate=drop_rate, regression_loss=regression_loss)
        self.prompt_embed_dim = 256
        self.encoder = ViTTCINetDeit(
            extract_layers=self.extract_layers, depth=self.depth, embed_dim=self.embed_dim,
            mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes, window_size=14,
            out_chans=self.prompt_embed_dim)
        self.classifier_head = (nn.Linear(self.prompt_embed_dim, num_tissue_classes)
                                 if num_tissue_classes > 0 else nn.Identity())

    def load_pretrained_encoder(self, model_path):
        state_dict = torch.load(str(model_path), map_location="cpu")
        print(f"Loading checkpoint: {self.encoder.load_state_dict(state_dict, strict=False)}")

    def forward(self, x, retrieve_tokens=False):
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0
        out_dict = {}
        classifier_logits, _, z = self.encoder(x)
        tissue_logits = self.classifier_head(classifier_logits)
        out_dict["tissue_types"] = tissue_logits
        z0, z1, z2, z3, z4 = x, *z
        z4=z4.permute(0,3,1,2); z3=z3.permute(0,3,1,2)
        z2=z2.permute(0,3,1,2); z1=z1.permute(0,3,1,2)
        if self.regression_loss:
            nb_map = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)
        if retrieve_tokens: out_dict["tokens"] = z4
        return out_dict

    def init_vit_b(self):
        self.embed_dim=768; self.depth=12; self.num_heads=12
        self.encoder_global_attn_indexes=[2,5,8,11]; self.extract_layers=[3,6,9,12]

    def init_vit_l(self):
        self.embed_dim=1024; self.depth=24; self.num_heads=16
        self.encoder_global_attn_indexes=[5,11,17,23]; self.extract_layers=[6,12,18,24]

    def init_vit_h(self):
        self.embed_dim=1280; self.depth=32; self.num_heads=16
        self.encoder_global_attn_indexes=[7,15,23,31]; self.extract_layers=[8,16,24,32]


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