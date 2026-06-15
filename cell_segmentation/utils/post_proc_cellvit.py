# -*- coding: utf-8 -*-
# PostProcessing Pipeline
#
# Adapted from HoverNet
# HoverNet Network (https://doi.org/10.1016/j.media.2019.101563)
# Code Snippet adapted from HoverNet implementation (https://github.com/vqdang/hover_net)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
#
# 参数版本说明（v4，Tissue-Aware）：
#
#   seed_thresh: 由 tissue 类型动态决定（默认 0.47）
#     密集腺体组织（Kidney/Testis/Esophagus等）→ 0.38–0.42
#     普通组织 → 0.47
#   GaussianBlur: (3,3)
#   min_size:     10

import warnings
from typing import Tuple, Literal, Optional

import cv2
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
import torch

from .tools import get_bounding_box, remove_small_objects


def noop(*args, **kargs):
    pass


warnings.warn = noop


# ============================================================
# Tissue-Aware seed threshold 映射表
# ============================================================
TISSUE_IDX = [
    "adrenal_gland", "bile-duct", "bladder", "breast", "cervix",
    "colon", "esophagus", "headneck", "kidney", "liver",
    "lung", "ovarian", "pancreatic", "prostate", "skin",
    "stomach", "testis", "thyroid", "uterus",
]

# 密集腺体/管状结构 → touching nuclei多 → 更低threshold让watershed分得更细
# 稀疏/混合组织 → 保持原值0.47
TISSUE_SEED_THRESH = {
    "kidney":        0.38,   # 肾小管，极密集
    "testis":        0.38,   # 生精小管，极密集
    "esophagus":     0.40,   # 食道腺体，密集
    "colon":         0.40,   # 结肠腺体，密集
    "prostate":      0.40,   # 前列腺腺泡，密集
    "bladder":       0.42,   # 膀胱上皮，中等
    "breast":        0.42,   # 乳腺腺体，中等
    "liver":         0.44,   # 肝细胞，中等
    "pancreatic":    0.44,   # 胰腺腺泡，中等
    "stomach":       0.44,   # 胃腺，中等
    "adrenal_gland": 0.47,
    "bile-duct":     0.47,
    "cervix":        0.47,
    "headneck":      0.47,
    "lung":          0.47,
    "ovarian":       0.47,
    "skin":          0.47,
    "thyroid":       0.47,
    "uterus":        0.47,
}


def get_tissue_seed_thresh(tissue_logits) -> float:
    """根据 tissue_logits 返回对应的 seed threshold。

    Args:
        tissue_logits: numpy array [19] 或 torch.Tensor [19] 或 [1, 19]
    Returns:
        seed threshold (float)
    """
    if isinstance(tissue_logits, torch.Tensor):
        tissue_logits = tissue_logits.detach().cpu().numpy()
    tissue_logits = np.array(tissue_logits).flatten()
    tissue_idx = int(np.argmax(tissue_logits))
    if tissue_idx < len(TISSUE_IDX):
        tissue_name = TISSUE_IDX[tissue_idx]
        return TISSUE_SEED_THRESH.get(tissue_name, 0.47)
    return 0.47


# ============================================================
# 主类
# ============================================================
class DetectionCellPostProcessor:
    def __init__(
        self,
        nr_types: int = None,
        magnification: Literal[20, 40] = 40,
        gt: bool = False,
        seed_thresh: Optional[float] = None,   # 新增：可外部传入，None则用默认0.47
    ) -> None:
        self.nr_types = nr_types
        self.magnification = magnification
        self.gt = gt
        self.seed_thresh = seed_thresh         # None = 用默认值 0.47

        if magnification == 40:
            self.object_size = 10
            self.k_size = 21
        elif magnification == 20:
            self.object_size = 3
            self.k_size = 11
        else:
            raise NotImplementedError("Unknown magnification")
        if gt:
            self.object_size = 100
            self.k_size = 21

    def post_process_cell_segmentation(
        self,
        pred_map: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        if self.nr_types is not None:
            pred_type = pred_map[..., :1]
            pred_inst = pred_map[..., 1:]
            pred_type = pred_type.astype(np.int32)
        else:
            pred_inst = pred_map

        pred_inst = np.squeeze(pred_inst)
        pred_inst = self.__proc_np_hv(
            pred_inst,
            object_size=self.object_size,
            ksize=self.k_size,
            seed_thresh=self.seed_thresh if self.seed_thresh is not None else 0.47,
        )

        inst_id_list = np.unique(pred_inst)[1:]
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0]: inst_bbox[1][0],
                inst_bbox[0][1]: inst_bbox[1][1],
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]
            inst_contour[:, 1] += inst_bbox[0][0]
            inst_centroid[0] += inst_bbox[0][1]
            inst_centroid[1] += inst_bbox[0][0]
            inst_info_dict[inst_id] = {
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = inst_map_crop == inst_id
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        return pred_inst, inst_info_dict

    def __proc_np_hv(
        self,
        pred: np.ndarray,
        object_size: int = 10,
        ksize: int = 21,
        seed_thresh: float = 0.47,
    ) -> np.ndarray:
        """Process Nuclei Prediction with HV Map using watershed

        Args:
            seed_thresh: watershed 种子阈值，由 tissue 类型动态决定
        """
        pred = np.array(pred, dtype=np.float32)

        blb_raw   = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        # ── Step 1: 二值化核预测 ──────────────────────────────────────────
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1

        # ── Step 2: HV 梯度图 ─────────────────────────────────────────────
        h_dir = cv2.normalize(
            h_dir_raw, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
        )

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(sobelh, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        )
        sobelv = 1 - (
            cv2.normalize(sobelv, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        # ── Step 3: 距离图 ────────────────────────────────────────────────
        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        # ── Step 4: Watershed 种子（Tissue-Aware threshold）────────────────
        # seed_thresh 由 tissue 类型决定：
        #   密集组织（kidney/testis 等）→ 0.38，允许更多种子，分得更细
        #   普通组织 → 0.47，保持原版行为
        overall = np.array(overall >= seed_thresh, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=10)

        # ── Step 5: Watershed ─────────────────────────────────────────────
        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred


# ============================================================
# GT 计算辅助函数（不改动）
# ============================================================
def calculate_instances(
    pred_types: torch.Tensor, pred_insts: torch.Tensor
) -> list[dict]:
    """Best used for GT"""
    type_preds = []
    pred_types = pred_types.permute(0, 2, 3, 1)
    for i in range(pred_types.shape[0]):
        pred_type = torch.argmax(pred_types, dim=-1)[i].detach().cpu().numpy()
        pred_inst = pred_insts[i].detach().cpu().numpy()
        inst_id_list = np.unique(pred_inst)[1:]
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0]: inst_bbox[1][0],
                inst_bbox[0][1]: inst_bbox[1][1],
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]
            inst_contour[:, 1] += inst_bbox[0][0]
            inst_centroid[0] += inst_bbox[0][1]
            inst_centroid[1] += inst_bbox[0][0]
            inst_info_dict[inst_id] = {
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = inst_map_crop == inst_id
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)
        type_preds.append(inst_info_dict)

    return type_preds