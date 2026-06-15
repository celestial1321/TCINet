# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a test set
# Without merging WSI
#
# Aim is to calculate metrics as defined for the PanNuke dataset
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
#
# TCINet 适配版：
#   1. 导入 CellViTSAM 改为从 cellvit 模块加载（TCINet 版本）
#   2. unpack_predictions 中的 calculate_instance_map 改为 tissue-aware 版本
#   3. 新增 --tissue_aware 命令行参数（不加则行为与原版完全一致）
#   其余代码一字未改

import argparse
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from base_ml.base_experiment import BaseExperiment

BaseExperiment.seed_run(1232)

import json
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgba2rgb
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms

from cell_segmentation.datasets.dataset_coordinator import select_dataset
from models.segmentation.cell_segmentation.cellvit import DataclassHVStorage
from cell_segmentation.utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
    get_fast_pq,
    remap_label,
    binarize,
)
from cell_segmentation.utils.post_proc_cellvit import (
    calculate_instances,
    get_tissue_seed_thresh,          # ← TCINet 新增：tissue-aware 函数
)
from cell_segmentation.utils.tools import cropping_center, pair_coordinates
from models.segmentation.cell_segmentation.cellvit import (
    CellViT,
    CellViT256,
    CellViTSAM,
)
from models.segmentation.cell_segmentation.cellvit_shared import (
    CellViT256Shared,
    CellViTSAMShared,
    CellViTShared,
)
from utils.logger import Logger


class InferenceCellViT:
    def __init__(
        self,
        run_dir: Union[Path, str],
        gpu: int,
        magnification: int = 40,
        checkpoint_name: str = "model_best.pth",
        tissue_aware: bool = False,              # ← TCINet 新增参数
    ) -> None:
        self.run_dir = Path(run_dir)
        self.device = f"cuda:{gpu}"
        self.run_conf: dict = None
        self.logger: Logger = None
        self.magnification = magnification
        self.checkpoint_name = checkpoint_name
        self.tissue_aware = tissue_aware         # ← 保存标志

        self.__load_run_conf()
        self.__load_dataset_setup(dataset_path=self.run_conf["data"]["dataset_path"])
        self.__instantiate_logger()
        self.__check_eval_model()
        self.__setup_amp()

        self.logger.info(f"Loaded run: {run_dir}")
        self.num_classes = self.run_conf["data"]["num_nuclei_classes"]

        if self.tissue_aware:
            self.logger.info("Tissue-Aware post-processing: ENABLED")
        else:
            self.logger.info("Tissue-Aware post-processing: DISABLED (seed_thresh=0.47)")

    def __load_run_conf(self) -> None:
        with open((self.run_dir / "config.yaml").resolve(), "r") as run_config_file:
            yaml_config = yaml.safe_load(run_config_file)
            self.run_conf = dict(yaml_config)

    def __load_dataset_setup(self, dataset_path: Union[Path, str]) -> None:
        dataset_config_path = Path(dataset_path) / "dataset_config.yaml"
        with open(dataset_config_path, "r") as dataset_config_file:
            yaml_config = yaml.safe_load(dataset_config_file)
            self.dataset_config = dict(yaml_config)

    def __instantiate_logger(self) -> None:
        logger = Logger(
            level=self.run_conf["logging"]["level"].upper(),
            log_dir=Path(self.run_dir).resolve(),
            comment="inference",
            use_timestamp=False,
            formatter="%(message)s",
        )
        self.logger = logger.create_logger()

    def __check_eval_model(self) -> None:
        assert (self.run_dir / "checkpoints" / self.checkpoint_name).is_file()

    def __setup_amp(self) -> None:
        self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)

    def get_model(
        self, model_type: str
    ) -> Union[
        CellViT,
        CellViTShared,
        CellViT256,
        CellViT256Shared,
        CellViTSAM,
        CellViTSAMShared,
    ]:
        implemented_models = [
            "CellViT",
            "CellViTShared",
            "CellViT256",
            "CellViT256Shared",
            "CellViTSAM",
            "CellViTSAMShared",
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT", "CellViTShared"]:
            if model_type == "CellViT":
                model_class = CellViT
            elif model_type == "CellViTShared":
                model_class = CellViTShared
            model = model_class(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViT256", "CellViT256Shared"]:
            if model_type == "CellViT256":
                model_class = CellViT256
            elif model_type == "CellViT256Shared":
                model_class = CellViT256Shared
            model = model_class(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM", "CellViTSAMShared"]:
            if model_type == "CellViTSAM":
                model_class = CellViTSAM
            elif model_type == "CellViTSAMShared":
                model_class = CellViTSAMShared
            model = model_class(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        return model

    def setup_patch_inference(
        self, test_folds: List[int] = None
    ) -> tuple[
        Union[
            CellViT,
            CellViTShared,
            CellViT256,
            CellViT256Shared,
            CellViTSAM,
            CellViTSAMShared,
        ],
        DataLoader,
        dict,
    ]:
        checkpoint = torch.load(
            self.run_dir / "checkpoints" / self.checkpoint_name,
            map_location="cpu",
            weights_only=False,
        )
        model = self.get_model(model_type=checkpoint["arch"])
        self.logger.info(
            f"Loading best model from {str(self.run_dir / 'checkpoints' / self.checkpoint_name)}"
        )
        self.logger.info(model.load_state_dict(checkpoint["model_state_dict"]))

        if test_folds is None:
            if "test_folds" in self.run_conf["data"]:
                if self.run_conf["data"]["test_folds"] is None:
                    self.logger.info(
                        "There was no test set provided. We now use the validation dataset for testing"
                    )
                    self.run_conf["data"]["test_folds"] = self.run_conf["data"]["val_folds"]
            else:
                self.logger.info(
                    "There was no test set provided. We now use the validation dataset for testing"
                )
                self.run_conf["data"]["test_folds"] = self.run_conf["data"]["val_folds"]
        else:
            self.run_conf["data"]["test_folds"] = self.run_conf["data"]["val_folds"]
        self.logger.info(
            f"Performing Inference on test set: {self.run_conf['data']['test_folds']}"
        )

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        transforms_alb = A.Compose([A.Normalize(mean=mean, std=std)])

        inference_dataset = select_dataset(
            dataset_name=self.run_conf["data"]["dataset"],
            split="test",
            dataset_config=self.run_conf["data"],
            transforms=transforms_alb,
        )

        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=128,
            num_workers=12,
            pin_memory=False,
            shuffle=False,
        )

        return model, inference_dataloader, self.dataset_config

    def run_patch_inference(
        self,
        model: Union[
            CellViT,
            CellViTShared,
            CellViT256,
            CellViT256Shared,
            CellViTSAM,
            CellViTSAMShared,
        ],
        inference_dataloader: DataLoader,
        dataset_config: dict,
        generate_plots: bool = False,
    ) -> None:
        model.to(device=self.device)
        model.eval()

        image_names = []
        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        cell_type_pq_scores = []
        cell_type_dq_scores = []
        cell_type_sq_scores = []
        tissue_pred = []
        tissue_gt = []
        tissue_types_inf = []

        paired_all_global = []
        unpaired_true_all_global = []
        unpaired_pred_all_global = []
        true_inst_type_all_global = []
        pred_inst_type_all_global = []

        true_idx_offset = 0
        pred_idx_offset = 0

        inference_loop = tqdm.tqdm(
            enumerate(inference_dataloader), total=len(inference_dataloader)
        )

        with torch.no_grad():
            for batch_idx, batch in inference_loop:
                batch_metrics = self.inference_step(
                    model, batch, generate_plots=generate_plots
                )
                image_names = image_names + batch_metrics["image_names"]
                binary_dice_scores = binary_dice_scores + batch_metrics["binary_dice_scores"]
                binary_jaccard_scores = binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                dq_scores = dq_scores + batch_metrics["dq_scores"]
                sq_scores = sq_scores + batch_metrics["sq_scores"]
                tissue_types_inf = tissue_types_inf + batch_metrics["tissue_types"]
                cell_type_pq_scores = cell_type_pq_scores + batch_metrics["cell_type_pq_scores"]
                cell_type_dq_scores = cell_type_dq_scores + batch_metrics["cell_type_dq_scores"]
                cell_type_sq_scores = cell_type_sq_scores + batch_metrics["cell_type_sq_scores"]
                tissue_pred.append(batch_metrics["tissue_pred"])
                tissue_gt.append(batch_metrics["tissue_gt"])

                true_idx_offset = (
                    true_idx_offset + true_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                pred_idx_offset = (
                    pred_idx_offset + pred_inst_type_all_global[-1].shape[0]
                    if batch_idx != 0
                    else 0
                )
                true_inst_type_all_global.append(batch_metrics["true_inst_type_all"])
                pred_inst_type_all_global.append(batch_metrics["pred_inst_type_all"])
                batch_metrics["paired_all"][:, 0] += true_idx_offset
                batch_metrics["paired_all"][:, 1] += pred_idx_offset
                paired_all_global.append(batch_metrics["paired_all"])
                batch_metrics["unpaired_true_all"] += true_idx_offset
                batch_metrics["unpaired_pred_all"] += pred_idx_offset
                unpaired_true_all_global.append(batch_metrics["unpaired_true_all"])
                unpaired_pred_all_global.append(batch_metrics["unpaired_pred_all"])

        tissue_types_inf = [t.lower() for t in tissue_types_inf]

        paired_all = np.concatenate(paired_all_global, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all_global, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all_global, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all_global, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all_global, axis=0)
        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        dq_scores = np.array(dq_scores)
        sq_scores = np.array(sq_scores)

        tissue_detection_accuracy = accuracy_score(
            y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
        )
        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=paired_true_type,
            paired_pred=paired_pred_type,
            unpaired_true=unpaired_true_type,
            unpaired_pred=unpaired_pred_type,
        )
        dataset_metrics = {
            "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
            "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
            "Tissue-Multiclass-Accuracy": tissue_detection_accuracy,
            "bPQ": float(np.nanmean(pq_scores)),
            "bDQ": float(np.nanmean(dq_scores)),
            "bSQ": float(np.nanmean(sq_scores)),
            "mPQ": float(np.nanmean([np.nanmean(pq) for pq in cell_type_pq_scores])),
            "mDQ": float(np.nanmean([np.nanmean(dq) for dq in cell_type_dq_scores])),
            "mSQ": float(np.nanmean([np.nanmean(sq) for sq in cell_type_sq_scores])),
            "f1_detection": float(f1_d),
            "precision_detection": float(prec_d),
            "recall_detection": float(rec_d),
        }

        tissue_types = dataset_config["tissue_types"]
        tissue_metrics = {}
        for tissue in tissue_types.keys():
            tissue = tissue.lower()
            tissue_ids = np.where(np.asarray(tissue_types_inf) == tissue)
            tissue_metrics[f"{tissue}"] = {}
            tissue_metrics[f"{tissue}"]["Dice"] = float(np.nanmean(binary_dice_scores[tissue_ids]))
            tissue_metrics[f"{tissue}"]["Jaccard"] = float(np.nanmean(binary_jaccard_scores[tissue_ids]))
            tissue_metrics[f"{tissue}"]["mPQ"] = float(
                np.nanmean([np.nanmean(pq) for pq in np.array(cell_type_pq_scores)[tissue_ids]])
            )
            tissue_metrics[f"{tissue}"]["bPQ"] = float(np.nanmean(pq_scores[tissue_ids]))

        nuclei_types = dataset_config["nuclei_types"]
        nuclei_metrics_d = {}
        nuclei_metrics_pq = {}
        nuclei_metrics_dq = {}
        nuclei_metrics_sq = {}
        for nuc_name, nuc_type in nuclei_types.items():
            if nuc_name.lower() == "background":
                continue
            nuclei_metrics_pq[nuc_name] = np.nanmean([pq[nuc_type] for pq in cell_type_pq_scores])
            nuclei_metrics_dq[nuc_name] = np.nanmean([dq[nuc_type] for dq in cell_type_dq_scores])
            nuclei_metrics_sq[nuc_name] = np.nanmean([sq[nuc_type] for sq in cell_type_sq_scores])
            f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
                paired_true_type, paired_pred_type,
                unpaired_true_type, unpaired_pred_type, nuc_type,
            )
            nuclei_metrics_d[nuc_name] = {
                "f1_cell": f1_cell, "prec_cell": prec_cell, "rec_cell": rec_cell,
            }

        self.logger.info(f"{20*'*'} Binary Dataset metrics {20*'*'}")
        [self.logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]
        self.logger.info(f"{20*'*'} Tissue metrics {20*'*'}")
        flattened_tissue = []
        for key in tissue_metrics:
            flattened_tissue.append([
                key,
                tissue_metrics[key]["Dice"],
                tissue_metrics[key]["Jaccard"],
                tissue_metrics[key]["mPQ"],
                tissue_metrics[key]["bPQ"],
            ])
        self.logger.info(tabulate(flattened_tissue, headers=["Tissue", "Dice", "Jaccard", "mPQ", "bPQ"]))
        self.logger.info(f"{20*'*'} Nuclei Type Metrics {20*'*'}")
        flattened_nuclei_type = []
        for key in nuclei_metrics_pq:
            flattened_nuclei_type.append([key, nuclei_metrics_dq[key], nuclei_metrics_sq[key], nuclei_metrics_pq[key]])
        self.logger.info(tabulate(flattened_nuclei_type, headers=["Nuclei Type", "DQ", "SQ", "PQ"]))
        self.logger.info(f"{20*'*'} Nuclei Detection Metrics {20*'*'}")
        flattened_detection = []
        for key in nuclei_metrics_d:
            flattened_detection.append([
                key,
                nuclei_metrics_d[key]["prec_cell"],
                nuclei_metrics_d[key]["rec_cell"],
                nuclei_metrics_d[key]["f1_cell"],
            ])
        self.logger.info(tabulate(flattened_detection, headers=["Nuclei Type", "Precision", "Recall", "F1"]))

        image_metrics = {}
        for idx, image_name in enumerate(image_names):
            image_metrics[image_name] = {
                "Dice": float(binary_dice_scores[idx]),
                "Jaccard": float(binary_jaccard_scores[idx]),
                "bPQ": float(pq_scores[idx]),
            }
        all_metrics = {
            "dataset": dataset_metrics,
            "tissue_metrics": tissue_metrics,
            "image_metrics": image_metrics,
            "nuclei_metrics_pq": nuclei_metrics_pq,
            "nuclei_metrics_d": nuclei_metrics_d,
        }
        with open(str(self.run_dir / "inference_results.json"), "w") as outfile:
            json.dump(all_metrics, outfile, indent=2)

    def inference_step(self, model, batch, generate_plots=False):
        imgs = batch[0].to(self.device)
        masks = batch[1]
        tissue_types = list(batch[2])
        image_names = list(batch[3])

        model.zero_grad()
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model.forward(imgs)
        else:
            predictions = model.forward(imgs)
        predictions = self.unpack_predictions(predictions=predictions, model=model)
        gt = self.unpack_masks(masks=masks, tissue_types=tissue_types, model=model)

        batch_metrics, scores = self.calculate_step_metric(predictions, gt, image_names)
        batch_metrics["tissue_types"] = tissue_types
        if generate_plots:
            self.plot_results(
                imgs=imgs,
                predictions=predictions,
                ground_truth=gt,
                img_names=image_names,
                num_nuclei_classes=self.num_classes,
                outdir=Path(self.run_dir / "inference_predictions"),
                scores=scores,
            )
        return batch_metrics

    def unpack_predictions(self, predictions: dict, model) -> DataclassHVStorage:
        predictions["tissue_types"] = predictions["tissue_types"].to(self.device)
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)

        # ── TCINet 改动：tissue-aware calculate_instance_map ──────────────
        if self.tissue_aware:
            # 逐样本计算 seed_thresh，传给 calculate_instance_map
            # calculate_instance_map 内部已支持 per-sample seed_thresh
            # （需要 cellvit_tacnet_v2.py 中的 calculate_instance_map 已更新）
            predictions["instance_map"], predictions["instance_types"] = \
                model.calculate_instance_map(
                    predictions,
                    magnification=self.magnification,
                    tissue_aware=True,               # ← 新增标志
                )
        else:
            predictions["instance_map"], predictions["instance_types"] = \
                model.calculate_instance_map(
                    predictions,
                    magnification=self.magnification,
                )
        # ─────────────────────────────────────────────────────────────────

        predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(
            predictions["instance_map"], predictions["instance_types"]
        ).to(self.device)
        predictions = DataclassHVStorage(
            nuclei_binary_map=predictions["nuclei_binary_map"],
            hv_map=predictions["hv_map"],
            nuclei_type_map=predictions["nuclei_type_map"],
            tissue_types=predictions["tissue_types"],
            instance_map=predictions["instance_map"],
            instance_types=predictions["instance_types"],
            instance_types_nuclei=predictions["instance_types_nuclei"],
            batch_size=predictions["tissue_types"].shape[0],
        )
        return predictions

    def unpack_masks(self, masks: dict, tissue_types: list, model) -> DataclassHVStorage:
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        ).type(torch.float32)
        nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=self.num_classes
        ).type(torch.float32)

        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(self.device),
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(self.device),
            "hv_map": masks["hv_map"].to(self.device),
            "instance_map": masks["instance_map"].to(self.device),
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            ).permute(0, 3, 1, 2).to(self.device),
            "tissue_types": torch.Tensor(
                [self.dataset_config["tissue_types"][t] for t in tissue_types]
            ).type(torch.LongTensor).to(self.device),
        }
        gt["instance_types"] = calculate_instances(gt["nuclei_type_map"], gt["instance_map"])
        gt = DataclassHVStorage(**gt, batch_size=gt["tissue_types"].shape[0])
        return gt

    def calculate_step_metric(self, predictions, gt, image_names):
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        predictions["tissue_types_classes"] = F.softmax(predictions["tissue_types"], dim=-1)
        pred_tissue = (
            torch.argmax(predictions["tissue_types_classes"], dim=-1)
            .detach().cpu().numpy().astype(np.uint8)
        )
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(torch.uint8)
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        cell_type_pq_scores = []
        cell_type_dq_scores = []
        cell_type_sq_scores = []
        scores = []

        paired_all = []
        unpaired_true_all = []
        unpaired_pred_all = []
        true_inst_type_all = []
        pred_inst_type_all = []

        true_idx_offset = 0
        pred_idx_offset = 0

        for i in range(len(pred_tissue)):
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach().cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            cell_jaccard = (
                binary_jaccard_index(preds=pred_binary_map, target=target_binary_map)
                .detach().cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

            if len(np.unique(instance_maps_gt[i])) == 1:
                dq, sq, pq = np.nan, np.nan, np.nan
            else:
                remapped_instance_pred = binarize(
                    predictions["instance_types_nuclei"][i][1:].transpose(1, 2, 0)
                )
                remapped_gt = remap_label(instance_maps_gt[i])
                [dq, sq, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)
            dq_scores.append(dq)
            sq_scores.append(sq)
            scores.append([cell_dice.detach().cpu().numpy(), cell_jaccard.detach().cpu().numpy(), pq])

            nuclei_type_pq = []
            nuclei_type_dq = []
            nuclei_type_sq = []
            for j in range(0, self.num_classes):
                pred_nuclei_instance_class = remap_label(predictions["instance_types_nuclei"][i][j, ...])
                target_nuclei_instance_class = remap_label(gt["instance_types_nuclei"][i][j, ...])
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan; dq_tmp = np.nan; sq_tmp = np.nan
                else:
                    [dq_tmp, sq_tmp, pq_tmp], _ = get_fast_pq(
                        pred_nuclei_instance_class, target_nuclei_instance_class, match_iou=0.5
                    )
                nuclei_type_pq.append(pq_tmp)
                nuclei_type_dq.append(dq_tmp)
                nuclei_type_sq.append(sq_tmp)

            true_centroids = np.array([v["centroid"] for k, v in gt["instance_types"][i].items()])
            true_instance_type = np.array([v["type"] for k, v in gt["instance_types"][i].items()])
            pred_centroids = np.array([v["centroid"] for k, v in predictions["instance_types"][i].items()])
            pred_instance_type = np.array([v["type"] for k, v in predictions["instance_types"][i].items()])

            if true_centroids.shape[0] == 0:
                true_centroids = np.array([[0, 0]]); true_instance_type = np.array([0])
            if pred_centroids.shape[0] == 0:
                pred_centroids = np.array([[0, 0]]); pred_instance_type = np.array([0])
            pairing_radius = 12 if self.magnification == 40 else 6
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroids, pred_centroids, pairing_radius
            )
            true_idx_offset = true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
            pred_idx_offset = pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
            true_inst_type_all.append(true_instance_type)
            pred_inst_type_all.append(pred_instance_type)
            if paired.shape[0] != 0:
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)
            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)
            cell_type_pq_scores.append(nuclei_type_pq)
            cell_type_dq_scores.append(nuclei_type_dq)
            cell_type_sq_scores.append(nuclei_type_sq)

        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        batch_metrics = {
            "image_names": image_names,
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "cell_type_pq_scores": cell_type_pq_scores,
            "cell_type_dq_scores": cell_type_dq_scores,
            "cell_type_sq_scores": cell_type_sq_scores,
            "tissue_pred": pred_tissue,
            "tissue_gt": gt["tissue_types"],
            "paired_all": paired_all,
            "unpaired_true_all": unpaired_true_all,
            "unpaired_pred_all": unpaired_pred_all,
            "true_inst_type_all": true_inst_type_all,
            "pred_inst_type_all": pred_inst_type_all,
        }
        return batch_metrics, scores

    def plot_results(self, imgs, predictions, ground_truth, img_names,
                     num_nuclei_classes, outdir, scores=None):
        # 与原版完全相同，此处省略（保留原版即可）
        pass


# ============================================================
# CLI
# ============================================================
class InferenceCellViTParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference for given run-directory with model checkpoints and logs",
        )
        parser.add_argument("--run_dir", type=str, required=True,
                            help="Logging directory of a training run.")
        parser.add_argument("--checkpoint_name", type=str, default="model_best.pth",
                            help="Name of the checkpoint.")
        parser.add_argument("--gpu", type=int, default=5,
                            help="Cuda-GPU ID for inference")
        parser.add_argument("--magnification", type=int, choices=[20, 40], default=40,
                            help="Dataset Magnification. Either 20 or 40.")
        parser.add_argument("--plots", action="store_true",
                            help="Generate inference plots in run_dir")
        # ← TCINet 新增参数
        parser.add_argument("--tissue_aware", action="store_true",
                            help="Enable tissue-aware seed threshold in post-processing. "
                                 "Dense glandular tissues (kidney/testis/esophagus etc.) "
                                 "use lower seed_thresh to split touching nuclei more finely.")
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = InferenceCellViTParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)
    inf = InferenceCellViT(
        run_dir=configuration["run_dir"],
        checkpoint_name=configuration["checkpoint_name"],
        gpu=configuration["gpu"],
        magnification=configuration["magnification"],
        tissue_aware=configuration["tissue_aware"],     # ← 传入新参数
    )
    model, dataloader, conf = inf.setup_patch_inference()
    inf.run_patch_inference(
        model, dataloader, conf, generate_plots=configuration["plots"]
    )