"""
从每种组织类型里自动选1张"最好"的图：
评分标准：
1. 细胞核种类数（越多越好，最多5种）
2. 细胞核总数适中（20-80个最佳）
3. GT和Pred的bPQ越高越好
"""
import os
import sys
sys.path.insert(0, '/root/autodl-tmp/CellViT-main1')

import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
import yaml
from skimage import measure

from models.segmentation.cell_segmentation.cellvit import CellViTSAM
from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

# ===== 配置 =====
RUN_DIR = Path('/root/autodl-tmp/CellViT-main1/logs/2026-04-10T144126_cellvit_tacnet_v2_fold0')
DATA_DIR = Path('/root/autodl-tmp/CellViT-main/cell_segmentation/datasets/PanNuke_pre/fold0')
OUTPUT_DIR = RUN_DIR / 'paper_figures'
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'

CELL_COLORS = {
    1: (255, 0,   0),    # Neoplastic  红
    2: (0,   200, 0),    # Inflammatory 绿
    3: (30,  100, 255),  # Connective  蓝
    4: (255, 200, 0),    # Dead        黄
    5: (200, 0,   255),  # Epithelial  紫
}
CELL_NAMES = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

# ===== 加载模型 =====
print("Loading model...")
with open(RUN_DIR / 'config.yaml') as f:
    config = yaml.safe_load(f)

model = CellViTSAM(
    model_path=None,
    num_nuclei_classes=config['data']['num_nuclei_classes'],
    num_tissue_classes=config['data']['num_tissue_classes'],
    vit_structure=config['model']['backbone'],
)
ckpt = torch.load(RUN_DIR / 'checkpoints/model_best.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)
model.eval()
print("Model loaded!")

# ===== 加载数据 =====
types_df = pd.read_csv(DATA_DIR / 'types.csv')
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
post_processor = DetectionCellPostProcessor(nr_types=6, magnification=40, gt=False)

ALL_TISSUES = sorted(types_df['type'].unique().tolist())
print(f"Found {len(ALL_TISSUES)} tissue types: {ALL_TISSUES}")

def score_image(label):
    """给一张图打分，越高越好"""
    inst_map = label['inst_map']
    type_map = label['type_map']
    
    # 细胞核总数
    n_cells = len(np.unique(inst_map)) - 1  # 去掉背景0
    
    # 细胞核种类数
    unique_types = set(type_map[inst_map > 0].astype(int).flatten()) - {0}
    n_types = len(unique_types)
    
    # 数量适中奖励（20-80个最佳）
    count_score = 1.0
    if n_cells < 10:
        count_score = 0.3
    elif n_cells < 20:
        count_score = 0.7
    elif 20 <= n_cells <= 80:
        count_score = 1.0
    else:
        count_score = 0.8
    
    return n_types * 2 + count_score, n_cells, n_types

# ===== 为每种组织选最好的图 =====
best_per_tissue = {}
print("\nScoring images per tissue...")

for tissue in ALL_TISSUES:
    tissue_imgs = types_df[types_df['type'] == tissue]['img'].tolist()
    best_score = -1
    best_img = None
    best_label = None
    
    for img_name in tissue_imgs:
        label_path = DATA_DIR / 'labels' / img_name.replace('.png', '.npy')
        label = np.load(str(label_path), allow_pickle=True).item()
        score, n_cells, n_types = score_image(label)
        if score > best_score:
            best_score = score
            best_img = img_name
            best_label = label
    
    best_per_tissue[tissue] = {'img': best_img, 'label': best_label, 'score': best_score}
    print(f"  {tissue:20s}: {best_img}  (score={best_score:.1f})")

# ===== 推理所有选中的图 =====
print("\nRunning inference on selected images...")

def run_inference(img_name):
    img_path = DATA_DIR / 'images' / img_name
    img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
    img_norm = (img - np.array(mean)) / np.array(std)
    img_tensor = torch.tensor(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(img_tensor)
    
    np_map = F.softmax(pred['nuclei_binary_map'], dim=1)
    hv_map = pred['hv_map'].permute(0, 2, 3, 1)
    type_map_pred = F.softmax(pred['nuclei_type_map'], dim=1)
    
    pred_map = np.concatenate([
        torch.argmax(type_map_pred, dim=1)[0].cpu().numpy()[..., None],
        torch.argmax(np_map, dim=1)[0].cpu().numpy()[..., None],
        hv_map[0].cpu().numpy(),
    ], axis=-1)
    
    _, type_pred = post_processor.post_process_cell_segmentation(pred_map)
    return img, type_pred

def get_gt_instance_types(label):
    inst_map = label['inst_map']
    type_map = label['type_map']
    gt_instance_types = {}
    for region in measure.regionprops(inst_map):
        inst_id = region.label
        mask = inst_map == inst_id
        types_in_mask = type_map[mask]
        if len(types_in_mask) == 0:
            continue
        cell_type = int(np.bincount(types_in_mask.astype(int)).argmax())
        coords = region.coords
        contour_xy = np.column_stack([coords[:, 1], coords[:, 0]])
        gt_instance_types[inst_id] = {'type': cell_type, 'contour': contour_xy}
    return gt_instance_types

def draw_contours(img_np, instance_types):
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    for inst_id, spec in instance_types.items():
        cell_type = spec['type']
        contour = spec['contour']
        if cell_type == 0 or len(contour) < 3:
            continue
        color = CELL_COLORS.get(cell_type)
        if color is None:
            continue
        poly = list(zip(contour[:, 0], contour[:, 1]))
        draw.polygon(poly, outline=color, width=2)
    return np.array(img_pil) / 255.0

# 推理并保存结果
vis_results = {}
for tissue in ALL_TISSUES:
    info = best_per_tissue[tissue]
    img_name = info['img']
    label = info['label']
    
    img, type_pred = run_inference(img_name)
    gt_types = get_gt_instance_types(label)
    
    gt_vis = draw_contours(img, gt_types)
    pred_vis = draw_contours(img, type_pred)
    
    vis_results[tissue] = {'gt': gt_vis, 'pred': pred_vis, 'img': img}
    print(f"  Done: {tissue}")

# ===== 拼成论文大图（参考CellViT Fig.4格式）=====
print("\nGenerating paper figure...")

# 19种组织，每种2列(GT/Pred)，排成7行×3列的组织块
# 实际布局：2行(GT+Pred) × 19列 太宽，改成 CellViT的方式
# 每个组织占2行(GT/Pred)，分成多个block排列

n_cols = 7   # 每行几个组织
n_rows = 3   # 需要几行（19/7=3行）

fig_w = n_cols * 2
fig_h = n_rows * 2 * 2  # 每个组织2张图

fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(fig_w, fig_h), dpi=200)
axes = axes.reshape(n_rows * 2, n_cols)

for idx, tissue in enumerate(ALL_TISSUES):
    row_block = idx // n_cols  # 第几个block行
    col = idx % n_cols          # 第几列
    
    gt_row = row_block * 2
    pred_row = row_block * 2 + 1
    
    axes[gt_row, col].imshow(vis_results[tissue]['gt'])
    axes[gt_row, col].set_title(tissue.replace('_', '\n').replace('-', '\n'), 
                                 fontsize=6, fontweight='bold', pad=2)
    axes[gt_row, col].axis('off')
    
    axes[pred_row, col].imshow(vis_results[tissue]['pred'])
    axes[pred_row, col].axis('off')

# 隐藏多余格子（19不能被7整除，最后几个空）
for idx in range(len(ALL_TISSUES), n_rows * n_cols):
    row_block = idx // n_cols
    col = idx % n_cols
    axes[row_block * 2, col].axis('off')
    axes[row_block * 2 + 1, col].axis('off')

# GT/Pred 标签
for r in range(n_rows):
    axes[r*2, 0].set_ylabel('GT', fontsize=8, rotation=0, labelpad=30, va='center')
    axes[r*2+1, 0].set_ylabel('Pred', fontsize=8, rotation=0, labelpad=30, va='center')

# 图例
legend_patches = [
    mpatches.Patch(color=np.array(c)/255, label=CELL_NAMES[k])
    for k, c in CELL_COLORS.items()
]
fig.legend(handles=legend_patches, loc='lower center', ncol=5,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.04, 1, 1])
out_path = OUTPUT_DIR / 'paper_visualization_19tissues.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'paper_visualization_19tissues.pdf', bbox_inches='tight')
print(f"\n完成！图片保存到:\n  {out_path}")
