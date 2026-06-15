"""
仿照CellViT Fig.4格式生成可视化图：
- 原图+细轮廓线（不填充）
- 每种组织2张patch并排
- GT和Pred各一行
- 竖向文字标签
- 右下角Legend
"""
import os
import sys
sys.path.insert(0, '/root/autodl-tmp/CellViT-main1')

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
import yaml
from skimage import measure, segmentation

from models.segmentation.cell_segmentation.cellvit import CellViTSAM
from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

# ===== 配置 =====
RUN_DIR = Path('/root/autodl-tmp/CellViT-main1/logs/2026-04-10T144126_cellvit_tacnet_v2_fold0')
DATA_DIR = Path('/root/autodl-tmp/CellViT-main/cell_segmentation/datasets/PanNuke_pre/fold0')
OUTPUT_DIR = RUN_DIR / 'paper_figures'
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'

# CellViT论文配色
CELL_COLORS = {
    1: (220, 50,  50),   # Neoplastic  红
    2: (230, 140, 30),   # Epithelial  橙
    3: (80,  180, 80),   # Inflammatory 绿
    4: (50,  100, 220),  # Connective  蓝
    5: (220, 200, 50),   # Dead        黄
}
CELL_NAMES = {
    1: 'Neoplastic',
    2: 'Epithelial',
    3: 'Inflammatory',
    4: 'Connective',
    5: 'Dead',
}
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

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
ckpt = torch.load(RUN_DIR / 'checkpoints/model_best.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)
model.eval()
print("Model loaded!")

post_processor = DetectionCellPostProcessor(nr_types=6, magnification=40, gt=False)
types_df = pd.read_csv(DATA_DIR / 'types.csv')
ALL_TISSUES = [
    'Adrenal gland', 'Bile duct', 'Bladder', 'Breast', 'Cervix',
    'Colon', 'Esophagus', 'Head & Neck', 'Kidney', 'Liver',
    'Lung', 'Ovarian', 'Pancreatic', 'Prostate', 'Skin',
    'Stomach', 'Testis', 'Thyroid', 'Uterus'
]
# types.csv里的实际名称映射
TISSUE_MAP = {t: t for t in types_df['type'].unique()}

# ===== 选图函数 =====
def score_image(label):
    inst_map = label['inst_map']
    type_map = label['type_map']
    n_cells = len(np.unique(inst_map)) - 1
    unique_types = set(type_map[inst_map > 0].astype(int).flatten()) - {0}
    n_types = len(unique_types)
    count_score = 1.0
    if n_cells < 10:
        count_score = 0.2
    elif n_cells < 20:
        count_score = 0.6
    elif 20 <= n_cells <= 70:
        count_score = 1.0
    else:
        count_score = 0.7
    return n_types * 2.0 + count_score, n_cells, n_types

def get_boundary_mask(inst_map):
    """获取实例边界mask"""
    boundary = np.zeros_like(inst_map, dtype=bool)
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        inst_mask = (inst_map == inst_id).astype(np.uint8)
        # 用形态学操作得到边界
        from scipy.ndimage import binary_dilation, binary_erosion
        dilated = binary_dilation(inst_mask, iterations=1)
        eroded = binary_erosion(inst_mask, iterations=1)
        boundary |= (dilated ^ eroded)
    return boundary

def draw_contours_outline(img_np, instance_types, line_width=2):
    """只画轮廓线，不填充，保留原图H&E纹理"""
    img_pil = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    for inst_id, spec in instance_types.items():
        cell_type = spec['type']
        contour = spec.get('contour', None)
        if cell_type == 0 or contour is None or len(contour) < 3:
            continue
        color = CELL_COLORS.get(cell_type)
        if color is None:
            continue
        poly = list(zip(contour[:, 0].tolist(), contour[:, 1].tolist()))
        if len(poly) >= 3:
            draw.polygon(poly, outline=color, width=line_width)
    return np.array(img_pil) / 255.0

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
        # 用marching squares获取平滑轮廓
        try:
            contours = measure.find_contours(mask.astype(float), 0.5)
            if len(contours) == 0:
                continue
            contour = contours[0]
            contour_xy = np.column_stack([contour[:, 1], contour[:, 0]])
        except:
            continue
        gt_instance_types[inst_id] = {'type': cell_type, 'contour': contour_xy}
    return gt_instance_types

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
    # 为pred添加contour
    for inst_id, spec in type_pred.items():
        if 'contour' not in spec:
            spec['contour'] = np.array([[0,0],[0,1],[1,0]])
    return img, type_pred

# ===== 为每种组织选2张最好的图 =====
print("Selecting best images per tissue...")
best_per_tissue = {}

for tissue_display in ALL_TISSUES:
    # 找types.csv中对应的名称
    tissue_csv = None
    for t in types_df['type'].unique():
        if t.lower().replace(' ','').replace('&','').replace('-','') == \
           tissue_display.lower().replace(' ','').replace('&','').replace('-',''):
            tissue_csv = t
            break
    if tissue_csv is None:
        # 模糊匹配
        for t in types_df['type'].unique():
            if tissue_display.lower()[:4] in t.lower():
                tissue_csv = t
                break
    if tissue_csv is None:
        print(f"  WARNING: {tissue_display} not found in types.csv, skipping")
        continue

    tissue_imgs = types_df[types_df['type'] == tissue_csv]['img'].tolist()
    scored = []
    for img_name in tissue_imgs:
        label_path = DATA_DIR / 'labels' / img_name.replace('.png', '.npy')
        try:
            label = np.load(str(label_path), allow_pickle=True).item()
            score, n_cells, n_types = score_image(label)
            scored.append((score, img_name, label))
        except:
            continue
    scored.sort(key=lambda x: -x[0])
    # 选top2，要求两张图不太相似（选不同rank的）
    top2 = scored[:2] if len(scored) >= 2 else scored
    best_per_tissue[tissue_display] = [
        {'img': s[1], 'label': s[2]} for s in top2
    ]
    print(f"  {tissue_display:20s}: {[s[1] for s in top2]}")

# ===== 推理所有选中的图 =====
print("\nRunning inference...")
vis_data = {}  # tissue -> list of (gt_vis, pred_vis, img)

for tissue, infos in best_per_tissue.items():
    vis_data[tissue] = []
    for info in infos:
        img, type_pred = run_inference(info['img'])
        gt_types = get_gt_instance_types(info['label'])
        gt_vis = draw_contours_outline(img, gt_types, line_width=2)
        pred_vis = draw_contours_outline(img, type_pred, line_width=2)
        vis_data[tissue].append((gt_vis, pred_vis))
        print(f"  Done: {tissue} - {info['img']}")

# ===== 拼图（仿CellViT Fig.4）=====
print("\nGenerating paper figure...")

tissues = [t for t in ALL_TISSUES if t in vis_data]
n_tissues = len(tissues)

# 布局：5列×4行block，每block = 2行(GT/Pred) × 2列(patch1/patch2)
# CellViT是5列，每列1种组织，每种2个patch
# 共19种，分5行排（5+5+5+4 = 19）
COLS = 5
import math
ROWS = math.ceil(n_tissues / COLS)

# 每个格子大小
PATCH = 1.8  # inches per patch
fig_w = COLS * 2 * PATCH + 0.4  # 2 patches per tissue + label space
fig_h = ROWS * 2 * PATCH + 0.6  # 2 rows(GT/Pred) per tissue row + legend

fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
fig.patch.set_facecolor('white')

# 用GridSpec精细控制
# 每个tissue占 2列(patch) + hspace，每个row占 2行(GT/Pred) + vspace
# 整体：COLS*2 列，ROWS*2 行

outer = gridspec.GridSpec(
    ROWS, COLS,
    figure=fig,
    hspace=0.15,
    wspace=0.08,
    left=0.06, right=0.98,
    top=0.94, bottom=0.08,
)

for t_idx, tissue in enumerate(tissues):
    row_block = t_idx // COLS
    col_block = t_idx % COLS
    patches = vis_data[tissue]  # list of (gt_vis, pred_vis)

    inner = gridspec.GridSpecFromSubplotSpec(
        2, len(patches),
        subplot_spec=outer[row_block, col_block],
        hspace=0.04,
        wspace=0.03,
    )

    for p_idx, (gt_vis, pred_vis) in enumerate(patches):
        # GT
        ax_gt = fig.add_subplot(inner[0, p_idx])
        ax_gt.imshow(gt_vis)
        ax_gt.axis('off')
        if p_idx == 0:
            ax_gt.set_ylabel('Ground-Truth', fontsize=5, rotation=90,
                             labelpad=3, va='center')

        # Pred
        ax_pred = fig.add_subplot(inner[1, p_idx])
        ax_pred.imshow(pred_vis)
        ax_pred.axis('off')
        if p_idx == 0:
            ax_pred.set_ylabel('Prediction', fontsize=5, rotation=90,
                               labelpad=3, va='center')

    # 组织名标题（只在第一个patch上方）
    ax_title = fig.add_subplot(inner[0, 0])
    ax_title.set_title(tissue, fontsize=6.5, fontweight='bold',
                       pad=3, loc='center')
    ax_title.axis('off')

# 图例（右下角方框，仿CellViT）
legend_patches = [
    mpatches.Patch(facecolor=np.array(CELL_COLORS[k])/255,
                   edgecolor='black', linewidth=0.5,
                   label=CELL_NAMES[k])
    for k in sorted(CELL_COLORS.keys())
]
legend = fig.legend(
    handles=legend_patches,
    loc='lower right',
    bbox_to_anchor=(0.98, 0.01),
    fontsize=7,
    title='Legend',
    title_fontsize=8,
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    ncol=1,
)

# 总标题
fig.suptitle(
    'Fig. X. Example of PanNuke patches with ground-truth annotations '
    'and TCINet predictions overlaid for each tissue type.',
    fontsize=7, y=0.01, va='bottom', style='italic'
)

out_png = OUTPUT_DIR / 'paper_visualization_v2.png'
out_pdf = OUTPUT_DIR / 'paper_visualization_v2.pdf'
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"\n完成！保存到:\n  {out_png}\n  {out_pdf}")
