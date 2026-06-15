import os, sys
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
from skimage import measure

from models.segmentation.cell_segmentation.cellvit import CellViTSAM
from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

RUN_DIR = Path('/root/autodl-tmp/CellViT-main1/logs/2026-04-10T144126_cellvit_tacnet_v2_fold0')
DATA_DIR = Path('/root/autodl-tmp/CellViT-main/cell_segmentation/datasets/PanNuke_pre/fold0')
OUTPUT_DIR = RUN_DIR / 'paper_figures'
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'

CELL_COLORS = {
    1: (220, 50,  50),
    2: (230, 140, 30),
    3: (80,  180, 80),
    4: (50,  100, 220),
    5: (220, 200, 50),
}
CELL_NAMES = {
    1: 'Neoplastic',
    2: 'Epithelial',
    3: 'Inflammatory',
    4: 'Connective',
    5: 'Dead',
}
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# 加载模型
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

post_processor = DetectionCellPostProcessor(nr_types=6, magnification=40, gt=False)
types_df = pd.read_csv(DATA_DIR / 'types.csv')

ALL_TISSUES_DISPLAY = [
    'Adrenal Gland', 'Bile Duct', 'Bladder', 'Breast', 'Cervix',
    'Colon', 'Esophagus', 'Head & Neck', 'Kidney', 'Liver',
    'Lung', 'Ovarian', 'Pancreatic', 'Prostate', 'Skin',
    'Stomach', 'Testis', 'Thyroid', 'Uterus'
]

def find_tissue_in_csv(tissue_display):
    for t in types_df['type'].unique():
        a = t.lower().replace(' ','').replace('&','').replace('-','').replace('_','')
        b = tissue_display.lower().replace(' ','').replace('&','').replace('-','').replace('_','')
        if a == b or a[:5] == b[:5]:
            return t
    return None

def score_image(label):
    inst_map = label['inst_map']
    type_map = label['type_map']
    n_cells = len(np.unique(inst_map)) - 1
    unique_types = set(type_map[inst_map > 0].astype(int).flatten()) - {0}
    n_types = len(unique_types)
    if n_cells < 10: count_score = 0.2
    elif n_cells < 20: count_score = 0.6
    elif 20 <= n_cells <= 70: count_score = 1.0
    else: count_score = 0.7
    return n_types * 2.0 + count_score

def get_gt_instance_types(label):
    inst_map = label['inst_map']
    type_map = label['type_map']
    result = {}
    for region in measure.regionprops(inst_map):
        inst_id = region.label
        mask = inst_map == inst_id
        types_in_mask = type_map[mask]
        if len(types_in_mask) == 0: continue
        cell_type = int(np.bincount(types_in_mask.astype(int)).argmax())
        try:
            contours = measure.find_contours(mask.astype(float), 0.5)
            if not contours: continue
            c = contours[0]
            contour_xy = np.column_stack([c[:, 1], c[:, 0]])
        except: continue
        result[inst_id] = {'type': cell_type, 'contour': contour_xy}
    return result

def draw_contours_outline(img_np, instance_types, line_width=2):
    img_pil = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    for inst_id, spec in instance_types.items():
        cell_type = spec['type']
        contour = spec.get('contour', None)
        if cell_type == 0 or contour is None or len(contour) < 3: continue
        color = CELL_COLORS.get(cell_type)
        if color is None: continue
        poly = list(zip(contour[:, 0].tolist(), contour[:, 1].tolist()))
        if len(poly) >= 3:
            draw.polygon(poly, outline=color, width=line_width)
    return np.array(img_pil) / 255.0

def run_inference(img_name):
    img = np.array(Image.open(DATA_DIR / 'images' / img_name).convert('RGB')).astype(np.float32) / 255.0
    img_norm = (img - np.array(mean)) / np.array(std)
    img_tensor = torch.tensor(img_norm).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(img_tensor)
    np_map = F.softmax(pred['nuclei_binary_map'], dim=1)
    hv_map = pred['hv_map'].permute(0,2,3,1)
    type_map_pred = F.softmax(pred['nuclei_type_map'], dim=1)
    pred_map = np.concatenate([
        torch.argmax(type_map_pred, dim=1)[0].cpu().numpy()[..., None],
        torch.argmax(np_map, dim=1)[0].cpu().numpy()[..., None],
        hv_map[0].cpu().numpy(),
    ], axis=-1)
    _, type_pred = post_processor.post_process_cell_segmentation(pred_map)
    return img, type_pred

# 选图
print("Selecting images...")
best_per_tissue = {}
for tissue_display in ALL_TISSUES_DISPLAY:
    tissue_csv = find_tissue_in_csv(tissue_display)
    if tissue_csv is None:
        print(f"  WARNING: {tissue_display} not found")
        continue
    tissue_imgs = types_df[types_df['type'] == tissue_csv]['img'].tolist()
    scored = []
    for img_name in tissue_imgs:
        label_path = DATA_DIR / 'labels' / img_name.replace('.png', '.npy')
        try:
            label = np.load(str(label_path), allow_pickle=True).item()
            scored.append((score_image(label), img_name, label))
        except: continue
    scored.sort(key=lambda x: -x[0])
    best_per_tissue[tissue_display] = [{'img': s[1], 'label': s[2]} for s in scored[:2]]
    print(f"  {tissue_display}: {[s[1] for s in scored[:2]]}")

# 推理
print("\nRunning inference...")
vis_data = {}
for tissue, infos in best_per_tissue.items():
    vis_data[tissue] = []
    for info in infos:
        img, type_pred = run_inference(info['img'])
        gt_types = get_gt_instance_types(info['label'])
        gt_vis = draw_contours_outline(img, gt_types, line_width=2)
        pred_vis = draw_contours_outline(img, type_pred, line_width=2)
        vis_data[tissue].append((gt_vis, pred_vis))
        print(f"  Done: {tissue}")

# 绘图
print("\nGenerating figure...")
tissues = [t for t in ALL_TISSUES_DISPLAY if t in vis_data]
COLS = 5
import math
ROWS = math.ceil(len(tissues) / COLS)

PATCH = 2.0
fig_w = COLS * 2 * PATCH + 0.5
fig_h = ROWS * 2 * PATCH + 0.3

fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
fig.patch.set_facecolor('white')

outer = gridspec.GridSpec(
    ROWS, COLS,
    figure=fig,
    hspace=0.22,
    wspace=0.06,
    left=0.04, right=0.97,
    top=0.97, bottom=0.10,
)

for t_idx, tissue in enumerate(tissues):
    row_block = t_idx // COLS
    col_block = t_idx % COLS
    patches_data = vis_data[tissue]

    inner = gridspec.GridSpecFromSubplotSpec(
        2, len(patches_data),
        subplot_spec=outer[row_block, col_block],
        hspace=0.05,
        wspace=0.03,
    )

    for p_idx, (gt_vis, pred_vis) in enumerate(patches_data):
        ax_gt = fig.add_subplot(inner[0, p_idx])
        ax_gt.imshow(gt_vis)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        for spine in ax_gt.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.5)

        ax_pred = fig.add_subplot(inner[1, p_idx])
        ax_pred.imshow(pred_vis)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        for spine in ax_pred.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.5)

        # GT / Pred 左侧小标签
        if p_idx == 0:
            ax_gt.set_ylabel('GT', fontsize=6, rotation=90,
                             labelpad=2, va='center', color='#444444')
            ax_pred.set_ylabel('Pred', fontsize=6, rotation=90,
                               labelpad=2, va='center', color='#444444')

    # 组织名标题 —— 横跨两个patch
    # 用 outer 坐标系添加一个不可见的大轴来放标题
    big_ax = fig.add_subplot(outer[row_block, col_block])
    big_ax.set_facecolor('none')
    for spine in big_ax.spines.values():
        spine.set_visible(False)
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_title(tissue, fontsize=9, fontweight='bold',
                     pad=5, color='#111111',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0',
                               edgecolor='#bbbbbb', linewidth=0.8))

# Legend
legend_patches = [
    mpatches.Patch(
        facecolor=np.array(CELL_COLORS[k])/255,
        edgecolor='#333333',
        linewidth=0.8,
        label=CELL_NAMES[k]
    )
    for k in sorted(CELL_COLORS.keys())
]
legend = fig.legend(
    handles=legend_patches,
    loc='lower right',
    bbox_to_anchor=(0.97, 0.01),
    fontsize=9,
    title='Legend',
    title_fontsize=10,
    frameon=True,
    framealpha=1.0,
    edgecolor='#333333',
    fancybox=False,
    ncol=1,
    handlelength=1.2,
    handleheight=1.2,
    borderpad=0.8,
    labelspacing=0.5,
)
legend.get_frame().set_linewidth(1.0)

out_png = OUTPUT_DIR / 'paper_visualization_v3.png'
out_pdf = OUTPUT_DIR / 'paper_visualization_v3.pdf'
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"\n完成！\n  {out_png}\n  {out_pdf}")
