import os, sys, re, urllib.request
sys.path.insert(0, '/root/autodl-tmp/CellViT-main1')

import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as F
import yaml
from skimage import measure

# ============================================================
# 全局字体: 优先 Arial, 没有就用视觉接近的 Liberation Sans / DejaVu Sans
#
# 在 Linux 服务器上,Arial 因版权原因不预装。有三种获取方式:
#   方式A(推荐,无需 root): 用 pip 装一个带 Arial 字体文件的包
#       pip install matplotlib-font-arial    (社区包)
#       或者手动把任何 Arial.ttf 文件放到  ARIAL_DIR
#   方式B(要 sudo): apt install ttf-mscorefonts-installer
#   方式C: 脚本会自动退回 Liberation Sans / DejaVu Sans
#         Liberation Sans 是 Arial 的免费度量兼容替代品,视觉几乎看不出差别
# ============================================================
ARIAL_DIR = Path('/root/autodl-tmp/CellViT-main1/_fonts_arial')
ARIAL_DIR.mkdir(parents=True, exist_ok=True)

def _setup_font():
    from matplotlib import font_manager

    # 1. 扫描用户放在 ARIAL_DIR 下的任何 .ttf / .otf
    for font_file in list(ARIAL_DIR.glob('*.ttf')) + list(ARIAL_DIR.glob('*.otf')):
        try:
            font_manager.fontManager.addfont(str(font_file))
            print(f"[字体] 已注册用户字体: {font_file.name}")
        except Exception as e:
            print(f"[字体] 注册 {font_file.name} 失败: {e}")

    # 2. 检查是否有 Arial
    for f in font_manager.fontManager.ttflist:
        if f.name.lower() == 'arial':
            print(f"[OK] 使用 Arial: {f.fname}")
            return 'Arial'

    # 3. 退回 Liberation Sans (度量 100% 兼容 Arial,视觉差别极小)
    for f in font_manager.fontManager.ttflist:
        if f.name.lower() == 'liberation sans':
            print(f"[INFO] 未找到 Arial, 使用度量兼容的 Liberation Sans: {f.fname}")
            return 'Liberation Sans'

    # 4. 最后回退
    print("[警告] 未找到 Arial / Liberation Sans, 使用 DejaVu Sans")
    print("  如需 Arial:")
    print(f"  1) 把任意 Arial.ttf 文件复制到: {ARIAL_DIR}/")
    print("  2) 或: sudo apt install -y fonts-liberation")
    print("  3) 然后重跑此脚本")
    return 'DejaVu Sans'

FONT_NAME = _setup_font()

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [FONT_NAME, 'Liberation Sans', 'DejaVu Sans']

# ====== TCINet ======
from models.segmentation.cell_segmentation.cellvit import CellViTSAM as TCINet_CellViTSAM
from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

# ====== baseline 动态加载 ======
BASELINE_SRC  = Path('/root/autodl-tmp/CellViT-main1/models/A cellvit baseline.py')
BASELINE_COPY = Path('/root/autodl-tmp/CellViT-main1/models/segmentation/cell_segmentation/cellvit_baseline_auto.py')

def _prepare_baseline_file():
    with open(BASELINE_SRC, 'r', encoding='utf-8') as f:
        src = f.read()
    ABS_PREFIX = 'models.segmentation.cell_segmentation'
    src = re.sub(r'^from\s+\.([A-Za-z_][\w\.]*)\s+import\s',
                 lambda m: f'from {ABS_PREFIX}.{m.group(1)} import ', src, flags=re.MULTILINE)
    src = re.sub(r'^from\s+\.\s+import\s', f'from {ABS_PREFIX} import ', src, flags=re.MULTILINE)
    src = re.sub(r'^import\s+\.([A-Za-z_][\w\.]*)',
                 lambda m: f'import {ABS_PREFIX}.{m.group(1)}', src, flags=re.MULTILINE)
    BASELINE_COPY.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_COPY, 'w', encoding='utf-8') as f:
        f.write(src)

def _load_baseline_module():
    _prepare_baseline_file()
    spec = importlib.util.spec_from_file_location("cellvit_baseline_auto", str(BASELINE_COPY))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cellvit_baseline_auto"] = mod
    spec.loader.exec_module(mod)
    return mod

print("Loading baseline CellViT module...")
_baseline_mod = _load_baseline_module()
Baseline_CellViTSAM = _baseline_mod.CellViTSAM
print("  done.\n")

# ========== 路径 ==========
TCINET_RUN_DIR  = Path('/root/autodl-tmp/CellViT-main1/logs/2026-04-10T144126_cellvit_tacnet_v2_fold0')
CELLVIT_RUN_DIR = Path('/root/autodl-tmp/CellViT-main1/logs/2026-04-14T140855_ablation_no_boundary_fold0')

DATA_ROOT = Path('/root/autodl-tmp/CellViT-main/cell_segmentation/datasets/PanNuke_pre')
FOLDS = ['fold0', 'fold1', 'fold2']
OUTPUT_DIR = TCINET_RUN_DIR / 'paper_figures'
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'

CELL_COLORS = {
    1: (220, 50,  50), 2: (220, 160, 30), 3: (50,  180, 50),
    4: (50,  80,  220), 5: (220, 210, 40),
}
CELL_NAMES = {1:'Neoplastic',2:'Epithelial',3:'Inflammatory',4:'Connective',5:'Dead'}
LEGEND_ORDER = [3, 4, 1, 5, 2]

mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)

ALL_TISSUES_DISPLAY = [
    'Adrenal','Bile-duct','Bladder','Breast','Cervix',
    'Colon','Esophagus','HeadNeck','Kidney','Liver',
    'Lung','Ovarian','Pancreatic','Prostate','Skin',
    'Stomach','Testis','Thyroid','Uterus'
]
TISSUE_CSV_MAP = {
    'Adrenal':'Adrenal gland','Bile-duct':'Bile duct','Bladder':'Bladder',
    'Breast':'Breast','Cervix':'Cervix','Colon':'Colon','Esophagus':'Esophagus',
    'HeadNeck':'Head & Neck','Kidney':'Kidney','Liver':'Liver','Lung':'Lung',
    'Ovarian':'Ovarian','Pancreatic':'Pancreatic','Prostate':'Prostate',
    'Skin':'Skin','Stomach':'Stomach','Testis':'Testis','Thyroid':'Thyroid','Uterus':'Uterus',
}

# ============================================================
# 模型加载
# ============================================================
def load_model(run_dir, name, model_class):
    print(f"Loading {name} from {run_dir.name}...")
    with open(run_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    m = model_class(
        model_path=None,
        num_nuclei_classes=cfg['data']['num_nuclei_classes'],
        num_tissue_classes=cfg['data']['num_tissue_classes'],
        vit_structure=cfg['model']['backbone'],
    )
    ckpt = torch.load(run_dir/'checkpoints/model_best.pth', map_location='cpu', weights_only=False)
    try:
        m.load_state_dict(ckpt['model_state_dict'], strict=True)
    except RuntimeError:
        m.load_state_dict(ckpt['model_state_dict'], strict=False)
    m.to(DEVICE); m.eval()
    return m

model_tcinet  = load_model(TCINET_RUN_DIR,  'TCINet',             TCINet_CellViTSAM)
model_cellvit = load_model(CELLVIT_RUN_DIR, 'CellViT (baseline)', Baseline_CellViTSAM)

post_processor = DetectionCellPostProcessor(nr_types=6, magnification=40, gt=False)

# ============================================================
# 数据汇总
# ============================================================
print("\n=== 汇总 fold 索引 ===")
all_dfs = []
for fd in FOLDS:
    csv_path = DATA_ROOT / fd / 'types.csv'
    if not csv_path.exists(): continue
    df = pd.read_csv(csv_path); df['fold'] = fd
    all_dfs.append(df)
    print(f"  {fd}: {len(df)}")
types_df = pd.concat(all_dfs, ignore_index=True)
print(f"  合计: {len(types_df)}\n")

actual_types = sorted(types_df['type'].unique())

def fuzzy_match_tissue(disp, types_list):
    mapped = TISSUE_CSV_MAP.get(disp)
    if mapped in types_list: return mapped
    key = disp.lower().replace('-','').replace(' ','').replace('&','')
    for t in types_list:
        t_key = t.lower().replace('-','').replace(' ','').replace('&','')
        if key in t_key or t_key in key: return t
    if disp == 'HeadNeck':
        for t in types_list:
            if 'head' in t.lower() or 'neck' in t.lower(): return t
    return None

FIXED_TISSUE_MAP = {t: fuzzy_match_tissue(t, actual_types) for t in ALL_TISSUES_DISPLAY}

# ============================================================
# 工具函数
# ============================================================
def load_label(fd, img_name):
    lp = DATA_ROOT/fd/'labels'/img_name.replace('.png','.npy')
    if not lp.exists(): return None
    try: return np.load(str(lp), allow_pickle=True).item()
    except: return None

def get_gt_instance_types(label):
    inst_map = label['inst_map']; type_map = label['type_map']
    result = {}
    for region in measure.regionprops(inst_map):
        inst_id = region.label
        mask = inst_map == inst_id
        tmask = type_map[mask]
        if len(tmask) == 0: continue
        cell_type = int(np.bincount(tmask.astype(int)).argmax())
        try:
            contours = measure.find_contours(mask.astype(float), 0.5)
            if not contours: continue
            c = contours[0]
            contour_xy = np.column_stack([c[:,1], c[:,0]])
        except: continue
        cy, cx = region.centroid
        result[inst_id] = {'type': cell_type, 'contour': contour_xy, 'centroid': (cx, cy)}
    return result

def draw_contours_outline(img_np, instance_types, line_width=2):
    img_pil = Image.fromarray((np.clip(img_np,0,1)*255).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    for inst_id, spec in instance_types.items():
        ct = spec['type']; contour = spec.get('contour', None)
        if ct == 0 or contour is None or len(contour) < 3: continue
        color = CELL_COLORS.get(ct)
        if color is None: continue
        poly = list(zip(contour[:,0].tolist(), contour[:,1].tolist()))
        if len(poly) >= 3:
            draw.polygon(poly, outline=color, width=line_width)
    return np.array(img_pil) / 255.0

def run_inference(model, fold_dir, img_name):
    img = np.array(Image.open(fold_dir/'images'/img_name).convert('RGB')).astype(np.float32)/255.0
    img_norm = (img - np.array(mean)) / np.array(std)
    t = torch.tensor(img_norm).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(t)
    np_map = F.softmax(pred['nuclei_binary_map'], dim=1)
    hv_map = pred['hv_map'].permute(0,2,3,1)
    tp = F.softmax(pred['nuclei_type_map'], dim=1)
    pred_map = np.concatenate([
        torch.argmax(tp,dim=1)[0].cpu().numpy()[...,None],
        torch.argmax(np_map,dim=1)[0].cpu().numpy()[...,None],
        hv_map[0].cpu().numpy(),
    ], axis=-1)
    _, type_pred = post_processor.post_process_cell_segmentation(pred_map)
    return img, type_pred

def match_and_score(gt_types, pred_instances, max_dist=12.0):
    gt_list = [(iid, s['centroid'], s['type']) for iid, s in gt_types.items() if s['type'] != 0]
    if not gt_list: return 0, 0
    pred_list = []
    for iid, s in pred_instances.items():
        if s['type'] == 0: continue
        c = s.get('centroid', None)
        if c is None:
            cont = s.get('contour')
            if cont is None or len(cont) == 0: continue
            c = (float(np.mean(cont[:,0])), float(np.mean(cont[:,1])))
        pred_list.append((iid, c, s['type']))

    correct = 0
    used_pred = set()
    for g_iid, g_c, g_t in gt_list:
        best_idx, best_d = -1, max_dist
        for i, (p_iid, p_c, p_t) in enumerate(pred_list):
            if i in used_pred: continue
            d = ((g_c[0]-p_c[0])**2 + (g_c[1]-p_c[1])**2) ** 0.5
            if d < best_d:
                best_d = d; best_idx = i
        if best_idx >= 0:
            used_pred.add(best_idx)
            if pred_list[best_idx][2] == g_t:
                correct += 1
    return correct, len(gt_list)

# ============================================================
# 选图
# ============================================================
MIN_GT_CELLS = 15
MAX_CANDIDATES = 100
REQUIRE_TCINET_BETTER = True

print("\n" + "="*60)
print("Smart image selection")
print("="*60)

best_per_tissue = {}
failure_cases = {}
skipped_tissues = []

for t_disp in ALL_TISSUES_DISPLAY:
    t_csv = FIXED_TISSUE_MAP.get(t_disp)
    if t_csv is None:
        skipped_tissues.append((t_disp,'无映射')); continue
    subset = types_df[types_df['type'] == t_csv].copy()

    filtered = []
    for _, row in subset.iterrows():
        fd = row['fold']; img_name = row['img']
        label = load_label(fd, img_name)
        if label is None: continue
        n_cells = len(np.unique(label['inst_map'])) - 1
        if n_cells < MIN_GT_CELLS: continue
        n_types = len(set(label['type_map'][label['inst_map']>0].astype(int).flatten()) - {0})
        filtered.append((n_types, n_cells, fd, img_name, label))
    filtered.sort(key=lambda x: (-x[0], -x[1]))
    candidates = filtered[:MAX_CANDIDATES]

    if not candidates:
        skipped_tissues.append((t_disp, '候选空')); continue

    print(f"\n  [{t_disp}] 评估 {len(candidates)} 候选...")
    scored = []
    for n_types, n_cells, fd, img_name, label in candidates:
        fold_dir = DATA_ROOT / fd
        try:
            img, pred_tci = run_inference(model_tcinet, fold_dir, img_name)
            _,   pred_cvt = run_inference(model_cellvit, fold_dir, img_name)
        except Exception:
            continue
        gt_types = get_gt_instance_types(label)
        if len(gt_types) == 0: continue
        c_tci, n_gt = match_and_score(gt_types, pred_tci)
        c_cvt, _    = match_and_score(gt_types, pred_cvt)
        diff = c_tci - c_cvt
        composite = c_tci * 0.3 + diff * 1.0 + n_types * 0.3
        scored.append({
            'composite': composite, 'diff': diff, 'c_tci': c_tci, 'c_cvt': c_cvt,
            'n_gt': n_gt, 'fd': fd, 'img_name': img_name,
            'img': img, 'pred_tcinet': pred_tci, 'pred_cellvit': pred_cvt,
            'gt_types': gt_types,
        })

    if not scored:
        skipped_tissues.append((t_disp,'全推理失败')); continue

    valid = [s for s in scored if s['diff'] > 0]
    if valid:
        valid.sort(key=lambda x: -x['composite'])
        b = valid[0]
        best_per_tissue[t_disp] = b
        print(f"    ✓ MAIN: {b['fd']}/{b['img_name']}  TCI={b['c_tci']}/{b['n_gt']}  CVT={b['c_cvt']}  diff=+{b['diff']}")
    else:
        skipped_tissues.append((t_disp, f'无 TCINet 胜出的候选 (best diff={max(s["diff"] for s in scored)})'))

    worst = min(scored, key=lambda x: x['diff'])
    if worst['diff'] < 0:
        failure_cases[t_disp] = worst

print("\n" + "="*60)
print(f"主图: {len(best_per_tissue)}/{len(ALL_TISSUES_DISPLAY)} 组织可用")
print(f"失败案例库: {len(failure_cases)} 个")
if skipped_tissues:
    print("跳过:")
    for t, r in skipped_tissues:
        print(f"  - {t}: {r}")
print("="*60 + "\n")

# ============================================================
# 准备可视化
# ============================================================
vis_data = {}
for t_disp, info in best_per_tissue.items():
    gt_vis      = draw_contours_outline(info['img'], info['gt_types'],     line_width=2)
    cellvit_vis = draw_contours_outline(info['img'], info['pred_cellvit'], line_width=2)
    tcinet_vis  = draw_contours_outline(info['img'], info['pred_tcinet'],  line_width=2)
    vis_data[t_disp] = (gt_vis, cellvit_vis, tcinet_vis)

tissues = [t for t in ALL_TISSUES_DISPLAY if t in vis_data]
n_tissues = len(tissues)

# ============================================================
# 主图: 扁平布局 + 底部横向 Legend
#
# 策略: 
#   - 2 个超级行, 每行放 COLS_TOP 个组织 (一行不够就分两行)
#   - Legend 作为独立横条,放在整张对比图下方,横向 5 列展开
#   - 这样对比网格本身是完整的,没有 "挤进空格" 的视觉突兀
# ============================================================
print("Generating main figure (wide layout + bottom legend)...")

COLS_TOP = 10
N_BANDS = 2

# ===== 尺寸参数 =====
PATCH = 1.15
LABEL_W = 0.50
INTRA_GAP = 0.025
COL_GAP = 0.06
TITLE_H = 0.24
BAND_VGAP = 0.25
LEGEND_H = 0.55          # ← 底部 legend 横条的高度
LEGEND_TOP_GAP = 0.22    # legend 与上方对比图之间的间距
# =====================

# ===== 字号 =====
FS_TITLE = 11
FS_ROW_LABEL = 11
FS_LEGEND = 15
# ================

BAND_H = PATCH * 3 + INTRA_GAP * 2 + TITLE_H

fig_w = LABEL_W + COLS_TOP * PATCH + (COLS_TOP - 1) * COL_GAP + 0.18
# 画布高度 = 顶部留白 + 2 个 band + band间距 + legend上方间距 + legend + 底部留白
fig_h = (N_BANDS * BAND_H + (N_BANDS - 1) * BAND_VGAP
         + LEGEND_TOP_GAP + LEGEND_H + 0.25)

print(f"  画布尺寸: {fig_w:.2f} × {fig_h:.2f} inch  (高/宽 = {fig_h/fig_w:.2f})")
print(f"  LaTeX \\textwidth=7inch 时预计显示高度 = {7.0 * fig_h/fig_w:.2f} inch")

fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
fig.patch.set_facecolor('white')

def nx(i): return i / fig_w
def ny(i): return i / fig_h

left_margin = nx(LABEL_W + 0.06)
top_margin  = 1.0 - ny(0.12)

patch_w_n = nx(PATCH)
patch_h_n = ny(PATCH)
title_h_n = ny(TITLE_H)
intra_gap_n = ny(INTRA_GAP)
col_gap_n = nx(COL_GAP)
band_h_n = ny(BAND_H)
band_vgap_n = ny(BAND_VGAP)

row_labels = ['Ground-Truth', 'CellViT', 'TCINet (Ours)']

# 19 个组织分配: 第一行 COLS_TOP=10 个, 第二行 剩下的 + Legend
n_last = n_tissues - COLS_TOP  # 第二行组织数
if n_last < 0:
    # 组织数比一行还少,只用一行
    bands_tissue_counts = [n_tissues]
elif n_last > COLS_TOP:
    # 组织数超过两行,需要三行(兜底)
    n_per_row = (n_tissues + N_BANDS - 1) // N_BANDS
    bands_tissue_counts = [n_per_row, n_tissues - n_per_row]
else:
    bands_tissue_counts = [COLS_TOP, n_last]

print(f"  每超级行组织数: {bands_tissue_counts}  (共 {sum(bands_tissue_counts)} 个)")

t_idx = 0
for b, n_in_band in enumerate(bands_tissue_counts):
    y_band_top = top_margin - b * (band_h_n + band_vgap_n)
    y_title_top = y_band_top
    y_title_bot = y_title_top - title_h_n
    patch_y_tops, patch_y_bots = [], []
    for i in range(3):
        y_top = y_title_bot - i * (patch_h_n + intra_gap_n)
        y_bot = y_top - patch_h_n
        patch_y_tops.append(y_top); patch_y_bots.append(y_bot)

    for i, lbl in enumerate(row_labels):
        y_center = (patch_y_tops[i] + patch_y_bots[i]) / 2
        color = '#B22222' if 'Ours' in lbl else '#111111'
        fig.text(left_margin - nx(0.08), y_center, lbl,
                 ha='right', va='center', fontsize=FS_ROW_LABEL, fontweight='bold',
                 rotation=90, color=color, family=FONT_NAME)

    for c in range(n_in_band):
        x_left = left_margin + c * (patch_w_n + col_gap_n)
        t_disp = tissues[t_idx]
        gt_vis, cellvit_vis, tcinet_vis = vis_data[t_disp]

        fig.text(x_left + patch_w_n/2, y_title_bot + title_h_n * 0.30,
                 t_disp, ha='center', va='bottom',
                 fontsize=FS_TITLE, fontweight='bold', color='#000000',
                 family=FONT_NAME)

        for i, patch_img in enumerate([gt_vis, cellvit_vis, tcinet_vis]):
            ax = fig.add_axes([x_left, patch_y_bots[i], patch_w_n, patch_h_n])
            ax.imshow(patch_img); ax.axis('off')

        t_idx += 1

# ============================================================
# 底部横向 Legend: 独立横条,横跨整张对比图宽度
# ============================================================
# 最后一个 band 的底部 y (patch_y_bots[-1] 在循环里最后一次赋值)
y_last_band_bot = patch_y_bots[-1]
legend_top_gap_n = ny(LEGEND_TOP_GAP)
legend_h_n = ny(LEGEND_H)

y_leg_top = y_last_band_bot - legend_top_gap_n
y_leg_bot = y_leg_top - legend_h_n

# 让 legend 横跨整个对比图的 patch 区域 (从 left_margin 到最右一列右端)
leg_x_left = left_margin
leg_x_right = left_margin + COLS_TOP * patch_w_n + (COLS_TOP - 1) * col_gap_n
leg_w_n = leg_x_right - leg_x_left

ax_leg = fig.add_axes([leg_x_left, y_leg_bot, leg_w_n, legend_h_n])
ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1); ax_leg.axis('off')

from matplotlib.patches import FancyBboxPatch, Ellipse
box = FancyBboxPatch(
    (0.003, 0.08), 0.994, 0.84,
    boxstyle="round,pad=0.005,rounding_size=0.015",
    linewidth=1.6, edgecolor='#000000', facecolor='#F7F7F7',
    transform=ax_leg.transAxes,
)
ax_leg.add_patch(box)

# 宽高比用于画正圆
leg_w_inch = leg_w_n * fig_w
leg_h_inch = legend_h_n * fig_h
aspect = leg_w_inch / leg_h_inch

# 5 个类别水平均匀分布
n_items = len(LEGEND_ORDER)
# 每个 item 占据一个 "槽位", 槽位均匀分布
slot_centers = [(i + 0.5) / n_items for i in range(n_items)]

# 圆圈半径 (y方向); 要比之前单列的圆大一点,因为 legend 横条比较矮
cr_y = 0.30
cr_x = cr_y / aspect

# 圆圈相对于 item 槽位中心的偏移(圆圈在文字左侧)
# 每个槽位宽度 = 1/n_items ≈ 0.2, 圆圈距离槽位中心左偏 1/8 槽宽
slot_w = 1.0 / n_items
for i, k in enumerate(LEGEND_ORDER):
    cx = slot_centers[i]
    # 圆圈在槽位靠左侧
    circle_x = cx - slot_w * 0.28
    # 文字在圆圈右侧
    text_x = cx - slot_w * 0.15

    color = tuple(c_/255 for c_ in CELL_COLORS[k])
    circ = Ellipse((circle_x, 0.5), width=2*cr_x, height=2*cr_y,
                   facecolor='none', edgecolor=color, linewidth=3.0,
                   transform=ax_leg.transAxes)
    ax_leg.add_patch(circ)
    ax_leg.text(text_x, 0.5, CELL_NAMES[k],
                ha='left', va='center',
                fontsize=FS_LEGEND, fontweight='bold',
                color='#1a1a1a', transform=ax_leg.transAxes,
                family=FONT_NAME)

out_png = OUTPUT_DIR / 'paper_visualization_v16.png'
out_pdf = OUTPUT_DIR / 'paper_visualization_v16.pdf'
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  主图完成: {out_png}")

# ============================================================
# Supplementary: 失败案例
# ============================================================
if failure_cases:
    print("\nGenerating supplementary (failure cases)...")
    fc_sorted = sorted(failure_cases.items(), key=lambda kv: kv[1]['diff'])[:6]
    n_fc = len(fc_sorted)

    FC_COLS = min(n_fc, 3)
    FC_ROWS = (n_fc + FC_COLS - 1) // FC_COLS

    fc_PATCH = 1.4
    fc_LABEL_W = 0.48
    fc_INTRA_GAP = 0.03
    fc_COL_GAP = 0.09
    fc_TITLE_H = 0.28
    fc_BAND_VGAP = 0.28
    fc_BAND_H = fc_PATCH * 3 + fc_INTRA_GAP * 2 + fc_TITLE_H

    fc_fig_w = fc_LABEL_W + FC_COLS * fc_PATCH + (FC_COLS - 1) * fc_COL_GAP + 0.18
    fc_fig_h = FC_ROWS * fc_BAND_H + (FC_ROWS - 1) * fc_BAND_VGAP + 0.25

    fig2 = plt.figure(figsize=(fc_fig_w, fc_fig_h), dpi=200)
    fig2.patch.set_facecolor('white')

    def nx2(i): return i / fc_fig_w
    def ny2(i): return i / fc_fig_h

    fc_left = nx2(fc_LABEL_W + 0.06)
    fc_top  = 1.0 - ny2(0.12)

    fc_pw = nx2(fc_PATCH); fc_ph = ny2(fc_PATCH)
    fc_th = ny2(fc_TITLE_H); fc_ig = ny2(fc_INTRA_GAP)
    fc_cg = nx2(fc_COL_GAP); fc_bh = ny2(fc_BAND_H); fc_bvg = ny2(fc_BAND_VGAP)

    k_idx = 0
    for br in range(FC_ROWS):
        y_band_top = fc_top - br * (fc_bh + fc_bvg)
        y_title_bot = y_band_top - fc_th
        p_tops, p_bots = [], []
        for i in range(3):
            yt = y_title_bot - i * (fc_ph + fc_ig); yb = yt - fc_ph
            p_tops.append(yt); p_bots.append(yb)
        for i, lbl in enumerate(row_labels):
            yc = (p_tops[i] + p_bots[i]) / 2
            col = '#B22222' if 'Ours' in lbl else '#111111'
            fig2.text(fc_left - nx2(0.08), yc, lbl, ha='right', va='center',
                     fontsize=10, fontweight='bold', rotation=90, color=col,
                     family=FONT_NAME)

        for cc in range(FC_COLS):
            if k_idx >= n_fc: break
            t_disp, info = fc_sorted[k_idx]
            x_left = fc_left + cc * (fc_pw + fc_cg)

            gt_vis      = draw_contours_outline(info['img'], info['gt_types'],     line_width=2)
            cellvit_vis = draw_contours_outline(info['img'], info['pred_cellvit'], line_width=2)
            tcinet_vis  = draw_contours_outline(info['img'], info['pred_tcinet'],  line_width=2)

            fig2.text(x_left + fc_pw/2, y_title_bot + fc_th*0.30,
                      f"{t_disp}  (TCI={info['c_tci']}, CVT={info['c_cvt']})",
                      ha='center', va='bottom', fontsize=10, fontweight='bold',
                      family=FONT_NAME)

            for i, patch_img in enumerate([gt_vis, cellvit_vis, tcinet_vis]):
                ax = fig2.add_axes([x_left, p_bots[i], fc_pw, fc_ph])
                ax.imshow(patch_img); ax.axis('off')
            k_idx += 1

    sup_png = OUTPUT_DIR / 'paper_visualization_v16_supplementary.png'
    sup_pdf = OUTPUT_DIR / 'paper_visualization_v16_supplementary.pdf'
    plt.savefig(sup_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(sup_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Supplementary: {sup_png}")

print("\n全部完成!")