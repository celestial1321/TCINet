# -*- coding: utf-8 -*-
"""Alpha可视化 v7：所有图片，无小字注释，干净布局"""

import os, sys
sys.path.insert(0, '/root/autodl-tmp/CellViT-main1')

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from collections import defaultdict

from models.segmentation.cell_segmentation.cellvit import CellViTSAM

TCINET_CKPT = '/root/autodl-tmp/CellViT-main1/logs/2026-04-24T231538_cellvit_tacnet_v2_bw8_fold0/checkpoints/model_best_bpq.pth'
TCINET_CFG  = '/root/autodl-tmp/CellViT-main1/logs/2026-04-24T231538_cellvit_tacnet_v2_bw8_fold0/config.yaml'
DATA_ROOT   = Path('/root/autodl-tmp/CellViT-main/cell_segmentation/datasets/PanNuke_pre')
OUT_DIR     = Path('/root/autodl-tmp/CellViT-main1/tacbi_visualization')
OUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'
MEAN = np.array([0.5, 0.5, 0.5])
STD  = np.array([0.5, 0.5, 0.5])

TISSUE_IDX = [
    'Adrenal','Bile Duct','Bladder','Breast','Cervix',
    'Colon','Esophagus','Head & Neck','Kidney','Liver',
    'Lung','Ovarian','Pancreatic','Prostate','Skin',
    'Stomach','Testis','Thyroid','Uterus'
]

TISSUE_CSV_MAP = {
    'Adrenal':    ['adrenal','adrenal gland'],
    'Bile Duct':  ['bile','bile duct','bile-duct'],
    'Bladder':    ['bladder'], 'Breast':['breast'],
    'Cervix':     ['cervix'], 'Colon':['colon'],
    'Esophagus':  ['esophagus'],
    'Head & Neck':['head','neck','headneck','head & neck'],
    'Kidney':     ['kidney'], 'Liver':['liver'],
    'Lung':       ['lung'], 'Ovarian':['ovarian'],
    'Pancreatic': ['pancreatic','pancreas'],
    'Prostate':   ['prostate'], 'Skin':['skin'],
    'Stomach':    ['stomach'], 'Testis':['testis'],
    'Thyroid':    ['thyroid'], 'Uterus':['uterus'],
}

A_COLORS  = ['#1565C0', '#2E7D32', '#C62828']
A_NEG_CLR = ['#BBDEFB', '#C8E6C9', '#FFCDD2']
A_LABELS  = [r'$\alpha_{\mathrm{NP}}$',
             r'$\alpha_{\mathrm{HV}}$',
             r'$\alpha_{\mathrm{NT}}$']


def get_tissue_rows(df, tissue_name):
    keywords = TISSUE_CSV_MAP.get(tissue_name, [tissue_name.lower()])
    results = []
    for _, row in df.iterrows():
        val = str(row['type']).lower().replace('-','').replace(' ','').replace('&','')
        for kw in keywords:
            kw2 = kw.lower().replace('-','').replace(' ','').replace('&','')
            if kw2 in val or val in kw2:
                results.append(row)
                break
    return results


def load_model():
    with open(TCINET_CFG) as f:
        cfg = yaml.safe_load(f)
    m = CellViTSAM(
        model_path=None,
        num_nuclei_classes=cfg['data']['num_nuclei_classes'],
        num_tissue_classes=cfg['data']['num_tissue_classes'],
        vit_structure=cfg['model']['backbone'],
    )
    ckpt = torch.load(TCINET_CKPT, map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'], strict=True)
    m.to(DEVICE).eval()
    print("Loaded TCINet")
    return m


def collect_alphas(model):
    captured = [None]
    def hook(module, input, output):
        captured[0] = output.detach().cpu().clone()
    model.tacbi.tissue_to_alpha.register_forward_hook(hook)

    import pandas as pd
    alpha_per_tissue = defaultdict(list)

    for fold in ['fold0', 'fold1', 'fold2']:
        fold_dir = DATA_ROOT / fold
        csv_path = fold_dir / 'types.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for tissue_name in TISSUE_IDX:
            rows = get_tissue_rows(df, tissue_name)
            for row in rows:  # 全部图片，无上限
                img_path = fold_dir / 'images' / row['img']
                if not img_path.exists():
                    continue
                try:
                    img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)/255
                    t = torch.tensor((img-MEAN)/STD).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        _ = model(t)
                    if captured[0] is not None:
                        alpha_per_tissue[tissue_name].append(captured[0][0].numpy().copy())
                except:
                    continue

    print("Samples per tissue:")
    for t in TISSUE_IDX:
        print(f"  {t:15s}: {len(alpha_per_tissue.get(t,[]))}")
    return alpha_per_tissue


def make_figure(alpha_per_tissue):
    valid = [t for t in TISSUE_IDX if len(alpha_per_tissue.get(t, [])) >= 3]
    all_arrs   = [np.stack(alpha_per_tissue[t]) for t in valid]
    means      = np.array([a.mean(0) for a in all_arrs])  # [n, 3]
    stds       = np.array([a.std(0)  for a in all_arrs])  # [n, 3]
    n          = len(valid)
    col_mean   = means.mean(0, keepdims=True)
    col_std    = means.std(0,  keepdims=True) + 1e-8
    means_z    = (means - col_mean) / col_std
    cross_std  = means.std(0)
    within_std = stds.mean(0)
    print(f"cross_std:  {cross_std}")
    print(f"within_std: {within_std}")

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 6.8), facecolor='white',
        gridspec_kw={'width_ratios': [2.0, 1.8, 1.3], 'wspace': 0.42}
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.06)
    ax0, ax1, ax2 = axes

    fig.text(0.5, 0.95,
             r'TC-TACBI: Tissue-Conditioned Dynamic $\alpha$ Routing',
             ha='center', va='top', fontsize=12, fontweight='bold')

    # ── (a) z-score 条形图 ───────────────────────────────────
    y       = np.arange(n)
    bar_h   = 0.22
    offsets = [-bar_h, 0, bar_h]
    for j in range(3):
        for i in range(n):
            v = means_z[i, j]
            ax0.barh(y[i]+offsets[j], v, height=bar_h*0.88,
                     color=A_COLORS[j] if v >= 0 else A_NEG_CLR[j],
                     alpha=0.90 if v >= 0 else 0.50)
    ax0.axvline(0, color='#333', lw=1.5, ls='--', zorder=5)
    ax0.set_yticks(y)
    ax0.set_yticklabels(valid, fontsize=8.5)
    ax0.invert_yaxis()
    ax0.set_xlabel('Z-score of α', fontsize=9)
    ax0.set_title('(a) Per-Tissue Routing Strength',
                  fontsize=9.5, fontweight='bold', pad=6)
    patches = [mpatches.Patch(color=A_COLORS[j], alpha=0.9, label=A_LABELS[j])
               for j in range(3)]
    ax0.legend(handles=patches, fontsize=8, loc='lower right',
               framealpha=0.85, handlelength=1.0)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.grid(axis='x', alpha=0.2, ls=':')

    # ── (b) diverging 热力图（无格内小字）────────────────────
    cmap_div = LinearSegmentedColormap.from_list(
        'div', ['#1565C0', '#FFFFFF', '#C62828'])
    vmax = max(float(np.abs(means_z).max()), 0.01)
    im = ax1.imshow(means_z, aspect='auto', cmap=cmap_div,
                    vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(A_LABELS, fontsize=11)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(valid, fontsize=8.5)
    ax1.set_title('(b) α Deviation from Mean',
                  fontsize=9.5, fontweight='bold', pad=6)
    # 格内数值：z-score + 原始值
    for i in range(n):
        for j in range(3):
            z   = means_z[i, j]
            raw = means[i, j]
            tc  = 'white' if abs(z) > vmax * 0.55 else 'black'
            ax1.text(j, i, f'{z:+.2f}\n({raw:.2f})',
                     ha='center', va='center',
                     fontsize=6.5, color=tc, fontweight='bold')
    cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label('Z-score', fontsize=8)

    # ── (c) cross vs within std ──────────────────────────────
    x3    = np.array([0.0, 1.0, 2.0])
    w3    = 0.30
    y_top = max(float(cross_std.max()), float(within_std.max()))
    ax2.bar(x3 - w3/2, cross_std,  w3, color=A_COLORS, alpha=0.90, zorder=3,
            label='Cross-tissue')
    ax2.bar(x3 + w3/2, within_std, w3, color=A_COLORS, alpha=0.38,
            hatch='xxx', edgecolor='white', zorder=3,
            label='Within-tissue')
    # ×N 倍数标在每组bar正上方
    for j in range(3):
        ratio = float(cross_std[j]) / (float(within_std[j]) + 1e-8)
        ax2.text(float(x3[j]),
                 max(float(cross_std[j]), float(within_std[j])) + y_top * 0.05,
                 f'×{ratio:.1f}', ha='center', fontsize=9,
                 color=A_COLORS[j], fontweight='bold')
    ax2.set_xticks(x3)
    ax2.set_xticklabels(A_LABELS, fontsize=11)
    ax2.set_ylabel('Std dev of α', fontsize=9)          # 缩短避免重叠
    ax2.set_title('(c) Cross- vs Within-Tissue α Variance',
                  fontsize=9.5, fontweight='bold', pad=6)
    ax2.set_xlim(-0.6, 2.6)
    ax2.set_ylim(0, y_top * 1.45)                       # 多留空间给×N标注
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.25, ls=':', zorder=0)
    ax2.legend(fontsize=8, framealpha=0.85, handlelength=1.2,
               loc='lower right')                       # 移到右下，避开×N标注

    out_png = OUT_DIR / 'alpha_v7.png'
    out_pdf = OUT_DIR / 'alpha_v7.pdf'
    plt.savefig(out_png, dpi=220, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    model = load_model()
    alpha_per_tissue = collect_alphas(model)
    make_figure(alpha_per_tissue)


if __name__ == '__main__':
    main()