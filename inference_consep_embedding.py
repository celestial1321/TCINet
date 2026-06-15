"""
CoNSeP Cell Embedding Evaluation — TCINet
复现 CellViT 原文 Section 5.4

CoNSeP 原始类别（mat文件）：
  1=other  2=inflammatory  3=healthy epithelial  4=dysplastic epithelial
  5=fibroblast  6=muscle  7=endothelial

原文合并：inflammatory(2)→0  epithelial(3+4)→1  spindle(5+6+7)→2

用法：
  cd /root/autodl-tmp/CellViT-main1
  python inference_consep_embedding.py \
      --model  logs/2026-04-10T144126_cellvit_tacnet_v2_fold0/checkpoints/model_best.pth \
      --data   /root/autodl-tmp/CoNSeP \
      --outdir results/consep/fold0 \
      --gpu 0 --umap
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

REPO = Path("/root/autodl-tmp/CellViT-main1")
sys.path.insert(0, str(REPO))
from models.cellvit_tacnet_v2 import CellViTSAM

# 类别映射：原始1-7 → 论文合并0/1/2
REMAP = {1: -1, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
CLASS_NAMES = ["Inflammatory", "Epithelial", "Spindle-shaped"]


def remap_labels(labels):
    out = np.array([REMAP.get(int(l), -1) for l in labels])
    return out, out >= 0


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model = CellViTSAM(
        model_path=None, num_nuclei_classes=6, num_tissue_classes=19,
        vit_structure="SAM-H", drop_rate=0.0,
    )
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def extract_one(model, img_np, type_map_np, device):
    # resize to 1024
    img_pil = Image.fromarray(img_np).resize((1024, 1024), Image.BILINEAR)
    img_t = torch.from_numpy(
        np.array(img_pil, dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    out = model(img_t, retrieve_tokens=True)

    # tokens [D, 64, 64]
    tokens = out["tokens"][0].cpu().numpy()
    D, th, tw = tokens.shape

    # post-process → instance map
    inst_maps, _ = model.calculate_instance_map(
        {k: out[k] for k in ["nuclei_binary_map","hv_map","nuclei_type_map"]},
        magnification=40
    )
    inst_map = inst_maps[0].numpy().astype(np.int32)

    # resize to token resolution
    inst_small = np.array(Image.fromarray(inst_map).resize((tw, th), Image.NEAREST))
    type_small = np.array(
        Image.fromarray(type_map_np.astype(np.int32)).resize((tw, th), Image.NEAREST)
    )

    embs, labels = [], []
    for nid in np.unique(inst_small):
        if nid == 0:
            continue
        mask = inst_small == nid
        types = type_small[mask]
        types = types[types > 0]
        if len(types) == 0:
            continue
        label = int(np.bincount(types, minlength=8)[1:].argmax()) + 1
        embs.append(tokens[:, mask].mean(axis=1))
        labels.append(label)

    if not embs:
        return np.zeros((0, D), np.float32), np.zeros(0, int)
    return np.stack(embs).astype(np.float32), np.array(labels, int)


def process_split(model, data_root, split, device):
    img_dir = data_root / split / "Images"
    lbl_dir = data_root / split / "Labels"
    files = sorted(img_dir.glob("*.png"))
    all_e, all_l = [], []
    for f in tqdm(files, desc=split):
        mat = sio.loadmat(str(lbl_dir / f"{f.stem}.mat"))
        img = np.array(Image.open(f).convert("RGB"), dtype=np.uint8)
        e, l = extract_one(model, img, mat["type_map"].astype(np.int32), device)
        all_e.append(e); all_l.append(l)
        tqdm.write(f"  {f.stem}: {len(l)} nuclei")
    return np.concatenate(all_e), np.concatenate(all_l)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--data",   required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--gpu",    type=int, default=0)
    p.add_argument("--umap",   action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    model = load_model(args.model, device)
    print(f"Embed dim: {model.embed_dim}\n")

    X_tr, y_tr_raw = process_split(model, Path(args.data), "Train", device)
    X_te, y_te_raw = process_split(model, Path(args.data), "Test",  device)

    y_tr, vtr = remap_labels(y_tr_raw); X_tr, y_tr = X_tr[vtr], y_tr[vtr]
    y_te, vte = remap_labels(y_te_raw); X_te, y_te = X_te[vte], y_te[vte]

    print(f"\nTrain: {len(X_tr)} nuclei")
    print(f"Test : {len(X_te)} nuclei")

    np.save(outdir/"train_embs.npy", X_tr); np.save(outdir/"train_lbls.npy", y_tr)
    np.save(outdir/"test_embs.npy",  X_te); np.save(outdir/"test_lbls.npy",  y_te)

    # 线性分类器
    print("\nTraining linear classifier...")
    clf = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  multi_class="multinomial", random_state=42, n_jobs=-1)),
    ])
    clf.fit(X_tr, y_tr)

    n_cls = len(np.unique(y_tr))
    y_prob = clf.predict_proba(X_te)
    auroc  = roc_auc_score(
        label_binarize(y_te, classes=list(range(n_cls))),
        y_prob, multi_class="ovr", average="macro"
    )

    print(f"\n{'='*45}")
    print(f"  AUROC (macro OVR): {auroc:.4f}")
    print(f"  Reference CellViT-SAM-H : 0.9630")
    print(f"{'='*45}")

    (outdir/"auroc_result.txt").write_text(
        f"Model: {args.model}\n"
        f"Train nuclei: {len(X_tr)}\nTest nuclei: {len(X_te)}\n"
        f"AUROC: {auroc:.4f}\nReference: 0.9630\n"
    )

    # UMAP
    if args.umap:
        print("\nGenerating UMAP...")
        try:
            import umap, matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = min(6000, len(X_te))
            idx = np.random.RandomState(42).choice(len(X_te), n, replace=False)
            e2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_te[idx])
            fig, ax = plt.subplots(figsize=(7, 6))
            for c, (cn, col) in enumerate(zip(CLASS_NAMES, ["#E74C3C","#3498DB","#2ECC71"])):
                m = y_te[idx] == c
                ax.scatter(e2d[m,0], e2d[m,1], c=col, label=cn, s=3, alpha=0.5)
            ax.legend(markerscale=4); ax.set_title(f"CoNSeP embeddings — TCINet  AUROC={auroc:.4f}")
            plt.tight_layout()
            plt.savefig(outdir/"umap_consep.pdf", dpi=200)
            plt.savefig(outdir/"umap_consep.png", dpi=200)
            print(f"  UMAP → {outdir}/umap_consep.pdf")
        except ImportError:
            print("  pip install umap-learn")

    print("\nDone.")

if __name__ == "__main__":
    main()