#!/usr/bin/env python3
"""
Multitask late-fusion model: radiomics (tabular) + masked image features.

Targets (derived from clinical CSV):
- ER_pos (binary)     : ER% >= 1 or textual positive
- PR_pos (binary)     : PR% >= 1 or textual positive
- HER2_pos (binary)   : IHC 3+ OR (IHC 2+ and FISH positive)
- Ki67  (regression)  : numeric % if present
- Grade (regression)  : numeric if present
- DCIS  (binary)      : numeric/textual presence

Notes
- One sample per record_id: choose radiomics row with largest original_shape2D_PixelSurface.
- Image is masked by PNG mask (background zero), then centrally resized to --img-size.
- Radiomics features = all columns starting with "original_".
- Train/val/test split by record_id (no leakage).
- Loss = sum of task losses computed only on samples where label is present.
- Metrics: AUROC for binary tasks, MAE for regressions.

python train_multitask_fusion.py \
  --radiomics /app/UC6_masks/radiomics_features.csv \
  --clinical /data/EuCanImageUseCase68.csv \
  --outdir /app/UC6_models/fusion_v1 \
  --epochs 20 --batch-size 8 --img-size 384

"""

import argparse, json, os, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import roc_auc_score, mean_absolute_error

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 13):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)

def mask_and_preprocess(image_path: Path, mask_path: Path, img_size: int) -> torch.Tensor:
    img = read_png(image_path).astype(np.float32)
    msk = (read_png(mask_path) > 0).astype(np.float32)
    img = img * msk
    ys, xs = np.where(msk > 0)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = 16
        y0 = max(0, y0 - pad); y1 = min(img.shape[0], y1 + pad)
        x0 = max(0, x0 - pad); x1 = min(img.shape[1], x1 + pad)
        img = img[y0:y1, x0:x1]
    im = Image.fromarray(np.clip(img, 0, np.max(img)).astype(np.float32))
    im = im.resize((img_size, img_size), resample=Image.BICUBIC)
    x = np.array(im, dtype=np.float32)
    if x.max() > 0:
        x /= x.max()
    x = np.stack([x, x, x], axis=0)
    return torch.from_numpy(x)

def standardize_train(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    return Xz, {"mean": mu, "std": sd}

def standardize_apply(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (X - stats["mean"]) / stats["std"]

def first_non_null(*vals):
    for v in vals:
        if v is None: continue
        if isinstance(v, str) and v.strip() == "": continue
        return v
    return None

def to_float_or_none(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.strip() == "": return None
        return float(x)
    except Exception:
        return None

def to_int_or_none(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.strip() == "": return None
        return int(float(x))
    except Exception:
        return None

def text_is_pos(x: Optional[str]) -> Optional[bool]:
    if x is None: return None
    s = str(x).strip().lower()
    if s in {"pos", "positive", "yes", "true", "1"}: return True
    if s in {"neg", "negative", "no", "false", "0"}: return False
    return None

# -----------------------------
# Targets derivation
# -----------------------------
def derive_targets(clin_row: pd.Series) -> Dict[str, Optional[float]]:
    def pick(*names):
        return first_non_null(*[clin_row.get(n) for n in names])

    er = pick("er", "er_2"); pr = pick("pr", "pr_2")
    er_f = to_float_or_none(er); pr_f = to_float_or_none(pr)
    er_t = text_is_pos(er); pr_t = text_is_pos(pr)
    ER_pos = None
    if er_f is not None: ER_pos = 1.0 if er_f >= 1.0 else 0.0
    elif er_t is not None: ER_pos = 1.0 if er_t else 0.0

    PR_pos = None
    if pr_f is not None: PR_pos = 1.0 if pr_f >= 1.0 else 0.0
    elif pr_t is not None: PR_pos = 1.0 if pr_t else 0.0

    her2ihc = pick("her2ihc", "her2ihc_2")
    fish = pick("her2fish", "her2fish_2")
    ihc_i = to_int_or_none(her2ihc)
    fish_t = text_is_pos(fish)
    HER2_pos = None
    if ihc_i is not None:
        if ihc_i >= 3: HER2_pos = 1.0
        elif ihc_i == 2 and (fish_t is True): HER2_pos = 1.0
        elif ihc_i in {0,1,2} and (fish_t is False or fish_t is None): HER2_pos = 0.0
    elif fish_t is not None:
        HER2_pos = 1.0 if fish_t else 0.0

    # Ki67 numeric
    Ki67 = to_float_or_none(pick("ki67", "ki67_2"))

    # Grade numeric
    grade = to_float_or_none(pick("grade", "grade_2"))
    Grade = grade

    # DCIS presence
    dcis = pick("dcis", "dcis_2")
    dcis_i = to_int_or_none(dcis)
    dcis_t = text_is_pos(dcis)
    DCIS = None
    if dcis_i is not None: DCIS = 1.0 if dcis_i >= 1 else 0.0
    elif dcis_t is not None: DCIS = 1.0 if dcis_t else 0.0

    return {
        "ER_pos": ER_pos,
        "PR_pos": PR_pos,
        "HER2_pos": HER2_pos,
        "Ki67": Ki67,
        "Grade": Grade,
        "DCIS": DCIS,
    }

# -----------------------------
# Dataset
# -----------------------------
class FusionDataset(Dataset):
    def __init__(self, rows: List[dict], rad_cols: List[str], img_size: int):
        self.rows = rows
        self.rad_cols = rad_cols
        self.img_size = img_size

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x_img = mask_and_preprocess(Path(r["image_path"]), Path(r["mask_path"]), self.img_size)  # [3,H,W]
        x_rad = torch.tensor(r["rad_vector"], dtype=torch.float32)  # [D]
        y = r["targets"]  # dict of task-> value or None
        return x_img, x_rad, y, r["record_id"]

# -----------------------------
# Model
# -----------------------------
class FusionModel(nn.Module):
    def __init__(self, rad_dim: int, out_tasks: Dict[str, str]):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.cnn = backbone
        in_features = backbone.fc.in_features
        self.cnn.fc = nn.Identity()

        self.img_proj = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.1))
        self.rad_proj = nn.Sequential(nn.Linear(rad_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.fuse = nn.Sequential(nn.Linear(256 + 128, 256), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(256, 128), nn.ReLU())

        self.tasks = out_tasks
        self.heads = nn.ModuleDict()
        for name, kind in out_tasks.items():
            if kind == "binary":
                self.heads[name] = nn.Linear(128, 1)
            elif kind == "regression":
                self.heads[name] = nn.Linear(128, 1)
            else:
                raise ValueError(f"Unknown task type: {kind}")

    def forward(self, x_img, x_rad):
        h_img = self.cnn(x_img)
        h_img = self.img_proj(h_img)
        h_rad = self.rad_proj(x_rad)
        z = self.fuse(torch.cat([h_img, h_rad], dim=1))
        out = {}
        for name in self.tasks.keys():
            out[name] = self.heads[name](z).squeeze(1)
        return out

# -----------------------------
# Metrics helpers
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, tasks):
    model.eval()
    preds = {t: [] for t in tasks}
    gts   = {t: [] for t in tasks}
    recs  = []
    for x_img, x_rad, ydict, rec_ids in loader:
        x_img = x_img.to(device)
        x_rad = x_rad.to(device)
        out = model(x_img, x_rad)
        for t in tasks:
            ys = torch.tensor([ (ydict_i[t] if ydict_i[t] is not None else np.nan) for ydict_i in ydict ], dtype=torch.float32)
            ps = out[t].detach().cpu().numpy()
            preds[t].extend(ps.tolist())
            gts[t].extend(ys.numpy().tolist())
        recs.extend(list(rec_ids))

    metrics = {}
    for t, kind in tasks.items():
        y_true = np.array(gts[t])
        y_pred = np.array(preds[t])
        mask = ~np.isnan(y_true)
        if mask.sum() < 5:
            metrics[t] = None
            continue
        if kind == "binary":
            try:
                auc = roc_auc_score(y_true[mask], 1/(1+np.exp(-y_pred[mask])))
                metrics[t] = {"AUROC": float(auc), "n": int(mask.sum())}
            except Exception:
                metrics[t] = {"AUROC": None, "n": int(mask.sum())}
        else:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            metrics[t] = {"MAE": float(mae), "n": int(mask.sum())}
    pred_rows = []
    for i, rid in enumerate(recs):
        row = {"record_id": rid}
        for t, kind in tasks.items():
            row[f"{t}_pred"] = preds[t][i]
            row[f"{t}_true"] = gts[t][i]
        pred_rows.append(row)
    return metrics, pd.DataFrame(pred_rows)

# -----------------------------
# Build table and splits
# -----------------------------
def build_record_table(radiomics_csv: Path, clinical_csv: Path, mlo_only=True):
    rad = pd.read_csv(radiomics_csv)
    clin = pd.read_csv(clinical_csv)
    if mlo_only and "view" in rad.columns:
        rad = rad[rad["view"].astype(str).str.upper() == "MLO"].copy()
    area_col = "original_shape2D_PixelSurface"
    if area_col not in rad.columns:
        raise SystemExit(f"Radiomics CSV missing {area_col}")
    rad["__area"] = rad[area_col].astype(float)
    rep = rad.sort_values("__area", ascending=False).groupby("record_id", as_index=False).first()

    clin_targets = []
    for _, r in clin.iterrows():
        t = derive_targets(r)
        t["record_id"] = r["record_id"]
        clin_targets.append(t)
    clin_t = pd.DataFrame(clin_targets)

    df = rep.merge(clin_t, on="record_id", how="inner")

    rad_cols = [c for c in df.columns if c.startswith("original_")]
    keep = ["record_id", "image_path", "mask_path"] + rad_cols + ["ER_pos","PR_pos","HER2_pos","Ki67","Grade","DCIS"]
    df = df[keep].copy()
    df = df[df["image_path"].apply(lambda p: Path(str(p)).exists()) &
            df["mask_path"].apply(lambda p: Path(str(p)).exists())]
    df = df.reset_index(drop=True)
    return df, rad_cols

def make_splits(record_ids: List[str], seed=13, test_frac=0.2, val_frac=0.15):
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(set(record_ids)))
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_frac))
    test_ids = set(ids[:n_test])
    rem = [i for i in ids if i not in test_ids]
    n_val = int(round(len(rem) * val_frac))
    val_ids = set(rem[:n_val])
    train_ids = set(rem[n_val:])
    return train_ids, val_ids, test_ids

# -----------------------------
# Collate (handles dict of labels)
# -----------------------------
def collate_fn(batch):
    imgs, rads, ys, rids = zip(*batch)
    x_img = torch.stack(imgs, dim=0)
    x_rad = torch.stack(rads, dim=0)
    return x_img, x_rad, list(ys), list(rids)

# -----------------------------
# Train
# -----------------------------
def train_epoch(model, loader, optimizer, device, tasks, w_bin=1.0, w_reg=1.0):
    model.train()
    loss_bce = nn.BCEWithLogitsLoss(reduction="none")
    loss_mse = nn.MSELoss(reduction="none")
    total = 0.0
    for x_img, x_rad, ydicts, _ in loader:
        x_img = x_img.to(device); x_rad = x_rad.to(device)
        outs = model(x_img, x_rad)
        loss = 0.0
        n_terms = 0
        for t, kind in tasks.items():
            y = torch.tensor([ (yd[t] if yd[t] is not None else np.nan) for yd in ydicts ],
                             dtype=torch.float32, device=device)
            m = ~torch.isnan(y)
            if m.sum() == 0: continue
            if kind == "binary":
                l = loss_bce(outs[t][m], y[m])
                loss = loss + w_bin * l.mean()
            else:
                l = loss_mse(outs[t][m], y[m])
                loss = loss + w_reg * l.mean()
            n_terms += 1
        if n_terms == 0:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radiomics", type=Path, required=True, help="radiomics_features.csv")
    ap.add_argument("--clinical",  type=Path, required=True, help="clinical.csv")
    ap.add_argument("--outdir",    type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df, rad_cols = build_record_table(args.radiomics, args.clinical, mlo_only=True)
    print(f"[data] records with image+mask+radiomics: {len(df)}")

    tr_ids, va_ids, te_ids = make_splits(df["record_id"].tolist(), seed=args.seed)
    json.dump({"train": sorted(list(tr_ids)), "val": sorted(list(va_ids)), "test": sorted(list(te_ids))},
              open(args.outdir/"splits.json","w"), indent=2)

    X = df[rad_cols].astype(float).values
    X_tr = df[df["record_id"].isin(tr_ids)][rad_cols].astype(float).values
    X_tr_z, stats = standardize_train(X_tr)
    X_z = standardize_apply(X, stats)
    np.save(args.outdir/"rad_stats_mean.npy", stats["mean"])
    np.save(args.outdir/"rad_stats_std.npy",  stats["std"])

    def row_to_targets(r):
        return {
            "ER_pos":  r["ER_pos"]  if pd.notna(r["ER_pos"])  else None,
            "PR_pos":  r["PR_pos"]  if pd.notna(r["PR_pos"])  else None,
            "HER2_pos":r["HER2_pos"]if pd.notna(r["HER2_pos"])else None,
            "Ki67":    r["Ki67"]    if pd.notna(r["Ki67"])    else None,
            "Grade":   r["Grade"]   if pd.notna(r["Grade"])   else None,
            "DCIS":    r["DCIS"]    if pd.notna(r["DCIS"])    else None,
        }

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "record_id": r["record_id"],
            "image_path": str(r["image_path"]),
            "mask_path":  str(r["mask_path"]),
            "rad_vector": X_z[i, :].astype(np.float32),
            "targets":    row_to_targets(r),
        })

    tasks = {"ER_pos":"binary","PR_pos":"binary","HER2_pos":"binary","Ki67":"regression","Grade":"regression","DCIS":"binary"}
    ds = FusionDataset(rows, rad_cols, img_size=args.img_size)
    idx_tr = [i for i, rw in enumerate(rows) if rw["record_id"] in tr_ids]
    idx_va = [i for i, rw in enumerate(rows) if rw["record_id"] in va_ids]
    idx_te = [i for i, rw in enumerate(rows) if rw["record_id"] in te_ids]

    dl_tr = DataLoader(Subset(ds, idx_tr), batch_size=args.batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dl_va = DataLoader(Subset(ds, idx_va), batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dl_te = DataLoader(Subset(ds, idx_te), batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(rad_dim=len(rad_cols), out_tasks=tasks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_va = -np.inf
    best_path = args.outdir/"ckpt_best.pt"

    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, dl_tr, opt, device, tasks)
        va_metrics, _ = evaluate(model, dl_va, device, tasks)

        scores = []
        for t, kind in tasks.items():
            m = va_metrics.get(t)
            if not m: continue
            if kind == "binary" and m["AUROC"] is not None:
                scores.append(m["AUROC"])
            elif kind == "regression" and m["MAE"] is not None:
                scores.append(1.0 / (1.0 + m["MAE"]))
        agg = float(np.mean(scores)) if scores else -np.inf

        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.4f}  val_score={agg:.4f}  val_metrics={va_metrics}")

        if agg > best_va:
            best_va = agg
            torch.save({"model": model.state_dict(), "rad_cols": rad_cols, "stats": stats, "tasks": tasks},
                       best_path)

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    te_metrics, te_preds = evaluate(model, dl_te, device, tasks)
    print("[test] metrics:", te_metrics)

    json.dump(te_metrics, open(args.outdir/"metrics_test.json","w"), indent=2)
    te_preds.to_csv(args.outdir/"preds_test.csv", index=False)
    print(f"[ok] saved: {best_path}, metrics_test.json, preds_test.csv, splits.json, rad_stats_*.npy")

if __name__ == "__main__":
    main()
