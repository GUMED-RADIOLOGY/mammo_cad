#!/usr/bin/env python3
"""
Multitask late-fusion model: radiomics (tabular) + masked image branch with mm-aware preprocessing.

Targets (derived from clinical CSV):
- ER_pos (binary)     : ER% >= 1 or textual positive
- PR_pos (binary)     : PR% >= 1 or textual positive
- HER2_pos (binary)   : IHC 3+ OR (IHC 2+ and FISH positive)
- Ki67  (regression)  : numeric % if present
- Grade (regression)  : numeric if present
- DCIS  (binary)      : numeric/textual presence

Key points
- Uses your image-metadata CSV (from DICOM headers) to get in-plane spacing (mm/px).
- Resamples image+mask to a fixed physical resolution (--target-mm), crops with mm padding (--pad-mm),
  then resizes to --img-size for the CNN.
- Radiomics features = all columns starting with "original_".
- One sample per record_id: radiomics row with the largest original_shape2D_PixelSurface.
- Train/val/test split by record_id to avoid leakage.
- Loss = sum of task losses; computed only where labels exist.
- Metrics: AUROC for binary; MAE for regression.
- Fixes PyTorch 2.6 torch.load default (weights_only=False).

Example:
python train_multitask_fusion.py \
  --radiomics /app/UC6_masks/radiomics_features.csv \
  --clinical  /data/EuCanImageUseCase68.csv \
  --image-metadata /app/UC6_converted/metadata_index.csv \
  --outdir    /app/UC6_models/fusion_v2 \
  --epochs 20 --batch-size 8 --img-size 384 \
  --target-mm 0.10 --pad-mm 10
"""

import argparse, json, os, random, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import roc_auc_score, mean_absolute_error


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 13):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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
        elif ihc_i in {0, 1, 2} and (fish_t is False or fish_t is None): HER2_pos = 0.0
    elif fish_t is not None:
        HER2_pos = 1.0 if fish_t else 0.0

    Ki67  = to_float_or_none(pick("ki67", "ki67_2"))
    Grade = to_float_or_none(pick("grade", "grade_2"))

    dcis = pick("dcis", "dcis_2")
    dcis_i = to_int_or_none(dcis)
    dcis_t = text_is_pos(dcis)
    DCIS = None
    if dcis_i is not None: DCIS = 1.0 if dcis_i >= 1 else 0.0
    elif dcis_t is not None: DCIS = 1.0 if dcis_t else 0.0

    return {"ER_pos": ER_pos, "PR_pos": PR_pos, "HER2_pos": HER2_pos,
            "Ki67": Ki67, "Grade": Grade, "DCIS": DCIS}


# -----------------------------
# Spacing-aware preprocessing
# -----------------------------
def mask_and_preprocess(image_path: Path,
                        mask_path: Path,
                        img_size: int,
                        spacing_rc: Tuple[float, float],
                        target_mm: float,
                        pad_mm: float) -> torch.Tensor:
    """Resample to target_mm, mm-pad crop to mask bbox, robust normalize, resize to img_size."""
    img = Image.open(image_path).convert("L")
    msk = Image.open(mask_path).convert("L")
    img = np.array(img, dtype=np.float32)
    msk = (np.array(msk) > 0).astype(np.uint8)

    row_mm, col_mm = spacing_rc
    row_mm = float(row_mm) if (row_mm and row_mm > 0) else 1.0
    col_mm = float(col_mm) if (col_mm and col_mm > 0) else 1.0
    H, W = img.shape[:2]
    new_h = max(8, int(round(H * row_mm / target_mm)))
    new_w = max(8, int(round(W * col_mm / target_mm)))
    img_r = np.array(Image.fromarray(img).resize((new_w, new_h), resample=Image.BICUBIC))
    msk_r = np.array(Image.fromarray(msk).resize((new_w, new_h), resample=Image.NEAREST)).astype(np.uint8)

    ys, xs = np.where(msk_r > 0)
    if len(xs) > 0 and len(ys) > 0:
        pad_px = int(round(pad_mm / target_mm))
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        x0 = max(0, x0 - pad_px); x1 = min(new_w, x1 + pad_px)
        y0 = max(0, y0 - pad_px); y1 = min(new_h, y1 + pad_px)
        img_r = img_r[y0:y1, x0:x1]
        msk_r = msk_r[y0:y1, x0:x1]

    if msk_r.sum() > 0:
        vals = img_r[msk_r > 0].astype(np.float32)
        p1, p99 = np.percentile(vals, [1, 99])
        if p99 > p1:
            img_r = np.clip((img_r - p1) / (p99 - p1), 0, 1)
        else:
            m = vals.max()
            img_r = img_r / m if m > 0 else img_r
    else:
        m = img_r.max()
        img_r = img_r / m if m > 0 else img_r

    img_r = np.array(
        Image.fromarray((img_r * 255).astype(np.uint8)).resize((img_size, img_size), resample=Image.BICUBIC),
        dtype=np.float32
    ) / 255.0

    x = np.stack([img_r, img_r, img_r], axis=0)
    return torch.from_numpy(x)


# -----------------------------
# Read image metadata (spacing)
# -----------------------------
def _parse_listish(s: str) -> Optional[Tuple[float, float]]:
    if not isinstance(s, str): return None
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(vals) >= 2:
        return float(vals[0]), float(vals[1])
    return None


def load_image_metadata(meta_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    keep = ["out_path", "pixel_spacing_row_mm", "pixel_spacing_col_mm", "imager_pixel_spacing"]
    meta = meta[[c for c in keep if c in meta.columns]].copy()

    r = meta.get("pixel_spacing_row_mm", pd.Series([np.nan]*len(meta)))
    c = meta.get("pixel_spacing_col_mm", pd.Series([np.nan]*len(meta)))
    meta["spacing_row_mm"] = pd.to_numeric(r, errors="coerce")
    meta["spacing_col_mm"] = pd.to_numeric(c, errors="coerce")

    need_fallback = meta["spacing_row_mm"].isna() | (meta["spacing_row_mm"] <= 0) | \
                    meta["spacing_col_mm"].isna() | (meta["spacing_col_mm"] <= 0)
    if "imager_pixel_spacing" in meta.columns and need_fallback.any():
        parsed = meta.loc[need_fallback, "imager_pixel_spacing"].apply(_parse_listish)
        meta.loc[need_fallback, "spacing_row_mm"] = parsed.apply(lambda x: x[0] if x else np.nan)
        meta.loc[need_fallback, "spacing_col_mm"] = parsed.apply(lambda x: x[1] if x else np.nan)

    meta = meta.rename(columns={"out_path": "image_path"})
    meta["image_path"] = meta["image_path"].astype(str)
    return meta[["image_path", "spacing_row_mm", "spacing_col_mm"]]


# -----------------------------
# Dataset
# -----------------------------
class FusionDataset(Dataset):
    def __init__(self, rows: List[dict], rad_cols: List[str], img_size: int,
                 target_mm: float = 0.10, pad_mm: float = 10.0):
        self.rows = rows
        self.rad_cols = rad_cols
        self.img_size = img_size
        self.target_mm = target_mm
        self.pad_mm = pad_mm

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        sr = r.get("spacing_row_mm", np.nan)
        sc = r.get("spacing_col_mm", np.nan)
        spacing_rc = (
            float(sr) if sr == sr else 1.0,
            float(sc) if sc == sc else 1.0
        )
        x_img = mask_and_preprocess(Path(r["image_path"]), Path(r["mask_path"]),
                                    self.img_size, spacing_rc,
                                    target_mm=self.target_mm, pad_mm=self.pad_mm)
        x_rad = torch.tensor(r["rad_vector"], dtype=torch.float32)
        y = r["targets"]
        return x_img, x_rad, y, r["record_id"]


# -----------------------------
# Model
# -----------------------------
class FusionModel(nn.Module):
    def __init__(self, rad_dim: int, out_tasks: Dict[str, str]):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import resnet18
            backbone = resnet18(weights=None)

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
            self.heads[name] = nn.Linear(128, 1)

    def forward(self, x_img, x_rad):
        h_img = self.cnn(x_img)
        h_img = self.img_proj(h_img)
        h_rad = self.rad_proj(x_rad)
        z = self.fuse(torch.cat([h_img, h_rad], dim=1))
        out = {name: head(z).squeeze(1) for name, head in self.heads.items()}
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
            ys = torch.tensor([(ydict_i[t] if ydict_i[t] is not None else np.nan)
                               for ydict_i in ydict], dtype=torch.float32)
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
                prob = 1.0 / (1.0 + np.exp(-y_pred[mask]))
                auc = roc_auc_score(y_true[mask], prob)
                metrics[t] = {"AUROC": float(auc), "n": int(mask.sum())}
            except Exception:
                metrics[t] = {"AUROC": None, "n": int(mask.sum())}
        else:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            metrics[t] = {"MAE": float(mae), "n": int(mask.sum())}

    pred_rows = []
    for i, rid in enumerate(recs):
        row = {"record_id": rid}
        for t in tasks:
            row[f"{t}_pred"] = preds[t][i]
            row[f"{t}_true"] = gts[t][i]
        pred_rows.append(row)
    return metrics, pd.DataFrame(pred_rows)


# -----------------------------
# Build table and splits
# -----------------------------
def build_record_table(radiomics_csv: Path, clinical_csv: Path, image_meta_csv: Path, mlo_only=True):
    rad = pd.read_csv(radiomics_csv)
    clin = pd.read_csv(clinical_csv)
    imgmeta = load_image_metadata(image_meta_csv)

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

    rep["image_path"] = rep["image_path"].astype(str)
    df = rep.merge(imgmeta, on="image_path", how="left")

    for col in ["spacing_row_mm", "spacing_col_mm"]:
        if col not in df.columns and col in rep.columns:
            df[col] = rep[col]
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.merge(clin_t, on="record_id", how="inner")

    rad_cols = [c for c in df.columns if c.startswith("original_")]
    keep = ["record_id", "image_path", "mask_path", "spacing_row_mm", "spacing_col_mm"] + \
           rad_cols + ["ER_pos", "PR_pos", "HER2_pos", "Ki67", "Grade", "DCIS"]
    df = df[[c for c in keep if c in df.columns]].copy()

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
    steps = 0
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
        steps += 1
    return total / max(steps, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radiomics", type=Path, required=True, help="radiomics_features.csv")
    ap.add_argument("--clinical",  type=Path, required=True, help="clinical.csv")
    ap.add_argument("--image-metadata", type=Path, required=True,
                    help="CSV with DICOM-derived metadata (out_path, pixel_spacing_* etc.)")
    ap.add_argument("--outdir",    type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--target-mm", type=float, default=0.10, help="Target in-plane spacing (mm/pixel)")
    ap.add_argument("--pad-mm",    type=float, default=10.0, help="Padding around bbox in mm")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df, rad_cols = build_record_table(args.radiomics, args.clinical, args.image_metadata, mlo_only=True)
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
            "spacing_row_mm": r.get("spacing_row_mm", np.nan),
            "spacing_col_mm": r.get("spacing_col_mm", np.nan),
            "rad_vector": X_z[i, :].astype(np.float32),
            "targets":    row_to_targets(r),
        })

    tasks = {"ER_pos":"binary","PR_pos":"binary","HER2_pos":"binary","Ki67":"regression","Grade":"regression","DCIS":"binary"}
    ds = FusionDataset(rows, rad_cols, img_size=args.img_size,
                       target_mm=args.target_mm, pad_mm=args.pad_mm)
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

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    te_metrics, te_preds = evaluate(model, dl_te, device, tasks)
    print("[test] metrics:", te_metrics)

    json.dump(te_metrics, open(args.outdir/"metrics_test.json","w"), indent=2)
    te_preds.to_csv(args.outdir/"preds_test.csv", index=False)
    print(f"[ok] saved: {best_path}, metrics_test.json, preds_test.csv, splits.json, rad_stats_*.npy")


if __name__ == "__main__":
    main()
