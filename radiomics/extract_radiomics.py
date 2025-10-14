# extract_radiomics.py
"""
Example:
python extract_radiomics.py \
  --images-root /app/UC6_converted \
  --masks-manifest /app/UC6_masks/masks_manifest.csv \
  --out-csv /app/UC6_masks/radiomics_features.csv \
  --qc-csv /app/UC6_masks/radiomics_qc.csv \
  --log-level INFO
"""
import argparse, json, re, logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm.auto import tqdm

logger = logging.getLogger("extract_radiomics")

def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=lvl)
    logger.setLevel(lvl)

def lwarn(msg): tqdm.write(f"[warn] {msg}")
def linfo(msg):
    if logger.isEnabledFor(logging.INFO):
        tqdm.write(f"[info] {msg}")
def ldebug(msg):
    if logger.isEnabledFor(logging.DEBUG):
        tqdm.write(f"[debug] {msg}")


def load_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def to_sitk(img: np.ndarray, spacing_rc=None, is_mask=False) -> sitk.Image:
    im = sitk.GetImageFromArray(img)
    if spacing_rc is not None:
        row_mm, col_mm = spacing_rc
        im.SetSpacing((float(col_mm), float(row_mm)))
    if is_mask:
        im = sitk.Cast(im, sitk.sitkUInt8)
    else:
        if img.dtype == np.uint16:
            im = sitk.Cast(im, sitk.sitkUInt16)
        elif img.dtype == np.int16:
            im = sitk.Cast(im, sitk.sitkInt16)
        else:
            im = sitk.Cast(im, sitk.sitkFloat32)
    return im

def read_sidecar(image_png: Path):
    j = Path(str(image_png) + ".json")
    return json.loads(j.read_text()) if j.exists() else {}

def read_spacing_from_sidecars(image_png: Path):
    meta = read_sidecar(image_png)
    r = meta.get("pixel_spacing_row_mm"); c = meta.get("pixel_spacing_col_mm")
    if isinstance(r, (int, float)) and isinstance(c, (int, float)):
        return float(r), float(c)
    ps = meta.get("PixelSpacing") or meta.get("pixel_spacing") or meta.get("spacing_mm")
    if isinstance(ps, dict) and "row" in ps and "col" in ps:
        return float(ps["row"]), float(ps["col"])
    if isinstance(ps, (list, tuple)) and len(ps) >= 2:
        return float(ps[0]), float(ps[1])
    return None

def matches_view(image_png: Path, allowed_views: set) -> bool:
    """Return True if image matches any allowed view(s). 'ALL' bypasses filtering."""
    if "ALL" in allowed_views:
        return True
    meta = read_sidecar(image_png)
    view = (meta.get("view_position") or meta.get("ViewPosition") or "").strip().upper()
    if view:
        return view in allowed_views
    sd = (meta.get("SeriesDescription") or "").upper()
    return any(v in sd for v in allowed_views)

def build_img_index(images_root: Path):
    return {p.stem: p for p in images_root.rglob("*.png")}


def preprocess_intensity(img: np.ndarray, mask: np.ndarray, photometric: str):
    """
    Returns: img_normalized (float32), qc_dict
    """
    qc = {}
    inverted = bool(photometric and photometric.upper() == "MONOCHROME1")
    qc["inverted"] = inverted

    if inverted:
        if np.issubdtype(img.dtype, np.integer):
            maxv = np.iinfo(img.dtype).max
        else:
            maxv = float(np.nanmax(img))
        imgf = (maxv - img).astype(np.float32, copy=False)
    else:
        imgf = img.astype(np.float32, copy=False)

    roi_mask = (mask > 0)
    roi_vals = imgf[roi_mask]
    qc["roi_npix"] = int(roi_vals.size)

    if roi_vals.size >= 10:
        pre_min, pre_max = float(roi_vals.min()), float(roi_vals.max())
        pre_mean, pre_std = float(roi_vals.mean()), float(roi_vals.std())
        qc.update({"pre_min": pre_min, "pre_max": pre_max, "pre_mean": pre_mean, "pre_std": pre_std})

        p1, p99 = np.percentile(roi_vals, [1, 99])
        qc["p1"] = float(p1); qc["p99"] = float(p99)
        qc["frac_clipped_low"]  = float((roi_vals < p1).mean())
        qc["frac_clipped_high"] = float((roi_vals > p99).mean())

        np.clip(imgf, p1, p99, out=imgf)
        roi_vals_clipped = np.clip(roi_vals, p1, p99)
        mu = float(roi_vals_clipped.mean())
        sd = float(roi_vals_clipped.std())
        if sd < 1e-6:
            sd = 1.0
            lwarn("ROI std ≈ 0 after winsorization; using sd=1 to avoid division by zero")
        qc["mu"] = mu; qc["sd"] = sd

        imgf = (imgf - mu) / sd
        roi_norm = (roi_vals_clipped - mu) / sd
        qc.update({
            "post_min": float(roi_norm.min()), "post_max": float(roi_norm.max()),
            "post_mean": float(roi_norm.mean()), "post_std": float(roi_norm.std())
        })
    else:
        lwarn("ROI has <10 pixels; skipping normalization and using raw float32 values")
        qc.update({
            "pre_min": float(roi_vals.min()) if roi_vals.size else np.nan,
            "pre_max": float(roi_vals.max()) if roi_vals.size else np.nan,
            "pre_mean": float(roi_vals.mean()) if roi_vals.size else np.nan,
            "pre_std": float(roi_vals.std()) if roi_vals.size else np.nan,
            "p1": np.nan, "p99": np.nan,
            "frac_clipped_low": np.nan, "frac_clipped_high": np.nan,
            "mu": np.nan, "sd": np.nan,
            "post_min": np.nan, "post_max": np.nan, "post_mean": np.nan, "post_std": np.nan
        })
    return imgf.astype(np.float32, copy=False), qc


def main():
    ap = argparse.ArgumentParser(description="Extract 2D radiomic features from PNG images + masks (MLO-only) with QC logs.")
    ap.add_argument("--images-root", required=True, type=Path, help="Root with converted PNG images")
    ap.add_argument("--masks-manifest", required=True, type=Path,
                    help="CSV from seg_rebuild with columns: mask_path, sop_instance_uid, roi_name, ...")
    ap.add_argument("--out-csv", required=True, type=Path, help="Where to save features CSV")
    ap.add_argument("--qc-csv", type=Path, default=None, help="Optional QC CSV with per-ROI normalization stats and diagnostics")
    ap.add_argument("--exclude-regex", default=r"(?i)\b(smart|paint|deleted|ignore)\b",
                    help="Regex of ROI names to skip")
    ap.add_argument("--views", default="MLO",
                    help="Comma-separated allowed views (default 'MLO'; use 'MLO,CC' or 'ALL')")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = ap.parse_args()

    setup_logging(args.log_level)

    df = pd.read_csv(args.masks_manifest)
    needed = {"mask_path", "sop_instance_uid", "roi_name"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {missing}")

    excl = re.compile(args.exclude_regex)
    start_n = len(df)
    df = df[~df["roi_name"].fillna("").apply(lambda s: bool(excl.search(str(s))))].copy()
    linfo(f"ROI filter: {start_n} → {len(df)}")

    img_idx = build_img_index(args.images_root)
    df = df[df["sop_instance_uid"].astype(str).isin(img_idx.keys())].copy()
    if df.empty:
        raise SystemExit("No matching images found for SOPInstanceUIDs under images-root.")
    df["image_path"] = df["sop_instance_uid"].astype(str).map(img_idx)

    vp_vals = []
    for p in df["image_path"]:
        m = read_sidecar(p)
        vp = (m.get("view_position") or m.get("ViewPosition") or "").strip().upper() or "<MISSING>"
        vp_vals.append(vp)
    vp_counts = pd.Series(vp_vals).value_counts(dropna=False)
    linfo(f"view_position distribution: {dict(vp_counts)}")

    allowed_views = set([v.strip().upper() for v in args.views.split(",") if v.strip()]) or {"MLO"}
    view_mask = df["image_path"].apply(lambda p: matches_view(p, allowed_views))
    linfo(f"View filter kept ({','.join(sorted(allowed_views))}): {int(view_mask.sum())} / {len(view_mask)}")
    df = df[view_mask].copy()
    if df.empty:
        raise SystemExit("No images after view filter. Tip: try --views ALL")

    settings = {
        "binWidth": 0.3,
        "resampledPixelSpacing": [0.10, 0.10],
        "interpolator": "sitkBSpline",
        "enableCExtensions": True,
        "force2D": True,
        "correctMask": True,
        "symmetricGLCM": True,
        "distances": [1, 2, 3],
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for cls in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape2D"):
        extractor.enableFeatureClassByName(cls)

    out_rows, qc_rows = [], []
    it = tqdm(df.itertuples(index=False), total=len(df), desc="Extracting")
    for r in it:
        uid = str(getattr(r, "sop_instance_uid"))
        img_path = Path(getattr(r, "image_path"))
        mask_path = Path(getattr(r, "mask_path"))

        if not mask_path.exists():
            lwarn(f"{uid}: missing mask: {mask_path}")
            continue

        meta = read_sidecar(img_path)

        spacing_rc = read_spacing_from_sidecars(img_path)
        if spacing_rc is None:
            lwarn(f"{uid}: missing PixelSpacing in sidecar → skip")
            continue

        img = load_png(img_path)
        mask = load_png(mask_path)
        if img.shape[:2] != mask.shape[:2]:
            lwarn(f"{uid}: shape mismatch image {img.shape} vs mask {mask.shape} → skip")
            continue
        mask = (mask > 0).astype(np.uint8)

        if img.dtype == np.uint8:
            lwarn(f"{uid}: 8-bit image detected; consider 16-bit export")

        img_norm, qc = preprocess_intensity(img, mask, photometric=meta.get("photometric", ""))
        row_mm, col_mm = spacing_rc
        qc.update({
            "sop_instance_uid": uid,
            "photometric": meta.get("photometric", ""),
            "dtype": str(img.dtype),
            "binWidth": settings["binWidth"],
            "roi_area_mm2": float(mask.sum() * row_mm * col_mm)
        })
        ldebug(f"{uid}: inv={qc['inverted']} roi_n={qc['roi_npix']} "
               f"p1/p99={qc.get('p1')}/{qc.get('p99')} mu/sd={qc.get('mu')}/{qc.get('sd')} "
               f"post_mean/std={qc.get('post_mean')}/{qc.get('post_std')}")

        if qc.get("frac_clipped_low", 0) and qc["frac_clipped_low"] > 0.10:
            lwarn(f"{uid}: >10% ROI below p1 ({qc['frac_clipped_low']*100:.1f}%)")
        if qc.get("frac_clipped_high", 0) and qc["frac_clipped_high"] > 0.10:
            lwarn(f"{uid}: >10% ROI above p99 ({qc['frac_clipped_high']*100:.1f}%)")

        itk_img = to_sitk(img_norm, spacing_rc=spacing_rc, is_mask=False)
        itk_msk = to_sitk(mask, spacing_rc=spacing_rc, is_mask=True)

        try:
            feats = extractor.execute(itk_img, itk_msk, label=1)
        except Exception as e:
            lwarn(f"{uid}: extraction error: {e}")
            continue

        rec = {
            "record_id": getattr(r, "record_id", None),
            "sop_instance_uid": uid,
            "roi_number": getattr(r, "roi_number", None),
            "roi_name": getattr(r, "roi_name", None),
            "mask_path": str(mask_path),
            "image_path": str(img_path),
            "spacing_row_mm": row_mm,
            "spacing_col_mm": col_mm,
            "view": "MLO",
            "manufacturer": meta.get("manufacturer", ""),
            "model_name": meta.get("model_name", ""),
            "photometric": meta.get("photometric", ""),
        }
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            try:
                rec[k] = float(v)
            except Exception:
                pass
        out_rows.append(rec)

        diag_keep = {k: v for k, v in feats.items() if k.startswith("diagnostics_")}
        qc_rows.append({**qc, **diag_keep})

    if not out_rows:
        raise SystemExit("No features extracted.")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)
    linfo(f"features CSV: {args.out_csv} (rows: {len(out_df)})")

    if args.qc_csv:
        qc_df = pd.DataFrame(qc_rows)
        qc_df.to_csv(args.qc_csv, index=False)
        linfo(f"QC CSV: {args.qc_csv} (rows: {len(qc_df)})")
        try:
            ok = qc_df["post_std"].dropna()
            linfo(f"QC summary: post_std median={ok.median():.3f}, IQR=({ok.quantile(0.25):.3f},{ok.quantile(0.75):.3f})")
        except Exception:
            pass

    linfo("Done.")

if __name__ == "__main__":
    main()
