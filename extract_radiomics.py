# extract_radiomics.py
"""
Example:
python extract_radiomics.py \
  --images-root /app/UC6_converted \
  --masks-manifest /app/UC6_masks/masks_manifest.csv \
  --out-csv /app/UC6_masks/radiomics_features.csv
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm.auto import tqdm


def load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))

def to_sitk(img: np.ndarray, spacing_rc=None, is_mask=False) -> sitk.Image:
    im = sitk.GetImageFromArray(img)
    if spacing_rc:
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

def read_spacing_from_sidecars(mask_png: Path, image_png: Path):
    mjson = Path(str(mask_png) + ".json")
    if mjson.exists():
        try:
            meta = json.loads(mjson.read_text())
            s = meta.get("spacing_mm")
            if s and "row" in s and "col" in s:
                return float(s["row"]), float(s["col"])
        except Exception:
            pass
    ijson = Path(str(image_png) + ".json")
    if ijson.exists():
        try:
            meta = json.loads(ijson.read_text())
            ps = meta.get("PixelSpacing") or meta.get("pixel_spacing") or meta.get("spacing_mm")
            if isinstance(ps, dict) and "row" in ps and "col" in ps:
                return float(ps["row"]), float(ps["col"])
            if isinstance(ps, (list, tuple)) and len(ps) >= 2:
                return float(ps[0]), float(ps[1])
        except Exception:
            pass
    return (1.0, 1.0)

def is_mlo(image_png: Path) -> bool:
    """Detect MLO view from image sidecar JSON using common keys/strings."""
    ijson = Path(str(image_png) + ".json")
    if not ijson.exists():
        return False
    try:
        meta = json.loads(ijson.read_text())
    except Exception:
        return False
    for key in ("ViewPosition", "ViewCodeMeaning", "ViewLabel", "SeriesDescription"):
        v = meta.get(key)
        if isinstance(v, str) and "MLO" in v.upper():
            return True
    for v in meta.values():
        if isinstance(v, str) and "MLO" in v.upper():
            return True
    return False

def build_img_index(images_root: Path):
    """Map SOPInstanceUID (png stem) -> PNG path."""
    return {p.stem: p for p in images_root.rglob("*.png")}


def main():
    ap = argparse.ArgumentParser(description="Extract 2D radiomic features from PNG images + masks (MLO only).")
    ap.add_argument("--images-root", required=True, type=Path, help="Root with converted PNG images")
    ap.add_argument("--masks-manifest", required=True, type=Path,
                    help="CSV from seg_rebuild with columns: mask_path, sop_instance_uid, roi_name, ...")
    ap.add_argument("--out-csv", required=True, type=Path, help="Where to save features CSV")
    ap.add_argument("--exclude-regex", default=r"(?i)\b(smart|paint|deleted|ignore)\b",
                    help="Regex of ROI names to skip")
    args = ap.parse_args()

    df = pd.read_csv(args.masks_manifest)
    needed = {"mask_path", "sop_instance_uid", "roi_name"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {missing}")

    excl = re.compile(args.exclude_regex)
    df = df[~df["roi_name"].fillna("").apply(lambda s: bool(excl.search(str(s))))].copy()
    if df.empty:
        raise SystemExit("Nothing to process after ROI filtering.")

    img_idx = build_img_index(args.images_root)
    df = df[df["sop_instance_uid"].astype(str).isin(img_idx.keys())].copy()
    if df.empty:
        raise SystemExit("No matching images found for SOPInstanceUIDs under images-root.")

    df["image_path"] = df["sop_instance_uid"].astype(str).map(img_idx)
    tqdm.write("[info] filtering MLO views…")
    df = df[df["image_path"].apply(is_mlo)].copy()
    if df.empty:
        raise SystemExit("No MLO images after filtering.")

    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": [0.10, 0.10],
        "interpolator": "sitkBSpline",
        "enableCExtensions": True,
        "force2D": True,
        "correctMask": True,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for cls in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape2D"):
        extractor.enableFeatureClassByName(cls)

    out_rows = []
    it = tqdm(df.itertuples(index=False), total=len(df), desc="Extracting")
    for r in it:
        uid = str(getattr(r, "sop_instance_uid"))
        img_path = Path(getattr(r, "image_path"))
        mask_path = Path(getattr(r, "mask_path"))

        if not mask_path.exists():
            tqdm.write(f"[skip] missing mask: {mask_path}")
            continue

        img = load_png(img_path)
        mask = load_png(mask_path)
        if img.shape[:2] != mask.shape[:2]:
            tqdm.write(f"[skip] shape mismatch: {uid}")
            continue
        mask = (mask > 0).astype(np.uint8)

        spacing_rc = read_spacing_from_sidecars(mask_path, img_path)
        itk_img = to_sitk(img, spacing_rc=spacing_rc, is_mask=False)
        itk_msk = to_sitk(mask, spacing_rc=spacing_rc, is_mask=True)

        try:
            feats = extractor.execute(itk_img, itk_msk, label=1)
        except Exception as e:
            tqdm.write(f"[err] {uid}: {e}")
            continue

        rec = {
            "record_id": getattr(r, "record_id", None),
            "sop_instance_uid": uid,
            "roi_number": getattr(r, "roi_number", None),
            "roi_name": getattr(r, "roi_name", None),
            "mask_path": str(mask_path),
            "image_path": str(img_path),
            "spacing_row_mm": spacing_rc[0],
            "spacing_col_mm": spacing_rc[1],
            "view": "MLO",
        }
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            try:
                rec[k] = float(v)
            except Exception:
                pass
        out_rows.append(rec)

    if not out_rows:
        raise SystemExit("No features extracted.")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)
    tqdm.write(f"[ok] wrote {len(out_df)} rows → {args.out_csv}")

if __name__ == "__main__":
    main()
