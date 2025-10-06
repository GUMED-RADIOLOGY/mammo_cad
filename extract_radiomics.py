#!/usr/bin/env python3
# extract_radiomics.py
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from radiomics import featureextractor

def load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))

def to_sitk(img: np.ndarray, spacing_rc=None, is_mask=False) -> sitk.Image:
    im = sitk.GetImageFromArray(img)  # array -> z,y,x with z=0 for 2D
    if spacing_rc:
        # SimpleITK spacing is (x, y) == (col, row)
        row_mm, col_mm = spacing_rc
        im.SetSpacing((float(col_mm), float(row_mm)))
    if is_mask:
        im = sitk.Cast(im, sitk.sitkUInt8)  # mask label=1
    else:
        if img.dtype == np.uint16:
            im = sitk.Cast(im, sitk.sitkUInt16)
        elif img.dtype == np.int16:
            im = sitk.Cast(im, sitk.sitkInt16)
        else:
            im = sitk.Cast(im, sitk.sitkFloat32)
    return im

def read_spacing_from_sidecars(mask_png: Path, image_png: Path):
    # 1) mask sidecar (from seg_rebuild): {... "spacing_mm": {"row": r, "col": c}}
    mjson = Path(str(mask_png) + ".json")
    if mjson.exists():
        try:
            meta = json.loads(mjson.read_text())
            s = meta.get("spacing_mm")
            if s and "row" in s and "col" in s:
                return float(s["row"]), float(s["col"])
        except Exception:
            pass
    # 2) image sidecar (converter placed PNG.json)
    ijson = Path(str(image_png) + ".json")
    if ijson.exists():
        try:
            meta = json.loads(ijson.read_text())
            # common keys: "PixelSpacing": [row, col] or similar
            ps = meta.get("PixelSpacing") or meta.get("pixel_spacing") or meta.get("spacing_mm")
            if isinstance(ps, dict) and "row" in ps and "col" in ps:
                return float(ps["row"]), float(ps["col"])
            if isinstance(ps, (list, tuple)) and len(ps) >= 2:
                return float(ps[0]), float(ps[1])
        except Exception:
            pass
    # 3) fallback
    return (1.0, 1.0)

def build_img_index(images_root: Path):
    return {p.stem: p for p in images_root.rglob("*.png")}

def main():
    ap = argparse.ArgumentParser(description="Extract 2D radiomic features from PNG images + masks.")
    ap.add_argument("--images-root", required=True, type=Path, help="Root with converted PNG images")
    ap.add_argument("--masks-manifest", "--manifest", dest="manifest", required=True, type=Path,
                    help="CSV from seg_rebuild with columns: mask_path, sop_instance_uid, roi_name, ...")
    ap.add_argument("--out-csv", required=True, type=Path, help="Where to save features CSV")
    ap.add_argument("--exclude-regex", default=r"(?i)smart|paint|deleted|ignore",
                    help="Regex of ROI names to skip")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    needed = {"mask_path", "sop_instance_uid", "roi_name"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {missing}")

    # filter ROIs
    excl = re.compile(args.exclude_regex)
    df = df[~df["roi_name"].fillna("").apply(lambda s: bool(excl.search(str(s))))].copy()
    if df.empty:
        raise SystemExit("Nothing to process after filtering.")

    # map images
    img_idx = build_img_index(args.images_root)
    df = df[df["sop_instance_uid"].astype(str).isin(img_idx.keys())]
    if df.empty:
        raise SystemExit("No matching images found for SOPInstanceUIDs under images-root.")

    # set up PyRadiomics extractor (2D)
    extractor = featureextractor.RadiomicsFeatureExtractor(
        **{
            "binWidth": 25,
            "resampledPixelSpacing": None,
            "interpolator": "sitkBSpline",
            "enableCExtensions": True,
            "force2D": True,  # treat as 2D
        }
    )
    extractor.disableAllFeatures()
    for cls in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape2D"):
        extractor.enableFeatureClassByName(cls)

    rows = []
    for i, row in df.iterrows():
        uid = str(row["sop_instance_uid"])
        img_path = img_idx[uid]
        mask_path = Path(row["mask_path"])
        if not mask_path.exists():
            print(f"[skip] missing mask: {mask_path}")
            continue

        img = load_png(img_path)
        mask = load_png(mask_path)
        if img.shape[:2] != mask.shape[:2]:
            print(f"[skip] shape mismatch: {uid}")
            continue

        # binarize mask to {0,1}
        mask = (mask > 0).astype(np.uint8)

        spacing_rc = read_spacing_from_sidecars(mask_path, img_path)
        itk_img = to_sitk(img, spacing_rc=spacing_rc, is_mask=False)
        itk_msk = to_sitk(mask, spacing_rc=spacing_rc, is_mask=True)

        try:
            feats = extractor.execute(itk_img, itk_msk, label=1)
        except Exception as e:
            print(f"[err] {uid}: {e}")
            continue

        # flatten/clean keys
        rec = {
            "record_id": row.get("record_id"),
            "sop_instance_uid": uid,
            "roi_number": row.get("roi_number"),
            "roi_name": row.get("roi_name"),
            "mask_path": str(mask_path),
            "image_path": str(img_path),
            "spacing_row_mm": spacing_rc[0],
            "spacing_col_mm": spacing_rc[1],
        }
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            try:
                rec[k] = float(v)
            except Exception:
                pass  # skip non-numerics
        rows.append(rec)

    if not rows:
        raise SystemExit("No features extracted.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {len(out_df)} rows → {args.out_csv}")

if __name__ == "__main__":
    main()
