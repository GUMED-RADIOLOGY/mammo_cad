#!/usr/bin/env python3
"""
DICOM → lossless 16-bit PNG/TIFF/NPY with a compact, harmonized manifest.

What it does
- Decodes pixels with pydicom (and available handlers like pylibjpeg/GDCM) respecting the Transfer Syntax.
- Preserves raw stored pixel values (no Modality LUT, no VOI LUT) in the lossless output.
- Normalizes dtype from DICOM header:
  - PixelRepresentation=0 → uint16
  - PixelRepresentation=1 → int16 (saved as uint16 for PNG/TIFF; prefer --format tiff/npy if signed)
- Writes one file per DICOM using SOPInstanceUID as the filename (avoids collisions on dotted UIDs like "2.25.*").
- Emits a JSON sidecar per image + a global CSV manifest.

Harmonization captured in the manifest (no pixel changes)
- Pixel spacing: prefers (0028,0030) PixelSpacing; falls back to (0018,1164) ImagerPixelSpacing. Both recorded.
- Core acquisition/display tags copied verbatim: Rows/Cols, BitsAllocated/BitsStored/HighBit,
  PixelRepresentation, PhotometricInterpretation, WindowCenter/WindowWidth, RescaleSlope/Intercept,
  LossyImageCompression, PixelPaddingValue, PixelIntensityRelationship(+Sign).
- Device/site context recorded: Manufacturer, ModelName, SoftwareVersions, UIDs (Study/Series/SOP).
- Study/patient context recorded: PatientSex/Age, dates.
- Deterministic file identity: SHA-256 hash of the raw pixel buffer, shape, and dtype.

Preview (human viewing only; does NOT affect lossless output)
- Optional “_vis.png” per DICOM (frame 0 if multi-frame):
  1) apply_modality_lut()
  2) apply_voi_lut() or WC/WW fallback; otherwise percentile [0.5, 99.5]
  3) if PixelIntensityRelationship == LOG and Sign>0 → log stretch
  4) normalize to 8-bit; invert if MONOCHROME1
- Purpose: quick QC; do not use for training/quant.

File layout
- out_root/<record_id>/<SOPInstanceUID>.<ext>
- Sidecar: same path + ".json"
- Manifest: one row per DICOM with paths + harmonized fields.

Performance
- Parallel decode with ProcessPoolExecutor.
- Lossless PNG (16-bit grayscale) or TIFF (16-bit, LZW); NPY stores exact ndarray.

Non-goals (by design)
- No de-identification, cropping, padding removal, denoising, equalization, or histogram changes.
- No vendor-specific display pipelines are applied to the lossless data.

python dicom2lossless.py \   --in-root /data/UC6 \   --out-root /app/UC6_converted \   --format png \   --manifest /app/UC6_converted/manifest.csv \   --jobs 16 \   --write-vis8bit \   --verbose

"""

import os, sys, glob, json, argparse, warnings, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# -------------------------- helpers -------------------------------

def record_id_from_path(dcm_path: Path, in_root: Path) -> str:
    """Pick the first path component under in_root as record_id."""
    try:
        rel = dcm_path.relative_to(in_root)
    except Exception:
        rel = Path(*dcm_path.parts[dcm_path.parts.index(in_root.name)+1:])
    return rel.parts[0]

def safe_get(ds, name, default=None):
    try: return getattr(ds, name)
    except Exception: return default

def to_str(x):
    if x is None: return ""
    if isinstance(x, (list, tuple)): return ",".join(map(str, x))
    return str(x)

def effective_pixel_spacing(ds):
    """Prefer PixelSpacing, else ImagerPixelSpacing."""
    ps = safe_get(ds, "PixelSpacing", None)
    if ps and len(ps) >= 2: return float(ps[0]), float(ps[1])
    ips = safe_get(ds, "ImagerPixelSpacing", None)
    if ips and len(ips) >= 2: return float(ips[0]), float(ips[1])
    return None, None

def window_for_visual(ds, arr):
    """Make an 8-bit view using modality + VOI LUTs (for humans)."""
    vis = arr.astype(np.float32)
    try: vis = apply_modality_lut(vis, ds)
    except Exception: pass
    try: vis = apply_voi_lut(vis, ds)
    except Exception:
        wc, ww = safe_get(ds, "WindowCenter", None), safe_get(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            wc = float(wc[0] if isinstance(wc, (list, tuple)) else wc)
            ww = float(ww[0] if isinstance(ww, (list, tuple)) else ww)
            lo, hi = wc - ww/2.0, wc + ww/2.0
            vis = np.clip(vis, lo, hi)
        else:
            lo, hi = np.percentile(vis, [0.5, 99.5])
            if hi <= lo: lo, hi = vis.min(), vis.max()
            vis = np.clip(vis, lo, hi)
    # LOG relationship -> log stretch
    if str(safe_get(ds, "PixelIntensityRelationship", "")).upper().startswith("LOG") and \
       str(safe_get(ds, "PixelIntensityRelationshipSign", "1")) in ("1", "None"):
        vis = np.log1p(np.maximum(vis - vis.min(), 0.0))
    # 0..255
    vmin, vmax = float(vis.min()), float(vis.max())
    if vmax <= vmin: vmax = vmin + 1.0
    vis = ((vis - vmin) / (vmax - vmin) * 255.0).round().astype(np.uint8)
    # Invert if MONOCHROME1
    if safe_get(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        vis = 255 - vis
    return vis

def pixel_hash(arr):
    h = hashlib.sha256()
    h.update(arr.tobytes()); h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode())
    return h.hexdigest()

# -------------------------- worker --------------------------

def process_one(dcm_path, in_root, out_root, fmt, write_vis8bit, verbose=False):
    dcm_path = Path(dcm_path); in_root = Path(in_root); out_root = Path(out_root)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(dcm_path, force=True)

        rid = record_id_from_path(dcm_path, in_root)
        sop = to_str(safe_get(ds, "SOPInstanceUID", dcm_path.stem))
        ext = { "png": ".png", "tiff": ".tiff", "npy": ".npy" }[fmt]
        out_path = out_root / rid / (sop + ext)

        # raw pixels (no LUTs)
        arr = ds.pixel_array
        # dtype align
        if int(safe_get(ds, "PixelRepresentation", 0)) == 0:
            arr = arr.astype(np.uint16, copy=False)
        else:
            arr = arr.astype(np.int16, copy=False)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # save lossless
        if fmt == "npy":
            np.save(out_path, arr)
        elif fmt == "png":
            # PNG is unsigned; if signed, cast
            img = Image.fromarray(arr.astype(np.uint16), mode="I;16")
            img.save(out_path)
        elif fmt == "tiff":
            img = Image.fromarray(arr.astype(np.uint16), mode="I;16")
            img.save(out_path, compression="tiff_lzw")

        # preview (unique per DICOM)
        vis_path = ""
        if write_vis8bit:
            try:
                if arr.ndim == 3:
                    vis = window_for_visual(ds, arr[0])
                else:
                    vis = window_for_visual(ds, arr)
                base_noext = Path(str(out_path)[:-len(ext)])
                vis_path = base_noext.with_name(base_noext.name + "_vis").with_suffix(".png")
                Image.fromarray(vis, mode="L").save(vis_path)
            except Exception as e:
                vis_path = ""
                if verbose: print(f"[warn] preview failed: {dcm_path} -> {e}")

        # metadata row
        row_mm, col_mm = effective_pixel_spacing(ds)
        row = {
            "dicom_path": str(dcm_path),
            "out_path": str(out_path),
            "record_id": rid,
            "sop_instance_uid": sop,
            "series_instance_uid": to_str(safe_get(ds, "SeriesInstanceUID","")),
            "study_instance_uid": to_str(safe_get(ds, "StudyInstanceUID","")),
            "modality": to_str(safe_get(ds, "Modality","")),
            "image_type": to_str(safe_get(ds, "ImageType","")),
            "presentation_intent": to_str(safe_get(ds, "PresentationIntentType","")),
            "manufacturer": to_str(safe_get(ds, "Manufacturer","")),
            "model_name": to_str(safe_get(ds, "ManufacturersModelName","")) or to_str(safe_get(ds,"ManufacturerModelName","")),
            "software_versions": to_str(safe_get(ds, "SoftwareVersions","")),
            "anatomic_region": to_str(safe_get(ds, "AnatomicRegionSequence","")),
            "view_position": to_str(safe_get(ds, "ViewPosition","")),
            "image_laterality": to_str(safe_get(ds, "ImageLaterality","")),
            "patient_sex": to_str(safe_get(ds, "PatientSex","")),
            "patient_age": to_str(safe_get(ds, "PatientAge","")),
            "study_date": to_str(safe_get(ds, "StudyDate","")),
            "acquisition_date": to_str(safe_get(ds, "AcquisitionDate","")),
            "rows": int(safe_get(ds, "Rows", arr.shape[-2] if arr.ndim>=2 else 0)),
            "cols": int(safe_get(ds, "Columns", arr.shape[-1] if arr.ndim>=2 else 0)),
            "bits_allocated": int(safe_get(ds, "BitsAllocated", 0)),
            "bits_stored": int(safe_get(ds, "BitsStored", 0)),
            "high_bit": int(safe_get(ds, "HighBit", 0)),
            "pixel_representation": int(safe_get(ds, "PixelRepresentation", 0)),
            "photometric": to_str(safe_get(ds, "PhotometricInterpretation","")),
            "pixel_spacing_row_mm": row_mm,
            "pixel_spacing_col_mm": col_mm,
            "imager_pixel_spacing": to_str(safe_get(ds, "ImagerPixelSpacing","")),
            "rescale_intercept": to_str(safe_get(ds, "RescaleIntercept","")),
            "rescale_slope": to_str(safe_get(ds, "RescaleSlope","")),
            "window_center": to_str(safe_get(ds, "WindowCenter","")),
            "window_width": to_str(safe_get(ds, "WindowWidth","")),
            "lossy_image_compression": to_str(safe_get(ds, "LossyImageCompression","")),
            "burned_in_annotation": to_str(safe_get(ds, "BurnedInAnnotation","")),
            "pixel_padding_value": to_str(safe_get(ds, "PixelPaddingValue","")),
            "pixel_intensity_relationship": to_str(safe_get(ds, "PixelIntensityRelationship","")),
            "pixel_intensity_relationship_sign": to_str(safe_get(ds, "PixelIntensityRelationshipSign","")),
            "kvp": to_str(safe_get(ds, "KVP","")),
            "exposure_time_ms": to_str(safe_get(ds, "ExposureTime","")),
            "xray_tube_current": to_str(safe_get(ds, "XRayTubeCurrent","")),
            "exposure": to_str(safe_get(ds, "Exposure","")),
            "compression_force": to_str(safe_get(ds, "CompressionForce","")),
            "detector_type": to_str(safe_get(ds, "DetectorType","")),
            "detector_description": to_str(safe_get(ds, "DetectorDescription","")),
            "pixel_hash_sha256": pixel_hash(arr),
            "sidecar_json": str(out_path) + ".json",
            "vis8bit_path": str(vis_path) if vis_path else "",
        }

        # write sidecar JSON
        Path(row["sidecar_json"]).write_text(json.dumps(row, indent=2, ensure_ascii=False))

        if verbose:
            print(f"[ok] {dcm_path} -> {out_path} ({arr.shape}, {arr.dtype})")
        return row, None

    except Exception as e:
        return None, f"{dcm_path}: {e}"

# -------------------------- main --------------------------

def find_dicoms(in_root: Path):
    pat = str(in_root / "*" / "**" / "resources" / "DICOM" / "files" / "*.dcm")
    return glob.glob(pat, recursive=True)

def main():
    ap = argparse.ArgumentParser(description="DICOM → lossless 16-bit images + manifest")
    ap.add_argument("--in-root", required=True, help="Input root, e.g. /data/UC6")
    ap.add_argument("--out-root", required=True, help="Output root")
    ap.add_argument("--format", default="png", choices=["png","tiff","npy"], help="Lossless format")
    ap.add_argument("--manifest", required=True, help="CSV manifest path")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count()//2), help="Workers")
    ap.add_argument("--write-vis8bit", action="store_true", help="Also write <SOP>_vis.png")
    ap.add_argument("--verbose", action="store_true", help="Print per-file info")
    args = ap.parse_args()

    in_root = Path(args.in_root); out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dcm_files = find_dicoms(in_root)
    if not dcm_files:
        print(f"No DICOMs under {in_root}", file=sys.stderr); sys.exit(2)

    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(process_one, p, str(in_root), str(out_root),
                          args.format, args.write_vis8bit, args.verbose)
                for p in dcm_files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Converting"):
            row, err = fut.result()
            if row is not None: rows.append(row)
            if err is not None: errors.append(err)

    if rows:
        pd.DataFrame(rows).to_csv(args.manifest, index=False)
        print(f"Manifest: {args.manifest} (rows: {len(rows)})")
    else:
        print("No successful conversions.", file=sys.stderr)

    if errors:
        err_log = Path(args.manifest).with_suffix(".errors.txt")
        err_log.write_text("\n".join(errors))
        print(f"Some files failed. See {err_log}")

if __name__ == "__main__":
    main()
