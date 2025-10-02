#!/usr/bin/env python3
"""
RTSTRUCT → per-image PNG masks (lossless) + CSV manifest.

What it does
- Recursively finds `segmentation.dcm` (SOP Class: RT Structure Set).
- For each ROI, rasterizes its polygons onto the referenced image(s).
- Uses DICOM geometry when available:
  * ContourData (mm, patient coords) + ImagePositionPatient + ImageOrientationPatient + PixelSpacing → pixel (row,col).
  * If geometry missing, falls back to simple mm-from-top-left using PixelSpacing.
- Honors CLOSED_PLANAR_XOR semantics: polygons are XOR-composited to form holes/islands.
- Saves binary masks as PNG (0 background, 255 ROI).
- Writes a CSV manifest with paths and key IDs.

Notes
- No pixel data from the images are read—only geometry/size/spacing/UIDs.
- PNG is lossless; mask values are exact.
"""

import os, glob, argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pydicom
import pandas as pd

# --------- helpers ---------

def record_id_from_path(dcm_path: Path, in_root: Path) -> str:
    try:
        return dcm_path.relative_to(in_root).parts[0]
    except Exception:
        # fallback: next segment after root name
        return Path(*dcm_path.parts[dcm_path.parts.index(in_root.name)+1:]).parts[0]

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or ""))[:64]

def get_spacing(ds):
    ps = getattr(ds, "PixelSpacing", None)
    if ps and len(ps) >= 2:  # [row_mm, col_mm]
        return float(ps[0]), float(ps[1])
    ips = getattr(ds, "ImagerPixelSpacing", None)
    if ips and len(ips) >= 2:
        return float(ips[0]), float(ips[1])
    return 1.0, 1.0

def load_image_header(path: Path):
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    rows = int(getattr(ds, "Rows", 0))
    cols = int(getattr(ds, "Columns", 0))
    rmm, cmm = get_spacing(ds)
    iop = getattr(ds, "ImageOrientationPatient", None)  # [r_x,r_y,r_z,c_x,c_y,c_z]
    ipp = getattr(ds, "ImagePositionPatient", None)     # [x,y,z]
    return ds, rows, cols, (rmm, cmm), iop, ipp

def mm_xyz_to_rc(xyz, rows, cols, spacing, iop, ipp):
    """
    Map patient (mm) coords to pixel row/col.
    If full geometry present: use ipp/iop; else assume top-left origin in mm.
    """
    x, y, z = xyz
    rmm, cmm = spacing
    if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
        r = np.array(iop[0:3], float)
        c = np.array(iop[3:6], float)
        p = np.array([x, y, z], float)
        o = np.array(ipp, float)
        d = p - o  # mm
        # components along row/col axes (mm) → pixels
        rr = (d @ r) / rmm
        cc = (d @ c) / cmm
    else:
        # fallback: treat (x=col_mm, y=row_mm) from top-left
        rr = y / rmm
        cc = x / cmm
    # clamp to image bounds
    rr = float(np.clip(rr, -1, rows))  # allow slight outside for polygon edges
    cc = float(np.clip(cc, -1, cols))
    return rr, cc

# --------- index all DICOM images under root ---------

def build_image_index(in_root: Path):
    print("[idx] scanning images…")
    pat = str(in_root / "*" / "**" / "resources" / "DICOM" / "files" / "*.dcm")
    idx = {}  # sop_uid -> dict(meta)
    for p in glob.iglob(pat, recursive=True):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            sop = getattr(ds, "SOPInstanceUID", None)
            if sop:
                rows = int(getattr(ds, "Rows", 0))
                cols = int(getattr(ds, "Columns", 0))
                if rows and cols:
                    idx[str(sop)] = {"path": Path(p)}
        except Exception:
            pass
    print(f"[idx] {len(idx)} image headers found")
    return idx

# --------- conversion ---------

def process_rtstruct(seg_path: Path, in_root: Path, out_root: Path, img_idx, manifest_rows):
    ds = pydicom.dcmread(seg_path, force=True)
    if str(getattr(ds, "SOPClassUID", "")) not in (
        pydicom.uid.RTStructureSetStorage, "1.2.840.10008.5.1.4.1.1.481.3"
    ):
        print(f"[skip] not RTSTRUCT: {seg_path}")
        return

    rid = record_id_from_path(seg_path, in_root)
    print(f"[seg] {seg_path} (RID={rid})")

    roi_meta = {}
    for item in getattr(ds, "StructureSetROISequence", []) or []:
        num = str(getattr(item, "ROINumber", ""))
        roi_meta[num] = {"name": getattr(item, "ROIName", f"ROI_{num}"), "color": None}

    for rc in getattr(ds, "ROIContourSequence", []) or []:
        ref_num = str(getattr(rc, "ReferencedROINumber", ""))
        color = getattr(rc, "ROIDisplayColor", None)
        if ref_num not in roi_meta:
            roi_meta[ref_num] = {"name": f"ROI_{ref_num}", "color": color}
        else:
            roi_meta[ref_num]["color"] = color

        # Each ContourSequence item belongs to a specific referenced image
        contours = getattr(rc, "ContourSequence", []) or []
        # group by referenced SOP
        per_sop = {}
        for c in contours:
            cis = getattr(c, "ContourImageSequence", []) or []
            if not cis:  # no image ref → skip
                continue
            rsop = str(getattr(cis[0], "ReferencedSOPInstanceUID", ""))
            data = list(map(float, getattr(c, "ContourData", []) or []))
            pts = [tuple(data[i:i+3]) for i in range(0, len(data), 3)]
            per_sop.setdefault(rsop, []).append(pts)

        # Rasterize XOR per referenced SOP
        for rsop, polys in per_sop.items():
            if rsop not in img_idx:
                # try to locate on the fly (rare)
                continue
            # load missing meta (once)
            meta = img_idx[rsop]
            if "rows" not in meta:
                ds_img, rows, cols, spacing, iop, ipp = load_image_header(meta["path"])
                meta.update({"rows": rows, "cols": cols, "spacing": spacing, "iop": iop, "ipp": ipp})
                img_idx[rsop] = meta

            rows, cols = meta["rows"], meta["cols"]
            spacing, iop, ipp = meta["spacing"], meta["iop"], meta["ipp"]
            if not rows or not cols:
                print(f"[warn] no size for {rsop}")
                continue

            mask = np.zeros((rows, cols), dtype=np.bool_)
            for pts in polys:
                # Map mm → pixels
                rc_pix = [mm_xyz_to_rc(p, rows, cols, spacing, iop, ipp) for p in pts]
                # PIL expects (x=col, y=row)
                poly_xy = [(c, r) for r, c in rc_pix]
                # draw filled polygon on temp mask
                tmp = Image.new("1", (cols, rows), 0)
                ImageDraw.Draw(tmp).polygon(poly_xy, outline=1, fill=1)
                mask ^= np.array(tmp, dtype=bool)  # XOR composition

            # Save mask PNG
            roi_name = sanitize(roi_meta.get(ref_num, {}).get("name", f"ROI_{ref_num}"))
            out_dir = out_root / rid / "masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_png = out_dir / f"{rsop}__roi-{ref_num}_{roi_name}.png"
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_png)

            # Sidecar (tiny)
            side = {
                "seg_source": str(seg_path),
                "record_id": rid,
                "sop_instance_uid": rsop,
                "rows": rows, "cols": cols,
                "roi_number": ref_num,
                "roi_name": roi_name,
                "roi_color_rgb": roi_meta.get(ref_num, {}).get("color", None),
                "spacing_mm": {"row": spacing[0], "col": spacing[1]},
                "geometry_mode": "iop+ipp" if (iop is not None and ipp is not None) else "top_left_mm_fallback"
            }
            Path(str(out_png) + ".json").write_text(json.dumps(side, indent=2))

            # Manifest row
            manifest_rows.append({
                "record_id": rid,
                "seg_path": str(seg_path),
                "mask_path": str(out_png),
                "sop_instance_uid": rsop,
                "roi_number": ref_num,
                "roi_name": roi_name,
                "rows": rows,
                "cols": cols
            })

def main():
    ap = argparse.ArgumentParser(description="Rebuild PNG masks from RTSTRUCT segmentation.dcm files.")
    ap.add_argument("--in-root", required=True, help="Input root (same as images root)")
    ap.add_argument("--out-root", required=True, help="Output root for masks")
    ap.add_argument("--manifest", required=True, help="CSV manifest path to write")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Find RTSTRUCT files by name; robust alternative: also accept any .dcm with RTSTRUCT SOPClass
    seg_paths = [Path(p) for p in glob.iglob(str(in_root / "**" / "segmentation.dcm"), recursive=True)]
    # Also include any .dcm that are RTSTRUCT even if not named 'segmentation.dcm'
    extra = []
    pat_all = str(in_root / "*" / "**" / "resources" / "DICOM" / "files" / "*.dcm")
    for p in glob.iglob(pat_all, recursive=True):
        if p.endswith("segmentation.dcm"):
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if str(getattr(ds, "SOPClassUID", "")) in (pydicom.uid.RTStructureSetStorage, "1.2.840.10008.5.1.4.1.1.481.3"):
                extra.append(Path(p))
        except Exception:
            pass
    seg_paths.extend(extra)
    seg_paths = sorted(set(seg_paths))

    if not seg_paths:
        print(f"[done] no RTSTRUCT files found under {in_root}")
        return

    # Build image index
    img_idx = build_image_index(in_root)

    manifest_rows = []
    for sp in seg_paths:
        try:
            process_rtstruct(sp, in_root, out_root, img_idx, manifest_rows)
        except Exception as e:
            print(f"[err] {sp}: {e}")

    if manifest_rows:
        df = pd.DataFrame(manifest_rows)
        df.to_csv(args.manifest, index=False)
        print(f"[ok] masks: {len(df)}  manifest: {args.manifest}")
    else:
        print("[done] no masks written")

if __name__ == "__main__":
    main()
