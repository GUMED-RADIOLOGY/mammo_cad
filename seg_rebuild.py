"""
RTSTRUCT -> per-image PNG masks + CSV manifest.

- Recursively finds RT Structure Sets.
- For each ROI, rasterizes its polygons onto referenced image.
- Uses DICOM geometry when available. Otherwise mm-from-top-left fallback.
- Composition: default is UNION (no XOR holes). Use --composite xor|union|auto.
- Skips vendor "Bounding box" measurement ROIs by default (toggle with --keep-bboxes).
- Saves binary masks as PNG (0 background, 255 ROI) and a JSON with metadata.
- Writes a CSV manifest.

Example:

"""

import os, glob, argparse, json, traceback
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
import pydicom
import pandas as pd
from tqdm.auto import tqdm

# --------- helpers ---------

def record_id_from_path(dcm_path: Path, in_root: Path) -> str:
    try:
        return dcm_path.relative_to(in_root).parts[0]
    except Exception:
        return Path(*dcm_path.parts[dcm_path.parts.index(in_root.name)+1:]).parts[0]

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or ""))[:64]

def get_spacing(ds):
    ps = getattr(ds, "PixelSpacing", None)
    if ps and len(ps) >= 2:
        return float(ps[0]), float(ps[1])
    ips = getattr(ds, "ImagerPixelSpacing", None)
    if ips and len(ips) >= 2:
        return float(ips[0]), float(ips[1])
    return 1.0, 1.0

def load_image_header(path: Path):
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)
    rmm, cmm = get_spacing(ds)
    iop = getattr(ds, "ImageOrientationPatient", None)  # [r_x,r_y,r_z,c_x,c_y,c_z]
    ipp = getattr(ds, "ImagePositionPatient", None)     # [x,y,z]
    return ds, rows, cols, (rmm, cmm), iop, ipp

def mm_xyz_to_rc(xyz, rows, cols, spacing, iop, ipp):
    """Map patient mm coords (x,y,z) → pixel (row, col)."""
    x, y, z = xyz
    rmm, cmm = spacing
    if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
        r = np.array(iop[0:3], float)
        c = np.array(iop[3:6], float)
        p = np.array([x, y, z], float)
        o = np.array(ipp[0:3], float)
        d = p - o
        rr = (d @ r) / max(rmm, 1e-9)
        cc = (d @ c) / max(cmm, 1e-9)
    else:
        rr = y / max(rmm, 1e-9)
        cc = x / max(cmm, 1e-9)
    rr = float(np.clip(rr, -0.5, rows - 0.5))
    cc = float(np.clip(cc, -0.5, cols - 0.5))
    return rr, cc

def to_jsonable(x):
    import numpy as _np
    from pydicom.valuerep import PersonName
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (PersonName,)):
        return str(x)
    if isinstance(x, (_np.generic,)):
        return x.item()
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    try:
        from collections.abc import Sequence
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray, dict)):
            return [to_jsonable(v) for v in x]
    except Exception:
        pass
    try:
        return json.loads(json.dumps(x))
    except Exception:
        return str(x)

def binary_fill_holes(mask_bool):
    """Fill holes in a boolean mask. Tries SciPy; falls back to pure-NumPy flood-fill."""
    try:
        from scipy.ndimage import binary_fill_holes as _fill
        return _fill(mask_bool)
    except Exception:
        h, w = mask_bool.shape
        visited = np.zeros_like(mask_bool, dtype=bool)
        q = deque()
        for r in range(h):
            for c in (0, w-1):
                if not mask_bool[r, c] and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))
        for c in range(w):
            for r in (0, h-1):
                if not mask_bool[r, c] and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if 0 <= nr < h and 0 <= nc < w and not mask_bool[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return mask_bool | (~visited & ~mask_bool)


def build_image_index(in_root: Path):
    print("[idx] scanning images…")
    pat = str(in_root / "*" / "**" / "resources" / "DICOM" / "files" / "*.dcm")
    paths = list(glob.iglob(pat, recursive=True))
    idx = {}
    for p in tqdm(paths, desc="Index images", unit="hdr"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            sop = getattr(ds, "SOPInstanceUID", None)
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            if sop and rows and cols:
                idx[str(sop)] = {"path": Path(p)}
        except Exception:
            pass
    print(f"[idx] {len(idx)} image headers found")
    return idx

# --------- conversion ---------

def should_skip_roi(name: str, desc: str, keep_bboxes: bool) -> bool:
    n = (name or "").lower()
    d = (desc or "").lower()
    if keep_bboxes:
        return False
    # Skip shit
    if "bounding box" in n or "bounding-box" in d or "2d bounding box" in d:
        return True
    return False

def process_rtstruct(seg_path: Path, in_root: Path, out_root: Path, img_idx, manifest_rows,
                     composite_mode: str = "union", fill_holes: bool = False, keep_bboxes: bool = False):
    ds = pydicom.dcmread(seg_path, force=True)

    if str(getattr(ds, "SOPClassUID", "")) not in (
        pydicom.uid.RTStructureSetStorage, "1.2.840.10008.5.1.4.1.1.481.3"
    ):
        print(f"[skip] not RTSTRUCT: {seg_path}")
        return

    rid = record_id_from_path(seg_path, in_root)
    print(f"[seg] {seg_path} (RID={rid})")

    # ROI metadata
    roi_meta = {}
    for item in getattr(ds, "StructureSetROISequence", []) or []:
        num = str(getattr(item, "ROINumber", ""))
        roi_meta[num] = {
            "name": getattr(item, "ROIName", f"ROI_{num}"),
            "desc": getattr(item, "ROIDescription", ""),
            "color": None,
        }

    # Iterate
    for rc in getattr(ds, "ROIContourSequence", []) or []:
        ref_num = str(getattr(rc, "ReferencedROINumber", ""))
        color = getattr(rc, "ROIDisplayColor", None)
        if ref_num not in roi_meta:
            roi_meta[ref_num] = {"name": f"ROI_{ref_num}", "desc": "", "color": color}
        else:
            roi_meta[ref_num]["color"] = color

        roi_name = roi_meta[ref_num]["name"]
        roi_desc = roi_meta[ref_num]["desc"]
        if should_skip_roi(roi_name, roi_desc, keep_bboxes):
            print(f"[skip] ROI {ref_num} '{roi_name}' (measurement/bounding-box)")
            continue

        contours = getattr(rc, "ContourSequence", []) or []
        if not contours:
            continue

        roi_modes = set()
        for c in contours:
            gtype = str(getattr(c, "ContourGeometricType", "") or "").upper()
            roi_modes.add(gtype)
        roi_composite = composite_mode
        if composite_mode == "auto":
            roi_composite = "xor" if any("XOR" in m for m in roi_modes) else "union"

        per_sop = {}
        for c in contours:
            cis = getattr(c, "ContourImageSequence", []) or []
            if not cis:
                continue
            rsop = str(getattr(cis[0], "ReferencedSOPInstanceUID", "") or "")
            data = list(map(float, getattr(c, "ContourData", []) or []))
            pts = [tuple(data[i:i+3]) for i in range(0, len(data), 3)]
            if len(pts) >= 3:
                per_sop.setdefault(rsop, []).append(pts)

        for rsop, polys in per_sop.items():
            if rsop not in img_idx:
                print(f"[warn] referenced image SOP not indexed: {rsop}")
                continue
            meta = img_idx[rsop]
            if "rows" not in meta:
                _, rows, cols, spacing, iop, ipp = load_image_header(meta["path"])
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
                rc_int = [(int(round(r)), int(round(c))) for r, c in rc_pix]
                rc_int = [(min(max(0, r), rows-1), min(max(0, c), cols-1)) for r, c in rc_int]

                poly_xy = [(c, r) for r, c in rc_int]
                tmp = Image.new("1", (cols, rows), 0)
                ImageDraw.Draw(tmp).polygon(poly_xy, outline=1, fill=1)
                tmp_arr = np.array(tmp, dtype=bool)

                if roi_composite == "xor":
                    mask ^= tmp_arr
                    mask |= tmp_arr

            if fill_holes:
                mask = binary_fill_holes(mask)

            # Save mask PNG
            out_dir = out_root / rid / "masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sanitize(roi_name)
            out_png = out_dir / f"{rsop}__roi-{ref_num}_{safe_name}.png"
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_png)

            side = {
                "seg_source": str(seg_path),
                "record_id": rid,
                "sop_instance_uid": rsop,
                "rows": rows,
                "cols": cols,
                "roi_number": ref_num,
                "roi_name": roi_name,
                "roi_desc": roi_desc,
                "roi_color_rgb": to_jsonable(roi_meta.get(ref_num, {}).get("color", None)),
                "spacing_mm": {"row": float(spacing[0]), "col": float(spacing[1])},
                "geometry_mode": "iop+ipp" if (meta["iop"] is not None and meta["ipp"] is not None) else "top_left_mm_fallback",
                "composite": roi_composite,
            }
            Path(str(out_png) + ".json").write_text(json.dumps(side, indent=2))

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
    ap.add_argument("--composite", choices=["union", "xor", "auto"], default="union",
                    help="How to combine polygons within an ROI per image (default: union)")
    ap.add_argument("--fill-holes", action="store_true", help="Fill interior holes in final mask")
    ap.add_argument("--keep-bboxes", action="store_true", help="Keep 'Bounding box' measurement ROIs")
    ap.add_argument("--trace", action="store_true", help="Print full tracebacks for errors")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seg_paths = [Path(p) for p in glob.iglob(str(in_root / "**" / "segmentation.dcm"), recursive=True)]
    pat_all = str(in_root / "*" / "**" / "resources" / "DICOM" / "files" / "*.dcm")
    for p in glob.iglob(pat_all, recursive=True):
        if p.endswith("segmentation.dcm"):
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if str(getattr(ds, "SOPClassUID", "")) in (pydicom.uid.RTStructureSetStorage, "1.2.840.10008.5.1.4.1.1.481.3"):
                seg_paths.append(Path(p))
        except Exception:
            pass

    seg_paths = sorted(set(seg_paths))
    if not seg_paths:
        print(f"[done] no RTSTRUCT files found under {in_root}")
        return

    img_idx = build_image_index(in_root)

    manifest_rows = []
    for sp in tqdm(seg_paths, desc="RTSTRUCTs", unit="file"):
        try:
            process_rtstruct(
                sp, in_root, out_root, img_idx, manifest_rows,
                composite_mode=args.composite, fill_holes=args.fill_holes, keep_bboxes=args.keep_bboxes
            )
        except Exception as e:
            if args.trace:
                print(f"[err] {sp}:\n{traceback.format_exc()}")
            else:
                print(f"[err] {sp}: {e}")

    if manifest_rows:
        df = pd.DataFrame(manifest_rows)
        df.to_csv(args.manifest, index=False)
        print(f"[ok] masks: {len(df)}  manifest: {args.manifest}")
    else:
        print("[done] no masks written")

if __name__ == "__main__":
    main()
