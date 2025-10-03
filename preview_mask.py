# preview_mask.py
import argparse, re, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def to_uint8(img: np.ndarray) -> np.ndarray:
    """Quick 16-bit→8-bit window using robust percentiles."""
    arr = img.astype(np.float32)
    p1, p99 = np.percentile(arr, (1, 99))
    if p99 <= p1:
        p1, p99 = arr.min(), arr.max() if arr.max() > arr.min() else (0.0, 1.0)
    arr = np.clip((arr - p1) / (p99 - p1), 0, 1) * 255.0
    return arr.astype(np.uint8)

def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    return np.array(im)

def load_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.dtype != np.bool_:
        m = (m > 0).astype(np.uint8)
    return m

def build_png_index(images_root: Path):
    """Map SOPInstanceUID (basename without extension) -> image PNG path."""
    idx = {}
    for p in images_root.rglob("*.png"):
        uid = p.stem
        idx[uid] = p
    return idx

def main():
    ap = argparse.ArgumentParser(description="Save quick previews of image/mask pairs.")
    ap.add_argument("--images-root", required=True, type=Path, help="Root with converted PNG images")
    ap.add_argument("--masks-manifest", "--manifest", dest="manifest", required=True, type=Path,
                    help="CSV with columns including: mask_path, sop_instance_uid, roi_name")
    ap.add_argument("--out-dir", type=Path, default=Path("./test_images"), help="Where to save preview PNGs")
    ap.add_argument("--samples", type=int, default=9, help="How many previews to save")
    ap.add_argument("--exclude-regex", type=str, default=r"(?i)smart|paint|deleted|ignore",
                    help="Regex of ROI names to exclude (case-insensitive)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--cols", type=int, default=3, help="Panels per row in the saved figure (3 = img/mask/overlay)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    df = pd.read_csv(args.manifest)
    needed_cols = {"mask_path", "sop_instance_uid", "roi_name"}
    if not needed_cols.issubset(df.columns):
        raise SystemExit(f"Manifest missing required columns: {needed_cols - set(df.columns)}")

    excl = re.compile(args.exclude_regex)
    df = df[~df["roi_name"].fillna("").apply(lambda s: bool(excl.search(str(s))))].copy()
    if df.empty:
        raise SystemExit("Nothing to preview after filtering — relax --exclude-regex or check manifest.")

    img_idx = build_png_index(args.images_root)

    df = df[df["sop_instance_uid"].astype(str).isin(img_idx.keys())]
    if df.empty:
        raise SystemExit("No matching image PNGs found for the manifest's SOPInstanceUIDs under images-root.")

    rows = df.sample(n=min(args.samples, len(df)), random_state=args.seed)

    for i, row in rows.reset_index(drop=True).iterrows():
        uid = str(row["sop_instance_uid"])
        img_path = img_idx[uid]
        mask_path = Path(row["mask_path"])
        if not mask_path.exists():
            print(f"[skip] missing mask: {mask_path}")
            continue

        img = load_gray(img_path)
        m = load_mask(mask_path)

        if img.shape[:2] != m.shape[:2]:
            print(f"[skip] shape mismatch for {uid}: img {img.shape} vs mask {m.shape}")
            continue

        img8 = to_uint8(img)
        overlay = np.stack([img8, img8, img8], axis=-1)
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        alpha = 0.35
        overlay = (overlay * (1 - alpha) + red * alpha * m[..., None]).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img8, cmap="gray")
        ax[0].set_title("Image")
        ax[1].imshow(m, cmap="gray")
        ax[1].set_title(f"Mask ({row['roi_name']})")
        ax[2].imshow(overlay)
        ax[2].set_title("Overlay")
        for a in ax:
            a.axis("off")

        out_name = f"{i+1:02d}__{uid}__roi-{row['roi_number']}_{str(row['roi_name']).replace(' ','_')}.png" \
                   if "roi_number" in rows.columns else f"{i+1:02d}__{uid}.png"
        out_path = args.out_dir / out_name
        fig.tight_layout(pad=0.1)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[saved] {out_path}")

    print(f"[done] previews in: {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()
