# rank_radiomics_vs_clinical.py
"""
Example:
python radiomics_vs_clinical.py \
  --radiomics /app/UC6_masks/radiomics_features.csv \
  --clinical  /data/EuCanImageUseCase68.csv \
  --out-dir   /app/UC6_masks/feature_ranking
"""
"""
Data flow
-------------------------------------------------
Inputs:
1) radiomics CSV: one or more rows per record_id (patient/case), typically multiple per ROI/image.
   Columns include record_id, ROI metadata, and many “original_*” radiomic features.
2) clinical CSV: exactly one row per record_id with clinical fields (e.g., ER, PR, HER2 IHC+FISH, Ki-67, grade, DCIS, histopath).

Processing steps:
1) Aggregate radiomics to the *record level* by taking the mean of each feature across all ROIs/images for a given record_id.
   Rationale: clinicians annotate a single biological lesion/state per patient, whereas radiomics may be computed many times
   (multiple views, multiple ROIs). Averaging prevents target leakage (multiple rows with same clinical label) and yields
   one feature vector per record_id. Alternatives like median/max are possible; mean is a simple, stable default.
2) Construct *clean clinical targets* that are biologically meaningful and potentially image-predictable:
   - grade (ordinal numeric), ki67% (numeric), ER/PR positivity (binary, ≥1% positive),
   - HER2 positivity (binary: IHC 3+ or IHC 2+ with FISH positive),
   - DCIS (binary), histopathology subtype (multiclass; light normalization of text labels).
   We deliberately ignore demographics and germline genetics here because they’re not “image-only” targets.
   When duplicate clinical fields exist (e.g., *_2 columns), we use the first non-missing value.
3) For each target, quantify *association* between every radiomic feature and the target with a method that matches the
   measurement scale:
   - Ordinal/numeric (grade, Ki-67): Spearman’s ρ (rank correlation). This is robust to monotonic nonlinearity
     and ignores absolute scaling.
   - Binary (ER, PR, HER2, DCIS): AUROC (threshold-free separability) + point-biserial correlation (linear effect size).
     AUROC is scale-invariant and easy to interpret (0.5=chance). We rank primarily by |AUROC – 0.5|.
   - Multiclass (histopath): mutual information (nonparametric dependence that captures nonlinear relations).
4) Rank radiomic features per target by an appropriate *score*:
   - |ρ| for Spearman, |AUROC–0.5| for binary, mutual information for multiclass.
   We write one CSV per target with the top-k features and a combined summary CSV for quick triage.

Interpretation (how to read the outputs)
----------------------------------------
- A high |ρ| (grade/Ki-67) means a feature increases or decreases consistently with the clinical measure.
- A high AUROC (>0.7 or <0.3) suggests a feature separates positive vs. negative cases well (the latter implies flipping sign).
- Higher mutual information indicates stronger dependence with histology classes.
These are *univariate* associations: each feature is tested in isolation. They highlight promising candidates but are not a model.

Design choices (why these statistics)
-------------------------------------
- Rank correlations (Spearman) avoid assumptions of linearity and are insensitive to monotone transforms.
- AUROC is threshold-agnostic and robust to class-imbalance compared to accuracy.
- Mutual information captures nonlinear, nonmonotonic patterns for multiclass labels.
- We aggregate to one row per record_id to avoid pseudo-replication and optimistic bias.

Limitations and caveats (what this analysis does NOT do)
--------------------------------------------------------
- No multiple-testing correction is applied; with hundreds of features, some strong-looking associations may be false positives.
  Use FDR control (e.g., Benjamini–Hochberg) in downstream work.
- No adjustment for confounders (scanner vendor/protocol, pixel spacing, lesion size, demographics). If confounding exists,
  associations may reflect site/protocol differences rather than biology.
- Radiomics were computed on 2D PNGs, not raw DICOMs; intensity transformations could affect first-order features.
  Shape features are typically more robust, but still depend on segmentation quality.
- Binary targets can be imbalanced. AUROC handles this better than accuracy, but small N may yield unstable estimates.
- Mutual information is not directional and harder to interpret clinically; it just signals dependence.

Recommended next steps (how to convert findings into a model)
-------------------------------------------------------------
1) Reduce to a short list of stable features (e.g., by bootstrap ranking stability).
2) Train simple baseline models per target (logistic regression for binary, ordinal regression for grade, etc.)
   with nested cross-validation to avoid overfitting.
3) Adjust for confounders (e.g., add scanner/site indicators, pixel spacing, or lesion size; or stratify/correct).
4) Control the false discovery rate across features/targets and validate externally on a held-out cohort.

Reproducibility notes
---------------------
- Outputs include per-target ranking CSVs and a summary. The script prints sample sizes and skips targets with insufficient data.
- All cleaning rules (e.g., HER2 positivity, ER/PR cutoff at 1%) are explicit here for transparency and can be modified as needed.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

NA_STR = {"", "na", "n/a", "nan", "none", "null", "missing"}

def to_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in NA_STR: return np.nan
    s = s.replace("%", "").replace("+", "")
    try: return float(s)
    except: return np.nan

def to_bin(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"1","yes","y","true","pos","positive"}: return 1.0
    if s in {"0","no","n","false","neg","negative"}: return 0.0
    v = to_num(s)
    if np.isnan(v): return np.nan
    return 1.0 if v >= 1 else 0.0

def her2_pos(ihc, fish):
    """HER2 positive if IHC 3+, or IHC 2+ with positive FISH."""
    ihc_n = to_num(ihc)
    if not np.isnan(ihc_n):
        if ihc_n >= 3: return 1.0
        if ihc_n <= 1: return 0.0
        if ihc_n == 2:
            f = to_bin(fish)
            return f
    ihc_s = str(ihc).strip().lower() if pd.notna(ihc) else ""
    if "3" in ihc_s: return 1.0
    if "2" in ihc_s:
        f = to_bin(fish)
        return f
    if "0" in ihc_s or "1" in ihc_s or "neg" in ihc_s: return 0.0
    return np.nan

def pick_first(*cols):
    out = None
    for c in cols:
        if c is None: continue
        out = c if out is None else out.fillna(c)
    return out

def clean_hist(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[^a-z0-9 ]+", " ", str(x).lower()).strip()
    if not s: return np.nan
    for k,v in [("duct", "ductal"), ("lobul","lobular"), ("medul","medullary"),
                ("mucin","mucinous"), ("metapl","metaplastic")]:
        if k in s: return v
    return s

def safe_auc(y_true, score):
    try:
        if len(np.unique(y_true)) < 2: return np.nan
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser(description="Rank radiomic features vs. meaningful clinical targets.")
    ap.add_argument("--radiomics", required=True, type=Path)
    ap.add_argument("--clinical",  required=True, type=Path)
    ap.add_argument("--out-dir",   required=True, type=Path)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    r = pd.read_csv(args.radiomics)
    c = pd.read_csv(args.clinical)

    key_cols = ["record_id"]
    meta_cols = ["record_id","sop_instance_uid","roi_name","mask_path","image_path","view",
                 "spacing_row_mm","spacing_col_mm"]
    feat_cols = [col for col in r.columns if col not in meta_cols]
    num_feats = r[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    r_agg = r.groupby("record_id")[num_feats].mean().reset_index()

    c["grade_num"]  = pick_first(c.get("grade"), c.get("grade_2")).apply(to_num)
    c["er_pct"]     = pick_first(c.get("er"), c.get("er_2")).apply(to_num)
    c["pr_pct"]     = pick_first(c.get("pr"), c.get("pr_2")).apply(to_num)
    c["ki67_pct"]   = pick_first(c.get("ki67"), c.get("ki67_2")).apply(to_num)
    c["dcis_bin"]   = pick_first(c.get("dcis"), c.get("dcis_2")).apply(to_bin)
    c["er_bin"]     = (c["er_pct"] >= 1).astype(float)
    c["pr_bin"]     = (c["pr_pct"] >= 1).astype(float)
    c["her2_pos"]   = [her2_pos(a,b) for a,b in zip(
                        pick_first(c.get("her2ihc"), c.get("her2ihc_2")),
                        pick_first(c.get("her2fish"), c.get("her2fish_2")))]
    c["histopath"]  = pick_first(c.get("histopat"), c.get("histopat_2"), c.get("histopat_3"),
                                 c.get("histopat_1_2"), c.get("histopat_2_2"), c.get("histopat_3_2")).apply(clean_hist)

    targets = {
        "grade_num":  {"type":"ordinal"},
        "ki67_pct":   {"type":"numeric"},
        "er_bin":     {"type":"binary"},
        "pr_bin":     {"type":"binary"},
        "her2_pos":   {"type":"binary"},
        "dcis_bin":   {"type":"binary"},
        "histopath":  {"type":"multiclass"},
    }

    df = pd.merge(r_agg, c[["record_id"] + list(targets.keys())], on="record_id", how="inner")

    all_rows = []
    for t, meta in targets.items():
        sub = df.dropna(subset=[t]).copy()
        if sub.empty: 
            tqdm.write(f"[skip] {t}: no data")
            continue

        kind = meta["type"]
        tqdm.write(f"[target] {t} ({kind})  n={len(sub)}")

        if kind in {"ordinal","numeric"}:
            vals = []
            for f in num_feats:
                x = sub[f].values
                y = sub[t].values
                if np.isnan(x).all(): 
                    vals.append((f, np.nan, np.nan))
                    continue
                rho, p = spearmanr(x, y, nan_policy="omit")
                vals.append((f, rho, p))
            res = (pd.DataFrame(vals, columns=["feature","spearman_rho","p_value"])
                     .assign(score=lambda d: d["spearman_rho"].abs())
                     .sort_values("score", ascending=False)
                     .head(args.topk))
        elif kind == "binary":
            vals = []
            y = sub[t].astype(float).values
            if len(np.unique(y)) < 2:
                tqdm.write(f"[skip] {t}: only one class present")
                continue
            for f in num_feats:
                x = sub[f].values
                auc = safe_auc(y, x)
                try:
                    r_pb, p = pointbiserialr(y, x)
                except Exception:
                    r_pb, p = (np.nan, np.nan)
                score = np.nan if np.isnan(auc) else abs(auc - 0.5)
                vals.append((f, auc, r_pb, p, score))
            res = (pd.DataFrame(vals, columns=["feature","auc","pointbiserial_r","p_value","score"])
                     .sort_values("score", ascending=False)
                     .head(args.topk))
        elif kind == "multiclass":
            y_raw = sub[t].astype("category")
            if y_raw.nunique() < 2:
                tqdm.write(f"[skip] {t}: <2 classes")
                continue
            y = y_raw.cat.codes.values
            X = sub[num_feats].fillna(sub[num_feats].median(numeric_only=True)).values
            mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
            res = (pd.DataFrame({"feature": num_feats, "mutual_info": mi})
                     .assign(score=lambda d: d["mutual_info"])
                     .sort_values("score", ascending=False)
                     .head(args.topk))
        else:
            continue

        res.insert(0, "target", t)
        all_rows.append(res)

        res.to_csv(args.out_dir / f"rank_{t}.csv", index=False)

    if not all_rows:
        raise SystemExit("No targets had sufficient data.")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.out_dir / "rank_summary_topk.csv", index=False)

    manifest = {
        "n_records_joined": int(len(df)),
        "targets_evaluated": [t for t in targets if (args.out_dir / f"rank_{t}.csv").exists()],
        "topk": args.topk,
        "radiomics_features": len(num_feats),
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] wrote rankings to {args.out_dir}")

if __name__ == "__main__":
    main()
