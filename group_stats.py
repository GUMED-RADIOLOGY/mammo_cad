#group_stats.py
"""
python group_stats.py \
  --meta_csv /data/EuCanImageUseCase68.csv \
  --features_csv /app/UC6_masks/radiomics_features.csv \
  --outdir /app/mammo_cad/group_stats \
  --save_filtered_meta \
  --ki67_cutoff 20 --er_cutoff 1 --pr_cutoff 1
"""
import argparse, os, numpy as np, pandas as pd
from scipy import stats
from math import sqrt

# Value maps (lean)
M_SEX = {"0":"male","1":"female"}
M_PATIENTCLASS = {"0":"Cancer","1":"Benign","2":"Normal"}
M_MENOP = {"0":"premenopause","1":"postmenopause"}
M_YNUNK = {"0":"yes","1":"no","2":"Unknown"}
M_FAM = {"0":"none","1":"first degree","2":"second degree","3":"Unknown"}
M_LATERALITY = {"0":"Left","1":"Right"}
M_GRADE = {"0":"G1","1":"G2","2":"G3","3":"Gx"}
M_DCIS = {"0":"Grade 1","1":"Grade 2","2":"Grade 3"}

CATEGORICAL_FEATURES = {
    "sex": M_SEX,
    "patientclass": M_PATIENTCLASS,
    "menop": M_MENOP,
    "lactation": M_YNUNK,
    "screening": M_YNUNK,
    "famhisto_b": M_FAM,
    "famhisto_o": M_FAM,
    "laterality": M_LATERALITY,
    "grade": M_GRADE,
    "dcis": M_DCIS,
}
NUMERIC_FEATURES = ["onsetage","n_preg"]

def read_csv_str(path, usecols=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, usecols=usecols)

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")

def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x, y = to_num(x).dropna(), to_num(y).dropna()
    if len(x) < 2 or len(y) < 2: return np.nan
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)) if (nx+ny-2)>0 else np.nan
    if sp == 0 or np.isnan(sp): return np.nan
    return (x.mean() - y.mean()) / sp

def cramers_v(ct: pd.DataFrame) -> float:
    if ct.size == 0: return np.nan
    chi2, _, _, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    if n == 0: return np.nan
    phi2 = chi2 / n
    r, k = ct.shape
    denom = min(r-1, k-1)
    return np.sqrt(phi2 / denom) if denom > 0 else np.nan

def ttest_welch(x: pd.Series, y: pd.Series) -> float:
    x, y = to_num(x).dropna(), to_num(y).dropna()
    if len(x) < 2 or len(y) < 2: return np.nan
    _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return p

def chi2_or_fisher(ct: pd.DataFrame) -> float:
    if ct.shape == (2, 2):
        _, p = stats.fisher_exact(ct.values)
        return p
    chi2, p, _, exp = stats.chi2_contingency(ct, correction=True)
    if (exp < 5).any() and ct.shape == (2, 2):
        _, p = stats.fisher_exact(ct.values)
    return p

def bh_fdr(pvals):
    idx = [i for i, p in enumerate(pvals) if not pd.isna(p)]
    m = len(idx)
    if m == 0: return pvals
    ps = np.array([pvals[i] for i in idx], dtype=float)
    order = np.argsort(ps)
    ranked = ps[order]
    adj = np.empty_like(ranked)
    for j, p in enumerate(ranked, start=1):
        adj[j-1] = p * m / j
    for i in range(len(adj)-2, -1, -1):
        adj[i] = min(adj[i], adj[i+1])
    out = [np.nan]*len(pvals)
    for pos, a in zip(order, adj):
        out[idx[pos]] = min(a, 1.0)
    return out

def label_ki67(df, cutoff):
    v = to_num(df.get("ki67", pd.Series(index=df.index)))
    return pd.Series(np.where(v >= cutoff, "High", "Low"), index=df.index)

def label_er(df, cutoff):
    v = to_num(df.get("er", pd.Series(index=df.index)))
    return pd.Series(np.where(v >= cutoff, "Positive", "Negative"), index=df.index)

def label_pr(df, cutoff):
    v = to_num(df.get("pr", pd.Series(index=df.index)))
    return pd.Series(np.where(v >= cutoff, "Positive", "Negative"), index=df.index)

def label_her2(df):
    ihc = pd.to_numeric(df.get("her2ihc", pd.Series(index=df.index)).replace("", np.nan), errors="coerce")
    fish = df.get("her2fish", pd.Series(index=df.index, dtype=str)).astype(str).replace("", np.nan)
    pos = (ihc == 3) | ((ihc == 2) & (fish == "0"))
    neg = (ihc.isin([0, 1])) | ((ihc == 2) & (fish == "1"))
    out = pd.Series(index=df.index, dtype=object)
    out[pos] = "Positive"; out[neg] = "Negative"
    return out.fillna("Unknown")

def mean_sd_text(s: pd.Series) -> str:
    x = to_num(s).dropna()
    return "NA" if x.empty else f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"

def build_numeric_block(df, group_col, g1, g2, col):
    x, y = df.loc[group_col==g1, col], df.loc[group_col==g2, col]
    p = ttest_welch(x, y)
    d = cohens_d(x, y)
    row = {"Variable": col,"Level":"",f"{g1}": mean_sd_text(x),f"{g2}": mean_sd_text(y),"Test":"Welch t-test","P":p,"Effect":d}
    return [row], p

def build_categorical_block(df, group_col, g1, g2, col, mapping, include_unknown=False):
    s = df[col].map(lambda v: mapping.get(str(v), str(v) if v != "" else "Unknown"))
    part = pd.DataFrame({col: s, "__group": group_col.astype(str)})
    disp = part.copy()
    if not include_unknown:
        disp = disp[disp[col] != "Unknown"]
    if disp.empty:
        return [], np.nan
    ct = pd.crosstab(disp[col], disp["__group"])
    if ct.shape[1] < 2:
        return [], np.nan
    col_tot = ct.sum(axis=0)
    pct = (ct.div(col_tot, axis=1) * 100).round(1)

    # test (exclude Unknown anyway)
    test_part = part[part[col] != "Unknown"]
    p = np.nan; v = np.nan
    if not test_part.empty and test_part[col].nunique() > 1:
        ct_test = pd.crosstab(test_part[col], test_part["__group"])
        if ct_test.shape[1] == 2:
            p = chi2_or_fisher(ct_test)
            v = cramers_v(ct_test)

    rows = []
    first = True
    for level in ct.index:
        c1 = int(ct.loc[level, g1]) if g1 in ct.columns else 0
        c2 = int(ct.loc[level, g2]) if g2 in ct.columns else 0
        p1 = pct.loc[level, g1] if (level in pct.index and g1 in pct.columns) else 0.0
        p2 = pct.loc[level, g2] if (level in pct.index and g2 in pct.columns) else 0.0
        rows.append({
            "Variable": col if first else "",
            "Level": level,
            f"{g1}": f"{c1} ({p1:.1f}%)",
            f"{g2}": f"{c2} ({p2:.1f}%)",
            "Test": "Chi-square/Fisher" if first else "",
            "P": p if first else np.nan,
            "Effect": v if first else np.nan,
        })
        first = False
    return rows, p

def make_table(df: pd.DataFrame, groups: pd.Series, title: str, cat_features: dict, num_features: list, include_unknown=False) -> pd.DataFrame:
    df = df.copy(); df["_group"] = groups
    vals = [str(g) for g in pd.unique(df["_group"]) if pd.notna(g) and str(g)!="Unknown"]
    vals = sorted(set(vals))[:2]
    if len(vals) < 2:
        return pd.DataFrame(columns=["Table","Variable","Level","A","B","Test","P","P_adj","Effect"])
    g1, g2 = vals[0], vals[1]
    n1, n2 = int((df["_group"].astype(str)==g1).sum()), int((df["_group"].astype(str)==g2).sum())
    col_g1 = f"{g1} (n={n1})"
    col_g2 = f"{g2} (n={n2})"

    all_rows, p_rows_idx, p_collect = [], [], []

    for col in num_features:
        if col not in df.columns: continue
        rows, p = build_numeric_block(df, df["_group"].astype(str), g1, g2, col)
        # rename keys to headers with n
        for r in rows: r[col_g1] = r.pop(g1); r[col_g2] = r.pop(g2)
        all_rows.extend(rows); p_collect.append(p); p_rows_idx.append(len(all_rows)-1)

    for col, mapping in cat_features.items():
        if col not in df.columns: continue
        rows, p = build_categorical_block(df, df["_group"].astype(str), g1, g2, col, mapping, include_unknown)
        for r in rows:
            r[col_g1] = r.pop(g1); r[col_g2] = r.pop(g2)
        if rows:
            all_rows.extend(rows); p_collect.append(p); p_rows_idx.append(len(all_rows)-len(rows))

    p_adj = bh_fdr(p_collect)
    for idx, padj in zip(p_rows_idx, p_adj):
        all_rows[idx]["P_adj"] = padj
    for i, r in enumerate(all_rows):
        if "P_adj" not in r: r["P_adj"] = np.nan

    out = pd.DataFrame(all_rows, columns=["Variable","Level",col_g1,col_g2,"Test","P","P_adj","Effect"])
    out.insert(0, "Table", title)

    def fmt_p(x): 
        if pd.isna(x): return "NA"
        return "<0.001" if x < 0.001 else f"{x:.3f}"
    def fmt_eff(x): 
        return "NA" if pd.isna(x) else f"{x:.3f}"

    out["P"] = out["P"].map(fmt_p)
    out["P_adj"] = out["P_adj"].map(fmt_p)
    out["Effect"] = out["Effect"].map(fmt_eff)
    return out

def main():
    ap = argparse.ArgumentParser(description="Filter metadata by features IDs and build readable clinical tables (Ki-67, HER2, ER, PR).")
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--save_filtered_meta", action="store_true")
    ap.add_argument("--filtered_meta_name", default="filtered_metadata.csv")
    ap.add_argument("--ki67_cutoff", type=float, default=20.0)
    ap.add_argument("--er_cutoff", type=float, default=1.0)
    ap.add_argument("--pr_cutoff", type=float, default=1.0)
    ap.add_argument("--include_unknown", action="store_true", help="Include 'Unknown' levels in output (default: hidden).")
    ap.add_argument("--extra_numeric", nargs="*", default=[])
    ap.add_argument("--extra_categorical", nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    feat_ids = read_csv_str(args.features_csv, usecols=["record_id"])
    if "record_id" not in feat_ids.columns: raise ValueError("features_csv missing 'record_id'")
    id_set = set(feat_ids["record_id"].str.strip().tolist())

    meta = read_csv_str(args.meta_csv)
    if "record_id" not in meta.columns: raise ValueError("meta_csv missing 'record_id'")
    meta["record_id"] = meta["record_id"].str.strip()
    df = meta[meta["record_id"].isin(id_set)].copy()

    if args.save_filtered_meta:
        df.to_csv(os.path.join(args.outdir, args.filtered_meta_name), index=False)

    for need in ["ki67","er","pr","her2ihc","her2fish"]:
        if need not in df.columns: df[need] = ""

    g_ki = label_ki67(df, args.ki67_cutoff)
    g_er = label_er(df, args.er_cutoff)
    g_pr = label_pr(df, args.pr_cutoff)
    g_her2 = label_her2(df)

    num_feats = list(NUMERIC_FEATURES) + [c for c in args.extra_numeric if c not in NUMERIC_FEATURES]
    cat_feats = dict(CATEGORICAL_FEATURES)
    for c in args.extra_categorical:
        if c not in cat_feats: cat_feats[c] = {}

    tabs = {
        "summary_ki67.csv": make_table(df, g_ki, "Ki-67 (High vs Low)", cat_feats, num_feats, include_unknown=args.include_unknown),
        "summary_her2.csv": make_table(df, g_her2, "HER2 (Positive vs Negative)", cat_feats, num_feats, include_unknown=args.include_unknown),
        "summary_er.csv": make_table(df, g_er, "ER (Positive vs Negative)", cat_feats, num_feats, include_unknown=args.include_unknown),
        "summary_pr.csv": make_table(df, g_pr, "PR (Positive vs Negative)", cat_feats, num_feats, include_unknown=args.include_unknown),
    }
    for name, tab in tabs.items():
        outp = os.path.join(args.outdir, name)
        tab.to_csv(outp, index=False)
        print(f"Saved: {outp} rows={len(tab)}")

    print(f"Input meta rows: {len(meta)} | Features IDs: {len(id_set)} | Kept: {len(df)}", flush=True)

if __name__ == "__main__":
    main()
