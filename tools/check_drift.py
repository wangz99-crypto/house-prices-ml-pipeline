# tools/check_drift.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _pct(x: float) -> str:
    return f"{x*100:.2f}%"


def summarize_numeric(s_ref: pd.Series, s_cur: pd.Series) -> dict:
    s_ref = pd.to_numeric(s_ref, errors="coerce")
    s_cur = pd.to_numeric(s_cur, errors="coerce")

    ref = s_ref.dropna()
    cur = s_cur.dropna()

    if len(ref) == 0 or len(cur) == 0:
        return {"note": "insufficient numeric data"}

    def q(s, p): return float(np.nanquantile(s, p))

    out = {
        "ref_mean": float(ref.mean()),
        "cur_mean": float(cur.mean()),
        "ref_std": float(ref.std(ddof=1)) if len(ref) > 1 else 0.0,
        "cur_std": float(cur.std(ddof=1)) if len(cur) > 1 else 0.0,
        "ref_q05": q(ref, 0.05),
        "cur_q05": q(cur, 0.05),
        "ref_q50": q(ref, 0.50),
        "cur_q50": q(cur, 0.50),
        "ref_q95": q(ref, 0.95),
        "cur_q95": q(cur, 0.95),
        "ref_na_rate": float(s_ref.isna().mean()),
        "cur_na_rate": float(s_cur.isna().mean()),
    }

    # Simple drift score: normalized mean shift + NA shift
    denom = abs(out["ref_std"]) + 1e-9
    mean_shift = abs(out["cur_mean"] - out["ref_mean"]) / denom
    na_shift = abs(out["cur_na_rate"] - out["ref_na_rate"])
    out["drift_score"] = float(mean_shift + 2.0 * na_shift)
    return out


def summarize_categorical(s_ref: pd.Series, s_cur: pd.Series, top_k: int = 10) -> dict:
    ref = s_ref.astype("string")
    cur = s_cur.astype("string")

    ref_na = float(ref.isna().mean())
    cur_na = float(cur.isna().mean())

    ref_vc = ref.value_counts(dropna=True)
    cur_vc = cur.value_counts(dropna=True)

    ref_top = ref_vc.head(top_k)
    cur_top = cur_vc.head(top_k)

    ref_set = set(ref_vc.index.tolist())
    cur_set = set(cur_vc.index.tolist())

    # unknown rate: values in current not seen in ref
    if len(cur.dropna()) == 0:
        unknown_rate = 0.0
    else:
        unknown_mask = ~cur.isna() & ~cur.isin(list(ref_set))
        unknown_rate = float(unknown_mask.mean())

    # distribution shift on intersection (L1 distance on top categories)
    cats = set(ref_top.index.tolist()) | set(cur_top.index.tolist())
    ref_dist = {c: float(ref_vc.get(c, 0)) for c in cats}
    cur_dist = {c: float(cur_vc.get(c, 0)) for c in cats}

    ref_sum = sum(ref_dist.values()) + 1e-9
    cur_sum = sum(cur_dist.values()) + 1e-9
    l1 = sum(abs(ref_dist[c]/ref_sum - cur_dist[c]/cur_sum) for c in cats)

    return {
        "ref_na_rate": ref_na,
        "cur_na_rate": cur_na,
        "unknown_rate": unknown_rate,
        "top_k": top_k,
        "l1_top_shift": float(l1),
        "ref_unique": int(ref.nunique(dropna=True)),
        "cur_unique": int(cur.nunique(dropna=True)),
        "drift_score": float(l1 + 3.0 * unknown_rate + 2.0 * abs(cur_na - ref_na)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-fingerprint", required=True, help="path to data_fingerprint.json")
    ap.add_argument("--current-csv", required=True, help="path to current csv (e.g., data/raw/train.csv)")
    ap.add_argument("--out", default="reports/drift_report.json")
    args = ap.parse_args()

    ref_fp = json.loads(Path(args.ref_fingerprint).read_text(encoding="utf-8"))
    df_cur = pd.read_csv(args.current_csv)

    # fingerprint schema assumption (adjust if yours differs):
    # ref_fp["columns"] = {col: {"dtype": "...", "stats": {...}}}
    ref_cols = ref_fp.get("columns", {})
    report = {"ref": args.ref_fingerprint, "current": args.current_csv, "columns": {}}

    total_score = 0.0
    scored_cols = 0

    for col, meta in ref_cols.items():
        if col not in df_cur.columns:
            report["columns"][col] = {"status": "missing_in_current", "drift_score": 10.0}
            total_score += 10.0
            scored_cols += 1
            continue

        dtype = str(meta.get("dtype", "")).lower()
        s_ref_type = dtype

        s_cur = df_cur[col]

        # heuristic: numeric if dtype hints numeric OR current is numeric-like
        is_numeric = ("float" in dtype) or ("int" in dtype) or pd.api.types.is_numeric_dtype(s_cur)

        if is_numeric:
            summary = summarize_numeric(pd.Series(meta.get("sample", [])), s_cur)  # fallback if no ref sample
            # If your fingerprint doesn't store sample, this line won't work well.
            # You can remove it and instead store ref stats in fingerprint.
            # We'll still keep drift_score logic based on current + ref stats if provided below.

            # Prefer ref stats if present
            ref_stats = meta.get("stats", {})
            if ref_stats:
                # override ref_mean/ref_std/ref_na_rate if available
                summary["ref_mean"] = float(ref_stats.get("mean", summary.get("ref_mean", 0.0)))
                summary["ref_std"] = float(ref_stats.get("std", summary.get("ref_std", 0.0)))
                summary["ref_na_rate"] = float(ref_stats.get("na_rate", summary.get("ref_na_rate", 0.0)))

                denom = abs(summary["ref_std"]) + 1e-9
                mean_shift = abs(summary["cur_mean"] - summary["ref_mean"]) / denom
                na_shift = abs(summary["cur_na_rate"] - summary["ref_na_rate"])
                summary["drift_score"] = float(mean_shift + 2.0 * na_shift)

            report["columns"][col] = {"type": "numeric", **summary, "ref_dtype": s_ref_type}
        else:
            # categorical drift
            # if fingerprint has category levels in meta["levels"], use that as ref set
            ref_levels = meta.get("levels", None)
            if ref_levels is not None:
                # build a pseudo ref series from levels (uniform) just for logic
                s_ref = pd.Series(ref_levels, dtype="string")
            else:
                s_ref = pd.Series([], dtype="string")

            summary = summarize_categorical(s_ref, s_cur)
            report["columns"][col] = {"type": "categorical", **summary, "ref_dtype": s_ref_type}

        ds = float(report["columns"][col].get("drift_score", 0.0))
        total_score += ds
        scored_cols += 1

    report["overall"] = {
        "avg_drift_score": float(total_score / max(scored_cols, 1)),
        "scored_columns": scored_cols,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Console summary
    print("=== Drift Summary ===")
    print("Ref fingerprint:", args.ref_fingerprint)
    print("Current data   :", args.current_csv)
    print("Avg drift score:", report["overall"]["avg_drift_score"])
    # top drifting columns
    cols = [(c, v.get("drift_score", 0.0)) for c, v in report["columns"].items()]
    cols.sort(key=lambda x: x[1], reverse=True)
    print("Top drifting columns:")
    for c, s in cols[:10]:
        print(f" - {c}: {s:.4f}")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
