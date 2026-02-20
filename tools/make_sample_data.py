from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_stratified_sample(df: pd.DataFrame, n: int, seed: int, q: int = 10) -> tuple[pd.DataFrame, dict]:
    """
    Stratified sampling by SalePrice quantile bins.
    Returns (sample_df, info_dict).
    """
    info = {"method": "random", "q": q, "bins_used": 0}

    if "SalePrice" not in df.columns or len(df) < 50:
        sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
        return sample, info

    # Build bins (may drop duplicates)
    s = pd.to_numeric(df["SalePrice"], errors="coerce")
    tmp = df.copy()
    tmp["_price_bin"] = pd.qcut(s, q=q, duplicates="drop")

    bins = tmp["_price_bin"].dropna().unique()
    if len(bins) < 3:
        # Too few bins -> fallback to random
        sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
        return sample, info

    # Count per bin
    groups = [g for _, g in tmp.groupby("_price_bin", observed=True)]
    sizes = np.array([len(g) for g in groups], dtype=float)
    props = sizes / sizes.sum()

    # Allocate counts: floor then distribute remainder by largest fractional parts
    raw = props * min(n, len(df))
    base = np.floor(raw).astype(int)
    remainder = int(min(n, len(df)) - base.sum())
    frac = raw - base

    if remainder > 0:
        add_idx = np.argsort(-frac)[:remainder]
        base[add_idx] += 1

    parts = []
    for k, g in zip(base, groups):
        if k <= 0:
            continue
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))

    sample = (
        pd.concat(parts, ignore_index=True)
          .drop(columns=["_price_bin"], errors="ignore")
          .sample(frac=1, random_state=seed)
          .head(min(n, len(df)))
          .reset_index(drop=True)
    )

    info = {"method": "stratified", "q": q, "bins_used": int(len(bins))}
    return sample, info


def main(n_train: int = 800, n_test: int = 400, seed: int = 42, q: int = 10):
    src_train = ROOT / "data" / "raw" / "train.csv"
    src_test = ROOT / "data" / "raw" / "test.csv"

    out_dir = ROOT / "tests" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_train = out_dir / "sample_train.csv"
    dst_test = out_dir / "sample_test.csv"
    meta = out_dir / "sample_data_meta.json"

    if not src_train.exists():
        raise FileNotFoundError(
            f"Missing Kaggle raw train.csv at: {src_train}\n"
            "Put Kaggle train.csv under data/raw/ (gitignored), then rerun this script."
        )

    df_train = pd.read_csv(src_train)

    # ---- train sample (keep SalePrice) ----
    train_sample, train_info = _safe_stratified_sample(df_train, n=n_train, seed=seed, q=q)
    train_sample.to_csv(dst_train, index=False)

    # ---- test sample (never include SalePrice) ----
    if src_test.exists():
        df_test = pd.read_csv(src_test)
        test_sample = df_test.sample(n=min(n_test, len(df_test)), random_state=seed).reset_index(drop=True)
        test_info = {"source": "raw_test.csv", "method": "random"}
    else:
        # fallback: resample from train sample and drop target
        tmp = train_sample.drop(columns=["SalePrice"], errors="ignore")
        test_sample = tmp.sample(n=min(n_test, len(tmp)), random_state=seed).reset_index(drop=True)
        test_info = {"source": "derived_from_sample_train", "method": "random"}

    test_sample = test_sample.drop(columns=["SalePrice"], errors="ignore")
    test_sample.to_csv(dst_test, index=False)

    meta_payload = {
        "generated_at": _utc_now(),
        "seed": int(seed),
        "train": {
            "source": str(src_train.as_posix()),
            "dest": str(dst_train.as_posix()),
            "n_requested": int(n_train),
            "n_saved": int(len(train_sample)),
            "sampling": train_info,
            "columns": list(train_sample.columns),
        },
        "test": {
            "source": str(src_test.as_posix()) if src_test.exists() else None,
            "dest": str(dst_test.as_posix()),
            "n_requested": int(n_test),
            "n_saved": int(len(test_sample)),
            "sampling": test_info,
            "columns": list(test_sample.columns),
        },
        "notes": [
            "Demo build uses tests/data/sample_train.csv and sample_test.csv only.",
            "Raw Kaggle files under data/raw are gitignored and not required for deployment.",
        ],
    }
    meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print(f"Saved: {dst_train}  shape={train_sample.shape}")
    print(f"Saved: {dst_test}   shape={test_sample.shape}")
    print(f"Meta : {meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q", type=int, default=10, help="quantile bins for stratified sampling")
    args = parser.parse_args()

    main(n_train=args.n_train, n_test=args.n_test, seed=args.seed, q=args.q)
