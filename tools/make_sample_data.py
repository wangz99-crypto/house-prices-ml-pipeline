from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main(n: int = 200, seed: int = 42):
    src = ROOT / "data" / "raw" / "train.csv"
    dst = ROOT / "tests" / "data" / "sample_train.csv"
    meta = ROOT / "tests" / "data" / "sample_train_meta.json"

    if not src.exists():
        raise FileNotFoundError(
            f"Missing Kaggle raw train.csv at: {src}\n"
            "Put Kaggle train.csv under data/raw/ (gitignored), then rerun this script."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)

    # Stratified sampling by SalePrice (more stable EDA/demo behavior)
    if "SalePrice" in df.columns and len(df) >= 50:
        df["_price_bin"] = pd.qcut(df["SalePrice"], q=10, duplicates="drop")
        # allocate samples proportionally per bin
        parts = []
        for _, g in df.groupby("_price_bin"):
            k = max(1, int(round(n * len(g) / len(df))))
            parts.append(g.sample(n=min(k, len(g)), random_state=seed))
        sample = (
            pd.concat(parts, ignore_index=True)
              .drop(columns=["_price_bin"])
              .sample(frac=1, random_state=seed)
              .head(min(n, len(df)))
              .reset_index(drop=True)
        )
    else:
        sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    sample.to_csv(dst, index=False)

    meta_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": str(src.as_posix()),
        "dest": str(dst.as_posix()),
        "n_requested": int(n),
        "n_saved": int(len(sample)),
        "seed": int(seed),
        "columns": list(sample.columns),
    }
    meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print(f"Saved: {dst}  shape={sample.shape}")
    print(f"Meta : {meta}")


if __name__ == "__main__":
    main()

