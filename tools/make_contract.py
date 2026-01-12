# tools/make_contract.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_train_test, split_xy
from src.pipelines import get_pipeline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ridge | lgbm | extratrees | xgb | voting_mean | stacking")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", type=int, default=12)
    args = ap.parse_args()

    train_df, _ = load_train_test()   # uses config.default_paths().data_raw by default
    X_raw, y_raw = split_xy(train_df)

    # match your pipeline training target (log1p)
    y = np.log1p(y_raw)

    # choose deterministic golden rows
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X_raw), size=args.rows, replace=False)
    golden = X_raw.iloc[idx].copy()

    pipe = get_pipeline(args.model, seed=args.seed)
    pipe.fit(X_raw, y)

    preds = pipe.predict(golden)

    # tolerance (tree models looser)
    if args.model in {"lgbm", "extratrees", "xgb", "stacking", "voting_mean"}:
        tol_rel = 0.03
        tol_abs = 0.03
    else:
        tol_rel = 0.01
        tol_abs = 0.005


    pred_lo = (preds - (np.abs(preds) * tol_rel + tol_abs)).tolist()
    pred_hi = (preds + (np.abs(preds) * tol_rel + tol_abs)).tolist()

    payload = {
        "model_name": args.model,
        "seed": int(args.seed),
        "n_rows": int(args.rows),
        "feature_columns": list(golden.columns),
        "golden_X": golden.to_dict(orient="records"),
        "expected": {"pred_lo": pred_lo, "pred_hi": pred_hi},
        "tolerance_policy": {
            "relative": tol_rel,
            "absolute": tol_abs,
            "note": "pass if pred within [lo, hi] for each golden row (log-space)",
        },
    }

    out = Path("tests/contracts") / f"{args.model}_contract.json"
    write_json(out, payload)

    print(f"Saved contract: {out.resolve()}")
    print(f"Rows: {args.rows}  Features: {len(payload['feature_columns'])}")
    print(
        f"Example pred: {float(preds[0]):.6f}  range: "
        f"[{payload['expected']['pred_lo'][0]:.6f}, {payload['expected']['pred_hi'][0]:.6f}]"
    )


if __name__ == "__main__":
    main()
