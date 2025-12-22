# src/predict.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_train_test, ID_COL
from .ensemble import blend_mean, blend_weighted, stacking_ridge

TARGET = "SalePrice"

def load_saved_preds(reports_dir: Path, model_names: list[str]):
    oofs, tests, rmses = [], [], []
    # read CV rmses from metrics.csv if present (for weighted blend)
    metrics_csv = reports_dir / "metrics.csv"
    if metrics_csv.exists():
        m = pd.read_csv(metrics_csv).set_index("model")
        rmses = [float(m.loc[name, "cv_rmse_log"]) for name in model_names]
    else:
        rmses = [1.0] * len(model_names)

    for name in model_names:
        oofs.append(np.load(reports_dir / f"oof_{name}.npy"))
        tests.append(np.load(reports_dir / f"testpred_{name}.npy"))
    return oofs, tests, rmses

def save_submission(test_ids: pd.Series, pred_log: np.ndarray, out_path: Path):
    pred = np.expm1(pred_log)  # back-transform
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: pred})
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Single model name: lgbm/xgb/ridge/extratrees")
    parser.add_argument("--ensemble", type=str, default=None, choices=[None, "blend_mean", "blend_weighted", "stack"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"

    train_df, test_df = load_train_test()
    test_ids = test_df[ID_COL].copy()

    model_names = ["lgbm", "xgb", "ridge", "extratrees"]

    if args.model:
        if args.model not in model_names:
            raise ValueError(f"Unknown model: {args.model}. Choose from {model_names}")
        _, tests, _ = load_saved_preds(reports_dir, [args.model])
        pred_log = tests[0]
        save_submission(test_ids, pred_log, project_root / f"submission_{args.model}.csv")
        return

    if args.ensemble is None:
        raise ValueError("Please provide either --model <name> or --ensemble <blend_mean|blend_weighted|stack>")

    oofs, tests, rmses = load_saved_preds(reports_dir, model_names)

    if args.ensemble == "blend_mean":
        pred_log = blend_mean(tests)
        save_submission(test_ids, pred_log, project_root / "submission_blend_mean.csv")

    elif args.ensemble == "blend_weighted":
        pred_log = blend_weighted(tests, rmses)
        save_submission(test_ids, pred_log, project_root / "submission_blend_weighted.csv")

    elif args.ensemble == "stack":
        # meta model trains on OOF predictions, target is log1p(SalePrice)
        y_log = np.log1p(train_df[TARGET].values.astype(float))
        pred_log = stacking_ridge(oofs, y_log, tests, seed=42)
        save_submission(test_ids, pred_log, project_root / "submission_stacking.csv")

if __name__ == "__main__":
    main()
