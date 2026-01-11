#!/usr/bin/env python3
"""
预测模块 - 支持 Kaggle 模式和 生产(Registry)模式

Kaggle:
  python -m src.predict kaggle --model lgbm
  python -m src.predict kaggle --ensemble blend_weighted

Prod:
  python -m src.predict prod --model-id ridge/2026-01-10_163501 --input data/new_data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

from .config import default_paths
from .data import load_train_test, ID_COL
from .ensemble import blend_mean, blend_weighted, stacking_ridge
from .registry import write_json, fingerprint_dataframe

TARGET = "SalePrice"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_saved_preds(reports_dir: Path, model_names: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    oofs, tests = [], []

    metrics_csv = reports_dir / "metrics.csv"
    if metrics_csv.exists():
        m = pd.read_csv(metrics_csv).set_index("model")
        rmses = [float(m.loc[name, "cv_rmse_log"]) if name in m.index else 1.0 for name in model_names]
    else:
        rmses = [1.0] * len(model_names)

    for name in model_names:
        oof_path = reports_dir / f"{name}_oof.npy"
        test_path = reports_dir / f"{name}_test_pred.npy"
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test predictions: {test_path}")
        oofs.append(np.load(oof_path))
        tests.append(np.load(test_path))

    return oofs, tests, rmses


def save_submission(test_ids: pd.Series, pred_log: np.ndarray, out_path: Path) -> None:
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)  # safety
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: pred})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


def parse_model_id(model_id: str) -> Tuple[str, str]:
    if "/" not in model_id:
        raise ValueError("model_id must look like '<model>/<run_id>' e.g. ridge/2026-01-10_163501")
    name, run_id = model_id.split("/", 1)
    if not name or not run_id:
        raise ValueError(f"Invalid model_id: {model_id}")
    return name, run_id


def get_model_expected_features(model) -> Optional[List[str]]:
    # If pipeline has feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Try steps
    if hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def validate_and_prepare_features(df: pd.DataFrame, expected_features: List[str], id_col: str = ID_COL) -> pd.DataFrame:
    if id_col not in df.columns:
        raise KeyError(f"Missing ID column '{id_col}' in input data")

    current = [c for c in df.columns if c != id_col]

    missing = list(set(expected_features) - set(current))
    extra = list(set(current) - set(expected_features))

    if missing:
        logger.warning(f"Missing {len(missing)} features. Adding them as 0. Example: {missing[:5]}")
        for c in missing:
            df[c] = 0

    if extra:
        logger.warning(f"Extra {len(extra)} features. Dropping them. Example: {extra[:5]}")
        df = df.drop(columns=extra)

    # order
    df = df[[id_col] + expected_features]
    return df


def score_batch_with_registry(model_id: str, input_csv: Path, out_path: Path) -> Dict[str, Any]:
    paths = default_paths()
    model_name, run_id = parse_model_id(model_id)

    run_dir = paths.models_registry / model_name / run_id
    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing registry model at: {model_path}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    ids = df[ID_COL].copy() if ID_COL in df.columns else None
    if ids is None:
        raise KeyError(f"Missing ID column '{ID_COL}' in input: {input_csv}")

    model = joblib.load(model_path)
    expected = get_model_expected_features(model)
    if expected:
        df = validate_and_prepare_features(df, expected)

    X = df.drop(columns=[ID_COL]).copy()

    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({ID_COL: ids, TARGET: pred, f"{TARGET}_log": pred_log, "model_id": model_id})
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved batch predictions: {out_path}")

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "model_id": model_id,
        "input_csv": str(input_csv),
        "input_fingerprint": fingerprint_dataframe(df),
        "output_csv": str(out_path),
        "rows": int(len(out_df)),
    }
    write_json(meta_path, meta)
    logger.info(f"Saved metadata: {meta_path}")

    return meta


def run_kaggle_mode(args: argparse.Namespace) -> None:
    paths = default_paths()
    reports_dir = paths.reports_dir
    project_root = paths.project_root

    train_df, test_df = load_train_test(paths.data_raw)
    test_ids = test_df[ID_COL].copy()

    # default model list for ensemble
    model_names = args.models

    if args.model:
        _, tests, _ = load_saved_preds(reports_dir, [args.model])
        pred_log = tests[0]
        save_submission(test_ids, pred_log, project_root / f"submission_{args.model}.csv")
        return

    if args.ensemble:
        oofs, tests, rmses = load_saved_preds(reports_dir, model_names)

        if args.ensemble == "blend_mean":
            pred_log = blend_mean(tests)
            out_path = project_root / "submission_blend_mean.csv"

        elif args.ensemble == "blend_weighted":
            pred_log = blend_weighted(tests, rmses)
            out_path = project_root / "submission_blend_weighted.csv"

        elif args.ensemble == "stack":
            y_log = np.log1p(train_df[TARGET].values.astype(float))
            pred_log = stacking_ridge(oofs, y_log, tests, seed=args.seed)
            out_path = project_root / "submission_stacking.csv"

        else:
            raise ValueError(f"Unknown ensemble: {args.ensemble}")

        save_submission(test_ids, pred_log, out_path)

        # Save ensemble config next to output
        cfg = {"ensemble": args.ensemble, "models": model_names, "seed": args.seed}
        with open(out_path.with_suffix(".config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return

    raise ValueError("Kaggle mode requires either --model <name> or --ensemble <blend_mean|blend_weighted|stack>")


def run_production_mode(args: argparse.Namespace) -> None:
    paths = default_paths()

    input_csv = Path(args.input) if args.input else (paths.data_raw / "test.csv")
    if args.out:
        out_path = Path(args.out)
    else:
        safe_model_id = args.model_id.replace("/", "_").replace("\\", "_")
        out_path = paths.reports_dir / "predictions" / f"preds_{safe_model_id}.csv"

    score_batch_with_registry(args.model_id, input_csv, out_path)


def main():
    parser = argparse.ArgumentParser(description="Make predictions using trained models (Kaggle + Registry).")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Kaggle mode
    kaggle = subparsers.add_parser("kaggle", help="Generate Kaggle submissions using saved predictions.")
    kaggle.add_argument("--model", type=str, default=None, help="Single model name (e.g., lgbm/xgb/ridge/extratrees)")
    kaggle.add_argument("--ensemble", type=str, default=None, choices=["blend_mean", "blend_weighted", "stack"])
    kaggle.add_argument("--models", type=str, nargs="+", default=["lgbm", "xgb", "ridge", "extratrees"])
    kaggle.add_argument("--seed", type=int, default=42)

    # Prod mode
    prod = subparsers.add_parser("prod", help="Batch scoring using registry model artifacts.")
    prod.add_argument("--model-id", type=str, required=True, help="Registry model id: <model>/<run_id>")
    prod.add_argument("--input", type=str, default=None, help="Input CSV path (default: data/raw/test.csv)")
    prod.add_argument("--out", type=str, default=None, help="Output CSV path")

    args = parser.parse_args()

    if args.mode == "kaggle":
        run_kaggle_mode(args)
    elif args.mode == "prod":
        run_production_mode(args)


if __name__ == "__main__":
    main()

