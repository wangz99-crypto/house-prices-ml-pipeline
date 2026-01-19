#!/usr/bin/env python3
"""
预测模块 - 支持 Kaggle 模式和 生产(Registry)模式

Kaggle:
  python -m src.predict kaggle --model lgbm
  python -m src.predict kaggle --ensemble blend_weighted

Prod (family selectors):
  python -m src.predict prod --model-id ridge/2026-01-10_163501 --input data/new_data.csv
  python -m src.predict prod --model-id ridge/latest --input data/new_data.csv
  python -m src.predict prod --model-id ridge/production --input data/new_data.csv

Prod (global selectors across all families):
  python -m src.predict prod --model-id global/latest --input data/new_data.csv
  python -m src.predict prod --model-id global/best --input data/new_data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import default_paths, ensure_dirs
from .data import ID_COL, load_train_test
from .ensemble import blend_mean, blend_weighted, stacking_ridge
from .registry import (
    fingerprint_dataframe,
    resolve_global_model_id,
    resolve_run_id,
    write_json,
)

TARGET = "SalePrice"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(":", "_")


def load_saved_preds(
    reports_dir: Path,
    model_names: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Load OOF and test predictions saved by training:
      artifacts/reports/<model>_oof.npy
      artifacts/reports/<model>_test_pred.npy
      artifacts/reports/metrics.csv (optional) -> provides cv_rmse_log per model for weighting
    """
    oofs: List[np.ndarray] = []
    tests: List[np.ndarray] = []

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
    ensure_dir(out_path.parent)
    sub.to_csv(out_path, index=False)
    logger.info(f"Saved submission: {out_path}")


def parse_model_id(model_id: str) -> Tuple[str, str]:
    """
    Returns (family, selector).
    model_id must look like '<family>/<selector>', e.g.:
      ridge/latest
      ridge/2026-01-14_165436
      global/best
    """
    if "/" not in model_id:
        raise ValueError("model_id must look like '<model>/<selector>' e.g. ridge/latest or global/best")
    name, selector = model_id.split("/", 1)
    if not name or not selector:
        raise ValueError(f"Invalid model_id: {model_id}")
    return name, selector


def resolve_any_model_id(model_id: str, registry_root: Path) -> str:
    """
    Supports:
      - global/best, global/latest  -> resolve_global_model_id(...)
      - <model>/<alias or run_id>  -> resolve_run_id(...)
    Returns resolved id: '<model>/<run_id>'
    """
    family, selector = parse_model_id(model_id)

    if family == "global":
        # selector must be best/latest
        resolved = resolve_global_model_id(selector, registry_root=registry_root)
        return resolved

    # family model
    resolved_run_id = resolve_run_id(family, selector, registry_root=registry_root)
    return f"{family}/{resolved_run_id}"


def get_model_expected_features(model) -> Optional[List[str]]:
    """
    Try to extract expected input features from model/pipeline.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


def validate_and_prepare_features(
    df: pd.DataFrame, expected_features: List[str], id_col: str = ID_COL
) -> pd.DataFrame:
    """
    Ensure df contains exactly expected_features (+ ID).
    - Add missing features as 0
    - Drop extra features
    - Reorder columns to match expected_features
    """
    if id_col not in df.columns:
        raise KeyError(f"Missing ID column '{id_col}' in input data")

    current = [c for c in df.columns if c != id_col]
    missing = sorted(list(set(expected_features) - set(current)))
    extra = sorted(list(set(current) - set(expected_features)))

    if missing:
        logger.warning(f"Missing {len(missing)} features. Adding them as 0. Example: {missing[:5]}")
        for c in missing:
            df[c] = 0

    if extra:
        logger.warning(f"Extra {len(extra)} features. Dropping them. Example: {extra[:5]}")
        df = df.drop(columns=extra)

    df = df[[id_col] + expected_features]
    return df


# ----------------------------
# Registry/Prod scoring
# ----------------------------
def score_batch_with_registry(model_id: str, input_csv: Path, out_path: Path) -> Dict[str, Any]:
    """
    model_id supports:
      - ridge/latest, ridge/best, ridge/production, ridge/<run_id>
      - global/latest, global/best
    """
    paths = default_paths()
    ensure_dirs(paths)

    registry_root = paths.models_registry
    resolved_model_id = resolve_any_model_id(model_id, registry_root=registry_root)

    model_name, resolved_run_id = parse_model_id(resolved_model_id)

    run_dir = registry_root / model_name / resolved_run_id
    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing registry model at: {model_path}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if ID_COL not in df.columns:
        raise KeyError(f"Missing ID column '{ID_COL}' in input: {input_csv}")

    ids = df[ID_COL].copy()

    model = joblib.load(model_path)
    expected = get_model_expected_features(model)
    if expected:
        df = validate_and_prepare_features(df, expected)

    X = df.drop(columns=[ID_COL]).copy()

    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    ensure_dir(out_path.parent)
    out_df = pd.DataFrame(
        {
            ID_COL: ids,
            TARGET: pred,
            f"{TARGET}_log": pred_log,
            "model_id": resolved_model_id,
        }
    )
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved batch predictions: {out_path}")

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "model_id_input": model_id,                 # may be alias/global
        "model_id_resolved": resolved_model_id,     # always explicit model/run
        "input_csv": str(input_csv),
        "input_fingerprint": fingerprint_dataframe(df),  # includes ID + (aligned) features
        "output_csv": str(out_path),
        "rows": int(len(out_df)),
        "model_artifact": str(model_path),
    }
    write_json(meta_path, meta)
    logger.info(f"Saved metadata: {meta_path}")

    return meta


# ----------------------------
# Kaggle mode
# ----------------------------
def run_kaggle_mode(args: argparse.Namespace) -> None:
    """
    Generate Kaggle submission files using saved predictions in artifacts/reports/.
    Default output location:
      artifacts/submissions/submission_<...>.csv
    """
    paths = default_paths()
    ensure_dirs(paths)

    reports_dir = paths.reports
    submissions_dir = paths.submissions

    # Load raw data to get test IDs and (if needed) y_log for stacking
    train_df, test_df = load_train_test(paths.data_raw)
    test_ids = test_df[ID_COL].copy()

    model_names = args.models

    # single model submission
    if args.model:
        _, tests, _ = load_saved_preds(reports_dir, [args.model])
        pred_log = tests[0]
        out_path = Path(args.out) if args.out else (submissions_dir / f"submission_{args.model}.csv")
        save_submission(test_ids, pred_log, out_path)
        return

    # ensemble submission
    if args.ensemble:
        oofs, tests, rmses = load_saved_preds(reports_dir, model_names)

        if args.ensemble == "blend_mean":
            pred_log = blend_mean(tests)
            out_path = Path(args.out) if args.out else (submissions_dir / "submission_blend_mean.csv")

        elif args.ensemble == "blend_weighted":
            pred_log = blend_weighted(tests, rmses)
            out_path = Path(args.out) if args.out else (submissions_dir / "submission_blend_weighted.csv")

        elif args.ensemble == "stack":
            y_log = np.log1p(train_df[TARGET].values.astype(float))
            pred_log = stacking_ridge(oofs, y_log, tests, seed=args.seed)
            out_path = Path(args.out) if args.out else (submissions_dir / "submission_stacking.csv")

        else:
            raise ValueError(f"Unknown ensemble: {args.ensemble}")

        save_submission(test_ids, pred_log, out_path)

        cfg = {"ensemble": args.ensemble, "models": model_names, "seed": args.seed}
        cfg_path = out_path.with_suffix(".config.json")
        ensure_dir(cfg_path.parent)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        logger.info(f"Saved submission config: {cfg_path}")
        return

    raise ValueError("Kaggle mode requires either --model <name> or --ensemble <blend_mean|blend_weighted|stack>")


# ----------------------------
# Production mode
# ----------------------------
def run_production_mode(args: argparse.Namespace) -> None:
    """
    Batch scoring using registry model artifacts.
    Default output location:
      artifacts/predictions/preds_<safe_model_id>.csv
    """
    paths = default_paths()
    ensure_dirs(paths)

    input_csv = Path(args.input) if args.input else (paths.data_raw / "test.csv")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = paths.predictions / f"preds_{safe_filename(args.model_id)}.csv"

    score_batch_with_registry(args.model_id, input_csv, out_path)


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Make predictions using trained models (Kaggle + Registry).")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Kaggle mode
    kaggle = subparsers.add_parser("kaggle", help="Generate Kaggle submissions using saved predictions.")
    kaggle.add_argument("--model", type=str, default=None, help="Single model name (e.g., lgbm/xgb/ridge/extratrees)")
    kaggle.add_argument("--ensemble", type=str, default=None, choices=["blend_mean", "blend_weighted", "stack"])
    kaggle.add_argument("--models", type=str, nargs="+", default=["lgbm", "xgb", "ridge", "extratrees"])
    kaggle.add_argument("--seed", type=int, default=42)
    kaggle.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for submission CSV (default: artifacts/submissions/...)",
    )

    # Prod mode
    prod = subparsers.add_parser("prod", help="Batch scoring using registry model artifacts.")
    prod.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model id: <model>/<run_id-or-alias> OR global/<best|latest>",
    )
    prod.add_argument("--input", type=str, default=None, help="Input CSV path (default: data/raw/test.csv)")
    prod.add_argument("--out", type=str, default=None, help="Output CSV path (default: artifacts/predictions/...)")

    args = parser.parse_args()

    if args.mode == "kaggle":
        run_kaggle_mode(args)
    elif args.mode == "prod":
        run_production_mode(args)


if __name__ == "__main__":
    main()
