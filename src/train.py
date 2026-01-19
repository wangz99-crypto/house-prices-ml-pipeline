from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import default_paths, ensure_dirs
from .data import ID_COL, load_train_test, split_xy
from .evaluate import kfold_oof_predict
from .pipelines import PIPELINES, get_pipeline
from .registry import (
    fingerprint_dataframe,
    make_run_dir,
    set_alias,
    write_json,
    ensure_aliases,
    read_json,
    set_global_alias,
    read_global_aliases,
)

REQUIRED_FILES = [
    "model.joblib",
    "metrics.json",
    "data_fingerprint.json",
    "train_args.json",
    "pipeline_repr.txt",
    "oof.npy",
    "test_pred.npy",
]


def assert_run_complete(run_path: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (run_path / f).exists()]
    if missing:
        raise FileNotFoundError(f"Run is incomplete. Missing: {missing}")


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prepare raw X/y/test for sklearn Pipelines.
    - drop Id column (not a feature)
    - y uses log1p (Kaggle standard)
    """
    X_raw, y_raw = split_xy(train_df)

    if ID_COL in X_raw.columns:
        X_raw = X_raw.drop(columns=[ID_COL])

    X_test = test_df.copy()
    if ID_COL in X_test.columns:
        X_test = X_test.drop(columns=[ID_COL])

    y = np.log1p(y_raw.astype(float))
    return X_raw, y, X_test


def save_metrics_compat(reports_dir: Path, model_name: str, cv_rmse: float, fold_scores: list[float]) -> None:
    payload = {
        "model": model_name,
        "cv_rmse": float(cv_rmse),
        "fold_rmse": [float(x) for x in fold_scores],
    }
    out = reports_dir / f"{model_name}_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_update_family_best(model_name: str, run_id: str, cv_rmse: float, registry_dir: Path) -> None:
    """
    Update per-family alias 'best' only if improved.
    """
    aliases = ensure_aliases(model_name, registry_root=registry_dir)
    best_run_id = aliases.get("best")

    if best_run_id is None:
        set_alias(model_name, "best", run_id, registry_root=registry_dir)
        return

    best_metrics_path = registry_dir / model_name / best_run_id / "metrics.json"
    if not best_metrics_path.exists():
        # historical best is broken -> repair
        set_alias(model_name, "best", run_id, registry_root=registry_dir)
        return

    best_metrics = read_json(best_metrics_path)
    best_rmse = float(best_metrics["cv_rmse"])
    if float(cv_rmse) < best_rmse:
        set_alias(model_name, "best", run_id, registry_root=registry_dir)


def _maybe_update_global_best(model_name: str, run_id: str, cv_rmse: float, registry_dir: Path) -> None:
    """
    Update global alias 'best' only if improved across all model families.
    """
    g = read_global_aliases(registry_root=registry_dir)
    g_best = g.get("best") or {}
    g_best_rmse = float(g_best["cv_rmse"]) if isinstance(g_best, dict) and "cv_rmse" in g_best else None

    if g_best_rmse is None or float(cv_rmse) < g_best_rmse:
        set_global_alias("best", model_name, run_id, float(cv_rmse), registry_root=registry_dir)


def run_one(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    reports_dir: Path,
    current_dir: Path | None = None,
    registry_dir: Path | None = None,
    seed: int = 42,
    n_splits: int = 5,
    export_compat_model: bool = True,  # 是否输出 artifacts/current/<model>.joblib
    *,
    # backward-compat: old code/tests used models_dir to mean "current snapshot dir"
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Train one model family with KFold OOF + fit final model, write artifacts.

    Compatibility:
    - Accepts `models_dir` as an alias for `current_dir`.
    - Returns `model_path` for old tests (points to current model if exported, else registry model).

    Safety:
    - Ensures all output directories exist before writing.
    """
    print(f"\n=== Training: {model_name} ===")

    # ------------------------------
    # Backward-compatible dir wiring
    # ------------------------------
    if models_dir is not None:
        # old interface: models_dir meant "current snapshot dir"
        current_dir = models_dir

    if current_dir is None:
        raise ValueError("current_dir is required (or pass models_dir for backward-compat).")
    if registry_dir is None:
        raise ValueError("registry_dir is required.")

    # ------------------------------
    # Ensure output dirs exist (avoid implicit assumptions)
    # ------------------------------
    reports_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Create a registry run folder for this training run
    run = make_run_dir(registry_dir, model_name)

    # model factory: each fold gets a fresh estimator
    model_fn = lambda: get_pipeline(model_name, seed=seed)

    # KFold OOF predictions + averaged test predictions
    oof, test_pred, cv_rmse, fold_scores = kfold_oof_predict(
        model=model_fn,
        X=X,
        y=y,
        X_test=X_test,
        n_splits=n_splits,
        seed=seed,
    )

    # ------------------------------
    # Save OOF + test preds
    #   1) reports/
    #   2) registry/<model>/<run_id>/
    # ------------------------------
    oof_path_reports = reports_dir / f"{model_name}_oof.npy"
    test_pred_path_reports = reports_dir / f"{model_name}_test_pred.npy"

    np.save(oof_path_reports, oof)
    np.save(test_pred_path_reports, test_pred)

    np.save(run.run_dir / "oof.npy", oof)
    np.save(run.run_dir / "test_pred.npy", test_pred)

    # ------------------------------
    # Save metrics (compat + registry)
    # ------------------------------
    save_metrics_compat(reports_dir, model_name, cv_rmse, fold_scores)

    write_json(
        run.run_dir / "metrics.json",
        {
            "model": model_name,
            "cv_rmse": float(cv_rmse),
            "fold_rmse": [float(x) for x in fold_scores],
        },
    )

    # ------------------------------
    # Fit full model on all data and save artifacts
    # ------------------------------
    final_model = get_pipeline(model_name, seed=seed)
    final_model.fit(X, y)

    # registry-style model file (versioned run)
    registry_model_path = run.run_dir / "model.joblib"
    joblib.dump(final_model, registry_model_path)

    # current-style model file (latest snapshot)
    current_model_path: Path | None = None
    if export_compat_model:
        current_model_path = current_dir / f"{model_name}.joblib"
        joblib.dump(final_model, current_model_path)

    # ------------------------------
    # Lineage / args / fingerprints
    # ------------------------------
    write_json(
        run.run_dir / "train_args.json",
        {"seed": int(seed), "n_splits": int(n_splits), "model_name": model_name},
    )

    write_json(
        run.run_dir / "data_fingerprint.json",
        {"X": fingerprint_dataframe(X), "y_rows": int(len(y))},
    )

    (run.run_dir / "pipeline_repr.txt").write_text(str(final_model), encoding="utf-8")

    # Final artifact completeness check
    assert_run_complete(run.run_dir)

    # ------------------------------
    # Aliases
    # ------------------------------
    # 1) family latest: always update
    set_alias(model_name, "latest", run.run_id, registry_root=registry_dir)

    # 2) family best: update only if improved
    _maybe_update_family_best(model_name, run.run_id, float(cv_rmse), registry_dir)

    # 3) global latest: always update
    set_global_alias("latest", model_name, run.run_id, float(cv_rmse), registry_root=registry_dir)

    # 4) global best: update only if improved
    _maybe_update_global_best(model_name, run.run_id, float(cv_rmse), registry_dir)

    print(f"[{model_name}] CV RMSE (log-space): {cv_rmse:.6f} | model_id={run.model_id}")

    # ------------------------------
    # Return payload (compat + new)
    # ------------------------------
    # model_path is what old tests often expect: prefer current snapshot if available
    model_path = current_model_path if current_model_path is not None else registry_model_path

    return {
        "model": model_name,
        "model_id": run.model_id,
        "run_dir": str(run.run_dir),
        "cv_rmse": float(cv_rmse),
        "fold_scores": [float(x) for x in fold_scores],
        "oof_path": str(oof_path_reports),
        "test_pred_path": str(test_pred_path_reports),
        "current_model_path": str(current_model_path) if current_model_path else None,
        "registry_model_path": str(registry_model_path),
        # backward-compat key
        "model_path": str(model_path),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help=f"one of {sorted(PIPELINES.keys())} or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing train.csv/test.csv (default: data/raw)",
    )

    # 控制是否生成 artifacts/current/<model>.joblib
    parser.add_argument(
        "--no-export-compat-model",
        dest="export_compat_model",
        action="store_false",
        help="Disable exporting artifacts/current/<model>.joblib; only write registry artifacts.",
    )
    parser.set_defaults(export_compat_model=True)

    args = parser.parse_args()

    paths = default_paths()
    ensure_dirs(paths)

    reports_dir = paths.reports
    current_dir = paths.models_current
    registry_dir = paths.models_registry

    # Load data
    data_dir = Path(args.data_dir) if args.data_dir else paths.data_raw
    train_df, test_df = load_train_test(data_dir)

    # Prepare raw features/target
    X, y, X_test = prepare_features(train_df, test_df)

    # Select models
    if args.model == "all":
        to_run = sorted(PIPELINES.keys())
    else:
        if args.model not in PIPELINES:
            raise ValueError(f"Unknown model: {args.model}. Choose from {sorted(PIPELINES.keys())} or 'all'")
        to_run = [args.model]

    # Train
    results: list[dict[str, Any]] = []
    for name in to_run:
        res = run_one(
            model_name=name,
            X=X,
            y=y,
            X_test=X_test,
            reports_dir=reports_dir,
            current_dir=current_dir,
            registry_dir=registry_dir,
            seed=args.seed,
            n_splits=args.folds,
            export_compat_model=bool(args.export_compat_model),
        )
        results.append(res)

    # Save metrics.csv for ensembling
    metrics_rows = [{"model": r["model"], "cv_rmse_log": float(r["cv_rmse"])} for r in results]
    metrics_path = reports_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    # Save summary
    summary_path = reports_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved CV summary to: {summary_path}")


if __name__ == "__main__":
    main()
