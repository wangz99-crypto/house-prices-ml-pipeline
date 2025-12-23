"""
feature_importance.py (v2)

Extract and export feature importance from trained models in this repo.

Key improvements vs v1:
- Better feature-name handling for top-level ensemble estimators (VotingRegressor/StackingRegressor)
  where the *top-level* object is not a Pipeline.
- For Voting/Stacking, feature names are taken from the first fitted base pipeline's
  preprocessing ('prep') step when possible.

Supports:
- Linear models with coef_  -> |coef|
- Tree/boosting models with feature_importances_
- VotingRegressor: weighted average across base estimators that expose importance
- StackingRegressor: meta-learner importance if available; otherwise average base importances

Outputs (to reports/ by default):
- <model>_feature_importance.csv (Top-K)
- <model>_feature_importance_meta.json
- <model>_feature_importance_top<k>.png (unless --no-plot)

Usage (repo root):
    python analysis/feature_importance.py --model extratrees
    python analysis/feature_importance.py --model ridge --topk 50
    python analysis/feature_importance.py --model voting_mean --topk 40
    python analysis/feature_importance.py --model stacking --topk 40

Notes:
- For StackingRegressor, if meta-learner importance is returned, features correspond to
  base-model prediction features (meta_feature_0..), not original engineered/OHE features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# ----------------------------
# Model artifact discovery
# ----------------------------

def _find_model_file(models_dir: Path, model_name: str) -> Path:
    candidates = [
        models_dir / f"{model_name}.joblib",
        models_dir / f"{model_name}_pipeline.joblib",
        models_dir / f"{model_name}_model.joblib",
        models_dir / f"{model_name}.pkl",
        models_dir / f"{model_name}_pipeline.pkl",
        models_dir / f"{model_name}_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits: List[Path] = []
    for pat in (f"*{model_name}*.joblib", f"*{model_name}*.pkl"):
        hits.extend(sorted(models_dir.glob(pat)))

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"Multiple model artifacts match '{model_name}' in {models_dir}:\n"
            + "\n".join(str(h) for h in hits)
            + "\nPlease specify an exact file name or clean the directory."
        )

    raise FileNotFoundError(
        f"Could not find model artifact for '{model_name}' in {models_dir}. "
        f"Tried: {', '.join(str(c.name) for c in candidates)}"
    )


# ----------------------------
# Feature name extraction
# ----------------------------

def _safe_get_feature_names_from_pipeline(p) -> Optional[np.ndarray]:
    """Best-effort feature name extraction from a *Pipeline* that contains a ColumnTransformer."""
    # 1) pipeline.get_feature_names_out
    try:
        if hasattr(p, "get_feature_names_out"):
            names = p.get_feature_names_out()
            if names is not None and len(names) > 0:
                return np.asarray(names)
    except Exception:
        pass

    # 2) common step names
    for key in ("prep", "preprocess", "preprocessor", "ct"):
        try:
            if hasattr(p, "named_steps") and key in p.named_steps:
                step = p.named_steps[key]
                if hasattr(step, "get_feature_names_out"):
                    names = step.get_feature_names_out()
                    if names is not None and len(names) > 0:
                        return np.asarray(names)
        except Exception:
            continue

    # 3) scan steps from the end
    try:
        if hasattr(p, "steps"):
            for _, step in reversed(p.steps):
                if hasattr(step, "get_feature_names_out"):
                    names = step.get_feature_names_out()
                    if names is not None and len(names) > 0:
                        return np.asarray(names)
    except Exception:
        pass

    return None


def _fallback_feature_names(n_features: int) -> np.ndarray:
    return np.asarray([f"x{i}" for i in range(n_features)])


def _get_feature_names_from_ensemble_base(ensemble) -> Optional[np.ndarray]:
    """Try to obtain feature names from the first fitted base estimator if it's a pipeline."""
    # VotingRegressor: estimators_ is list of fitted estimators (pipelines in your repo)
    # StackingRegressor: estimators_ is list of fitted estimators
    base_list = getattr(ensemble, "estimators_", None)
    if not base_list:
        return None

    first = base_list[0]
    # If base is a Pipeline, extract names from its preprocessing.
    return _safe_get_feature_names_from_pipeline(first)


# ----------------------------
# Importance extraction
# ----------------------------

def _importance_from_estimator(est) -> Optional[np.ndarray]:
    if hasattr(est, "coef_"):
        coef = np.asarray(getattr(est, "coef_"))
        if coef.ndim == 2:
            return np.mean(np.abs(coef), axis=0)
        return np.abs(coef)

    if hasattr(est, "feature_importances_"):
        return np.asarray(getattr(est, "feature_importances_"))

    return None


def _aggregate_importance_from_voting(voting) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    if not hasattr(voting, "estimators_"):
        return None

    try:
        names = [n for n, _ in voting.estimators]
    except Exception:
        names = [f"est{i}" for i in range(len(voting.estimators_))]

    w = getattr(voting, "weights", None)
    if w is None or len(w) != len(voting.estimators_):
        w = [1.0] * len(voting.estimators_)

    imps, weights, used = [], [], {}
    for name, est, weight in zip(names, voting.estimators_, w):
        # base estimator may be a Pipeline; unwrap if so
        base_model = est.named_steps["model"] if hasattr(est, "named_steps") and "model" in est.named_steps else est
        imp = _importance_from_estimator(base_model)
        if imp is None:
            continue
        imps.append(imp.astype(float))
        weights.append(float(weight))
        used[name] = float(weight)

    if not imps:
        return None

    W = np.asarray(weights, dtype=float)
    W = W / (W.sum() if W.sum() != 0 else 1.0)
    agg = (np.vstack(imps).T @ W).ravel()
    return agg, used


def _aggregate_importance_from_stacking(stack) -> Optional[Tuple[np.ndarray, Dict[str, float], str]]:
    """Return (importance, used_estimators, mode) where mode is 'meta' or 'base'."""
    used: Dict[str, float] = {}

    # 1) meta-learner importance
    final_est = getattr(stack, "final_estimator_", None)
    if final_est is not None:
        meta_imp = _importance_from_estimator(final_est)
        if meta_imp is not None:
            used["_meta_learner_"] = 1.0
            return meta_imp, used, "meta"

    # 2) base estimators importance (average)
    estimators_ = getattr(stack, "estimators_", None)
    if estimators_ is None:
        return None

    try:
        names = [n for n, _ in stack.estimators]
    except Exception:
        names = [f"est{i}" for i in range(len(estimators_))]

    imps = []
    for name, est in zip(names, estimators_):
        base_model = est.named_steps["model"] if hasattr(est, "named_steps") and "model" in est.named_steps else est
        imp = _importance_from_estimator(base_model)
        if imp is None:
            continue
        imps.append(imp.astype(float))
        used[name] = 1.0

    if not imps:
        return None

    agg = np.mean(np.vstack(imps), axis=0).ravel()
    return agg, used, "base"


# ----------------------------
# Public API
# ----------------------------

def extract_feature_importance(model_obj) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return (importance_vector, feature_names, metadata)."""
    meta: Dict = {}

    cls = model_obj.__class__.__name__

    # --- VotingRegressor ---
    if cls == "VotingRegressor":
        out = _aggregate_importance_from_voting(model_obj)
        if out is None:
            raise ValueError("VotingRegressor base estimators do not expose coef_ / feature_importances_.")
        imp, used = out
        meta["aggregation"] = "voting_weighted_average"
        meta["used_estimators"] = used

        names = _get_feature_names_from_ensemble_base(model_obj)
        return np.asarray(imp), (names if names is not None else _fallback_feature_names(len(imp))), meta

    # --- StackingRegressor ---
    if cls == "StackingRegressor":
        out = _aggregate_importance_from_stacking(model_obj)
        if out is None:
            raise ValueError("StackingRegressor does not expose usable importance from meta/base estimators.")
        imp, used, mode = out
        meta["aggregation"] = "stacking_meta_or_base_average"
        meta["used_estimators"] = used
        meta["mode"] = mode

        if mode == "meta":
            names = np.asarray([f"meta_feature_{i}" for i in range(len(imp))])
            meta["note"] = "Meta-learner importance is over base-model prediction features (not original features)."
            return np.asarray(imp), names, meta

        # mode == 'base'
        names = _get_feature_names_from_ensemble_base(model_obj)
        return np.asarray(imp), (names if names is not None else _fallback_feature_names(len(imp))), meta

    # --- Pipeline (your single models) ---
    if hasattr(model_obj, "named_steps"):
        # try to unwrap estimator
        est = model_obj.named_steps.get("model", model_obj)
        imp = _importance_from_estimator(est)
        if imp is None:
            raise ValueError(f"Estimator {est.__class__.__name__} does not expose coef_ or feature_importances_.")

        names = _safe_get_feature_names_from_pipeline(model_obj)
        if names is None:
            names = _fallback_feature_names(len(imp))
            meta["warning"] = "Could not extract feature names from pipeline; using x0.."

        return np.asarray(imp), np.asarray(names), meta

    # --- Plain estimator ---
    imp = _importance_from_estimator(model_obj)
    if imp is None:
        raise ValueError(f"Estimator {cls} does not expose coef_ or feature_importances_.")

    names = _fallback_feature_names(len(imp))
    meta["warning"] = "Model is not a pipeline; using x0.."
    return np.asarray(imp), names, meta


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name (e.g., ridge, extratrees, lgbm, xgb, voting_mean, stacking).")
    ap.add_argument("--models-dir", default="models", help="Directory where trained model artifacts are saved.")
    ap.add_argument("--reports-dir", default="reports", help="Directory to write outputs (csv/json/plot).")
    ap.add_argument("--topk", type=int, default=30, help="Top-K features to export.")
    ap.add_argument("--no-plot", action="store_true", help="Do not generate a bar plot PNG.")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_file = _find_model_file(models_dir, args.model)
    model_obj = joblib.load(model_file)

    imp, names, meta = extract_feature_importance(model_obj)
    imp = np.asarray(imp, dtype=float).ravel()
    names = np.asarray(names)

    if len(names) != len(imp):
        meta["warning"] = (
            f"feature_names length ({len(names)}) != importance length ({len(imp)}); falling back to x0.."
        )
        names = _fallback_feature_names(len(imp))

    df_imp = pd.DataFrame({"feature": names, "importance": imp})
    df_imp["abs_importance"] = np.abs(df_imp["importance"])
    df_imp = df_imp.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    out_csv = reports_dir / f"{args.model}_feature_importance.csv"
    df_imp.head(args.topk).to_csv(out_csv, index=False)

    out_json = reports_dir / f"{args.model}_feature_importance_meta.json"
    out_json.write_text(
        json.dumps(
            {"model": args.model, "model_artifact": str(model_file), "topk": args.topk, "meta": meta},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Saved: {out_csv}")
    print(f"[OK] Saved: {out_json}")
    print(df_imp.head(args.topk).to_string(index=False))

    if not args.no_plot:
        import matplotlib.pyplot as plt

        top = df_imp.head(args.topk).iloc[::-1]
        plt.figure(figsize=(10, max(4, 0.28 * len(top))))
        plt.barh(top["feature"], top["abs_importance"])
        plt.title(f"Top {args.topk} Feature Importance: {args.model}")
        plt.xlabel("Absolute importance")
        plt.tight_layout()

        out_png = reports_dir / f"{args.model}_feature_importance_top{args.topk}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
