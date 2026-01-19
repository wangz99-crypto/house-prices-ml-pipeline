# analysis/feature_importance.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import sys

def _inject_repo_root_to_syspath():
    cur = Path(__file__).resolve()
    # analysis/feature_importance.py -> repo_root is parent of "analysis"
    repo_root = cur.parents[1]
    if (repo_root / "src").exists():
        sys.path.insert(0, str(repo_root))
    else:
        # fallback: walk up a few levels
        p = cur
        for _ in range(6):
            p = p.parent
            if (p / "src").exists():
                sys.path.insert(0, str(p))
                break

_inject_repo_root_to_syspath()



# ----------------------------
# Repo root (robust)
# ----------------------------
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "src").exists() and (cur / "artifacts").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(f"Cannot locate repo root from {start}")


# ----------------------------
# Feature name extraction
# ----------------------------
def _safe_get_feature_names_from_pipeline(p) -> Optional[np.ndarray]:
    try:
        if hasattr(p, "get_feature_names_out"):
            names = p.get_feature_names_out()
            if names is not None and len(names) > 0:
                return np.asarray(names)
    except Exception:
        pass

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
    base_list = getattr(ensemble, "estimators_", None)
    if not base_list:
        return None
    first = base_list[0]
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
    used: Dict[str, float] = {}

    final_est = getattr(stack, "final_estimator_", None)
    if final_est is not None:
        meta_imp = _importance_from_estimator(final_est)
        if meta_imp is not None:
            used["_meta_learner_"] = 1.0
            return meta_imp, used, "meta"

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


def extract_feature_importance(model_obj) -> Tuple[np.ndarray, np.ndarray, Dict]:
    meta: Dict = {}
    cls = model_obj.__class__.__name__

    if cls == "VotingRegressor":
        out = _aggregate_importance_from_voting(model_obj)
        if out is None:
            raise ValueError("VotingRegressor base estimators do not expose coef_ / feature_importances_.")
        imp, used = out
        meta["aggregation"] = "voting_weighted_average"
        meta["used_estimators"] = used
        names = _get_feature_names_from_ensemble_base(model_obj)
        return np.asarray(imp), (names if names is not None else _fallback_feature_names(len(imp))), meta

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

        names = _get_feature_names_from_ensemble_base(model_obj)
        return np.asarray(imp), (names if names is not None else _fallback_feature_names(len(imp))), meta

    if hasattr(model_obj, "named_steps"):
        est = model_obj.named_steps.get("model", model_obj)
        imp = _importance_from_estimator(est)
        if imp is None:
            raise ValueError(f"Estimator {est.__class__.__name__} does not expose coef_ or feature_importances_.")

        names = _safe_get_feature_names_from_pipeline(model_obj)
        if names is None:
            names = _fallback_feature_names(len(imp))
            meta["warning"] = "Could not extract feature names from pipeline; using x0.."

        return np.asarray(imp), np.asarray(names), meta

    imp = _importance_from_estimator(model_obj)
    if imp is None:
        raise ValueError(f"Estimator {cls} does not expose coef_ or feature_importances_.")
    names = _fallback_feature_names(len(imp))
    meta["warning"] = "Model is not a pipeline; using x0.."
    return np.asarray(imp), names, meta


# ----------------------------
# Registry helpers (global best)
# ----------------------------
def load_global_best(repo_root: Path) -> Dict:
    aliases_path = repo_root / "artifacts" / "registry" / "_global" / "aliases.json"
    if not aliases_path.exists():
        raise FileNotFoundError(f"Missing {aliases_path}")

    d = json.loads(aliases_path.read_text(encoding="utf-8"))
    best = d.get("best")
    if not isinstance(best, dict):
        raise ValueError("Global aliases.json missing 'best' dict.")

    # tolerate different key names
    model = best.get("model") or best.get("model_name") or best.get("name")
    run_id = best.get("run_id") or best.get("run") or best.get("id")

    if not model or not run_id:
        raise KeyError(f"Cannot parse global best keys in {aliases_path}. Got: {best.keys()}")

    return {"model": model, "run_id": run_id, "raw": best}


def registry_model_path(repo_root: Path, model: str, run_id: str) -> Path:
    p = repo_root / "artifacts" / "registry" / model / run_id / "model.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Missing model artifact: {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="__global_best__", help="Model name, or __global_best__ (default).")
    ap.add_argument("--run-id", default=None, help="Optional run id. If omitted and model is __global_best__, read from global aliases.")
    ap.add_argument("--topk", type=int, default=30, help="Top-K features to export.")
    ap.add_argument("--no-plot", action="store_true", help="Do not generate a bar plot PNG.")
    args = ap.parse_args()

    repo_root = find_repo_root(Path.cwd())

    # ---- resolve model + run_id ----
    if args.model == "__global_best__":
        info = load_global_best(repo_root)
        model = info["model"]
        run_id = info["run_id"]
        best_meta = info["raw"]
    else:
        model = args.model
        run_id = args.run_id
        best_meta = {"note": "manual selection", "model": model, "run_id": run_id}

        if run_id is None:
            # allow using "current" artifact as fallback
            current = repo_root / "artifacts" / "current" / f"{model}.joblib"
            if current.exists():
                model_file = current
            else:
                raise ValueError("When --model is not __global_best__, please provide --run-id (or ensure artifacts/current/<model>.joblib exists).")
        else:
            model_file = registry_model_path(repo_root, model, run_id)

    if args.model == "__global_best__":
        model_file = registry_model_path(repo_root, model, run_id)

    # ---- output dir ----
    out_dir = repo_root / "artifacts" / "reports" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load + extract ----
    model_obj = joblib.load(model_file)
    imp, names, meta = extract_feature_importance(model_obj)

    imp = np.asarray(imp, dtype=float).ravel()
    names = np.asarray(names)

    if len(names) != len(imp):
        meta["warning"] = f"feature_names length ({len(names)}) != importance length ({len(imp)}); falling back to x0.."
        names = _fallback_feature_names(len(imp))

    df_imp = pd.DataFrame({"feature": names, "importance": imp})
    df_imp["abs_importance"] = np.abs(df_imp["importance"])
    df_imp = df_imp.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    # ---- write artifacts ----
    tag = f"{model}__{run_id}" if run_id else model
    out_csv = out_dir / f"{tag}__top{args.topk}.csv"
    out_json = out_dir / f"{tag}__meta.json"

    df_imp.head(args.topk).to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "model": model,
                "run_id": run_id,
                "model_artifact": str(model_file),
                "topk": args.topk,
                "best_meta": best_meta,
                "meta": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Saved CSV : {out_csv}")
    print(f"[OK] Saved META: {out_json}")
    print(df_imp.head(args.topk).to_string(index=False))

    if not args.no_plot:
        import matplotlib.pyplot as plt

        top = df_imp.head(args.topk).iloc[::-1]
        plt.figure(figsize=(10, max(4, 0.28 * len(top))))
        plt.barh(top["feature"], top["abs_importance"])
        plt.title(f"Top {args.topk} Feature Importance: {model}")
        plt.xlabel("Absolute importance")
        plt.tight_layout()

        out_png = out_dir / f"{tag}__top{args.topk}.png"
        plt.savefig(out_png, dpi=180)
        plt.close()
        print(f"[OK] Saved PNG : {out_png}")


if __name__ == "__main__":
    main()
