from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import joblib


# -----------------------------
# Path bootstrap (fix: pickle references "src")
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # ✅ so "src" can be imported when unpickling

DEMO = REPO_ROOT / "artifacts_demo"

FIG_DIR = DEMO / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = DEMO / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = REPO_ROOT / "data" / "raw" / "train.csv"

SUMMARY_CSV = REPORTS_DIR / "model_performance_summary.csv"

REGISTRY_DIR = DEMO / "registry"

FEAT_IMP_DIR = REPORTS_DIR / "feature_importance"
FEAT_IMP_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Recommended: clear old FI csvs each run so "latest" doesn't pick stale files
CLEAN_OLD_FEATIMP_CSVS = True


# -----------------------------
# Utilities
# -----------------------------
def _safe_read_json(p: Path) -> Any:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt_money_axis(ax):
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))


def _pick_latest_csv(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    csvs = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _normalize_stage(s: str) -> str:
    s0 = str(s).strip().replace(" ", "")
    s0 = s0.lower()
    if s0 == "stage1":
        return "Stage1"
    if s0 == "stage2":
        return "Stage2"
    if "1" in s0 and "stage" in s0:
        return "Stage1"
    if "2" in s0 and "stage" in s0:
        return "Stage2"
    return str(s).strip()


def _load_model_summary() -> Optional[pd.DataFrame]:
    if not SUMMARY_CSV.exists():
        print(f"[skip] missing {SUMMARY_CSV}")
        return None

    df = pd.read_csv(SUMMARY_CSV)
    df.columns = [c.strip() for c in df.columns]

    required = {"stage", "model", "candidate", "rmse_mean", "rmse_std"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{SUMMARY_CSV} missing columns: {sorted(missing)}")

    df["stage"] = df["stage"].apply(_normalize_stage)
    df["rmse_mean"] = pd.to_numeric(df["rmse_mean"], errors="coerce")
    df["rmse_std"] = pd.to_numeric(df["rmse_std"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["rmse_mean"]).copy()
    return df


def _unwrap_fitted_pipeline(model_obj: Any) -> Any:
    """
    Your models are often wrapped by AsRegressor:
      - fitted pipeline stored in estimator_
      - base pipeline stored in estimator
    We prefer estimator_ (fitted).
    """
    if hasattr(model_obj, "estimator_"):
        return getattr(model_obj, "estimator_")
    if hasattr(model_obj, "estimator"):
        return getattr(model_obj, "estimator")
    return model_obj


def _get_transformed_feature_names(model_obj: Any) -> Optional[List[str]]:
    """
    For your pipelines:
      Pipeline([("shared", ...), ("prep", ColumnTransformer), ("model", ...)])
    We want the output feature names after 'prep' (including OHE expansion).
    """
    pipe = _unwrap_fitted_pipeline(model_obj)
    if not hasattr(pipe, "named_steps"):
        return None
    if "prep" not in pipe.named_steps:
        return None

    prep = pipe.named_steps["prep"]
    if not hasattr(prep, "get_feature_names_out"):
        return None

    try:
        names = prep.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return None


def _unwrap_estimator(model_obj: Any) -> Any:
    """
    Try to extract an estimator that actually has coef_ or feature_importances_.
    Handles:
      - AsRegressor wrapper
      - sklearn Pipeline
      - common wrapper attributes
    """
    obj = _unwrap_fitted_pipeline(model_obj)

    # sklearn Pipeline: last step is model
    if hasattr(obj, "named_steps"):
        last = list(obj.named_steps.values())[-1]
        return last

    for attr in ["estimator_", "model_", "clf_", "regressor_"]:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    return obj


def _load_run_dir(model_name: str, run_id: str) -> Path:
    """
    artifacts_demo/registry/<model_name>/<run_id>/
    """
    return REGISTRY_DIR / model_name / run_id


def _read_alias_run_id(model_name: str, alias: str) -> Optional[str]:
    """
    aliases.json like:
      {"best": {"run_id": "..."} , "latest": {"run_id": "..."}}
    or {"best": "..."}.
    """
    p = REGISTRY_DIR / model_name / "aliases.json"
    if not p.exists():
        return None
    obj = _safe_read_json(p)
    if not obj:
        return None

    v = obj.get(alias)
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and "run_id" in v:
        return str(v["run_id"])
    return None


def _load_model_and_feature_names(run_dir: Path) -> Tuple[Any, List[str]]:
    """
    ✅ Important fix:
    - Prefer transformed feature names from fitted pipeline's 'prep'
      (this matches coef_/feature_importances_ length)
    - Fallback to feature_columns.json only if needed
    """
    model_path = run_dir / "model.joblib"
    feat_path = run_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"missing {model_path}")

    model_obj = joblib.load(model_path)

    transformed = _get_transformed_feature_names(model_obj)
    if transformed and len(transformed) > 0:
        return model_obj, transformed

    # fallback (rare): if no pipeline feature names available
    if not feat_path.exists():
        raise FileNotFoundError(f"missing {feat_path} (and pipeline has no feature names)")

    feat_names = _safe_read_json(feat_path)
    if not isinstance(feat_names, list) or not all(isinstance(x, str) for x in feat_names):
        raise RuntimeError(f"feature_columns.json must be list[str]. got: {type(feat_names)}")

    return model_obj, feat_names


def _compute_feature_importance(model_obj: Any, feat_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Returns df with columns: feature, importance
    """
    est = _unwrap_estimator(model_obj)

    # linear models: coef_
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_).reshape(-1)
        if len(coef) != len(feat_names):
            return None
        imp = np.abs(coef)
        return pd.DataFrame({"feature": feat_names, "importance": imp})

    # tree models: feature_importances_
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_).reshape(-1)
        if len(imp) != len(feat_names):
            return None
        return pd.DataFrame({"feature": feat_names, "importance": imp})

    # xgboost booster fallback (if needed)
    if hasattr(est, "get_booster"):
        try:
            booster = est.get_booster()
            score = booster.get_score(importance_type="gain")
            imp = np.zeros(len(feat_names), dtype=float)
            for k, v in score.items():
                if k.startswith("f"):
                    idx = int(k[1:])
                    if 0 <= idx < len(imp):
                        imp[idx] = float(v)
            return pd.DataFrame({"feature": feat_names, "importance": imp})
        except Exception:
            return None

    return None


def _write_featimp_csv(df_imp: pd.DataFrame, model_name: str, run_id: str, topk: int = 30) -> Path:
    df2 = df_imp.copy()
    df2["importance"] = pd.to_numeric(df2["importance"], errors="coerce").fillna(0.0)
    df2 = df2.sort_values("importance", ascending=False).head(topk)
    out = FEAT_IMP_DIR / f"{model_name}__{run_id}__top{topk}.csv"
    df2.to_csv(out, index=False)
    return out


def _clear_old_featimp_csvs():
    if not FEAT_IMP_DIR.exists():
        return
    for p in FEAT_IMP_DIR.glob("*.csv"):
        try:
            p.unlink()
        except Exception:
            pass


# -----------------------------
# Figure generators
# -----------------------------
def fig_price_distribution():
    if not TRAIN_CSV.exists():
        print(f"[skip] missing {TRAIN_CSV}")
        return

    df = pd.read_csv(TRAIN_CSV)
    if "SalePrice" not in df.columns:
        print("[skip] train.csv missing SalePrice")
        return

    y = df["SalePrice"].astype(float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(y, bins=60)
    ax.set_title("House price distribution")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    _fmt_money_axis(ax)

    fig.tight_layout()
    out = FIG_DIR / "eda_price_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote", out)


def fig_model_comparison():
    summary = _load_model_summary()
    if summary is None or len(summary) == 0:
        print("[skip] no model summary found (model_performance_summary.csv).")
        return

    def _plot_stage(stage_name: str, out_name: str):
        if not (summary["stage"] == stage_name).any():
            print(f"[skip] no rows for {stage_name}")
            return

        df = summary[summary["stage"] == stage_name].copy()
        df = df.sort_values("rmse_mean", ascending=True)

        labels = (df["model"].astype(str) + " • " + df["candidate"].astype(str)).tolist()
        means = df["rmse_mean"].astype(float).to_numpy()
        stds = df["rmse_std"].astype(float).to_numpy()

        best_idx = int(np.argmin(means))

        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        y = np.arange(len(labels))

        ax.errorbar(
            means,
            y,
            xerr=np.where(stds > 0, stds, 0),
            fmt="o",
            markersize=8,
            elinewidth=2,
            capsize=4,
        )
        ax.plot(means[best_idx], y[best_idx], "o", markersize=11)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        left = float(np.min(means - np.maximum(stds, 0))) - 0.0015
        right = float(np.max(means + np.maximum(stds, 0))) + 0.0015
        ax.set_xlim(left, right)

        ax.set_title(f"Model comparison — {stage_name} (mean ± std)")
        ax.set_xlabel("CV RMSE on log1p(SalePrice) (lower = better)")
        ax.grid(axis="x", alpha=0.20)

        # ✅ Remove right-side numeric annotations (cleaner plot)
        # (No ax.text loop here)

        ax.text(
            0.0, -0.14,
            "Note: x-axis is zoomed to make small differences visible.",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.7,
        )

        # ✅ Avoid bbox_inches='tight' to prevent unexpected cropping
        fig.tight_layout()
        out = FIG_DIR / out_name
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("[ok] wrote", out)

    _plot_stage("Stage1", "model_comparison_stage1.png")
    _plot_stage("Stage2", "model_comparison_stage2.png")



def export_registry_feature_importance(topk: int = 30):
    """
    Export feature importance for model families using best/latest runs.

    Your models are wrapped by AsRegressor and use:
      ("prep", ColumnTransformer(... OneHotEncoder ...))
      ("model", <Ridge/ExtraTrees/XGB/LGBM>)
    So we use prep.get_feature_names_out() to align dimensions.
    """
    if not REGISTRY_DIR.exists():
        print(f"[skip] missing registry dir: {REGISTRY_DIR}")
        return

    if CLEAN_OLD_FEATIMP_CSVS:
        _clear_old_featimp_csvs()

    model_families = [p.name for p in REGISTRY_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    model_families = [m for m in model_families if m != "_global"]

    if not model_families:
        print("[skip] no model families under registry/")
        return

    for model_name in sorted(model_families):
        # You can keep skipping these in demo (no canonical FI)
        if model_name in ["voting_mean", "stacking"]:
            print(f"[skip] {model_name}: no canonical feature importance (demo)")
            continue

        for alias in ["best", "latest"]:
            run_id = _read_alias_run_id(model_name, alias)
            if not run_id:
                print(f"[skip] {model_name}/{alias}: alias not set")
                continue

            run_dir = _load_run_dir(model_name, run_id)
            if not run_dir.exists():
                print(f"[skip] {model_name}/{alias}: missing run_dir {run_dir}")
                continue

            try:
                model_obj, feat_names = _load_model_and_feature_names(run_dir)
            except Exception as e:
                print(f"[skip] {model_name}/{alias}: cannot load model or features ({e})")
                continue

            df_imp = _compute_feature_importance(model_obj, feat_names)
            if df_imp is None or df_imp.empty:
                print(f"[skip] {model_name}/{alias}: cannot compute feature importance")
                continue

            out = _write_featimp_csv(df_imp, model_name=model_name, run_id=run_id, topk=topk)
            print(f"[ok] wrote {out}")


def fig_feature_importance_top20_latest():
    """
    Pick latest CSV in feature_importance and make feat_importance_top20.png
    """
    csv_path = _pick_latest_csv(FEAT_IMP_DIR)
    if csv_path is None:
        print(f"[skip] no feature importance CSV under {FEAT_IMP_DIR}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[skip] feature importance CSV is empty:", csv_path)
        return

    if "feature" not in df.columns or "importance" not in df.columns:
        print("[skip] feature importance CSV missing columns feature/importance:", csv_path)
        return

    top = df.sort_values("importance", ascending=False).head(20).copy()
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(top["feature"].astype(str), top["importance"].astype(float))
    ax.set_title(f"Top 20 feature importance (latest exported) — {csv_path.stem.split('__')[0]}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    out = FIG_DIR / "feat_importance_top20.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote", out, "(from", csv_path.name, ")")


def fig_feature_importance_top20_per_model():
    """
    Make per-model FI figures so Streamlit dropdown can switch without recomputing.
    Output:
      feat_importance_top20__ridge.png
      feat_importance_top20__xgb.png
      ...
    """
    csvs = sorted(FEAT_IMP_DIR.glob("*.csv"))
    if not csvs:
        print(f"[skip] no feature_importance CSV files under {FEAT_IMP_DIR}")
        return

    by_model: Dict[str, Path] = {}
    for p in csvs:
        parts = p.stem.split("__")
        if len(parts) >= 2:
            model_name = parts[0]
            prev = by_model.get(model_name)
            if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                by_model[model_name] = p

    for model_name, csv_path in sorted(by_model.items()):
        try:
            df = pd.read_csv(csv_path)
            if df.empty or "feature" not in df.columns or "importance" not in df.columns:
                print(f"[skip] bad FI csv for {model_name}: {csv_path.name}")
                continue

            top = df.sort_values("importance", ascending=False).head(20).copy()
            top = top.iloc[::-1]

            fig, ax = plt.subplots(figsize=(10, 6.2))
            ax.barh(top["feature"].astype(str), top["importance"].astype(float))
            ax.set_title(f"Top 20 feature importance — {model_name}")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")

            fig.tight_layout()
            out = FIG_DIR / f"feat_importance_top20__{model_name}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("[ok] wrote", out, "(from", csv_path.name, ")")
        except Exception as e:
            print(f"[skip] {model_name}: failed to plot ({e})")


# -----------------------------
# Main
# -----------------------------
def main():
    print("REPO_ROOT:", REPO_ROOT)
    print("DEMO     :", DEMO)
    print("FIG_DIR  :", FIG_DIR)
    print("SUMMARY  :", SUMMARY_CSV)
    print("REGISTRY :", REGISTRY_DIR)

    # 1) export FI csvs first
    export_registry_feature_importance(topk=30)

    # 2) figures
    fig_price_distribution()
    fig_model_comparison()

    # 3) FI plots
    fig_feature_importance_top20_latest()
    fig_feature_importance_top20_per_model()

    print("Done.")


if __name__ == "__main__":
    main()
