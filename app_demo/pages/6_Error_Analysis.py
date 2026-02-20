from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from lib.ui_style import hero, section

# Demo registry IO
from registry_io_demo import (
    RegistryLayout,
    RunRef,
    get_alias_runref,
    list_model_names,
    load_run_bundle,
    read_aliases,
)

# --------------------------------------------------
# Config
# --------------------------------------------------
ALLOWED_MODELS = {"ridge", "xgb", "lgbm"}  # demo-safe families


# --------------------------------------------------
# Layout bootstrap
# --------------------------------------------------
def _ensure_layout() -> RegistryLayout:
    layout = st.session_state.get("REGISTRY_LAYOUT")
    if layout is None:
        st.warning("Please start from the Home page to initialize the demo assets.")
        with st.expander("Why am I seeing this?", expanded=False):
            st.write(
                "This page reads precomputed evaluation artifacts (OOF predictions, metrics, and reports). "
                "The Home page initializes the artifact layout for the session."
            )
        st.stop()
    return layout


def _get_demo_dir(layout: RegistryLayout) -> Path:
    """
    Resolve the artifacts root that contains:
      - reports/feature_importance
      - reports/figures
      - registry/
    """
    ss = st.session_state
    for k in ("DEMO", "DEMO_DIR", "ARTIFACTS_DEMO", "ARTIFACTS_DIR"):
        v = ss.get(k)
        if isinstance(v, (str, Path)) and Path(v).exists():
            return Path(v)

    v = getattr(layout, "artifacts_dir", None)
    if isinstance(v, Path) and v.exists():
        return v

    reg = getattr(layout, "registry_dir", None)
    if isinstance(reg, Path) and reg.exists():
        parent = reg.parent
        if parent.exists():
            return parent

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "artifacts_demo"


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def _money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "N/A"


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


@st.cache_data(show_spinner=False)
def _load_train(repo_root: Path) -> pd.DataFrame:
    """
    Demo-friendly data loader: prefers a small reproducible sample.
    Priority:
      1) tests/data/sample_train.csv
      2) data/raw/train.csv (local-only fallback)
    """
    p1 = repo_root / "tests" / "data" / "sample_train.csv"
    p2 = repo_root / "data" / "raw" / "train.csv"

    if p1.exists():
        return pd.read_csv(p1)
    if p2.exists():
        return pd.read_csv(p2)

    raise FileNotFoundError("Training data is not available in this build.")


def _detect_log_space(oof: np.ndarray, y_true_price: np.ndarray) -> bool:
    """
    Heuristic: in Ames, log1p(SalePrice) tends to be ~10-13.
    If OOF median is in that band and true prices are large, treat OOF as log-space.
    """
    o = np.asarray(oof).reshape(-1)
    y = np.asarray(y_true_price).reshape(-1)

    if len(o) == 0 or len(y) == 0:
        return True

    o_med = float(np.nanmedian(o))
    y_med = float(np.nanmedian(y))

    if 6.0 <= o_med <= 16.0 and y_med > 1000:
        return True

    return False


def _prepare_error_df(train_df: pd.DataFrame, oof: np.ndarray) -> tuple[pd.DataFrame, bool]:
    df = train_df.copy()
    if "SalePrice" not in df.columns:
        raise RuntimeError("Training data is missing the target column (SalePrice).")

    y_true = pd.to_numeric(df["SalePrice"], errors="coerce").to_numpy()
    oof = np.asarray(oof).reshape(-1)

    n = min(len(y_true), len(oof))
    y_true = y_true[:n]
    oof = oof[:n]
    df = df.iloc[:n].copy()

    is_log = _detect_log_space(oof, y_true)
    y_pred = np.expm1(oof) if is_log else oof

    out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["ape"] = out["abs_error"] / out["y_true"].clip(lower=1.0)

    keep = [
        c
        for c in [
            "Neighborhood",
            "OverallQual",
            "OverallCond",
            "GrLivArea",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "YearBuilt",
            "YrSold",
        ]
        if c in df.columns
    ]
    for c in keep:
        out[c] = df[c].values

    if all(c in out.columns for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        out["TotalSF"] = (
            pd.to_numeric(out["TotalBsmtSF"], errors="coerce").fillna(0)
            + pd.to_numeric(out["1stFlrSF"], errors="coerce").fillna(0)
            + pd.to_numeric(out["2ndFlrSF"], errors="coerce").fillna(0)
        )

    return out, is_log


def _top_segment(err: pd.DataFrame, seg_col: str) -> dict | None:
    if seg_col not in err.columns:
        return None
    g = (
        err.groupby(seg_col, observed=True)
        .agg(
            n=("abs_error", "size"),
            mean_abs=("abs_error", "mean"),
            p90_abs=("abs_error", lambda s: float(pd.Series(s).quantile(0.90))),
            mean_ape=("ape", "mean"),
        )
        .reset_index()
    )
    if g.empty:
        return None
    g = g.sort_values("mean_abs", ascending=False)
    return g.iloc[0].to_dict()


def _spearman_insights(err: pd.DataFrame, candidate_cols: list[str], topk: int = 8) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(err["abs_error"], errors="coerce")
    for c in candidate_cols:
        if c not in err.columns:
            continue
        x = pd.to_numeric(err[c], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 50:
            continue
        corr = x[ok].corr(y[ok], method="spearman")
        rows.append({"feature": c, "spearman_abs_error": float(corr), "n": int(ok.sum())})
    if not rows:
        return pd.DataFrame(columns=["feature", "spearman_abs_error", "n"])
    df = pd.DataFrame(rows)
    df["abs_corr"] = df["spearman_abs_error"].abs()
    df = df.sort_values("abs_corr", ascending=False).head(topk)
    return df.drop(columns=["abs_corr"])


def _load_top_featimp_csv(demo_dir: Path, model_name: str, run_id: str) -> pd.DataFrame | None:
    fi_dir = demo_dir / "reports" / "feature_importance"
    if not fi_dir.exists():
        return None

    exact = fi_dir / f"{model_name}__{run_id}__top30.csv"
    if exact.exists():
        try:
            df = pd.read_csv(exact)
            if "importance" in df.columns:
                df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
            return df
        except Exception:
            return None

    cands = sorted(
        fi_dir.glob(f"{model_name}__*__top30.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        return None

    try:
        df = pd.read_csv(cands[0])
        if "importance" in df.columns:
            df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
        return df
    except Exception:
        return None


def _build_tiers(err: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    # price tiers
    try:
        err["price_tier"] = pd.qcut(
            pd.to_numeric(err["y_true"], errors="coerce"),
            4,
            labels=["Q1 (lower)", "Q2", "Q3", "Q4 (higher)"],
            duplicates="drop",
        )
    except Exception:
        err["price_tier"] = pd.Series([pd.NA] * len(err))

    # size tiers
    size_col = "TotalSF" if "TotalSF" in err.columns else ("GrLivArea" if "GrLivArea" in err.columns else None)
    if size_col is not None:
        try:
            err["size_tier"] = pd.qcut(
                pd.to_numeric(err[size_col], errors="coerce"),
                4,
                labels=["Q1 (smaller)", "Q2", "Q3", "Q4 (larger)"],
                duplicates="drop",
            )
        except Exception:
            err["size_tier"] = pd.Series([pd.NA] * len(err))

    # quality tiers
    if "OverallQual" in err.columns:
        oq = pd.to_numeric(err["OverallQual"], errors="coerce")
        try:
            err["qual_tier"] = pd.cut(
                oq, bins=[0, 4, 6, 10], labels=["Low (1–4)", "Mid (5–6)", "High (7–10)"]
            )
        except Exception:
            err["qual_tier"] = pd.Series([pd.NA] * len(err))

    return err, size_col


def _risk_table(err: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["price_tier", "size_tier", "qual_tier"]:
        if col not in err.columns:
            continue
        g = (
            err.groupby(col, observed=True)
            .agg(
                n=("abs_error", "size"),
                mean_abs=("abs_error", "mean"),
                p90_abs=("abs_error", lambda s: float(pd.Series(s).quantile(0.90))),
                mean_ape=("ape", "mean"),
            )
            .reset_index()
        )
        if len(g):
            g = g.rename(columns={col: "segment"})
            g.insert(0, "group", col)
            rows.append(g)

    # Neighborhood (if enough coverage)
    if "Neighborhood" in err.columns and err["Neighborhood"].notna().mean() > 0.6:
        g = (
            err.groupby("Neighborhood", observed=True)
            .agg(
                n=("abs_error", "size"),
                mean_abs=("abs_error", "mean"),
                p90_abs=("abs_error", lambda s: float(pd.Series(s).quantile(0.90))),
                mean_ape=("ape", "mean"),
            )
            .reset_index()
        )
        g = g[g["n"] >= 10].sort_values("mean_abs", ascending=False)
        if len(g):
            g = g.rename(columns={"Neighborhood": "segment"})
            g.insert(0, "group", "Neighborhood")
            rows.append(g)

    if not rows:
        return pd.DataFrame(columns=["group", "segment", "n", "mean_abs", "p90_abs", "mean_ape"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["mean_abs", "p90_abs"], ascending=False)
    return out


# --------------------------------------------------
# Page
# --------------------------------------------------
layout = _ensure_layout()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = _get_demo_dir(layout)

hero("🚨 Error Analysis", "Where does the selected model miss the most? (OOF view)")

st.info(
    "Demo build note: for fast startup, this page analyzes lightweight model families only "
    f"({', '.join(sorted(ALLOWED_MODELS))}). Full evaluation across all models is covered elsewhere in the app."
)

# --- model + alias ---
model_names_all = list_model_names(layout)
model_names = sorted([m for m in model_names_all if m in ALLOWED_MODELS])

if not model_names:
    st.warning("No deployment-friendly model families are available in this build.")
    st.stop()

section("Select a model", "Choose a model family and version to inspect.")
model_name = st.selectbox("Model family", model_names, index=0)

aliases = read_aliases(layout, model_name)
alias_key = st.selectbox("Version", ["best", "latest"], index=0)

ref: RunRef | None = get_alias_runref(aliases, alias_key, default_model_name=model_name)
if ref is None:
    st.info("This version label is not set for the selected model yet.")
    st.stop()

bundle = load_run_bundle(layout, ref)
run_dir = Path(bundle["run_dir"])

# --- OOF ---
oof_path = run_dir / "oof.npy"
if not oof_path.exists():
    st.warning("This build does not include OOF predictions for the selected run.")
    with st.expander("Details (optional)", expanded=False):
        st.write("Expected file:", str(oof_path))
    st.stop()

oof = np.load(oof_path)

# --- train + error df ---
try:
    train_df = _load_train(REPO_ROOT)
    err, was_log = _prepare_error_df(train_df, oof)
except Exception as e:
    st.warning("Error analysis data is not available in this build.")
    with st.expander("Details (optional)", expanded=False):
        st.write(str(e))
    st.stop()

err, size_col = _build_tiers(err)

# --------------------------------------------------
# 1) Explanation
# --------------------------------------------------
section("What this page shows", "OOF errors are a realistic proxy for performance on new data.")
st.markdown(
    """
- **Out-of-Fold (OOF):** each home is predicted by a model that **did not train on that home**  
- This reduces “self-scoring” and provides a more honest view of model behavior  
- The goal is to highlight **where the model is reliable vs. risky**, not only a single score
""".strip()
)

if was_log:
    st.caption("Scale note: OOF predictions appear to be stored in **log-space** and are converted back to **dollars** on this page.")
else:
    st.caption("Scale note: OOF predictions appear to be stored in **dollar scale** on this page.")

# --------------------------------------------------
# 2) KPIs
# --------------------------------------------------
rmse = _rmse(err["y_true"], err["y_pred"])
mae = _mae(err["y_true"], err["y_pred"])
p90 = float(err["abs_error"].quantile(0.90))
p95 = float(err["abs_error"].quantile(0.95))

c1, c2, c3, c4 = st.columns(4)
c1.metric("OOF RMSE", _money(rmse))
c2.metric("OOF MAE", _money(mae))
c3.metric("90th % abs error", _money(p90))
c4.metric("95th % abs error", _money(p95))
st.caption("MAE ≈ typical miss. 90/95% highlight the scale of large misses.")

st.divider()

# --------------------------------------------------
# 3) Risk flags
# --------------------------------------------------
section("Risk flags", "Plain-language summary of where this model is most risky.", "⚠️")

risk_items = []
r = _top_segment(err, "price_tier")
if r:
    risk_items.append(("Price tier", r["price_tier"], r["n"], r["mean_abs"], r["p90_abs"]))
r = _top_segment(err, "size_tier")
if r:
    risk_items.append(("Size tier", r["size_tier"], r["n"], r["mean_abs"], r["p90_abs"]))
r = _top_segment(err, "qual_tier")
if r:
    risk_items.append(("Quality tier", r["qual_tier"], r["n"], r["mean_abs"], r["p90_abs"]))

# Neighborhood risk (only if meaningful coverage)
if "Neighborhood" in err.columns and err["Neighborhood"].notna().mean() > 0.6:
    g = (
        err.groupby("Neighborhood", observed=True)
        .agg(
            n=("abs_error", "size"),
            mean_abs=("abs_error", "mean"),
            p90_abs=("abs_error", lambda s: float(pd.Series(s).quantile(0.90))),
        )
        .reset_index()
    )
    g = g[g["n"] >= 10].sort_values("mean_abs", ascending=False)
    if len(g) > 0:
        row = g.iloc[0]
        risk_items.append(("Neighborhood", row["Neighborhood"], int(row["n"]), float(row["mean_abs"]), float(row["p90_abs"])))

if not risk_items:
    st.info("Not enough segment information is available to generate risk flags.")
else:
    cards = st.columns(min(3, len(risk_items)))
    for i, (group, seg, n, mean_abs, p90_abs) in enumerate(risk_items[:3]):
        with cards[i]:
            st.metric(f"Highest-risk {group}", str(seg))
            st.caption(f"avg miss ≈ {_money(mean_abs)} • 90% ≤ {_money(p90_abs)} • n={int(n)}")

    with st.expander("Show all segments (details)", expanded=False):
        rt = _risk_table(err).copy()
        if not rt.empty:
            rt["mean_abs"] = rt["mean_abs"].map(_money)
            rt["p90_abs"] = rt["p90_abs"].map(_money)
            rt["mean_ape"] = (rt["mean_ape"] * 100).round(2).astype(str) + "%"
        st.dataframe(rt, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# 4) Why it happens
# --------------------------------------------------
section("Why these misses happen", "Connect error patterns to model-relevant signals.", "🧠")

num_candidates = [
    c for c in ["GrLivArea", "TotalSF", "OverallQual", "OverallCond", "YearBuilt", "YrSold", "TotalBsmtSF"]
    if c in err.columns
]
corr_df = _spearman_insights(err, num_candidates, topk=6)

colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.markdown("**Which factors are associated with larger errors?**")
    if corr_df.empty:
        st.info("Not enough numeric fields are available to compute correlations.")
    else:
        show = corr_df.copy()
        show["Interpretation"] = np.where(
            show["spearman_abs_error"] >= 0,
            "Higher values tend to align with larger errors",
            "Lower values tend to align with larger errors",
        )
        show["spearman_abs_error"] = show["spearman_abs_error"].round(3)
        st.dataframe(show[["feature", "spearman_abs_error", "Interpretation", "n"]], use_container_width=True)

with colR:
    st.markdown("**What the model relies on (top features)**")
    fi_df = _load_top_featimp_csv(DEMO, model_name, ref.run_id)
    if fi_df is None or fi_df.empty or "feature" not in fi_df.columns:
        st.info("Feature importance is not available for this run in the demo reports.")
    else:
        topk = min(12, len(fi_df))
        show = fi_df.sort_values("importance", ascending=False).head(topk).copy()
        if "importance" in show.columns:
            show["importance"] = pd.to_numeric(show["importance"], errors="coerce").fillna(0.0)
        st.dataframe(show[["feature", "importance"]], use_container_width=True)

st.divider()

# --------------------------------------------------
# 5) Overall behavior charts
# --------------------------------------------------
section("Overall behavior", "Does the model track reality, and how are errors distributed?", "📈")

colA, colB = st.columns(2, gap="large")

with colA:
    fig, ax = plt.subplots()
    ax.scatter(err["y_true"], err["y_pred"], alpha=0.35)
    lo = float(min(err["y_true"].min(), err["y_pred"].min()))
    hi = float(max(err["y_true"].max(), err["y_pred"].max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title("Actual vs Predicted (OOF)")
    st.pyplot(fig)
    plt.close(fig)

with colB:
    fig, ax = plt.subplots()
    ax.hist(err["error"], bins=60)
    ax.set_title("Residual distribution (Pred − Actual)")
    ax.set_xlabel("Error ($)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# --------------------------------------------------
# 6) Segment breakdown
# --------------------------------------------------
section("Where it struggles", "Error by price tier, size tier, and quality tier.", "🔍")


def segment_table(col: str) -> pd.DataFrame:
    g = err.groupby(col, observed=True)["abs_error"].agg(["count", "mean", "median"]).reset_index()
    g = g.rename(columns={"mean": "mean_abs_error", "median": "median_abs_error"})
    g = g.sort_values("mean_abs_error", ascending=False)
    return g


def segment_bar(g: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots()
    ax.bar(g[col].astype(str), g["mean_abs_error"].astype(float))
    ax.set_title(title)
    ax.set_ylabel("Mean absolute error ($)")
    ax.set_xlabel("")
    # Optional labels (safe across matplotlib versions)
    try:
        ax.bar_label(ax.containers[0], fmt="%.0f", padding=2)
    except Exception:
        pass
    return fig


a, b, c = st.columns(3, gap="large")

with a:
    if "price_tier" not in err.columns:
        st.info("Price tiers are not available.")
    else:
        g = segment_table("price_tier")
        fig = segment_bar(g, "price_tier", "Error by price tier")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(g, use_container_width=True)

with b:
    if "size_tier" not in err.columns or size_col is None:
        st.info("Size tiers are not available.")
    else:
        g = segment_table("size_tier")
        fig = segment_bar(g, "size_tier", f"Error by size tier ({size_col})")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(g, use_container_width=True)

with c:
    if "qual_tier" not in err.columns:
        st.info("Quality tiers are not available.")
    else:
        g = segment_table("qual_tier")
        fig = segment_bar(g, "qual_tier", "Error by quality tier (OverallQual)")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(g, use_container_width=True)

st.divider()

# --------------------------------------------------
# 7) Worst cases
# --------------------------------------------------
section("Hardest cases", "The largest misses — useful for boundary thinking and future improvements.", "🚩")

topn = st.slider("Show top N worst cases", 10, 100, 20, step=10)
cols_show = [c for c in ["Neighborhood", "OverallQual", "OverallCond", "GrLivArea", "TotalSF", "YearBuilt", "YrSold"] if c in err.columns]

worst = err.sort_values("abs_error", ascending=False).head(int(topn)).copy()
worst["Actual"] = worst["y_true"].map(_money)
worst["Predicted"] = worst["y_pred"].map(_money)
worst["AbsError"] = worst["abs_error"].map(_money)
worst["Error%"] = (worst["ape"] * 100).round(1).astype(str) + "%"

st.dataframe(worst[["Actual", "Predicted", "AbsError", "Error%"] + cols_show], use_container_width=True)

with st.expander("Technical details (optional)", expanded=False):
    st.write("Model:", model_name)
    st.write("Version:", alias_key)
    st.write("Run:", ref.run_id)
    st.write("Rows analyzed:", len(err))
