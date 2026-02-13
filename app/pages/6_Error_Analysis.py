from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from lib.ui_style import hero, section

from registry_io import (
    RegistryLayout,
    RunRef,
    get_alias_runref,
    list_model_names,
    load_run_bundle,
    read_aliases,
)


# -----------------------------
# Layout bootstrap
# -----------------------------
def _ensure_layout() -> RegistryLayout:
    layout = st.session_state.get("REGISTRY_LAYOUT")
    if layout is None:
        st.error("Registry layout not initialized. Please open the Home page first (app/app.py).")
        st.stop()
    return layout


def _get_demo_dir(layout: RegistryLayout) -> Path:
    """
    Robustly find the demo/artifacts root for reports/feature_importance.

    Tries (in order):
      1) st.session_state["DEMO"] or ["DEMO_DIR"]
      2) layout.demo_dir / layout.DEMO / layout.artifacts_demo (common patterns)
      3) layout.artifacts_dir (often points to artifacts_demo)
      4) infer from registry_dir (artifacts_demo/registry -> artifacts_demo)
    """
    ss = st.session_state
    for k in ("DEMO", "DEMO_DIR", "ARTIFACTS_DEMO", "ARTIFACTS_DIR"):
        v = ss.get(k)
        if isinstance(v, (str, Path)) and Path(v).exists():
            return Path(v)

    # attribute-based
    for attr in ("demo_dir", "DEMO", "artifacts_demo", "artifacts_dir"):
        v = getattr(layout, attr, None)
        if isinstance(v, Path) and v.exists():
            return v

    # infer from registry_dir if available
    reg = getattr(layout, "registry_dir", None)
    if isinstance(reg, Path) and reg.exists():
        # registry is usually: <artifacts_demo>/registry
        parent = reg.parent
        if parent.exists():
            return parent

    # last resort: try common relative to this page
    repo_root = Path(__file__).resolve().parents[2]
    guess = repo_root / "artifacts_demo"
    return guess


# -----------------------------
# Utilities
# -----------------------------
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


def _load_train(repo_root: Path) -> pd.DataFrame:
    p = repo_root / "data" / "raw" / "train.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing train.csv: {p}")
    return pd.read_csv(p)


def _detect_log_space(oof: np.ndarray, y_true_price: np.ndarray) -> bool:
    """
    Heuristic:
    - If oof values look like ~[10, 14] and prices are ~[50k, 500k], it's log1p space.
    """
    o = np.asarray(oof).reshape(-1)
    y = np.asarray(y_true_price).reshape(-1)

    if len(o) == 0 or len(y) == 0:
        return True

    o_med = float(np.nanmedian(o))
    y_med = float(np.nanmedian(y))

    # log1p prices: around 10~13
    if 6.0 <= o_med <= 16.0 and y_med > 1000:
        return True

    # if oof already in dollars, it will be same scale as y
    return False


def _prepare_error_df(train_df: pd.DataFrame, oof: np.ndarray) -> pd.DataFrame:
    """
    Create analysis dataframe:
      - y_true (dollars)
      - y_pred (dollars)
      - error, abs_error, ape
      - attach a subset of helpful fields if exist
      - compute engineered TotalSF if possible
    """
    df = train_df.copy()
    if "SalePrice" not in df.columns:
        raise RuntimeError("train.csv missing SalePrice")

    y_true = pd.to_numeric(df["SalePrice"], errors="coerce").to_numpy()
    oof = np.asarray(oof).reshape(-1)

    # align length
    n = min(len(y_true), len(oof))
    y_true = y_true[:n]
    oof = oof[:n]
    df = df.iloc[:n].copy()

    # detect log-space and convert to dollars
    is_log = _detect_log_space(oof, y_true)
    if is_log:
        y_pred = np.expm1(oof)
    else:
        y_pred = oof

    out = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["ape"] = out["abs_error"] / out["y_true"].clip(lower=1.0)

    # attach helpful columns if exist
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

    # compute TotalSF if possible
    if all(c in out.columns for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        out["TotalSF"] = (
            pd.to_numeric(out["TotalBsmtSF"], errors="coerce").fillna(0)
            + pd.to_numeric(out["1stFlrSF"], errors="coerce").fillna(0)
            + pd.to_numeric(out["2ndFlrSF"], errors="coerce").fillna(0)
        )

    return out


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "N/A"


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
    """
    Load top30 feature importance CSV from artifacts_demo/reports/feature_importance.

    Prefer the CSV matching current run:
      <model>__<run_id>__top30.csv

    Fallback:
      latest modified <model>__*__top30.csv
    """
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

    cands = sorted(fi_dir.glob(f"{model_name}__*__top30.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
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
    """
    Create tier columns ONCE before risk flags and segment charts.

    Returns:
      (err, size_col)
    """
    # price tier (use qcut; duplicates drop avoids crash if many identical prices)
    try:
        err["price_tier"] = pd.qcut(
            pd.to_numeric(err["y_true"], errors="coerce"),
            4,
            labels=["Q1 (lower)", "Q2", "Q3", "Q4 (higher)"],
            duplicates="drop",
        )
    except Exception:
        # if qcut fails, keep as missing
        err["price_tier"] = pd.Series([pd.NA] * len(err))

    # size tier
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

    # quality tier
    if "OverallQual" in err.columns:
        oq = pd.to_numeric(err["OverallQual"], errors="coerce")
        try:
            err["qual_tier"] = pd.cut(oq, bins=[0, 4, 6, 10], labels=["Low (1–4)", "Mid (5–6)", "High (7–10)"])
        except Exception:
            err["qual_tier"] = pd.Series([pd.NA] * len(err))

    return err, size_col


# -----------------------------
# Page
# -----------------------------
layout = _ensure_layout()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = _get_demo_dir(layout)

hero("🧯 Error Analysis", "Where does the *best* model miss the most? (OOF, no leakage)")

# --- pick model + alias ---
model_names = list_model_names(layout)
if not model_names:
    st.warning("No model families found in registry. Check artifacts_demo/registry.")
    st.stop()

model_name = st.selectbox("Model family", model_names, index=0)
aliases = read_aliases(layout, model_name)
alias_key = st.selectbox("Model version", ["best", "latest"], index=0)

ref: RunRef | None = get_alias_runref(aliases, alias_key, default_model_name=model_name)
if ref is None:
    st.warning("Alias not set for this model family.")
    st.stop()

bundle = load_run_bundle(layout, ref)
run_dir = Path(bundle["run_dir"])

# --- load OOF ---
oof_path = run_dir / "oof.npy"
if not oof_path.exists():
    st.error(f"Missing OOF file: {oof_path}")
    st.stop()

oof = np.load(oof_path)

# --- load train.csv + build error df ---
try:
    train_df = _load_train(REPO_ROOT)
    err = _prepare_error_df(train_df, oof)
except Exception as e:
    st.error(f"Failed to build error table: {e}")
    st.stop()

# --- build tiers ONCE (FIX: risk flags needs these columns) ---
err, size_col = _build_tiers(err)

# -----------------------------
# 1) Explain (product tone)
# -----------------------------
section("What this page shows", "OOF errors are a realistic proxy for performance on new data.", "🧾")
st.markdown(
    """
- **Out-of-Fold (OOF)** means: each row is predicted by a model that **did not train on that row**.
- This avoids “self-scoring” and gives a more honest view of error patterns.
- The goal is not only a single score — it’s to show **where the model is reliable vs risky**.
""".strip()
)

# -----------------------------
# 2) KPIs
# -----------------------------
rmse = _rmse(err["y_true"], err["y_pred"])
mae = _mae(err["y_true"], err["y_pred"])
p90 = float(err["abs_error"].quantile(0.90))
p95 = float(err["abs_error"].quantile(0.95))

c1, c2, c3, c4 = st.columns(4)
c1.metric("OOF RMSE", _money(rmse))
c2.metric("OOF MAE", _money(mae))
c3.metric("90th % abs error", _money(p90))
c4.metric("95th % abs error", _money(p95))

st.caption("MAE≈ typical miss in dollars. 90/95% show how bad large misses can get.")

st.divider()

# =============================
# Upgrade A: Risk Flags (product)
# =============================
section("Risk flags (automatic)", "Plain-language summary: where the model is most risky.", "⚠️")

risk_items = []

# price tier
r = _top_segment(err, "price_tier")
if r:
    risk_items.append(("Price tier", r["price_tier"], r["n"], r["mean_abs"], r["p90_abs"], r["mean_ape"]))

# size tier
r = _top_segment(err, "size_tier")
if r:
    risk_items.append(("Size tier", r["size_tier"], r["n"], r["mean_abs"], r["p90_abs"], r["mean_ape"]))

# quality tier
r = _top_segment(err, "qual_tier")
if r:
    risk_items.append(("Quality tier", r["qual_tier"], r["n"], r["mean_abs"], r["p90_abs"], r["mean_ape"]))

# neighborhood (only if exists + not too many missing)
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
    g = g[g["n"] >= 25].sort_values("mean_abs", ascending=False)  # avoid tiny-sample noise
    if len(g) > 0:
        row = g.iloc[0]
        risk_items.append(
            ("Neighborhood (>=25 homes)", row["Neighborhood"], int(row["n"]), float(row["mean_abs"]), float(row["p90_abs"]), np.nan)
        )

if not risk_items:
    st.info("Not enough segment columns found to produce risk flags.")
else:
    cards = st.columns(min(3, len(risk_items)))
    for i, (group, seg, n, mean_abs, p90_abs, mean_ape) in enumerate(risk_items[:3]):
        with cards[i]:
            st.metric(f"{group} most risky", str(seg))
            st.caption(f"avg miss ≈ {_money(mean_abs)} • 90% ≤ {_money(p90_abs)} • n={int(n)}")

    with st.expander("All risk flags (details)", expanded=False):
        out_rows = []
        for (group, seg, n, mean_abs, p90_abs, mean_ape) in risk_items:
            out_rows.append(
                {
                    "Group": group,
                    "Worst segment": str(seg),
                    "n": int(n),
                    "Mean abs error": float(mean_abs),
                    "90% abs error": float(p90_abs),
                    "Mean error %": float(mean_ape) if pd.notna(mean_ape) else np.nan,
                }
            )
        df_flags = pd.DataFrame(out_rows)
        df_flags["Mean abs error"] = df_flags["Mean abs error"].map(_money)
        df_flags["90% abs error"] = df_flags["90% abs error"].map(_money)
        df_flags["Mean error %"] = df_flags["Mean error %"].map(_fmt_pct)
        st.dataframe(df_flags, width="stretch")

st.markdown("---")

# =============================
# Upgrade B: Why it happens (explainability)
# =============================
section("Why these misses happen", "Link error patterns to model-relevant signals.", "🧠")

num_candidates = [c for c in ["GrLivArea", "TotalSF", "OverallQual", "OverallCond", "YearBuilt", "YrSold", "TotalBsmtSF"] if c in err.columns]
corr_df = _spearman_insights(err, num_candidates, topk=6)

colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.markdown("**Which factors relate most to large errors?**")
    if corr_df.empty:
        st.info("Not enough numeric fields to compute correlations.")
    else:
        show = corr_df.copy()
        show["Direction"] = np.where(show["spearman_abs_error"] >= 0, "↑ larger value → larger error", "↓ larger value → larger error")
        show["spearman_abs_error"] = show["spearman_abs_error"].round(3)
        st.dataframe(show[["feature", "spearman_abs_error", "Direction", "n"]], width="stretch")
        st.caption("Spearman is rank-based, so it captures monotonic patterns even when relationships are not linear.")

with colR:
    st.markdown("**What the model relies on (top features, if available)**")
    fi_df = _load_top_featimp_csv(DEMO, model_name, ref.run_id)
    if fi_df is None or fi_df.empty or "feature" not in fi_df.columns:
        st.info("No feature importance CSV found for this model in demo reports.")
        st.caption(f"Looked under: {DEMO / 'reports' / 'feature_importance'}")
    else:
        topk = min(12, len(fi_df))
        show = fi_df.sort_values("importance", ascending=False).head(topk).copy()
        if "importance" in show.columns:
            show["importance"] = pd.to_numeric(show["importance"], errors="coerce").fillna(0.0)
        st.dataframe(show[["feature", "importance"]], width="stretch")
        st.caption(f"Run-matched: {model_name}__{ref.run_id}__top30.csv (or latest fallback).")

st.markdown("**Plain-language takeaway**")
takeaways = []

worst_price = _top_segment(err, "price_tier")
if worst_price:
    takeaways.append(
        f"- **Hardest price tier:** `{worst_price['price_tier']}` (avg miss ≈ {_money(worst_price['mean_abs'])}). "
        "High-end homes are often a long-tail segment → fewer examples + more unique upgrades."
    )

if "OverallQual" in err.columns:
    takeaways.append(
        "- **Quality & size interactions matter:** a large home with high quality can jump price non-linearly. "
        "That’s why features like `OverallQual`, `GrLivArea/TotalSF` often drive both predictions and errors."
    )

if "Neighborhood" in err.columns:
    takeaways.append(
        "- **Neighborhood premium is real:** two similar houses can price very differently by location. "
        "If a neighborhood is rare (few sales), models can struggle more there."
    )

if corr_df is not None and not corr_df.empty:
    top_feat = str(corr_df.iloc[0]["feature"])
    takeaways.append(
        f"- **Error is most sensitive to:** `{top_feat}` — meaning extreme values of this factor tend to be where the model is less certain."
    )

st.markdown("\n".join(takeaways) if takeaways else "- No automatic narrative available for current selection.")

# -----------------------------
# 3) Overall behavior charts
# -----------------------------
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
    ax.set_title("Residual distribution (Pred - Actual)")
    ax.set_xlabel("Error ($)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# -----------------------------
# 4) Segment breakdown (audience-friendly)
# -----------------------------
section("Where it struggles", "Error by price tier, size tier, and quality tier.", "🔍")


def segment_table(col: str) -> pd.DataFrame:
    g = err.groupby(col, observed=True)["abs_error"].agg(["count", "mean", "median"]).reset_index()
    g = g.rename(columns={"mean": "mean_abs_error", "median": "median_abs_error"})
    return g


def segment_bar(g: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots()
    ax.bar(g[col].astype(str), g["mean_abs_error"].astype(float))
    ax.set_title(title)
    ax.set_ylabel("Mean absolute error ($)")
    ax.set_xlabel("")
    return fig


a, b, c = st.columns(3, gap="large")

with a:
    if "price_tier" not in err.columns:
        st.info("price_tier not available.")
    else:
        g = segment_table("price_tier")
        fig = segment_bar(g, "price_tier", "Error by price tier")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(g, width="stretch")

with b:
    if "size_tier" not in err.columns or size_col is None:
        st.info("No TotalSF/GrLivArea available for size tiers.")
    else:
        g = segment_table("size_tier")
        fig = segment_bar(g, "size_tier", f"Error by size tier ({size_col})")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(g, width="stretch")

with c:
    if "qual_tier" not in err.columns:
        st.info("No OverallQual available for quality tiers.")
    else:
        g = segment_table("qual_tier")
        fig = segment_bar(g, "qual_tier", "Error by quality tier (OverallQual)")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(g, width="stretch")

st.divider()

# -----------------------------
# 5) Worst cases (actionable)
# -----------------------------
section("Hardest cases", "Which homes were the biggest misses?", "🚨")

topn = st.slider("Show top N worst cases", 10, 100, 20, step=10)
cols_show = [c for c in ["Neighborhood", "OverallQual", "OverallCond", "GrLivArea", "TotalSF", "YearBuilt", "YrSold"] if c in err.columns]

worst = err.sort_values("abs_error", ascending=False).head(int(topn)).copy()
worst["Actual"] = worst["y_true"].map(_money)
worst["Predicted"] = worst["y_pred"].map(_money)
worst["AbsError"] = worst["abs_error"].map(_money)
worst["Error%"] = (worst["ape"] * 100).round(1).astype(str) + "%"

st.dataframe(worst[["Actual", "Predicted", "AbsError", "Error%"] + cols_show], width="stretch")

with st.expander("Technical details (optional)", expanded=False):
    st.write("Run used:", ref.run_id)
    st.write("Run dir:", str(run_dir))
    st.write("OOF file:", str(oof_path))
    st.write("Demo dir:", str(DEMO))
    st.write("Rows:", len(err))
