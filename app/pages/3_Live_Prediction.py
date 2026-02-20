# app/pages/3_Live_Prediction.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app.lib.ui_style import hero, section
from registry_io import (
    RegistryLayout,
    RunRef,
    get_alias_runref,
    list_model_names,
    load_run_bundle,
    read_aliases,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def apply_derivations(row: dict):
    # internal derived fields (hidden from users)
    for k in list(row.keys()):
        if k.endswith("_log") and row.get(k) is None:
            base = k[:-4]
            v = row.get(base)
            row[k] = float(np.log1p(max(v, 0))) if isinstance(v, (int, float)) else 0.0

    yr_sold = row.get("YrSold")
    if isinstance(yr_sold, (int, float)):
        if row.get("HouseAge") is None and isinstance(row.get("YearBuilt"), (int, float)):
            row["HouseAge"] = float(max(0, yr_sold - row["YearBuilt"]))
        if row.get("RemodAge") is None and isinstance(row.get("YearRemodAdd"), (int, float)):
            row["RemodAge"] = float(max(0, yr_sold - row["YearRemodAdd"]))

    for k in list(row.keys()):
        if k.endswith("_Qual") and row.get(k) is None:
            row[k] = 0.0


def build_input_df(user_fields: dict, expected_cols: list, defaults: dict) -> pd.DataFrame:
    row = {c: defaults.get(c, None) for c in expected_cols}
    row.update(user_fields)
    apply_derivations(row)
    return pd.DataFrame([row], columns=expected_cols)


def _ensure_layout() -> RegistryLayout:
    """
    Ensure registry layout exists even if user opens this page first.
    Preference order:
      - st.session_state['REGISTRY_LAYOUT']
      - repo_root/artifacts_demo
      - repo_root/artifacts
    """
    layout = st.session_state.get("REGISTRY_LAYOUT")
    if isinstance(layout, RegistryLayout):
        return layout

    app_dir = Path(__file__).resolve().parents[1]  # .../app
    repo_root = app_dir.parent

    # you have both artifacts_demo and artifacts in repo root
    candidates = [
        repo_root / "artifacts_demo",
        repo_root / "artifacts",
    ]
    artifacts_dir = next((p for p in candidates if p.exists()), repo_root / "artifacts_demo")

    layout = RegistryLayout(artifacts_dir=artifacts_dir)
    st.session_state["REGISTRY_LAYOUT"] = layout
    return layout


def _pick_model_file(layout: RegistryLayout, ref: RunRef, bundle: dict) -> Path:
    # prefer artifacts/current/<model>.joblib
    p = Path(layout.current_model_path(ref.model_name))
    if p.exists():
        return p

    # fallback to run folder model.joblib
    bp = bundle.get("model_path")
    if isinstance(bp, str) and bp:
        p2 = Path(bp)
        if p2.exists():
            return p2

    return p


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------

layout = _ensure_layout()

hero("🧪 Live Price Prediction", "A guided interface. The system fills technical inputs automatically.")

# Optional diagnostics for you (collapsed)
with st.expander("🔧 System paths (optional)", expanded=False):
    st.write("Artifacts dir:", str(layout.artifacts_dir))
    st.write("Registry dir:", str(layout.registry_dir))
    st.write("Current dir:", str(layout.current_dir))
    st.write("Reports dir:", str(layout.reports_dir))

# --- Model picker ---
model_names = list_model_names(layout)
if not model_names:
    st.error(
        "No models found in the registry.\n\n"
        f"Expected: {layout.registry_dir}\n\n"
        "Tip: For the full app, you probably want `artifacts/registry/...`."
    )
    st.stop()

model_name = st.selectbox("Model version family", model_names, index=0)

aliases = read_aliases(layout, model_name)
alias_key = st.selectbox("Select model version", ["best", "latest"], index=0)
ref = get_alias_runref(aliases, alias_key, default_model_name=model_name)

if ref is None:
    st.warning("This model version is not set yet. Please check aliases.json for this family.")
    st.stop()

bundle = load_run_bundle(layout, ref)
run_dir = Path(bundle["run_dir"])

model_path = _pick_model_file(layout, ref, bundle)
if not model_path.exists():
    st.error(
        "Model file not found.\n\n"
        f"Tried:\n- {layout.current_model_path(ref.model_name)}\n"
        f"- {bundle.get('model_path')}\n\n"
        "Fix: ensure artifacts/current/<model>.joblib OR registry/<model>/<run>/model.joblib exists."
    )
    st.stop()

model = joblib.load(model_path)

# internal schema files
feature_cols_path = run_dir / "feature_columns.json"
defaults_path = run_dir / "defaults.json"

if not feature_cols_path.exists() or not defaults_path.exists():
    st.error(
        "Schema files missing for this run.\n\n"
        f"Expected:\n- {feature_cols_path}\n- {defaults_path}\n\n"
        "Tip: Your `artifacts_demo` has these, but `artifacts` (full) may not."
    )
    st.stop()

expected_cols = load_json(feature_cols_path)
defaults = load_json(defaults_path)


# ------------------------------------------------------------
# Model summary (product-facing)
# ------------------------------------------------------------
section("Model summary", "Designed for non-technical viewers.", "🧾")
m = bundle.get("metrics", {}) or {}
fp = bundle.get("data_fingerprint", {}) or {}

colA, colB, colC, colD = st.columns(4)

with colA:
    st.metric("Model family", ref.model_name)
with colB:
    st.metric("Selected version", alias_key.upper())
with colC:
    score = m.get("cv_rmse")
    st.metric("Quality score (lower is better)", f"{score:.6f}" if isinstance(score, (int, float)) else "N/A")
with colD:
    rows = fp.get("X", {}).get("rows")
    cols = fp.get("X", {}).get("cols")
    st.metric("Training data", f"{rows} rows / {cols} fields" if rows and cols else "N/A")

st.markdown("---")
section("Tell us about the house", "Inputs are grouped by how people think about homes.", "🏠")

st.info(
    "Demo mode note: This interactive prediction page uses lightweight models for fast startup. "
    "Full model evaluation (including Stacking and XGBoost) is shown in **Model Behavior** and **Experiments & Analysis**."
)

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
with st.expander("🏡 Essentials (required)", expanded=True):
    overall_qual = st.slider("Overall build quality", 1, 10, 5)
    overall_cond = st.slider("Overall condition", 1, 10, 5)
    living_area = st.slider("Living area (sqft)", 300, 6000, 1500, step=10)
    basement = st.slider("Basement area (sqft)", 0, 3000, 800, step=10)
    garage_cars = st.selectbox("Garage capacity (cars)", [0, 1, 2, 3, 4], index=2)
    year_built = st.slider("Year built", 1870, 2010, 1970)
    year_sold = st.slider("Year sold", 2006, 2010, 2008)

with st.expander("🛏️ Layout (recommended)", expanded=False):
    bedrooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5, 6], index=3)
    total_rooms = st.selectbox("Total rooms (above ground)", list(range(2, 13)), index=4)
    full_bath = st.selectbox("Full bathrooms", [0, 1, 2, 3, 4], index=1)
    half_bath = st.selectbox("Half bathrooms", [0, 1, 2], index=0)

with st.expander("📍 Location & lot (optional)", expanded=False):
    lot_area = st.slider("Lot size (sqft)", 1000, 50000, 9000, step=50)
    central_air = st.checkbox("Central air", value=True)
    nb_default = defaults.get("Neighborhood", "NAmes")
    neighborhood = st.text_input(
        "Neighborhood (optional)",
        value=str(nb_default),
        help="If unknown, keep the default."
    )

with st.expander("⚙️ Advanced assumptions (optional)", expanded=False):
    st.caption("For most users, leaving these as default is recommended.")
    remodeled = st.selectbox("Recently remodeled?", ["Unknown / default", "Yes", "No"], index=0)
    has_fireplace = st.selectbox("Has fireplace?", ["Unknown / default", "Yes", "No"], index=0)

# map to model fields
user_fields = {
    "OverallQual": overall_qual,
    "OverallCond": overall_cond,
    "GrLivArea": living_area,
    "TotalBsmtSF": basement,
    "GarageCars": garage_cars,
    "YearBuilt": year_built,
    "YrSold": year_sold,
    "BedroomAbvGr": bedrooms,
    "TotRmsAbvGrd": total_rooms,
    "FullBath": full_bath,
    "HalfBath": half_bath,
    "LotArea": lot_area,
    "CentralAir": "Y" if central_air else "N",
    "Neighborhood": neighborhood,
}

if remodeled != "Unknown / default":
    user_fields["IsRemodeled"] = 1 if remodeled == "Yes" else 0

if has_fireplace != "Unknown / default":
    user_fields["HasFireplace"] = 1 if has_fireplace == "Yes" else 0

st.markdown("---")
section("Get an estimate", "The system will generate a price estimate and show the version used.", "🧾")

if st.button("Estimate price"):
    X = build_input_df(user_fields, expected_cols, defaults)
    yhat = float(np.asarray(model.predict(X)).ravel()[0])
    est_price = float(np.expm1(yhat))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"Estimated sale price: **${est_price:,.0f}**")
        st.caption("This is an automated estimate based on historical patterns in the training data.")
    with col2:
        st.info("Model version used")
        st.write(f"- family: `{ref.model_name}`")
        st.write(f"- alias: `{alias_key}`")
        st.write(f"- run: `{ref.run_id}`")

    with st.expander("Technical details (optional)", expanded=False):
        st.write("Internal prediction value (model output):", yhat)
        st.write("Input row sent to the model:")
        st.dataframe(X, use_container_width=True)

def _reliability_from_inputs(user_fields: dict, defaults: dict) -> tuple[str, int]:
    """
    Heuristic reliability score based on how far key inputs are from training defaults.
    Returns: (label, score_0_to_100)
    """
    keys = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt", "LotArea"]
    deltas = []

    for k in keys:
        u = user_fields.get(k, None)
        d = defaults.get(k, None)
        if u is None or d is None:
            continue
        if isinstance(u, (int, float)) and isinstance(d, (int, float)):
            # scale by a soft denominator so one feature doesn't dominate
            denom = max(1.0, abs(float(d)))
            deltas.append(min(3.0, abs(float(u) - float(d)) / denom))

    if not deltas:
        return ("Medium", 60)

    avg = float(np.mean(deltas))
    # map avg delta to score
    score = int(max(0, min(100, 100 * (1.0 - avg / 1.5))))

    if score >= 75:
        return ("High", score)
    if score >= 50:
        return ("Medium", score)
    return ("Low", score)


def _typical_range_from_cv_rmse(est_price: float, cv_rmse_log: float) -> tuple[float, float]:
    """
    Convert log-space RMSE to a multiplicative typical error band around est_price.
    NOTE: This is NOT a formal prediction interval. It's an error-informed typical range.
    """
    factor = float(np.exp(cv_rmse_log))
    lo = est_price / factor
    hi = est_price * factor
    return lo, hi


# --- After est_price is computed ---
cv_rmse = m.get("cv_rmse") if isinstance(m, dict) else None

label, score = _reliability_from_inputs(user_fields, defaults)

col1, col2 = st.columns([2, 1])
with col1:
    st.success(f"Estimated sale price: **${est_price:,.0f}**")

    if isinstance(cv_rmse, (int, float)) and cv_rmse > 0:
        lo, hi = _typical_range_from_cv_rmse(est_price, float(cv_rmse))
        st.caption(f"Typical range (CV-informed): **${lo:,.0f} – ${hi:,.0f}**")
        st.caption("Note: this is an error-informed typical band, not a formal prediction interval.")
    else:
        st.caption("Typical range is unavailable (CV RMSE not found for this run).")

with col2:
    st.markdown("**Reliability (demo heuristic)**")
    st.progress(score)
    st.write(f"**{label}** ({score}/100)")
    st.caption("Based on how typical the inputs are relative to training defaults.")
