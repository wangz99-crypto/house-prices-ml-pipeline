# app_demo/pages/3_Live_Prediction_Demo.py
# Demo-only: real inference using lightweight models only (ridge, xgb, lgbm).
# No huge ensemble artifacts. Instant startup: load model only on button click.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from lib.ui_style import hero, section
from registry_io import (
    RegistryLayout,
    RunRef,
    get_alias_runref,
    list_model_names,
    load_run_bundle,
    read_aliases,
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

ALLOWED_MODELS = {"ridge", "xgb", "lgbm"}


def _ensure_layout() -> RegistryLayout:
    layout = st.session_state.get("REGISTRY_LAYOUT")
    if layout is None:
        st.error("Registry layout not initialized. Please open Home page first.")
        st.stop()
    return layout


def load_json(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def apply_derivations(row: dict):
    # keep consistent with full build
    for k in list(row.keys()):
        if k.endswith("_log") and row.get(k) is None:
            base = k[:-4]
            v = row.get(base)
            row[k] = float(np.log1p(max(v, 0))) if isinstance(v, (int, float)) else 0.0

    yr = row.get("YrSold")
    if isinstance(yr, (int, float)):
        if isinstance(row.get("YearBuilt"), (int, float)):
            row["HouseAge"] = max(0, yr - row["YearBuilt"])
        if isinstance(row.get("YearRemodAdd"), (int, float)):
            row["RemodAge"] = max(0, yr - row["YearRemodAdd"])


def build_input_df(user_fields, expected_cols, defaults):
    row = {c: defaults.get(c, None) for c in expected_cols}
    row.update(user_fields)
    apply_derivations(row)
    return pd.DataFrame([row], columns=expected_cols)


@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path: str):
    return joblib.load(model_path)


def _pick_allowed_models(layout: RegistryLayout) -> list[str]:
    all_models = list_model_names(layout)
    picked = [m for m in all_models if m in ALLOWED_MODELS]
    # stable ordering for demos
    order = {"ridge": 0, "xgb": 1, "lgbm": 2}
    picked.sort(key=lambda x: order.get(x, 99))
    return picked


def _quality_label(cv_rmse: float) -> str:
    if cv_rmse < 0.13:
        return "🟢 Excellent"
    if cv_rmse < 0.14:
        return "🟡 Strong"
    return "🔴 Baseline"


# --------------------------------------------------
# UI — Header
# --------------------------------------------------

layout = _ensure_layout()

hero(
    "🧪 Live Price Prediction (Demo)",
    "Interview demo — real inference using **deployment-friendly models only** (Ridge, XGBoost, LightGBM). "
    "Large ensemble artifacts are intentionally excluded to keep the online demo fast."
)

st.caption(
    "Demo build includes **ridge / xgb / lgbm** only. "
    "Other models (ExtraTrees / Voting / Stacking) are excluded due to artifact size."
)

# --------------------------------------------------
# Model picker (restricted)
# --------------------------------------------------

model_names = _pick_allowed_models(layout)
if not model_names:
    st.warning("No allowed demo models found in artifacts_demo/registry.")
    st.stop()

model_name = st.selectbox("Model family", model_names)

aliases = read_aliases(layout, model_name)
alias_key = st.selectbox("Version", ["best", "latest"])

ref: RunRef | None = get_alias_runref(aliases, alias_key, default_model_name=model_name)
if ref is None:
    st.warning("Version alias missing.")
    st.stop()

bundle = load_run_bundle(layout, ref)
run_dir = Path(bundle["run_dir"])

# schema
expected_cols = bundle.get("feature_columns") or {}
defaults = bundle.get("defaults") or {}

if not expected_cols:
    expected_cols = load_json(run_dir / "feature_columns.json")
if not defaults:
    defaults = load_json(run_dir / "defaults.json")

# --------------------------------------------------
# Model summary — PRODUCT VIEW
# --------------------------------------------------

section("Model summary", "High-level model quality overview.", "🧾")

metrics = bundle.get("metrics", {}) or {}
fingerprint = bundle.get("data_fingerprint", {}) or {}

colA, colB, colC, colD = st.columns(4)

with colA:
    st.metric("Model family", ref.model_name)

with colB:
    st.metric("Version", alias_key.upper())

with colC:
    score = metrics.get("cv_rmse")
    if isinstance(score, (int, float)):
        st.metric("Model reliability", _quality_label(float(score)), help=f"Cross-validated RMSE = {score:.5f}")
    else:
        st.metric("Model reliability", "N/A")

with colD:
    rows = fingerprint.get("X", {}).get("rows")
    st.metric("Training data", f"{rows:,} rows" if rows else "N/A", help="Dataset size used during training")

st.caption("This demo runs **real inference** from the selected model’s registry run (model.joblib).")

with st.expander("📦 Dataset fingerprint (technical details)", expanded=False):
    st.json(fingerprint)

st.markdown("---")

# --------------------------------------------------
# Inputs — HUMAN VIEW
# --------------------------------------------------

section("Tell us about the house", "Inputs organized for clarity.", "🏠")

with st.expander("🏡 Essentials", expanded=True):
    overall_qual = st.slider("Build quality", 1, 10, 5)
    living_area = st.slider("Living area (sqft)", 300, 6000, 1500)
    garage = st.selectbox("Garage cars", [0, 1, 2, 3], 2)
    year_built = st.slider("Year built", 1870, 2010, 1970)
    year_sold = st.slider("Year sold", 2006, 2010, 2008)

user_fields = {
    "OverallQual": overall_qual,
    "GrLivArea": living_area,
    "GarageCars": garage,
    "YearBuilt": year_built,
    "YrSold": year_sold,
}

st.markdown("---")

# --------------------------------------------------
# Prediction
# --------------------------------------------------

section("Get estimate", "Run model inference (on click).", "🧾")

colX, colY = st.columns([2, 1])
with colY:
    st.info("Demo deployment")
    st.write("Loads only on demand")
    st.write("Cached after first load")

if st.button("Estimate price"):
    # Build input with full schema + derivations (keeps alignment with contracts)
    X = build_input_df(user_fields, expected_cols, defaults)

    # For demo: load from run dir (preferred)
    model_path = run_dir / "model.joblib"

    if not model_path.exists():
        st.error(
            "Model artifact missing for demo.\n\n"
            f"Expected: {model_path}\n\n"
            "Tip: ensure you copied `model.joblib` for ridge/xgb/lgbm into `artifacts_demo/registry/...` "
            "and allowed them through .gitignore."
        )
        st.stop()

    try:
        model = _load_model_cached(str(model_path))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    try:
        pred = model.predict(X)
        yhat = float(np.asarray(pred).ravel()[0])
        est_price = float(np.expm1(yhat))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"Estimated price: **${est_price:,.0f}**")

    with col2:
        st.info("Model version")
        st.write(ref.model_name, alias_key, ref.run_id)

    with st.expander("Technical details", expanded=False):
        st.write("Raw prediction (log1p):", yhat)
        st.write("Model file:", str(model_path))
        st.dataframe(X, use_container_width=True)
