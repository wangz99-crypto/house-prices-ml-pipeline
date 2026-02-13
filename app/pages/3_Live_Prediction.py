# app/pages/3_Live_Prediction.py

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


# --------------------------------------------------
# UI — Header
# --------------------------------------------------

layout = _ensure_layout()

hero(
    "🧪 Live Price Prediction",
    "Guided interface — the system handles technical inputs automatically."
)

# --------------------------------------------------
# Model picker
# --------------------------------------------------

model_names = list_model_names(layout)

if not model_names:
    st.warning("No models found.")
    st.stop()

model_name = st.selectbox("Model family", model_names)

aliases = read_aliases(layout, model_name)
alias_key = st.selectbox("Version", ["best", "latest"])

ref: RunRef | None = get_alias_runref(
    aliases,
    alias_key,
    default_model_name=model_name
)

if ref is None:
    st.warning("Version alias missing.")
    st.stop()

bundle = load_run_bundle(layout, ref)
run_dir = Path(bundle["run_dir"])

# Prefer current snapshot
model_path = Path(layout.current_model_path(ref.model_name))

# Fallback to run bundle model.joblib if "current" snapshot isn't present
if not model_path.exists():
    mp = bundle.get("model_path")  # usually <run_dir>/model.joblib
    if isinstance(mp, str):
        model_path = Path(mp)

# Final fallback: run_dir / model.joblib
if not model_path.exists():
    model_path = Path(bundle["run_dir"]) / "model.joblib"

if not model_path.exists():
    st.error(
        "Model artifact missing.\n\n"
        f"Checked:\n- {layout.current_model_path(ref.model_name)}\n"
        f"- {bundle.get('model_path')}\n"
        f"- {Path(bundle['run_dir']) / 'model.joblib'}"
    )
    st.stop()

# Load model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


expected_cols = load_json(run_dir / "feature_columns.json")
defaults = load_json(run_dir / "defaults.json")

# --------------------------------------------------
# Model summary — PRODUCT VIEW
# --------------------------------------------------

section("Model summary", "High-level model quality overview.", "🧾")

metrics = bundle.get("metrics", {}) or {}
fingerprint = bundle.get("data_fingerprint", {}) or {}

colA, colB, colC, colD = st.columns(4)

# --- model info

with colA:
    st.metric("Model family", ref.model_name)

with colB:
    st.metric("Version", alias_key.upper())

# --- quality interpretation

with colC:
    score = metrics.get("cv_rmse")

    if isinstance(score, (int, float)):

        if score < 0.13:
            label = "🟢 Excellent"
        elif score < 0.14:
            label = "🟡 Strong"
        else:
            label = "🔴 Baseline"

        st.metric(
            "Model reliability",
            label,
            help=f"Cross-validated RMSE = {score:.5f}"
        )
    else:
        st.metric("Model reliability", "N/A")

# --- dataset size

with colD:
    rows = fingerprint.get("X", {}).get("rows")

    st.metric(
        "Training data",
        f"{rows:,} rows" if rows else "N/A",
        help="Dataset size used during training"
    )

st.caption(
"""
Model performance is validated using cross-validation.
Lower RMSE indicates more stable predictions.
"""
)

# --- fingerprint (technical)

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

section("Get estimate", "Run model inference.", "🧾")

if st.button("Estimate price"):

    X = build_input_df(user_fields, expected_cols, defaults)

    pred = model.predict(X)
    yhat = float(np.asarray(pred).ravel()[0])
    est_price = float(np.expm1(yhat))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.success(f"Estimated price: **${est_price:,.0f}**")

    with col2:
        st.info("Model version")
        st.write(ref.model_name, alias_key, ref.run_id)

    with st.expander("Technical details", expanded=False):
        st.write("Raw prediction:", yhat)
        st.dataframe(X)
