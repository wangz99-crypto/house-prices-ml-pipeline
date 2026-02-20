# app_demo/pages/3_Live_Prediction_Demo.py
from __future__ import annotations

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
# Lightweight online build: restrict models
# --------------------------------------------------
ALLOWED_MODELS = {"ridge", "xgb", "lgbm"}

MODEL_DISPLAY = {
    "ridge": "Ridge Regression",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

# Human-friendly labels for ordinal quality codes (UI shows this, model receives code)
QUAL_LABEL = {
    "Ex": "Excellent (Ex)",
    "Gd": "Good (Gd)",
    "TA": "Typical/Average (TA)",
    "Fa": "Fair (Fa)",
    "Po": "Poor (Po)",
}
QUAL_ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _ensure_layout() -> RegistryLayout:
    layout = st.session_state.get("REGISTRY_LAYOUT")
    if layout is None:
        st.warning("Please initialize the system from the main entry page.")
        st.stop()
    return layout


def load_json(p: Path):
    if not p.exists():
        raise FileNotFoundError("Required configuration files are not available in this build.")
    return json.loads(p.read_text(encoding="utf-8"))


def _stability_label(cv_rmse: float) -> str:
    if cv_rmse < 0.13:
        return "🟢 Excellent"
    if cv_rmse < 0.14:
        return "🟡 Strong"
    return "🟠 Standard"


def _format_currency(x: float) -> str:
    return f"${x:,.0f}"


def _pick_allowed_models(layout: RegistryLayout) -> list[str]:
    all_models = list_model_names(layout)
    picked = [m for m in all_models if m in ALLOWED_MODELS]
    order = {"ridge": 0, "xgb": 1, "lgbm": 2}
    picked.sort(key=lambda x: order.get(x, 99))
    return picked


def _pick_qual_codes(series: pd.Series) -> list[str]:
    vals = [v for v in series.dropna().astype(str).unique().tolist() if v in set(QUAL_ORDER)]
    return [v for v in QUAL_ORDER if v in vals] or vals


def qual_select(label: str, codes: list[str], default_code: str) -> str:
    """Selectbox that shows friendly text but returns original code."""
    if not codes:
        return default_code
    display = [QUAL_LABEL.get(c, c) for c in codes]
    try:
        idx = codes.index(default_code)
    except ValueError:
        idx = 0
    chosen = st.selectbox(label, display, index=idx)
    return codes[display.index(chosen)]


def apply_derivations(row: dict) -> None:
    # Fill *_log fields if present
    for k in list(row.keys()):
        if k.endswith("_log") and row.get(k) is None:
            base = k[:-4]
            v = row.get(base)
            row[k] = float(np.log1p(max(v, 0))) if isinstance(v, (int, float)) else 0.0

    yr = row.get("YrSold")
    if isinstance(yr, (int, float)):
        if isinstance(row.get("YearBuilt"), (int, float)):
            row["HouseAge"] = max(0, int(yr) - int(row["YearBuilt"]))
        if isinstance(row.get("YearRemodAdd"), (int, float)):
            row["RemodAge"] = max(0, int(yr) - int(row["YearRemodAdd"]))


def build_input_df(user_fields: dict, expected_cols: list, defaults: dict) -> pd.DataFrame:
    row = {c: defaults.get(c, None) for c in expected_cols}
    row.update(user_fields)
    apply_derivations(row)
    return pd.DataFrame([row], columns=expected_cols)


@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path: str):
    return joblib.load(model_path)


def percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float((s < value).mean() * 100.0)


# --------------------------------------------------
# Confidence-style helpers (distribution-aware typicality)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def build_feature_profile(df: pd.DataFrame, features: list[str]) -> dict:
    """
    Build a lightweight distribution profile from sample data.
    Returns: {feature: {"p05":..., "p50":..., "p95":..., "iqr":...}}
    """
    prof = {}
    for f in features:
        s = pd.to_numeric(df.get(f, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(s) < 20:
            continue
        p05 = float(s.quantile(0.05))
        p50 = float(s.quantile(0.50))
        p95 = float(s.quantile(0.95))
        p25 = float(s.quantile(0.25))
        p75 = float(s.quantile(0.75))
        iqr = max(1e-9, p75 - p25)  # avoid div by zero
        prof[f] = {"p05": p05, "p50": p50, "p95": p95, "iqr": iqr}
    return prof


def reliability_from_profile(user_fields: dict, profile: dict, keys: list[str]) -> tuple[str, int, list[str]]:
    """
    Percentile-based typicality score (heuristic, distribution-aware).
    - Score starts at 100.
    - For each key feature:
        - if value within [p05, p95] => small/no penalty
        - if outside => penalty grows with distance scaled by IQR
    Returns: (label, score, reasons)
    """
    score = 100.0
    reasons: list[str] = []
    used = 0

    for k in keys:
        if k not in profile:
            continue
        v = user_fields.get(k, None)
        if v is None or not isinstance(v, (int, float)):
            continue

        used += 1
        p05 = profile[k]["p05"]
        p95 = profile[k]["p95"]
        iqr = profile[k]["iqr"]

        if v < p05:
            dist = (p05 - float(v)) / iqr
            penalty = min(18.0, 8.0 + 10.0 * dist)
            score -= penalty
            reasons.append(f"{k} is below typical range (p05≈{p05:.1f}).")
        elif v > p95:
            dist = (float(v) - p95) / iqr
            penalty = min(18.0, 8.0 + 10.0 * dist)
            score -= penalty
            reasons.append(f"{k} is above typical range (p95≈{p95:.1f}).")

    if used == 0:
        return ("Medium", 60, ["Not enough numeric inputs to assess typicality."])

    score_i = int(max(0, min(100, round(score))))

    if score_i >= 80:
        return ("High", score_i, reasons)
    if score_i >= 55:
        return ("Medium", score_i, reasons)
    return ("Low", score_i, reasons)


def typical_range_from_cv_rmse(est_price: float, cv_rmse_log: float) -> tuple[float, float]:
    """
    Convert log-space RMSE to a multiplicative typical error band around est_price.
    NOTE: Not a formal prediction interval. A CV-informed typical band.
    """
    factor = float(np.exp(cv_rmse_log))
    lo = est_price / factor
    hi = est_price * factor
    return lo, hi


# --------------------------------------------------
# Paths (sample data used for presets + percentiles + dropdown options)
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO_ROOT / "tests" / "data" / "sample_train.csv"


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


# --------------------------------------------------
# UI — Header
# --------------------------------------------------
layout = _ensure_layout()

hero(
    "Live Price Prediction",
    "Enter property details to generate a market-based estimate.",
)

st.markdown(
    """
This page demonstrates the end-to-end prediction experience.
Inputs are easy to understand while the system applies defaults and derived fields behind the scenes.
"""
)

# --------------------------------------------------
# Model selection
# --------------------------------------------------
section("Model selection", "Choose a model and version for the estimate.", "🧾")

model_keys = _pick_allowed_models(layout)
if not model_keys:
    st.warning("No models available.")
    st.stop()

label_to_key = {MODEL_DISPLAY.get(k, k): k for k in model_keys}
model_label = st.selectbox("Model", list(label_to_key.keys()), index=0)
model_key = label_to_key[model_label]

aliases = read_aliases(layout, model_key)
alias_key = st.selectbox("Version", ["best", "latest"], index=0)

ref: RunRef | None = get_alias_runref(aliases, alias_key, default_model_name=model_key)
if ref is None:
    st.warning("The selected model version is not available.")
    st.stop()

bundle = load_run_bundle(layout, ref)
run_dir = Path(bundle["run_dir"])

# Load schema + defaults
expected_cols = bundle.get("feature_columns") or []
defaults = bundle.get("defaults") or {}

try:
    if not expected_cols:
        expected_cols = load_json(run_dir / "feature_columns.json")
    if not defaults:
        defaults = load_json(run_dir / "defaults.json")
except Exception:
    st.warning("Model configuration is not available for the selected version.")
    st.stop()

# Ensure expected_cols is a list of strings
if isinstance(expected_cols, dict):
    expected_cols = list(expected_cols.keys())
elif not isinstance(expected_cols, list):
    expected_cols = list(expected_cols)

# Load model (cached on demand, but we keep it ready for smooth UX)
model_path = run_dir / "model.joblib"
if not model_path.exists():
    st.warning("The selected model artifact is not available in this build.")
    st.stop()

model = _load_model_cached(str(model_path))

# --------------------------------------------------
# Model summary (RMSE not truncated)
# --------------------------------------------------
section("Model summary", "A compact view of model stability and training context.", "📌")

metrics = bundle.get("metrics", {}) or {}
fingerprint = bundle.get("data_fingerprint", {}) or {}

cv_rmse = metrics.get("cv_rmse")
rows = fingerprint.get("X", {}).get("rows")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", MODEL_DISPLAY.get(ref.model_name, ref.model_name))
col2.metric("Version", alias_key.upper())

with col3:
    if isinstance(cv_rmse, (int, float)):
        st.metric("Stability", _stability_label(float(cv_rmse)))
        st.caption(f"CV RMSE: {float(cv_rmse):.4f} (lower is better)")
    else:
        st.metric("Stability", "N/A")

col4.metric("Training rows", f"{rows:,}" if rows else "N/A")

st.caption(
    "Stability is derived from cross-validation. It reflects consistency, not a guarantee for any individual property."
)

with st.expander("Training snapshot (technical)", expanded=False):
    st.json(fingerprint)

st.markdown("---")

st.info(
    "Demo mode note: This interactive prediction page uses lightweight models for fast startup. "
    "Full model evaluation (including Stacking and XGBoost) is shown in **Model Behavior** and **Experiments & Analysis**."
)

# --------------------------------------------------
# Sample-driven features: presets + percentiles + dropdowns
# --------------------------------------------------
df_sample = load_sample()

# Neighborhood dropdown options (safe, no free text)
neighborhood_options = sorted(df_sample["Neighborhood"].dropna().astype(str).unique().tolist())

# Quality code options (sorted in sensible order)
kitchen_codes = _pick_qual_codes(df_sample.get("KitchenQual", pd.Series(dtype=str)))
exter_codes = _pick_qual_codes(df_sample.get("ExterQual", pd.Series(dtype=str)))
heat_codes = _pick_qual_codes(df_sample.get("HeatingQC", pd.Series(dtype=str)))

# Distribution profile for reliability (typicality check)
RELIABILITY_KEYS = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt", "LotArea"]
feature_profile = build_feature_profile(df_sample, RELIABILITY_KEYS)

# --------------------------------------------------
# Quick start presets
# --------------------------------------------------
section("Quick start", "Start with a realistic example or customize manually.", "⚡")

preset = st.selectbox("Example property", ["Custom", "Starter home", "Typical home", "Premium home"], index=0)


def _preset_row(name: str) -> pd.Series | None:
    if name == "Custom":
        return None
    df = df_sample.copy()
    df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
    df = df.dropna(subset=["SalePrice"])
    if len(df) == 0:
        return None
    if name == "Starter home":
        return df.nsmallest(1, "SalePrice").iloc[0]
    if name == "Premium home":
        return df.nlargest(1, "SalePrice").iloc[0]
    med = df["SalePrice"].median()
    idx = (df["SalePrice"] - med).abs().idxmin()
    return df.loc[idx]


preset_row = _preset_row(preset)


def _preset_or_default_num(col: str, default: int) -> int:
    if preset_row is not None and col in preset_row and pd.notna(preset_row[col]):
        try:
            return int(float(preset_row[col]))
        except Exception:
            return default
    return default


def _preset_or_default_cat(col: str, default: str) -> str:
    if preset_row is not None and col in preset_row and pd.notna(preset_row[col]):
        return str(preset_row[col])
    return default


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
section("Property details", "Adjust features to reflect the property.", "🏠")

with st.expander("Essentials", expanded=True):
    overall_qual = st.slider(
        "Build quality",
        1,
        10,
        _preset_or_default_num("OverallQual", 5),
        help="Overall material and finish quality (1–10).",
    )
    overall_cond = st.slider("Overall condition", 1, 10, _preset_or_default_num("OverallCond", 5))
    living_area = st.slider("Living area (sqft)", 300, 6000, _preset_or_default_num("GrLivArea", 1500), step=10)
    basement = st.slider("Basement area (sqft)", 0, 3000, _preset_or_default_num("TotalBsmtSF", 800), step=10)

    garage_val = _preset_or_default_num("GarageCars", 2)
    garage_choices = [0, 1, 2, 3, 4]
    garage_idx = garage_choices.index(garage_val) if garage_val in garage_choices else 2
    garage_cars = st.selectbox("Garage capacity (cars)", garage_choices, index=garage_idx)

    year_built = st.slider("Year built", 1870, 2010, _preset_or_default_num("YearBuilt", 1970))
    year_sold = st.slider("Year sold", 2006, 2010, _preset_or_default_num("YrSold", 2008))

with st.expander("Layout (optional, helps accuracy)", expanded=False):
    bedrooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5, 6],
                            index=min(max(_preset_or_default_num("BedroomAbvGr", 3), 0), 6))
    total_rooms = st.selectbox("Total rooms (above ground)", list(range(2, 13)),
                               index=max(0, min(10, _preset_or_default_num("TotRmsAbvGrd", 6) - 2)))
    full_bath = st.selectbox("Full bathrooms", [0, 1, 2, 3, 4],
                             index=min(max(_preset_or_default_num("FullBath", 1), 0), 4))
    half_bath = st.selectbox("Half bathrooms", [0, 1, 2],
                             index=min(max(_preset_or_default_num("HalfBath", 0), 0), 2))

with st.expander("Quality & finish", expanded=False):
    kitchen_default = _preset_or_default_cat("KitchenQual", str(defaults.get("KitchenQual", "TA")))
    exter_default = _preset_or_default_cat("ExterQual", str(defaults.get("ExterQual", "TA")))
    heat_default = _preset_or_default_cat("HeatingQC", str(defaults.get("HeatingQC", "TA")))

    kitchen_qual = qual_select("Kitchen quality", kitchen_codes, kitchen_default)
    exter_qual = qual_select("Exterior quality", exter_codes, exter_default)
    heating_qc = qual_select("Heating quality", heat_codes, heat_default)

with st.expander("Lot & location (optional)", expanded=False):
    lot_area = st.slider("Lot size (sqft)", 1000, 50000, _preset_or_default_num("LotArea", 9000), step=50)
    central_air = st.checkbox("Central air", value=True)

    nb_default = _preset_or_default_cat("Neighborhood", str(defaults.get("Neighborhood", neighborhood_options[0])))
    nb_default = nb_default if nb_default in neighborhood_options else neighborhood_options[0]
    neighborhood = st.selectbox("Neighborhood", neighborhood_options, index=neighborhood_options.index(nb_default))

with st.expander("Home preferences (optional)", expanded=False):
    remodeled = st.selectbox("Recently remodeled?", ["Use system default", "Yes", "No"], index=0)
    has_fireplace = st.selectbox("Fireplace?", ["Use system default", "Yes", "No"], index=0)

# --------------------------------------------------
# Compare to typical (percentiles)
# --------------------------------------------------
st.markdown("---")
st.subheader("How this property compares")

p_area = percentile_rank(df_sample.get("GrLivArea", pd.Series(dtype=float)), float(living_area))
p_qual = percentile_rank(df_sample.get("OverallQual", pd.Series(dtype=float)), float(overall_qual))

if not np.isnan(p_area):
    st.write(f"- Living area is larger than **{p_area:.0f}%** of homes in this sample")
if not np.isnan(p_qual):
    st.write(f"- Build quality is above **{p_qual:.0f}%** of homes in this sample")

st.markdown("---")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
section("Price estimate", "Generate an estimate and show which model version was used.", "🧾")

if st.button("Generate estimate", use_container_width=True):
    # Map to model fields
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
        "KitchenQual": kitchen_qual,
        "ExterQual": exter_qual,
        "HeatingQC": heating_qc,
    }

    # Add remodeled / fireplace only if the schema supports it
    if remodeled != "Use system default":
        if "IsRemodeled" in expected_cols:
            user_fields["IsRemodeled"] = 1 if remodeled == "Yes" else 0
        elif "YearRemodAdd" in expected_cols:
            if remodeled == "Yes":
                user_fields["YearRemodAdd"] = max(int(year_built), int(year_sold) - 5)
            else:
                user_fields["YearRemodAdd"] = int(year_built)

    if has_fireplace != "Use system default":
        if "HasFireplace" in expected_cols:
            user_fields["HasFireplace"] = 1 if has_fireplace == "Yes" else 0
        elif "Fireplaces" in expected_cols:
            user_fields["Fireplaces"] = 1 if has_fireplace == "Yes" else 0
        elif "FireplaceQu" in expected_cols and has_fireplace == "No":
            user_fields["FireplaceQu"] = "NA"

    X = build_input_df(user_fields, expected_cols, defaults)

    try:
        pred = model.predict(X)
        yhat = float(np.asarray(pred).ravel()[0])
        est_price = float(np.expm1(yhat))
    except Exception:
        st.warning("Prediction could not be generated with the current settings. Try adjusting inputs or model.")
        st.stop()

    # Confidence-style signals
    rel_label, rel_score, rel_reasons = reliability_from_profile(
        user_fields=user_fields,
        profile=feature_profile,
        keys=RELIABILITY_KEYS,
    )

    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.success(f"Estimated price: **{_format_currency(est_price)}**")

        if isinstance(cv_rmse, (int, float)) and float(cv_rmse) > 0:
            lo, hi = typical_range_from_cv_rmse(est_price, float(cv_rmse))
            st.caption(f"Typical range (CV-informed): **{_format_currency(lo)} – {_format_currency(hi)}**")
            st.caption("Note: this is an error-informed typical band, not a formal prediction interval.")
        else:
            st.caption("Typical range is unavailable (CV RMSE not found for this run).")

        st.caption("This is an automated estimate based on historical patterns in the training data.")

    with colB:
        st.markdown("**Reliability (typicality check)**")
        st.progress(rel_score)
        st.write(f"**{rel_label}** ({rel_score}/100)")
        st.caption("Heuristic based on whether inputs fall within the typical range of training data.")

        if rel_reasons:
            with st.expander("Why this score?", expanded=False):
                for r in rel_reasons[:4]:
                    st.write(f"- {r}")
                if len(rel_reasons) > 4:
                    st.write(f"- …and {len(rel_reasons) - 4} more.")

        st.markdown("**Model used**")
        st.write(f"{MODEL_DISPLAY.get(ref.model_name, ref.model_name)} • {alias_key.upper()}")
        with st.expander("Details", expanded=False):
            st.write(f"Run ID: {ref.run_id}")

    with st.expander("Inputs used (after defaults & derived fields)", expanded=False):
        st.dataframe(X, use_container_width=True)

    with st.expander("Prediction output (technical)", expanded=False):
        st.write("Model output (log scale):", yhat)
