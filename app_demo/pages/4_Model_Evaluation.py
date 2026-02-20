# app/pages/4_Model_Behavior.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# Optional dependencies (silent fallbacks)
# =========================
try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except Exception:
    sm = None
    _HAS_STATSMODELS = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline
    _HAS_SKLEARN_EXTRAS = True
except Exception:
    permutation_importance = None
    train_test_split = None
    mean_squared_error = None
    Pipeline = None
    _HAS_SKLEARN_EXTRAS = False

# =========================
# Runtime constraints (kept internal, not shown as "policy")
# =========================
ALLOWED_MODELS = {"ridge", "xgb", "lgbm"}  # only these may be loaded interactively (demo-safe)
MAX_MODEL_MB = 50.0

# =========================
# Page config
# =========================
st.set_page_config(page_title="Model Behavior", layout="wide")
st.title("🧠 Model Behavior")
st.caption(
    "See which model families perform best, what features matter most, and how to interpret a readable linear baseline (Ridge). "
    "This page prioritizes clarity for non-technical viewers while preserving evaluation rigor."
)

# =========================
# Sidebar: optional debug
# =========================
with st.sidebar:
    st.markdown("## Model Behavior")
    DEBUG = st.toggle("Debug mode", value=False, help="Show detailed error traces for troubleshooting.")
    st.caption("Tip: For a quick tour, scan sections 1 → 2, then open 3 only if you want deeper interpretation.")

# =========================
# Resolve artifact paths
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO1 = REPO_ROOT / "artifacts_demo"
DEMO2 = REPO_ROOT / "app" / "artifacts_demo"
DEMO = DEMO1 if DEMO1.exists() else DEMO2

REPORTS = DEMO / "reports"
FIG_DIR = REPORTS / "figures"
FI_DIR = REPORTS / "feature_importance"
SUMMARY_CSV = REPORTS / "model_performance_summary.csv"
REGISTRY_DIR = DEMO / "registry"

if not DEMO.exists():
    st.error("Required demo assets are not available in this build.")
    st.stop()

# ============================================================
# Feature dictionary (Ames + engineered layer)
# ============================================================
AMES_DESC = {
    "MSSubClass": "Dwelling type code (1-story, 2-story, duplex, etc.).",
    "MSZoning": "Zoning classification of the property.",
    "LotFrontage": "Street frontage: linear feet of street connected to the lot.",
    "LotArea": "Lot size in square feet.",
    "Street": "Road access type (paved vs gravel).",
    "Alley": "Alley access type (or no alley access).",
    "LotShape": "Lot shape regularity (regular vs irregular).",
    "LandContour": "Lot flatness / contour (level, hillside, depression...).",
    "Utilities": "Utilities available (all public vs limited).",
    "LotConfig": "Lot configuration (inside lot, corner, cul-de-sac...).",
    "LandSlope": "Slope of property (gentle/moderate/severe).",
    "Neighborhood": "Neighborhood location within Ames city limits.",
    "Condition1": "Proximity to roads/rail/parks etc. (primary).",
    "Condition2": "Additional proximity condition (if multiple).",
    "BldgType": "Building type (single-family, townhouse, duplex...).",
    "HouseStyle": "House style (1Story, 2Story, SFoyer, SLvl...).",
    "OverallQual": "Overall build quality rating (1–10).",
    "OverallCond": "Overall condition rating (1–10).",
    "YearBuilt": "Original construction year.",
    "YearRemodAdd": "Remodel/addition year (same as YearBuilt if never remodeled).",
    "RoofStyle": "Roof style.",
    "RoofMatl": "Roof material.",
    "Exterior1st": "Primary exterior covering.",
    "Exterior2nd": "Secondary exterior covering (if multiple).",
    "MasVnrType": "Masonry veneer type.",
    "MasVnrArea": "Masonry veneer area (sq ft).",
    "ExterQual": "Exterior material quality (Ex/Gd/TA/Fa/Po).",
    "ExterCond": "Exterior material condition (Ex/Gd/TA/Fa/Po).",
    "Foundation": "Foundation type.",
    "BsmtQual": "Basement height/quality (Ex/Gd/TA/Fa/Po/NA).",
    "BsmtCond": "Basement condition (Ex/Gd/TA/Fa/Po/NA).",
    "BsmtExposure": "Basement exposure / walkout (Gd/Av/Mn/No/NA).",
    "BsmtFinType1": "Basement finish rating (GLQ/ALQ/Rec/Unf/NA...).",
    "BsmtFinSF1": "Finished basement area (type 1) in sq ft.",
    "BsmtFinType2": "Second basement finish rating (if applicable).",
    "BsmtFinSF2": "Finished basement area (type 2) in sq ft.",
    "BsmtUnfSF": "Unfinished basement area in sq ft.",
    "TotalBsmtSF": "Total basement area in sq ft.",
    "Heating": "Heating type.",
    "HeatingQC": "Heating quality/condition (Ex/Gd/TA/Fa/Po).",
    "CentralAir": "Central air conditioning (Y/N).",
    "Electrical": "Electrical system type.",
    "1stFlrSF": "First floor area in sq ft.",
    "2ndFlrSF": "Second floor area in sq ft.",
    "LowQualFinSF": "Low quality finished area in sq ft.",
    "GrLivArea": "Above-grade living area in sq ft.",
    "BsmtFullBath": "Basement full bathrooms.",
    "BsmtHalfBath": "Basement half bathrooms.",
    "FullBath": "Full bathrooms above grade.",
    "HalfBath": "Half bathrooms above grade.",
    "BedroomAbvGr": "Bedrooms above grade (excluding basement).",
    "KitchenAbvGr": "Kitchens above grade.",
    "KitchenQual": "Kitchen quality (Ex/Gd/TA/Fa/Po).",
    "TotRmsAbvGrd": "Total rooms above grade (excluding bathrooms).",
    "Functional": "Overall home functionality.",
    "Fireplaces": "Number of fireplaces.",
    "FireplaceQu": "Fireplace quality (Ex/Gd/TA/Fa/Po/NA).",
    "GarageType": "Garage location/type.",
    "GarageYrBlt": "Garage build year.",
    "GarageFinish": "Garage interior finish.",
    "GarageCars": "Garage capacity (# of cars).",
    "GarageArea": "Garage area in sq ft.",
    "GarageQual": "Garage quality.",
    "GarageCond": "Garage condition.",
    "PavedDrive": "Driveway paving (Y/P/N).",
    "WoodDeckSF": "Wood deck area in sq ft.",
    "OpenPorchSF": "Open porch area in sq ft.",
    "EnclosedPorch": "Enclosed porch area in sq ft.",
    "3SsnPorch": "Three-season porch area in sq ft.",
    "ScreenPorch": "Screen porch area in sq ft.",
    "PoolArea": "Pool area in sq ft.",
    "PoolQC": "Pool quality.",
    "Fence": "Fence quality.",
    "MiscFeature": "Miscellaneous feature (shed, tennis court...).",
    "MiscVal": "Value of miscellaneous feature ($).",
    "MoSold": "Month sold (1–12).",
    "YrSold": "Year sold.",
    "SaleType": "Type of sale.",
    "SaleCondition": "Sale condition.",
}

ENGINEERED_DESC = {
    "HouseAge": "Engineered: house age at sale (YrSold − YearBuilt).",
    "IsNewHouse": "Engineered: 1 if house is relatively new (e.g., age ≤ 5).",
    "RemodAge": "Engineered: years since remodel (YrSold − YearRemodAdd).",
    "IsRemodeled": "Engineered: 1 if remodeled (YearBuilt != YearRemodAdd).",
    "TotalSF": "Engineered: total footprint (basement + 1st + 2nd floors).",
    "TotalBathrooms": "Engineered: total baths (Full + 0.5×Half + basement baths).",
    "TotalPorchSF": "Engineered: total porch area (open + enclosed + 3-season + screen).",
    "HasBasement": "Engineered: 1 if basement exists (TotalBsmtSF > 0).",
    "HasGarage": "Engineered: 1 if garage exists (GarageArea > 0).",
    "HasFireplace": "Engineered: 1 if fireplace exists (Fireplaces > 0).",
    "HasPool": "Engineered: 1 if pool exists (PoolArea > 0).",
    "HasPorch": "Engineered: 1 if any porch exists (TotalPorchSF > 0).",
    "HasDeck": "Engineered: 1 if deck exists (WoodDeckSF > 0).",
    "HasMasonryVeneer": "Engineered: 1 if veneer exists (MasVnrArea > 0).",
    "LuxuryAmenityScore": "Engineered: simple luxury score (pool + veneer + deck + porch + fireplace).",
    "QualGrLiv": "Engineered: quality × living area (OverallQual × GrLivArea).",
    "QualTotalSF": "Engineered: quality × total area (OverallQual × TotalSF).",
    "QualGarage": "Engineered: quality × garage area (OverallQual × GarageArea).",
    "IsHighQuality": "Engineered: 1 if OverallQual ≥ 7.",
    "IsLargeHouse": "Engineered: 1 if GrLivArea ≥ 2000 sq ft.",
    "IsLuxury": "Engineered: 1 if high quality AND large (OverallQual ≥ 8 and GrLivArea ≥ 2500).",
    "GrLivAreaBin": "Engineered: living area bucket (small / mid / large).",
    "Neighborhood_Qual": "Engineered: neighborhood premium signal combined with quality.",
}

QUALITY_LEVEL = {
    "Ex": "Excellent",
    "Gd": "Good",
    "TA": "Typical/Average",
    "Fa": "Fair",
    "Po": "Poor",
    "NA": "Not applicable / none",
}
YESNO_LEVEL = {"Y": "Yes", "N": "No"}


def _strip_nested_prefix(name: str) -> str:
    s = str(name)
    while "__" in s:
        _, s = s.split("__", 1)
    return s


def describe_feature(name: str) -> str:
    s = _strip_nested_prefix(name)

    if s in ENGINEERED_DESC:
        return ENGINEERED_DESC[s]

    if s.endswith("_log"):
        base = s[:-4]
        base_h = AMES_DESC.get(base, f"{base} (raw field)")
        return f"Engineered: log transform of {base}. {base_h}"

    # One-hot: Field_Level
    if "_" in s:
        field, level = s.split("_", 1)
        if field in AMES_DESC:
            if level in QUALITY_LEVEL:
                return f"{field} = {QUALITY_LEVEL[level]} ({level}). {AMES_DESC[field]}"
            if level in YESNO_LEVEL:
                return f"{field} = {YESNO_LEVEL[level]} ({level}). {AMES_DESC[field]}"
            return f"{field} = {level}. {AMES_DESC[field]}"
        return f"Categorical indicator: {s}."

    if s in AMES_DESC:
        return AMES_DESC[s]

    if s.startswith("Has"):
        return f"Engineered indicator for an amenity: {s}."
    if s.startswith("Is"):
        return f"Engineered flag: {s}."
    if s.endswith("Bin"):
        return f"Engineered bucket: {s}."

    return f"Model feature: {s}."


def add_description_column(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    if feature_col not in df.columns:
        return df
    out = df.copy()
    out["description"] = out[feature_col].map(describe_feature)
    return out


# =========================
# Utilities: unwrap + split
# =========================
def _unwrap_estimator(obj: object):
    cur = obj
    for _ in range(10):
        if hasattr(cur, "steps") and hasattr(cur, "named_steps"):
            return cur
        for attr in ("estimator_", "estimator", "model", "pipeline", "base_estimator", "regressor", "regressor_"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
        else:
            return cur
    return cur


def _split_pipeline(obj: object):
    pipe = _unwrap_estimator(obj)
    if hasattr(pipe, "steps") and hasattr(pipe, "named_steps"):
        est = pipe.steps[-1][1]
        pre = Pipeline(pipe.steps[:-1]) if (Pipeline is not None and len(pipe.steps) > 1) else None
        return pre, est
    return None, pipe


@st.cache_data(show_spinner=False)
def _load_train_data(repo_root: Path) -> Tuple[pd.DataFrame, pd.Series]:
    p_sample = repo_root / "tests" / "data" / "sample_train.csv"
    p_raw = repo_root / "data" / "raw" / "train.csv"

    if p_sample.exists():
        df = pd.read_csv(p_sample)
    elif p_raw.exists():
        df = pd.read_csv(p_raw)
    else:
        raise FileNotFoundError("Training dataset is not available in this build.")

    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    return X, y


def _get_shared_and_ct(pre):
    if pre is None or not hasattr(pre, "named_steps"):
        return None, None
    shared = pre.named_steps.get("shared")
    ct = pre.named_steps.get("prep")
    return shared, ct


def _find_run_dirs(model_name: str) -> List[Path]:
    if model_name not in ALLOWED_MODELS:
        return []
    base = REGISTRY_DIR / model_name
    if not base.exists():
        return []
    run_dirs = [d for d in base.iterdir() if d.is_dir() and d.name not in {"_global"}]
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs


@st.cache_data(show_spinner=False)
def _load_latest_pipeline_and_run(model_name: str):
    if model_name not in ALLOWED_MODELS:
        raise ValueError("This model family is not available for interactive inspection in this build.")

    run_dirs = _find_run_dirs(model_name)
    if not run_dirs:
        raise FileNotFoundError("No saved run is available for this model family.")

    run_dir = run_dirs[0]
    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Saved model file is not available for this run.")

    mb = model_path.stat().st_size / 1024 / 1024
    if mb > MAX_MODEL_MB:
        raise RuntimeError("This model is not included for interactive loading in this build.")

    pipe = joblib.load(model_path)
    return pipe, run_dir


def _normalize_stage(s: str) -> str:
    s0 = str(s).strip().lower().replace(" ", "")
    if s0 in {"stage1", "1", "s1"} or ("stage" in s0 and "1" in s0):
        return "Stage 1"
    if s0 in {"stage2", "2", "s2"} or ("stage" in s0 and "2" in s0):
        return "Stage 2"
    return str(s).strip()


def _format_currency(x: float) -> str:
    return f"${x:,.0f}"


# ============================================================
# 0) Demo note (align expectations)
# ============================================================
st.info(
    "Demo build note: this page can **interactively load and inspect** lightweight models (Ridge/XGB/LGBM) for speed and stability. "
    "Full model evaluation (ExtraTrees/Voting/Stacking, etc.) is still reflected in the **comparison charts and tables**."
)

# ============================================================
# 1) Model Comparison
# ============================================================
st.subheader("1) Model selection overview")

st.markdown(
    """
This system uses a **two-stage selection** approach to reduce “lucky” results:

- **Stage 1 (Screening):** quickly tests many candidates and filters out weak configurations  
- **Stage 2 (Stability):** re-tests the strongest candidates and selects the most consistent winner  
- The chosen winner becomes the system’s default (**Best**) and the most recent run becomes (**Latest**)
"""
)

c1, c2 = st.columns(2, gap="large")
with c1:
    p = FIG_DIR / "model_comparison_stage1.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.info("Stage 1 comparison chart is not available in this build.")
with c2:
    p = FIG_DIR / "model_comparison_stage2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.info("Stage 2 comparison chart is not available in this build.")

# Quick intuition card (kept short, non-technical + safer wording)
try:
    if SUMMARY_CSV.exists():
        X_all, y_all = _load_train_data(REPO_ROOT)
        p50 = float(np.median(y_all))

        df_sum_all = pd.read_csv(SUMMARY_CSV)
        df_sum_all.columns = [c.strip() for c in df_sum_all.columns]
        if "stage" in df_sum_all.columns:
            df_sum_all["stage"] = df_sum_all["stage"].apply(_normalize_stage)

        stage2 = df_sum_all[df_sum_all.get("stage", "Stage 2") == "Stage 2"] if "stage" in df_sum_all.columns else df_sum_all
        if len(stage2) and "rmse_mean" in stage2.columns:
            best = stage2.sort_values("rmse_mean", ascending=True).iloc[0]
            e = float(best["rmse_mean"])
            approx_pct = float(np.expm1(e))
            approx_usd = p50 * approx_pct

            st.info(
                f"""
**Quick intuition**

Best Stage 2 RMSE ≈ **{e:.4f}** (log scale).

For a median-priced home (~**{_format_currency(p50)}**),  
this corresponds to a **rough error scale** on the order of **±{_format_currency(approx_usd)}**.

*(Intuition aid only — not a guarantee.)*
"""
            )
except Exception as e:
    if DEBUG:
        st.exception(e)

st.success("Takeaway: Stage 2 prioritizes stability across repeated evaluation over a single best run.")

# Evaluation table: default Stage 2 + Top 30, with optional full view
if SUMMARY_CSV.exists():
    df_sum = pd.read_csv(SUMMARY_CSV)
    df_sum.columns = [c.strip() for c in df_sum.columns]
    if "stage" in df_sum.columns:
        df_sum["stage"] = df_sum["stage"].apply(_normalize_stage)

    with st.expander("Show evaluation table (details)"):
        show_all = st.checkbox("Show all rows", value=False)
        stage_filter = None
        if "stage" in df_sum.columns:
            stage_filter = st.selectbox("Stage filter", ["Stage 2 (recommended)", "Stage 1", "All stages"], index=0)
        view = df_sum.copy()

        if stage_filter == "Stage 2 (recommended)":
            view = view[view["stage"] == "Stage 2"]
        elif stage_filter == "Stage 1":
            view = view[view["stage"] == "Stage 1"]

        if {"stage", "rmse_mean"}.issubset(view.columns):
            view = view.sort_values(["stage", "rmse_mean"], ascending=[True, True])

        if not show_all:
            view = view.head(30)

        st.dataframe(view, use_container_width=True)

st.divider()

# ============================================================
# 2) Feature Importance (Tree/Boosting)
# ============================================================
st.subheader("2) What drives predictions (Feature importance)")

st.markdown(
    """
These charts highlight the features that most influence model predictions in **tree/boosting** models.
They help explain “what the model pays attention to” at a high level.
"""
)

imgs = sorted(FIG_DIR.glob("feat_importance_top20__*.png"))
names_all = [p.stem.replace("feat_importance_top20__", "") for p in imgs]
names = [n for n in names_all if (n.lower() in {"xgb", "lgbm"})]

if not names:
    st.info("Feature-importance charts are not available in this build.")
else:
    model = st.selectbox("Model", names, index=0)  # friendlier than "family"
    p = FIG_DIR / f"feat_importance_top20__{model}.png"
    if p.exists():
        st.image(str(p), use_container_width=True)

    csv_candidates = sorted(FI_DIR.glob(f"{model}__*__top30.csv"))
    if csv_candidates:
        with st.expander("Show top features as a table (with explanations)"):
            try:
                df = pd.read_csv(csv_candidates[-1])
                if "feature" in df.columns:
                    df = add_description_column(df, "feature")
                st.dataframe(df.head(30), use_container_width=True)
            except Exception as e:
                if DEBUG:
                    st.exception(e)
                st.info("Feature table is not available for this model in the current build.")

st.success("Takeaway: size + quality signals tend to dominate, consistent with real-world housing pricing drivers.")
st.divider()

# ============================================================
# 3) Ridge interpretation
# ============================================================
st.subheader("3) Ridge model interpretation")

st.markdown(
    """
Ridge is a linear baseline, so we interpret it differently:

- **Coefficients:** direction and strength of influence (after preprocessing)
- **Optional deeper views:** an OLS refit (interpretation aid) and permutation importance (slower)
"""
)

FAST_VIEW = st.toggle("Quick view (recommended)", value=True)

try:
    ridge_pipe, _ridge_run_dir = _load_latest_pipeline_and_run("ridge")
except Exception as e:
    if DEBUG:
        st.exception(e)
    st.info("Ridge interpretation is not available in this build.")
    st.stop()

pre, est = _split_pipeline(ridge_pipe)
try:
    X, y = _load_train_data(REPO_ROOT)
except Exception as e:
    if DEBUG:
        st.exception(e)
    st.info("Training data is not available in this build.")
    st.stop()

# transform for interpretation layers
try:
    X_proc = pre.transform(X) if pre is not None else X.to_numpy()
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
except Exception as e:
    if DEBUG:
        st.exception(e)
    st.info("This build does not include the required preprocessing artifacts for Ridge interpretation.")
    st.stop()

coef = np.asarray(getattr(est, "coef_", np.array([]))).ravel()

shared, ct = _get_shared_and_ct(pre)
final_feature_names = None
if ct is not None and hasattr(ct, "get_feature_names_out"):
    try:
        final_feature_names = list(ct.get_feature_names_out())
    except Exception:
        final_feature_names = None

if final_feature_names is None or len(final_feature_names) != len(coef):
    final_feature_names = [f"f{i}" for i in range(len(coef))]

# 3.1 Coefficients
st.markdown("### 3.1 Strongest linear signals (coefficients)")

if len(coef) == 0:
    st.info("Coefficient view is not available for this Ridge estimator.")
else:
    coef_df = pd.DataFrame(
        {"feature": final_feature_names, "ridge_coef": coef, "abs_coef": np.abs(coef)}
    ).sort_values("abs_coef", ascending=False)
    coef_df = add_description_column(coef_df, "feature")

    st.caption(
        "Larger absolute coefficients usually indicate stronger influence (after scaling and one-hot encoding). "
        "The sign indicates direction (up/down). For one-hot features, a positive coefficient means that category "
        "is associated with higher prices relative to the baseline category."
    )
    st.dataframe(coef_df.head(30), use_container_width=True)

# 3.2 OLS refit (rename + safer framing)
st.markdown("### 3.2 Optional: coefficient evidence view (OLS refit)")
if FAST_VIEW:
    st.info("Quick view is ON — skipping this section.")
elif not _HAS_STATSMODELS:
    st.info("This build does not include the package needed for OLS refit.")
else:
    try:
        X_sm = sm.add_constant(X_proc, has_constant="add")
        ols = sm.OLS(y, X_sm).fit()

        rows = []
        for i, f in enumerate(final_feature_names, start=1):
            if i >= len(ols.params):
                break
            rows.append(
                {
                    "feature": f,
                    "coef": float(ols.params[i]),
                    "t": float(ols.tvalues[i]),
                    "p_value": float(ols.pvalues[i]),
                }
            )
        ols_df = pd.DataFrame(rows).sort_values("p_value", ascending=True)
        ols_df = add_description_column(ols_df, "feature")

        st.caption(
            "This is an interpretation aid (a separate OLS refit), not the deployed model. "
            "It provides a correlation-style signal of which processed features appear strongest."
        )
        st.dataframe(ols_df.head(30), use_container_width=True)
    except Exception as e:
        if DEBUG:
            st.exception(e)
        st.info("OLS refit could not be computed in this build.")

# 3.3 Permutation importance
st.markdown("### 3.3 Optional: sensitivity test (permutation importance)")
if FAST_VIEW:
    st.info("Quick view is ON — skipping this section.")
elif not _HAS_SKLEARN_EXTRAS:
    st.info("This build does not include the packages needed for permutation importance.")
else:
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred_log = ridge_pipe.predict(X_te)
        base_rmse_price = float(np.sqrt(mean_squared_error(y_te, np.expm1(y_pred_log))))

        if shared is None or ct is None:
            st.info("Permutation importance is not available for this Ridge pipeline in the current build.")
        else:
            X_shared_te = shared.transform(X_te)
            X_final_te = ct.transform(X_shared_te)
            if hasattr(X_final_te, "toarray"):
                X_final_te = X_final_te.toarray()

            names_pi = None
            if hasattr(ct, "get_feature_names_out"):
                try:
                    names_pi = list(ct.get_feature_names_out())
                except Exception:
                    names_pi = None
            if names_pi is None or len(names_pi) != X_final_te.shape[1]:
                names_pi = [f"f{i}" for i in range(X_final_te.shape[1])]

            perm = permutation_importance(
                est,
                X_final_te,
                np.log1p(y_te),
                n_repeats=8,
                random_state=42,
                scoring="neg_root_mean_squared_error",
            )

            perm_df = pd.DataFrame(
                {
                    "feature": names_pi,
                    "rmse_increase": -perm.importances_mean,
                    "std": perm.importances_std,
                }
            ).sort_values("rmse_increase", ascending=False)
            perm_df = add_description_column(perm_df, "feature")

            st.caption(
                f"Baseline error scale on this split is roughly **±{_format_currency(base_rmse_price)}**. "
                "Permutation importance estimates how much error increases when a feature is randomized."
            )
            st.dataframe(perm_df.head(30), use_container_width=True)
    except Exception as e:
        if DEBUG:
            st.exception(e)
        st.info("Permutation importance could not be computed in this build.")

st.success("Takeaway: Ridge is a readable baseline. Non-linear models typically win by capturing interactions (quality × size, etc.).")
st.divider()

st.caption(
    "Recommended reading order: start from the charts (Stage 1/2 + feature importance), "
    "then expand the detailed tables if you want the full evaluation record."
)
