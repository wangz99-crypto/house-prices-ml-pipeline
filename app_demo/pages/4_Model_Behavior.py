# app/pages/4_Model_Behavior.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import re

# =========================
# Optional dependencies
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
# Demo policy: allowlist + size guard
# =========================
ALLOWED_MODELS = {"ridge", "xgb", "lgbm"}     # ✅ only these can be joblib.load()
MAX_MODEL_MB = 50.0                          # ✅ demo safety guard (avoid 300MB artifacts)

# =========================
# Page config
# =========================
st.set_page_config(page_title="Model Behavior", layout="wide")
st.title("🧠 Model Behavior")
st.caption(
    "Product-friendly interpretation of model performance and drivers. "
    "Includes two-stage selection, feature importance, and Ridge interpretation layers."
)

st.info(
    f"Demo load policy: joblib.load() is allowed only for {sorted(ALLOWED_MODELS)} "
    f"and artifacts must be <= {MAX_MODEL_MB:.0f} MB. All other families are report-only."
)

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
    st.error("❌ artifacts_demo directory not found.")
    st.stop()

# ============================================================
# 0) Product-friendly feature dictionary (Ames + your engineered layer)
# ============================================================

AMES_DESC = {
    "MSSubClass": "Dwelling type code (1-story, 2-story, duplex, PUD, etc.).",
    "MSZoning": "Zoning classification of the property (RL/RM/FV/etc.).",
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
    "OverallQual": "Overall build quality rating (1–10; higher is better).",
    "OverallCond": "Overall condition rating (1–10; higher is better).",
    "YearBuilt": "Original construction year.",
    "YearRemodAdd": "Remodel/addition year (same as YearBuilt if never remodeled).",
    "RoofStyle": "Roof style (gable, hip, flat...).",
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
    "Functional": "Overall home functionality (Typ, Min, Mod, Maj...).",
    "Fireplaces": "Number of fireplaces.",
    "FireplaceQu": "Fireplace quality (Ex/Gd/TA/Fa/Po/NA).",
    "GarageType": "Garage location/type (attached, detached, none...).",
    "GarageYrBlt": "Garage build year.",
    "GarageFinish": "Garage interior finish (Fin/RFn/Unf/NA).",
    "GarageCars": "Garage capacity (# of cars).",
    "GarageArea": "Garage area in sq ft.",
    "GarageQual": "Garage quality (Ex/Gd/TA/Fa/Po/NA).",
    "GarageCond": "Garage condition (Ex/Gd/TA/Fa/Po/NA).",
    "PavedDrive": "Driveway paving (Y/P/N).",
    "WoodDeckSF": "Wood deck area in sq ft.",
    "OpenPorchSF": "Open porch area in sq ft.",
    "EnclosedPorch": "Enclosed porch area in sq ft.",
    "3SsnPorch": "Three-season porch area in sq ft.",
    "ScreenPorch": "Screen porch area in sq ft.",
    "PoolArea": "Pool area in sq ft.",
    "PoolQC": "Pool quality (Ex/Gd/TA/Fa/NA).",
    "Fence": "Fence quality (GdPrv/MnPrv/...).",
    "MiscFeature": "Miscellaneous feature (elevator, shed, tennis court...).",
    "MiscVal": "Value of miscellaneous feature ($).",
    "MoSold": "Month sold (1–12).",
    "YrSold": "Year sold.",
    "SaleType": "Type of sale (WD, New, COD, ...).",
    "SaleCondition": "Sale condition (Normal, Partial, Abnorml...).",
}

ENGINEERED_DESC = {
    "HouseAge": "Engineered: house age at sale (YrSold − YearBuilt).",
    "IsNewHouse": "Engineered: 1 if HouseAge ≤ 5 (new-ish home).",
    "RemodAge": "Engineered: years since remodel (YrSold − YearRemodAdd).",
    "IsRemodeled": "Engineered: 1 if remodeled (YearBuilt != YearRemodAdd).",
    "TotalSF": "Engineered: total footprint (basement + 1st + 2nd floors).",
    "TotalBathrooms": "Engineered: total baths (Full + 0.5*Half + basement baths).",
    "TotalPorchSF": "Engineered: total porch area (open + enclosed + 3-season + screen).",
    "HasBasement": "Engineered: 1 if basement exists (TotalBsmtSF > 0).",
    "HasGarage": "Engineered: 1 if garage exists (GarageArea > 0).",
    "HasFireplace": "Engineered: 1 if fireplace exists (Fireplaces > 0).",
    "HasPool": "Engineered: 1 if pool exists (PoolArea > 0).",
    "HasPorch": "Engineered: 1 if any porch exists (TotalPorchSF > 0).",
    "HasDeck": "Engineered: 1 if deck exists (WoodDeckSF > 0).",
    "HasMasonryVeneer": "Engineered: 1 if masonry veneer exists (MasVnrArea > 0).",
    "LuxuryAmenityScore": "Engineered: simple luxury score (pool + veneer + deck + porch + fireplace).",
    "QualGrLiv": "Engineered: quality × living area (OverallQual × GrLivArea).",
    "QualTotalSF": "Engineered: quality × total area (OverallQual × TotalSF).",
    "QualGarage": "Engineered: quality × garage area (OverallQual × GarageArea).",
    "IsHighQuality": "Engineered: 1 if OverallQual ≥ 7.",
    "IsLargeHouse": "Engineered: 1 if GrLivArea ≥ 2000 sq ft.",
    "IsLuxury": "Engineered: 1 if high quality AND large (OverallQual ≥ 8 and GrLivArea ≥ 2500).",
    "GrLivAreaBin": "Engineered: living area bucket (small / mid / large).",
    "Neighborhood_Qual": "Engineered: Neighborhood combined with OverallQual (captures local premium by quality).",
}

QUALITY_LEVEL = {
    "Ex": "Excellent",
    "Gd": "Good",
    "TA": "Typical/Average",
    "Fa": "Fair",
    "Po": "Poor",
    "NA": "Not applicable / None",
}
YESNO_LEVEL = {"Y": "Yes", "N": "No"}


def _strip_nested_prefix(name: str) -> str:
    """Normalize sklearn names like num__LotArea -> LotArea."""
    s = str(name)
    while "__" in s:
        _, s = s.split("__", 1)
    return s


def describe_feature(name: str) -> str:
    """
    Product-friendly description for:
    - base features
    - engineered features
    - *_log
    - one-hot fields: Field_Level
    """
    s = _strip_nested_prefix(name)

    if s in ENGINEERED_DESC:
        return ENGINEERED_DESC[s]

    if s.endswith("_log"):
        base = s[:-4]
        base_h = AMES_DESC.get(base, f"{base} (raw field)")
        return f"Engineered: log transform of {base} (reduces skew, stabilizes relationships). {base_h}"

    # One-hot: try split Field_Level
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
        return f"Engineered indicator for amenity presence: {s}."
    if s.startswith("Is"):
        return f"Engineered flag feature: {s}."
    if s.endswith("Bin"):
        return f"Engineered bucket feature: {s}."

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
    """Unwrap wrapper objects (e.g., AsRegressor) down to a sklearn Pipeline or estimator."""
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
    """Return (preprocessor, estimator)."""
    pipe = _unwrap_estimator(obj)
    if hasattr(pipe, "steps") and hasattr(pipe, "named_steps"):
        est = pipe.steps[-1][1]
        pre = Pipeline(pipe.steps[:-1]) if (Pipeline is not None and len(pipe.steps) > 1) else None
        return pre, est
    return None, pipe


@st.cache_data(show_spinner=False)
def _load_train_data(repo_root: Path) -> Tuple[pd.DataFrame, pd.Series]:
    # ✅ demo-safe: prefer sample_train; raw only as local fallback
    p_sample = repo_root / "tests" / "data" / "sample_train.csv"
    p_raw = repo_root / "data" / "raw" / "train.csv"

    if p_sample.exists():
        df = pd.read_csv(p_sample)
    elif p_raw.exists():
        df = pd.read_csv(p_raw)
    else:
        raise FileNotFoundError(
            "Missing training dataset.\n"
            f"Checked:\n- {p_sample}\n- {p_raw}\n\n"
            "Demo build expects tests/data/sample_train.csv."
        )

    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    return X, y


def _get_shared_and_ct(pre):
    """
    Your ridge preprocessor structure:
      pre = Pipeline([('shared', Pipeline([...MissingValueHandler, FeatureEngineerV2...])),
                      ('prep', ColumnTransformer(...))])
    """
    if pre is None or not hasattr(pre, "named_steps"):
        return None, None
    shared = pre.named_steps.get("shared")
    ct = pre.named_steps.get("prep")
    return shared, ct


def _find_run_dirs(model_name: str) -> List[Path]:
    # ✅ hard deny non-allowed models for any loader logic
    if model_name not in ALLOWED_MODELS:
        return []
    base = REGISTRY_DIR / model_name
    if not base.exists():
        return []
    run_dirs = [d for d in base.iterdir() if d.is_dir() and d.name not in {"_global"}]
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs


@st.cache_data(show_spinner=False)
def _load_latest_pipeline_and_run(demo_dir: Path, model_name: str):
    # ✅ allowlist enforcement
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"Demo build only allows loading models: {sorted(ALLOWED_MODELS)}. Requested: {model_name}"
        )

    run_dirs = _find_run_dirs(model_name)
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under registry/{model_name}/")
    run_dir = run_dirs[0]
    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.joblib in: {run_dir}")

    # ✅ size guard
    mb = model_path.stat().st_size / 1024 / 1024
    if mb > MAX_MODEL_MB:
        raise RuntimeError(
            f"Demo safety guard: {model_name} model is too large ({mb:.1f} MB > {MAX_MODEL_MB} MB).\n"
            "This demo build intentionally excludes large ensembles."
        )

    pipe = joblib.load(model_path)
    return pipe, run_dir


def _normalize_stage(s: str) -> str:
    s0 = str(s).strip().lower().replace(" ", "")
    if s0 in {"stage1", "1", "s1"} or ("stage" in s0 and "1" in s0):
        return "Stage1"
    if s0 in {"stage2", "2", "s2"} or ("stage" in s0 and "2" in s0):
        return "Stage2"
    return str(s).strip()


# ============================================================
# 1) Model Comparison (productized)
# ============================================================
st.subheader("1) Model Comparison")

st.markdown(
    """
**What you’re seeing:** we don’t pick a model from a single lucky run.  
Instead, we use a **two-stage selection** to balance **speed + stability + generalization**.

- **Stage 1 (Screening):** quickly test many candidates → keep only the most promising ones  
- **Stage 2 (Stability check):** re-test the top candidates with more randomness → pick the most reliable winner  
- The winner is recorded in the **Model Registry** as `best` (and the newest run as `latest`)
"""
)

a1, a2, a3 = st.columns(3)
with a1:
    st.metric("Stage 1 goal", "Eliminate weak configs")
with a2:
    st.metric("Stage 2 goal", "Stability / robustness")
with a3:
    st.metric("Output", "Registry aliases: best/latest")

# --- show figures if present ---
c1, c2 = st.columns(2, gap="large")
with c1:
    p = FIG_DIR / "model_comparison_stage1.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.info("Stage1 figure not found. Run: python tools/make_app_figures.py")
with c2:
    p = FIG_DIR / "model_comparison_stage2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.info("Stage2 figure not found. Run: python tools/make_app_figures.py")

# --- lightweight “intuition card” for RMSE (demo-safe: no raw train dependency) ---
try:
    if SUMMARY_CSV.exists():
        _, y_all = _load_train_data(REPO_ROOT)  # ✅ sample_train preferred
        p50 = float(np.median(y_all))

        df_sum = pd.read_csv(SUMMARY_CSV)
        df_sum.columns = [c.strip() for c in df_sum.columns]
        if "stage" in df_sum.columns:
            df_sum["stage"] = df_sum["stage"].apply(_normalize_stage)
        stage2 = df_sum[df_sum["stage"] == "Stage2"] if "stage" in df_sum.columns else df_sum

        if len(stage2) and "rmse_mean" in stage2.columns:
            best = stage2.sort_values("rmse_mean", ascending=True).iloc[0]
            e = float(best["rmse_mean"])
            approx_pct = float(np.expm1(e))
            approx_usd = p50 * approx_pct

            st.info(
                f"**RMSE intuition (for quick understanding):** "
                f"best Stage2 RMSE ≈ **{e:.4f}** (log space). "
                f"At a median-priced home (~**${p50:,.0f}**), that’s roughly an error scale of **±${approx_usd:,.0f}**. "
                f"*(This is for intuition, not a guarantee.)*"
            )
except Exception:
    pass

# --- table moved into expander (advanced) ---
if SUMMARY_CSV.exists():
    df_sum = pd.read_csv(SUMMARY_CSV)
    df_sum.columns = [c.strip() for c in df_sum.columns]
    if "stage" in df_sum.columns:
        df_sum["stage"] = df_sum["stage"].apply(_normalize_stage)

    # keep it tidy: sort by stage then rmse
    if {"stage", "rmse_mean"}.issubset(df_sum.columns):
        df_sum = df_sum.sort_values(["stage", "rmse_mean"], ascending=[True, True])

    with st.expander("Show evaluation table (advanced)"):
        st.caption("Sorted by Stage then RMSE (lower is better). Showing top 30 rows per stage.")
        if "stage" in df_sum.columns:
            parts = []
            for s in sorted(df_sum["stage"].unique()):
                top = df_sum[df_sum["stage"] == s].head(30)
                parts.append(top)
            show_df = pd.concat(parts, axis=0) if parts else df_sum.head(60)
        else:
            show_df = df_sum.head(60)
        st.dataframe(show_df, use_container_width=True)

st.divider()

# ============================================================
# 2) Feature Importance (Tree / Boosting Models) — exclude Ridge
# ============================================================
st.subheader("2) Feature Importance (Tree / Boosting Models)")

imgs = sorted(FIG_DIR.glob("feat_importance_top20__*.png"))
names_all = [p.stem.replace("feat_importance_top20__", "") for p in imgs]

# ✅ keep report-driven view but restrict to allowed families (excluding ridge)
names = [n for n in names_all if (n.lower() in {"xgb", "lgbm"})]

if not names:
    st.info("No tree/boosting feature-importance figures found (xgb/lgbm).")
else:
    model = st.selectbox("Select a model family", names, index=0)
    p = FIG_DIR / f"feat_importance_top20__{model}.png"
    if p.exists():
        st.image(str(p), use_container_width=True)

    # optional CSV table with descriptions
    csv_candidates = sorted(FI_DIR.glob(f"{model}__*__top30.csv"))
    if csv_candidates:
        with st.expander("Show top features as a table (with descriptions)"):
            try:
                df = pd.read_csv(csv_candidates[-1])
                if "feature" in df.columns:
                    df = add_description_column(df, "feature")
                st.dataframe(df.head(30), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not read importance CSV: {e}")

st.divider()

# ============================================================
# 3) Linear Model (Ridge) — interpretation layers
# ============================================================
st.subheader("3) Linear Model (Ridge) — Interpretation Layers")
st.caption(
    "Ridge is interpreted differently from trees: we look at coefficients, a refit OLS layer, "
    "and permutation importance (which measures error increase when a feature is shuffled)."
)

DEMO_FAST = st.toggle("Fast demo mode (skip OLS + permutation importance)", value=True)

try:
    ridge_pipe, _ridge_run_dir = _load_latest_pipeline_and_run(DEMO, "ridge")  # ✅ allowlisted + size-guarded
except Exception as e:
    st.error(f"Could not load Ridge run: {e}")
    st.stop()

pre, est = _split_pipeline(ridge_pipe)
X, y = _load_train_data(REPO_ROOT)  # ✅ sample_train preferred

# transform for interpretation layers
try:
    X_proc = pre.transform(X) if pre is not None else X.to_numpy()
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
except Exception as e:
    st.error(f"Failed to preprocess training data for Ridge interpretation: {e}")
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

# --- 3.1 Ridge coefficients ---
st.markdown("### 3.1 Ridge coefficients (trained model)")
if len(coef) == 0:
    st.warning("Ridge coefficients not found on this estimator.")
else:
    coef_df = pd.DataFrame(
        {"feature": final_feature_names, "ridge_coef": coef, "abs_coef": np.abs(coef)}
    ).sort_values("abs_coef", ascending=False)
    coef_df = add_description_column(coef_df, "feature")

    st.caption(
        "Bigger absolute coefficients typically mean stronger influence (after scaling + one-hot encoding). "
        "Sign tells direction (up/down)."
    )
    st.dataframe(coef_df.head(30), use_container_width=True)

# --- 3.2 OLS p-values (interpretation layer) ---
st.markdown("### 3.2 OLS refit p-values (interpretation layer)")
if DEMO_FAST:
    st.info("Fast demo mode is ON — skipping OLS refit.")
elif not _HAS_STATSMODELS:
    st.info("statsmodels is not installed; skipping p-values.")
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
            "This is a separate OLS refit used as an interpretation aid. "
            "It’s not the deployed model, but helps explain which signals are statistically strong."
        )
        st.dataframe(ols_df.head(30), use_container_width=True)
    except Exception as e:
        st.warning(f"OLS refit failed: {e}")

# --- 3.3 Permutation importance (final feature space) ---
st.markdown("### 3.3 Permutation importance (RMSE increase) — final feature space")

if DEMO_FAST:
    st.info("Fast demo mode is ON — skipping permutation importance.")
elif not _HAS_SKLEARN_EXTRAS:
    st.info("sklearn extras not available; skipping permutation importance.")
else:
    try:
        # split raw data
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # baseline RMSE in original price space for human readability
        y_pred_log = ridge_pipe.predict(X_te)
        base_rmse_price = float(np.sqrt(mean_squared_error(y_te, np.expm1(y_pred_log))))

        if shared is None or ct is None:
            st.warning("Could not access Ridge preprocessing steps ('shared' / 'prep'). Skipping permutation importance.")
        else:
            # final feature matrix
            X_shared_te = shared.transform(X_te)
            X_final_te = ct.transform(X_shared_te)
            if hasattr(X_final_te, "toarray"):
                X_final_te = X_final_te.toarray()

            # aligned names
            names_pi = None
            if hasattr(ct, "get_feature_names_out"):
                try:
                    names_pi = list(ct.get_feature_names_out())
                except Exception:
                    names_pi = None
            if names_pi is None or len(names_pi) != X_final_te.shape[1]:
                names_pi = [f"f{i}" for i in range(X_final_te.shape[1])]

            # permute on estimator only (expects final numeric matrix)
            perm = permutation_importance(
                est,
                X_final_te,
                np.log1p(y_te),  # estimator trained in log space
                n_repeats=8,
                random_state=42,
                scoring="neg_root_mean_squared_error",
            )

            perm_df = pd.DataFrame(
                {
                    "feature": names_pi,
                    "rmse_increase": -perm.importances_mean,  # positive means worse when shuffled
                    "std": perm.importances_std,
                }
            ).sort_values("rmse_increase", ascending=False)
            perm_df = add_description_column(perm_df, "feature")

            st.caption(
                f"Baseline (rough) error scale on this holdout split: **±${base_rmse_price:,.0f}**. "
                "Permutation importance shows how much error increases when a feature is randomized."
            )
            st.dataframe(perm_df.head(30), use_container_width=True)
    except Exception as e:
        st.warning(f"Permutation importance failed: {e}")

st.divider()

st.caption(
    "Tip: This page is designed for mixed audiences. "
    "Non-technical viewers can focus on the charts + the RMSE intuition card; "
    "advanced viewers can expand the full evaluation tables."
)
