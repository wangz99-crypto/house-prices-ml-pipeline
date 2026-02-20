from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

from lib.ui_style import hero, section

# --------------------------------------------------
# Data loading (robust for demo/portfolio builds)
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "tests" / "data" / "sample_train.csv"
RAW = REPO_ROOT / "data" / "raw" / "train.csv"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if SAMPLE.exists():
        return pd.read_csv(SAMPLE)
    if RAW.exists():
        return pd.read_csv(RAW)
    raise FileNotFoundError("Dataset files are not available in this build.")


# --------------------------------------------------
# UI helpers
# --------------------------------------------------
def money_fmt(x: float, _pos: int) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def safe_pearsonr(x: pd.Series, y: pd.Series) -> float | None:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return None
    return float(np.corrcoef(x[m].to_numpy(), y[m].to_numpy())[0, 1])


def build_display_maps(columns: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      raw_to_display: raw feature name -> display label
      display_to_raw: display label -> raw feature name
    """
    raw_to_display = {
        # Core, audience-friendly (high value to show in demo)
        "OverallQual": "Overall Quality (1–10)",
        "GrLivArea": "Above-Ground Living Area (sq ft)",
        "TotalBsmtSF": "Basement Area (sq ft)",
        "GarageCars": "Garage Capacity (cars)",
        "GarageArea": "Garage Area (sq ft)",
        "LotArea": "Lot Size (sq ft)",
        "LotFrontage": "Street Frontage (ft)",
        "YearBuilt": "Year Built",
        "YearRemodAdd": "Year Remodeled",
        "FullBath": "Full Bathrooms",
        "HalfBath": "Half Bathrooms",
        "TotRmsAbvGrd": "Total Rooms Above Grade",
        "Fireplaces": "Number of Fireplaces",
        "1stFlrSF": "First Floor Area (sq ft)",
        "2ndFlrSF": "Second Floor Area (sq ft)",
        "BsmtFinSF1": "Finished Basement Area (Type 1, sq ft)",
        "BsmtFinSF2": "Finished Basement Area (Type 2, sq ft)",
        "BsmtUnfSF": "Unfinished Basement Area (sq ft)",
        "WoodDeckSF": "Deck Area (sq ft)",
        "OpenPorchSF": "Open Porch Area (sq ft)",
        "EnclosedPorch": "Enclosed Porch Area (sq ft)",
        "ScreenPorch": "Screen Porch Area (sq ft)",
        "PoolArea": "Pool Area (sq ft)",
        "MasVnrArea": "Masonry Veneer Area (sq ft)",
        "BedroomAbvGr": "Bedrooms Above Grade",
        "KitchenAbvGr": "Kitchens Above Grade",
        # Engineered (optional advanced)
        "TotalSF": "Total Finished SF (engineered)",
        "HouseAge": "House Age (years, engineered)",
        "RemodAge": "Years Since Remodel (engineered)",
        "LuxuryAmenityScore": "Luxury Amenity Score (engineered)",
        "QualTotalSF": "Quality × TotalSF (engineered)",
        "QualGrLiv": "Quality × Living Area (engineered)",
        "QualGarage": "Quality × Garage Area (engineered)",
        "TotalBathrooms": "Total Bathrooms (engineered)",
        "TotalPorchSF": "Total Porch SF (engineered)",
        "ServiceCount": "Service Count (engineered)",  # if exists in your pipeline; safe to ignore otherwise
        "IsNewHouse": "New House Flag (engineered)",
        "IsRemodeled": "Remodeled Flag (engineered)",
        "IsHighQuality": "High Quality Flag (engineered)",
        "IsLargeHouse": "Large House Flag (engineered)",
        "IsLuxury": "Luxury Flag (engineered)",
    }

    # Only keep entries that exist in current df
    raw_to_display = {k: v for k, v in raw_to_display.items() if k in columns}

    # Ensure unique display labels (avoid collisions)
    seen = set()
    for k, v in list(raw_to_display.items()):
        if v in seen:
            raw_to_display[k] = f"{v} [{k}]"
        seen.add(raw_to_display[k])

    display_to_raw = {v: k for k, v in raw_to_display.items()}
    return raw_to_display, display_to_raw


# --------------------------------------------------
# Page header
# --------------------------------------------------
hero(
    "Data Understanding",
    "A visual, interactive look at the patterns behind the prediction task.",
)

st.markdown(
    """
This section helps you build intuition about the data.
Explore how prices are distributed, how key features relate to price,
and how lower- and higher-priced homes differ in measurable ways.
"""
)

# Load data with product-friendly handling
try:
    df = load_data()
except Exception:
    st.warning(
        "Data is not available in this build. If you are running locally, ensure the dataset files are present."
    )
    st.stop()

if "SalePrice" not in df.columns:
    st.warning("This dataset does not include the target column **SalePrice**.")
    st.stop()

prices = pd.to_numeric(df["SalePrice"], errors="coerce").dropna()
if prices.empty:
    st.warning("SalePrice contains no usable numeric values in this dataset.")
    st.stop()

# --------------------------------------------------
# Feature selection policy (product-first)
# --------------------------------------------------
# Default: show only highly interpretable numeric features
CORE_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "LotArea",
    "LotFrontage",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "HalfBath",
    "TotRmsAbvGrd",
    "Fireplaces",
    "1stFlrSF",
    "2ndFlrSF",
]

# Advanced engineered features (only shown if present and toggle is on)
ENGINEERED_FEATURES = [
    "TotalSF",
    "HouseAge",
    "RemodAge",
    "LuxuryAmenityScore",
    "QualTotalSF",
    "QualGrLiv",
    "QualGarage",
    "TotalBathrooms",
    "TotalPorchSF",
    "IsNewHouse",
    "IsRemodeled",
    "IsHighQuality",
    "IsLargeHouse",
    "IsLuxury",
]

# Keep only what exists in df and is numeric
numeric_cols = set(df.select_dtypes(include=np.number).columns.tolist())
numeric_cols.discard("SalePrice")

core_available = [c for c in CORE_FEATURES if c in numeric_cols]
eng_available = [c for c in ENGINEERED_FEATURES if c in numeric_cols]

raw_to_display, display_to_raw = build_display_maps(list(df.columns))

# Global UI toggles
with st.expander("Display options", expanded=False):
    use_log = st.checkbox("Use log scale for SalePrice (helps long-tail readability)", value=False)
    show_money_axis = st.checkbox("Format axes as currency", value=True)
    show_engineered = st.checkbox("Show engineered features (advanced)", value=False, disabled=(len(eng_available) == 0))
    if len(eng_available) == 0:
        st.caption("No engineered numeric features detected in this dataset build.")

# Final ordered feature list for dropdowns
ordered = core_available + (eng_available if show_engineered else [])
if not ordered:
    st.warning("No numeric features are available for interactive exploration in this build.")
    st.stop()

# Build UI display options (friendly labels)
display_options = {}
for raw in ordered:
    label = raw_to_display.get(raw, raw)
    display_options[label] = raw

# --------------------------------------------------
# Section A — Price distribution
# --------------------------------------------------
section("Price distribution", "How are home prices spread across the market?", "💰")

st.markdown(
    """
Home prices often cluster around a mid-range band, with a smaller number of high-priced properties.
This view helps you see the overall spread and the presence of a long right tail.
"""
)

bins = st.slider("Histogram detail", 20, 120, 60)

plot_prices = np.log1p(prices) if use_log else prices

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.hist(plot_prices, bins=bins)
ax.set_title("Home price distribution" + (" (log scale)" if use_log else ""))
ax.set_xlabel("SalePrice" + (" (log1p)" if use_log else ""))
ax.set_ylabel("Count")

# Median marker
med = float(np.median(plot_prices))
ax.axvline(med, linestyle="--")
ax.text(med, ax.get_ylim()[1] * 0.95, "median", rotation=90, va="top", ha="right")

if show_money_axis and not use_log:
    ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))

st.pyplot(fig)
st.caption("A visible long right tail indicates fewer, much higher-priced homes.")

# --------------------------------------------------
# Section B — Feature vs price
# --------------------------------------------------
section("Feature relationships", "How do individual features relate to price?", "📈")

st.markdown(
    """
Select a feature to see how it moves with price.
Clear upward patterns often indicate a useful signal for prediction.
"""
)

selected_display = st.selectbox("Select a feature", list(display_options.keys()))
feature = display_options[selected_display]

x = pd.to_numeric(df[feature], errors="coerce")
y_raw = pd.to_numeric(df["SalePrice"], errors="coerce")
y = np.log1p(y_raw) if use_log else y_raw

mask = x.notna() & y.notna()

r = safe_pearsonr(x[mask], y_raw[mask])  # show correlation vs raw price for interpretability
if r is None:
    st.caption("Correlation: not enough valid data points to compute.")
else:
    st.caption(f"Correlation (Pearson r, vs raw SalePrice): **{r:.3f}**")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.scatter(x[mask], y[mask], alpha=0.35)
ax.set_xlabel(selected_display)
ax.set_ylabel("SalePrice" + (" (log1p)" if use_log else ""))
ax.set_title(f"{selected_display} vs SalePrice" + (" (log scale)" if use_log else ""))

if show_money_axis and not use_log:
    ax.yaxis.set_major_formatter(FuncFormatter(money_fmt))

st.pyplot(fig)
st.caption("This is a raw view (no interactions). Real models learn from many signals at once.")

# --------------------------------------------------
# Section C — Price segment comparison
# --------------------------------------------------
section("Market tiers", "How do lower- and higher-priced homes differ?", "🔍")

st.markdown(
    """
To make comparisons easier, the dataset is split into price tiers.
Compare a feature across tiers to see how typical homes differ by segment.
"""
)

split_mode = st.radio(
    "Tier split",
    options=["Median (50/50)", "Top quartile (25% high-end)"],
    horizontal=True,
)

if split_mode.startswith("Median"):
    threshold = float(prices.median())
    high_mask = pd.to_numeric(df["SalePrice"], errors="coerce") > threshold
else:
    threshold = float(prices.quantile(0.75))
    high_mask = pd.to_numeric(df["SalePrice"], errors="coerce") >= threshold

seg = np.where(high_mask, "Higher-priced", "Lower-priced")

selected_display2 = st.selectbox("Select a feature to compare", list(display_options.keys()), key="compare")
compare_feature = display_options[selected_display2]

x2 = pd.to_numeric(df[compare_feature], errors="coerce")
m2 = x2.notna()

plot_df = pd.DataFrame(
    {"Segment": seg[m2.to_numpy()], selected_display2: x2[m2].to_numpy()}
)
plot_df["Segment"] = pd.Categorical(
    plot_df["Segment"],
    categories=["Lower-priced", "Higher-priced"],
    ordered=True,
)

fig, ax = plt.subplots(figsize=(8.5, 4.4))
plot_df.boxplot(column=selected_display2, by="Segment", ax=ax)
plt.suptitle("")
ax.set_title(selected_display2)
ax.set_xlabel("")
ax.set_ylabel(selected_display2)
st.pyplot(fig)

st.caption("Shifts in the distributions suggest measurable differences between market tiers.")

# --------------------------------------------------
# Closing
# --------------------------------------------------
st.markdown("---")
st.info(
    "Key takeaway: housing prices reflect measurable structure (space, quality, amenities). "
    "This is what enables a predictive model to learn consistent patterns."
)
