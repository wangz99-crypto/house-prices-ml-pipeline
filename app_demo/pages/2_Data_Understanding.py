import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lib.ui_style import hero, section

# --------------------------------------------------
# Paths
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
    raise FileNotFoundError(
        "No dataset found.\n"
        f"Checked:\n- {SAMPLE}\n- {RAW}\n\n"
        "Demo build expects tests/data/sample_train.csv to exist."
    )


df = load_data()

# --------------------------------------------------
# Hero
# --------------------------------------------------
hero(
    "📊 Data Understanding",
    "Learn how housing prices behave and why predictive modeling is possible.",
)

st.caption(
    "Demo build uses a small reproducible sample dataset: `tests/data/sample_train.csv` "
    "(raw Kaggle files are not shipped online)."
)

# --------------------------------------------------
# Overview stats
# --------------------------------------------------
section("Dataset snapshot", "Quick summary of the demo dataset.", "🧾")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{len(df):,}")
with col2:
    st.metric("Columns", f"{df.shape[1]:,}")
with col3:
    st.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
with col4:
    st.metric("Target present", "SalePrice" if "SalePrice" in df.columns else "N/A")

st.dataframe(df.head(30), use_container_width=True)

# --------------------------------------------------
# Target distribution
# --------------------------------------------------
if "SalePrice" in df.columns:
    section("Target distribution", "SalePrice is right-skewed, motivating log transforms.", "📈")

    prices = df["SalePrice"].astype(float)

    fig = plt.figure()
    plt.hist(prices, bins=30)
    plt.title("SalePrice distribution (sample)")
    plt.xlabel("SalePrice")
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)

    fig2 = plt.figure()
    plt.hist(np.log1p(prices), bins=30)
    plt.title("log1p(SalePrice) distribution (sample)")
    plt.xlabel("log1p(SalePrice)")
    plt.ylabel("Count")
    st.pyplot(fig2, clear_figure=True)

# --------------------------------------------------
# Simple relationships (safe subset)
# --------------------------------------------------
section("A few key drivers", "Small sample correlations for intuition only.", "🔍")

features = [c for c in ["OverallQual", "GrLivArea", "GarageCars", "YearBuilt"] if c in df.columns]
if "SalePrice" in df.columns and features:
    sub = df[features + ["SalePrice"]].dropna()
    corr = sub.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
    st.write("Correlation with SalePrice (sample):")
    st.dataframe(corr.to_frame("corr"), use_container_width=True)
else:
    st.info("Sample file does not contain expected columns for the quick correlation view.")
