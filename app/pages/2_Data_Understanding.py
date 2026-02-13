import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lib.ui_style import hero, section

# --------------------------------------------------
# Load data
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "raw" / "train.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    return df

df = load_data()

# --------------------------------------------------
# Hero
# --------------------------------------------------

hero(
    "📊 Data Understanding",
    "Learn how housing prices behave and why predictive modeling is possible.",
)

st.markdown(
"""
This section builds intuition about the dataset before modeling.

We explore:

• Market price structure  
• Feature–price relationships  
• Differences between low and high price homes  

The goal is to understand **why** machine learning can predict prices — not just that it can.
"""
)

# ==================================================
# Section A — Price distribution
# ==================================================

section("Market price structure", "How are home prices distributed?", "💰")

st.markdown(
"""
### What to look for

Housing prices are rarely evenly distributed.  
Real markets show clustering, skew, and outliers.

Understanding this helps:

✅ Detect luxury outliers  
✅ Understand scale differences  
✅ Choose modeling strategies
"""
)

bins = st.slider("Histogram bins", 20, 120, 60)

fig, ax = plt.subplots()

ax.hist(df["SalePrice"], bins=bins)
ax.set_title("House price distribution")
ax.set_xlabel("Price ($)")
ax.set_ylabel("Count")

st.pyplot(fig)

st.caption(
    "Most homes fall into a mid-market cluster, with a long right tail representing premium properties."
)

# ==================================================
# Section B — Feature vs price
# ==================================================

section("Feature–price relationship", "Which variables drive price?", "📈")

st.markdown(
"""
### What to look for

If price rises consistently with a feature:

👉 The model has a learnable signal.

Weak or noisy relationships:

👉 Require more complex modeling.

This visualization reveals raw predictive structure.
"""
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove("SalePrice")

feature = st.selectbox("Choose a feature", numeric_cols)

fig, ax = plt.subplots()

ax.scatter(df[feature], df["SalePrice"], alpha=0.4)
ax.set_xlabel(feature)
ax.set_ylabel("Price ($)")
ax.set_title(f"{feature} vs Price")

st.pyplot(fig)

st.caption(
    "Clear upward patterns indicate strong predictors of value."
)

# ==================================================
# Section C — Cheap vs expensive comparison
# ==================================================

section("Price segment comparison", "How do market tiers differ structurally?", "🔍")

st.markdown(
"""
### What to look for

Homes in different price tiers often show:

• Structural differences  
• Quality gaps  
• Amenity variations  

This confirms that pricing reflects measurable characteristics —
not random variation.
"""
)

threshold = df["SalePrice"].median()

df["Segment"] = np.where(
    df["SalePrice"] > threshold,
    "High price",
    "Low price"
)

compare_feature = st.selectbox(
    "Feature to compare",
    numeric_cols,
    key="compare"
)

fig, ax = plt.subplots()

df.boxplot(column=compare_feature, by="Segment", ax=ax)

plt.suptitle("")
ax.set_title(compare_feature)

st.pyplot(fig)

st.caption(
    "Distribution shifts reveal structural differences between market tiers."
)

# --------------------------------------------------
# Closing insight
# --------------------------------------------------

st.markdown("---")

st.info(
"""
✅ Key takeaway:

Housing price is not random.

It reflects measurable structure — size, quality, amenities —  
which machine learning models can systematically learn.

Next step → see how models capture these relationships.
"""
)
