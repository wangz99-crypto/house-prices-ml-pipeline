from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Data Drift Monitor", layout="wide")

# ------------------------------------------------
# Paths
# ------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = REPO_ROOT / "data" / "raw" / "test.csv"

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def psi(expected, actual, bins=10):
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    e_hist, _ = np.histogram(expected, bins=breakpoints)
    a_hist, _ = np.histogram(actual, bins=breakpoints)

    e_perc = e_hist / max(e_hist.sum(), 1)
    a_perc = a_hist / max(a_hist.sum(), 1)

    psi_val = np.sum((a_perc - e_perc) * np.log((a_perc + 1e-6) / (e_perc + 1e-6)))
    return psi_val


def cat_shift(base, new, topk=5):
    base_dist = base.value_counts(normalize=True)
    new_dist = new.value_counts(normalize=True)

    cats = set(base_dist.index) | set(new_dist.index)

    rows = []
    for c in cats:
        rows.append({
            "category": c,
            "train_pct": base_dist.get(c, 0),
            "test_pct": new_dist.get(c, 0),
            "shift": abs(base_dist.get(c, 0) - new_dist.get(c, 0)),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("shift", ascending=False).head(topk)


def drift_flag(score):
    if np.isnan(score):
        return "—"
    if score < 0.1:
        return "🟢 Stable"
    if score < 0.25:
        return "🟡 Moderate"
    return "🔴 Severe"


# ------------------------------------------------
# Load data
# ------------------------------------------------
st.title("📊 Data Drift Monitor")

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    st.error("train/test CSV not found.")
    st.stop()

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

st.success("Loaded baseline (train) vs new data (test)")

# ------------------------------------------------
# Numeric Drift — PSI
# ------------------------------------------------
st.subheader("1) Numeric Drift (PSI)")

num_cols = train.select_dtypes(include=np.number).columns

psi_rows = []
for col in num_cols:
    if col not in test.columns:
        continue
    score = psi(train[col], test[col])
    psi_rows.append({
        "feature": col,
        "psi": score,
        "status": drift_flag(score),
    })

psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)

st.dataframe(psi_df, use_container_width=True)

# ------------------------------------------------
# Missing Drift
# ------------------------------------------------
st.subheader("2) Missing Value Drift")

missing_rows = []
for col in train.columns:
    if col not in test.columns:
        continue

    train_m = train[col].isna().mean()
    test_m = test[col].isna().mean()

    missing_rows.append({
        "feature": col,
        "train_missing": train_m,
        "test_missing": test_m,
        "shift": abs(train_m - test_m),
    })

missing_df = pd.DataFrame(missing_rows).sort_values("shift", ascending=False)

st.dataframe(missing_df.head(20), use_container_width=True)

# ------------------------------------------------
# Categorical Drift
# ------------------------------------------------
st.subheader("3) Categorical Drift")

cat_cols = train.select_dtypes(include="object").columns

col_choice = st.selectbox(
    "Choose categorical feature",
    cat_cols
)

if col_choice:
    drift_table = cat_shift(train[col_choice], test[col_choice])
    st.dataframe(drift_table, use_container_width=True)

# ------------------------------------------------
# Summary
# ------------------------------------------------
st.subheader("System Drift Summary")

severe = (psi_df["psi"] > 0.25).sum()
moderate = ((psi_df["psi"] >= 0.1) & (psi_df["psi"] <= 0.25)).sum()

st.metric("Severe drift features", severe)
st.metric("Moderate drift features", moderate)

st.info(
"""
🧠 Interpretation:

🟢 Stable — model likely safe  
🟡 Moderate — monitor closely  
🔴 Severe — retraining recommended  

This simulates production data monitoring.
"""
)
