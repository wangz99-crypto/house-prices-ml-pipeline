# app/pages/1_Overview.py
from __future__ import annotations

from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Overview", layout="wide")

st.title("🏠 House Price Prediction — ML Engineering Demo")

st.markdown(
"""
## What is this system?

This project is a **production-style machine learning system** built around a classic business problem:

👉 **Predicting house prices from structured real-estate data**

Instead of focusing only on model accuracy (like a typical Kaggle notebook),
this demo shows how a predictive model can be engineered into a **reliable, reproducible system** — closer to how ML is deployed in real organizations.

The dataset comes from the Kaggle competition:

**House Prices — Advanced Regression Techniques**

which simulates real-world property valuation scenarios.
"""
)

# --------- Resolve repo/demo paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO1 = REPO_ROOT / "artifacts_demo"
DEMO2 = REPO_ROOT / "app" / "artifacts_demo"
DEMO = DEMO1 if DEMO1.exists() else DEMO2
REPORTS = DEMO / "reports"

left, right = st.columns([2, 1], gap="large")

# ================= LEFT =================

with left:

    st.subheader("Why this system is different")

    st.markdown(
"""
Most ML demos stop at training a model.

This system demonstrates what happens **after training** — the engineering needed to make ML reliable:

### 🔹 Reproducible training pipeline  
Models can be rebuilt consistently across environments.

### 🔹 Model version management  
Every model run is tracked and stored — similar to software version control.

### 🔹 Cross-validated evaluation  
Performance is measured rigorously, not by a single lucky split.

### 🔹 Ensemble modeling  
Multiple model families combine to improve robustness.

### 🔹 Engineering safeguards  
Model contracts, sanity checks, and drift monitoring prevent silent failures.

In short:

👉 **This is not just a model — it’s a miniature ML platform.**
"""
    )

    st.subheader("What you can explore")

    st.markdown(
"""
Use the sidebar to navigate:

### 📊 Data Understanding  
Understand the housing dataset and key signals.

### 🔮 Live Prediction  
Enter property features and get a price prediction.

### 🧠 Model Behavior  
Compare models, feature importance, and diagnostics.

### 🧪 Experiments & Analysis  
See how evaluation and iteration work.

### 🗂 Model Registry  
View stored model versions and metadata.

Each page reveals a different layer of the ML lifecycle —
from raw data → modeling → system reliability.
"""
    )

# ================= RIGHT =================

with right:

    st.subheader("System status")

    st.write("**Repository root:**", str(REPO_ROOT))
    st.write("**Demo artifacts:**", str(DEMO))

    if not DEMO.exists():
        st.error("Artifacts not found — generate demo files first.")
    else:
        figs = REPORTS / "figures"
        fi = REPORTS / "feature_importance"

        st.write("Figures:", "✅ Ready" if figs.exists() else "❌ Missing")
        st.write("Feature reports:", "✅ Ready" if fi.exists() else "❌ Missing")

    st.markdown("---")

    st.info(
"""
💡 Think of this app as a walkthrough of how an ML system moves from
data → modeling → production readiness.
"""
    )
