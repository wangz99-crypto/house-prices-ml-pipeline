# app/pages/1_Overview.py
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Overview", layout="wide")

st.title("Overview")
st.caption("Use this page to navigate the demo by time available and by system layer.")
st.markdown(
    """
- Want a fast end-to-end walkthrough? Follow **Guided demo**.  
- Want to understand the engineering design? Follow **Engineering deep dive**.
"""
)

st.markdown("---")

# ----------------------------
# Two natural paths
# ----------------------------
c1, c2 = st.columns(2, gap="large")

with c1:
    st.subheader("Guided demo (2–3 minutes)")
    st.markdown(
        """
1) **Live Prediction** — enter features and generate a price estimate  
2) **Model Behavior** — see what drives predictions and performance trade-offs  
3) **Model Registry** — inspect how runs and results are tracked  
"""
    )

with c2:
    st.subheader("Engineering deep dive")
    st.markdown(
        """
1) **Model Registry** — versioning, runs, and evaluation tracking  
2) **Model Contract** — input schema and validation expectations  
3) **Data Drift** — monitoring signals for reliability over time  
"""
    )

st.info("Short on time? Start here: **Live Prediction → Model Behavior → Model Registry**.")
st.markdown("---")


# ----------------------------
# Helper: one navigation card
# ----------------------------
def nav_card(
    *,
    title: str,
    value_line: str,
    what_to_notice: str,
    target_page: str,
    button_label: str = "Open",
) -> None:
    with st.container(border=True):
        left, right = st.columns([3, 1], gap="large")
        with left:
            st.markdown(f"### {title}")
            st.markdown(f"**Purpose:** {value_line}")
            st.markdown(f"**What to notice:** {what_to_notice}")
        with right:
            st.write("")
            st.write("")
            key = f"btn_{title}__{target_page}"
            if st.button(button_label, use_container_width=True, key=key):
                st.switch_page(target_page)


# ----------------------------
# Cards grid
# ----------------------------
st.subheader("Explore by module")

row1 = st.columns(2, gap="large")
with row1[0]:
    nav_card(
        title="Live Prediction",
        value_line="Generate a price estimate from property features.",
        what_to_notice="Input validation → consistent output experience.",
        target_page="pages/3_Live_Prediction_Demo.py",
    )
with row1[1]:
    nav_card(
        title="Model Registry",
        value_line="Browse versioned runs, metrics, and stored artifacts.",
        what_to_notice="Reproducibility and comparison across experiments.",
        target_page="pages/5_Model_Registry.py",
    )

row2 = st.columns(2, gap="large")
with row2[0]:
    nav_card(
        title="Model Behavior",
        value_line="Compare models and interpret feature impact.",
        what_to_notice="Performance trade-offs and sensitivity to key features.",
        target_page="pages/4_Model_Behavior.py",
    )
with row2[1]:
    nav_card(
        title="Data Drift",
        value_line="See how data shifts can reduce reliability over time.",
        what_to_notice="Signals that suggest review or retraining.",
        target_page="pages/7_Data_Drift.py",
    )

row3 = st.columns(2, gap="large")
with row3[0]:
    nav_card(
        title="Data Understanding",
        value_line="Explore dataset structure and modeling variables.",
        what_to_notice="Which signals matter and how data quality shapes outcomes.",
        target_page="pages/2_Data_Understanding.py",
    )
with row3[1]:
    nav_card(
        title="Model Contract",
        value_line="Review input schema and validation rules.",
        what_to_notice="How the system prevents silent failures at inference time.",
        target_page="pages/8_Model_Contract.py",
    )

row4 = st.columns(2, gap="large")
with row4[0]:
    nav_card(
        title="Error Analysis",
        value_line="Inspect where predictions are strong or weak.",
        what_to_notice="Residual patterns, outliers, and iteration opportunities.",
        target_page="pages/6_Error_Analysis.py",
    )
with row4[1]:
    nav_card(
        title="Experiments & Analysis",
        value_line="See how evaluation and iteration improve robustness.",
        what_to_notice="Experiment structure and validation of conclusions.",
        target_page="pages/9_Analysis_Experiments.py",
    )

st.markdown("---")
