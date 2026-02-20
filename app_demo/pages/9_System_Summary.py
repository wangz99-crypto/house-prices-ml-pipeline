# app_demo/pages/0_Project_Summary.py  (or your last page)
from __future__ import annotations

import streamlit as st

from lib.ui_style import hero, section
from lib.notebook_links import default_notebooks

# ==============================
# Config
# ==============================
REPO_URL = "https://github.com/wangz99-crypto/house-prices-ml-pipeline"
BRANCH = "main"

def gh(path: str) -> str:
    """GitHub link helper."""
    return f"{REPO_URL}/blob/{BRANCH}/{path}"

def gh_tree(path: str) -> str:
    return f"{REPO_URL}/tree/{BRANCH}/{path}"

REPO_README = f"{REPO_URL}#readme"
REPO_ACTIONS = f"{REPO_URL}/actions"
REPO_TESTS = gh_tree("tests")
REPO_TOOLS = gh_tree("tools")
REPO_SRC = gh_tree("src")
REPO_REGISTRY_DEMO = gh_tree("artifacts_demo/registry")
REPO_ARTIFACTS_DEMO = gh_tree("artifacts_demo")

# ==============================
# Hero
# ==============================
hero(
    "Project Summary",
    "A concise overview of the system, with optional links for deeper technical verification."
)

# ==============================
# One-minute understanding
# ==============================
section("One-minute understanding", "If you only read one section, read this.", "⏱️")

st.markdown(
    """
**What this is:** an end-to-end machine learning system built on the Ames Housing dataset.  
**What it does:** trains multiple models, evaluates them reliably, saves reproducible artifacts, and serves predictions through a lightweight demo interface.  
**Why it matters:** it demonstrates an ML workflow as a structured system — not just a notebook experiment.
"""
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Dataset", "Ames Housing")
with k2:
    st.metric("Evaluation", "K-Fold + OOF")
with k3:
    st.metric("Versioning", "Registry + aliases")
with k4:
    st.metric("Safeguards", "Tests + contracts")

st.caption(
    "Demo mode note: interactive pages load **lightweight models for fast startup**, while **full model evaluation** "
    "(including ensembles) is documented in Model Behavior / Experiments."
)

st.divider()

# ==============================
# Choose your focus
# ==============================
section(
    "How would you like to explore this project?",
    "Different readers often focus on different aspects of the system."
)

tab1, tab2, tab3 = st.tabs(
    [
        "Practical value",
        "Evaluation quality",
        "System structure",
    ]
)

with tab1:
    st.markdown(
        """
If you're mainly interested in **decision support and interpretation**, start here:

- **Live Prediction** → Try inputs and see an estimate  
- **Error Analysis** → Understand where the model is reliable vs. risky  
- **Data Drift Monitor** → See how the system detects distribution shift  

These sections focus on responsible interpretation, not just a single number.
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Open README (overview)", REPO_README)
    with c2:
        st.link_button("Artifacts snapshot (demo)", REPO_ARTIFACTS_DEMO)

with tab2:
    st.markdown(
        """
If you're mainly interested in **methodology and reliability**, review:

- **Model Behavior** → Cross-validation results and drivers  
- **Model Registry** → Versioned runs, metrics, data fingerprints  
- **Contract Validation** → Golden-row consistency checks  

This shows how evaluation is stabilized beyond a single split.
"""
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("CI / Tests evidence", REPO_ACTIONS)
    with c2:
        st.link_button("Tests folder", REPO_TESTS)
    with c3:
        st.link_button("Registry (demo)", REPO_REGISTRY_DEMO)

with tab3:
    st.markdown(
        """
If you're mainly interested in **implementation and maintainability**, explore:

- `src/` → modular pipeline components  
- `tools/` → reproducible generators for artifacts and reports  
- `artifacts_demo/` → portable snapshot powering this app  

This demonstrates engineering structure and reproducibility.
"""
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("src/ (pipeline)", REPO_SRC)
    with c2:
        st.link_button("tools/ (repro commands)", REPO_TOOLS)
    with c3:
        st.link_button("Repository root", REPO_URL)

st.divider()

# ==============================
# What the system demonstrates
# ==============================
section("What this system demonstrates", "Capabilities summarized in plain language.", "🧩")

st.markdown(
    """
- **Reproducibility:** Figures, schemas, and summaries are generated in a consistent, repeatable way.  
- **Reliable evaluation:** K-Fold CV + OOF predictions reduce sampling bias.  
- **Model versioning:** Each run is stored with metrics and metadata, addressable by aliases.  
- **Explainability:** Drivers and behavior patterns are surfaced in the interface.  
- **Monitoring mindset:** Contracts and drift checks guard against silent regressions.
"""
)

st.markdown("### If you’re especially interested in…")
st.write("• How structured features translate into pricing estimates")
st.write("• How model risk is communicated (error patterns, drift flags, contracts)")
st.write("• How an ML workflow can be organized end-to-end as a system")

st.divider()

# ==============================
# Optional supporting material
# ==============================
section("Optional supporting material", "For readers who prefer deeper technical traceability.", "📚")

with st.expander("Show evidence notebooks and links", expanded=False):
    notebooks = default_notebooks()

    for nb in notebooks:
        with st.container(border=True):
            st.markdown(f"**{nb.title}**")
            st.caption(nb.desc)
            # notebooks store paths like "notebooks/....ipynb"
            st.link_button("Open notebook", gh(nb.path))

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("Open README", REPO_README)
    with c2:
        st.link_button("CI evidence", REPO_ACTIONS)
    with c3:
        st.link_button("Registry snapshot", REPO_REGISTRY_DEMO)

st.divider()

# ==============================
# Next step
# ==============================
section(
    "If deployed in a production environment",
    "How this system could scale in a real-world setting.",
    "🚀",
)

st.markdown(
    """
If deployed in an organizational setting, the natural next steps would include:

- Connecting artifacts to cloud storage (e.g., S3 / Blob storage)
- Scheduling automated training + scoring jobs via orchestration
- Adding structured monitoring for performance and drift
- Integrating alerting and operational dashboards
"""
)

st.info(
    "The architecture is modular and artifact-driven, so productionization is primarily an "
    "infrastructure extension rather than a redesign."
)
