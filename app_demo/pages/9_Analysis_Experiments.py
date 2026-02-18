import streamlit as st
from lib.ui_style import hero, section
from lib.notebook_links import default_notebooks

# ==============================
# Config
# ==============================
REPO_URL = "https://github.com/wangz99-crypto/house-prices-ml-pipeline"
REPO_NOTEBOOK_BASE = f"{REPO_URL}/blob/main/"
REPO_README = f"{REPO_URL}#readme"
REPO_ACTIONS = f"{REPO_URL}/actions"
REPO_REGISTRY_DOC = f"{REPO_URL}/tree/main/artifacts_demo/registry"
REPO_TESTS = f"{REPO_URL}/tree/main/tests"
REPO_TOOLS = f"{REPO_URL}/tree/main/tools"

# ==============================
# Hero
# ==============================
hero("✅ Project Summary", "A product-style wrap-up with links to the full engineering evidence.")

# ==============================
# Section: One-minute understanding
# ==============================
section("One-minute understanding", "If you only read one section, read this.", "⏱️")

st.markdown(
    """
**What this is:** an end-to-end machine learning system built on the **Kaggle House Prices** dataset.  
**What it does:** trains multiple models, selects the best one reliably, saves everything as artifacts, and serves predictions in a demo app.  
**Why it matters:** it demonstrates **ML engineering**, not just a single notebook — reproducibility, registry, safeguards, and explainability.
"""
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Data source", "Kaggle (Ames Housing)")
with k2:
    st.metric("Training style", "K-Fold CV + OOF")
with k3:
    st.metric("Model lifecycle", "Registry + aliases")
with k4:
    st.metric("Safety layer", "tests + contracts")

st.markdown("---")

# ==============================
# Section: Who are you? pick a path
# ==============================
section("Choose your path", "Different viewers care about different proof.", "🧭")

tab1, tab2, tab3 = st.tabs(["👩‍💼 Business viewer", "🧑‍🏫 Reviewer / professor", "🧑‍💻 Engineer"])

with tab1:
    st.markdown(
        """
**If you’re here for product value:**
- Go to **Live Prediction** → try inputs → see a price estimate  
- Go to **Error Analysis** → see where the model is risky (which house types)  
- Go to **Data Drift** → see how the system detects “new data looks different”
"""
    )
    st.link_button("Open README (product overview)", REPO_README)

with tab2:
    st.markdown(
        """
**If you’re evaluating rigor:**
- **Model Behavior** → model comparison + importance + Ridge interpretation  
- **Model Registry** → best/latest aliases, run folders, metrics + fingerprint  
- **Tests evidence** → contracts + performance + smoke tests
"""
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("CI / Tests (GitHub Actions)", REPO_ACTIONS)
    with c2:
        st.link_button("Tests folder", REPO_TESTS)
    with c3:
        st.link_button("Registry artifacts (demo)", REPO_REGISTRY_DOC)

with tab3:
    st.markdown(
        """
**If you care about implementation details:**
- `src/` contains reusable pipeline code  
- `tools/` contains reproducible command-line generators  
- `artifacts_demo/` is the portable demo snapshot for this Streamlit app
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Tools (repro commands)", REPO_TOOLS)
    with c2:
        st.link_button("Repository root", REPO_URL)

st.markdown("---")

# ==============================
# Section: What this system demonstrates (engineering bullets)
# ==============================
section("What this system demonstrates", "Engineering features explained in plain language.", "🧩")

st.markdown(
    """
- **Reproducibility:** one command regenerates artifacts (figures, schemas, summaries).  
- **Reliable evaluation:** K-Fold CV + OOF predictions reduce “lucky split” bias.  
- **Model registry:** every training run is saved with metrics + fingerprint; aliases point to best/latest.  
- **Explainability:** model behavior and feature importance are surfaced in-app (for non-technical viewers).  
- **Safeguards:** tests and model contracts protect against silent regressions.
"""
)

st.markdown("---")

# ==============================
# Section: Evidence notebooks (secondary)
# ==============================
section("Evidence notebooks (optional)", "Deep dives for people who want the full trace.", "📚")
st.caption("Notebooks are supportive evidence — the system does not depend on notebooks to run.")

notebooks = default_notebooks()

for nb in notebooks:
    with st.container(border=True):
        st.markdown(f"### {nb.title}")
        st.caption(nb.desc)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"Category: **{nb.kind}**")
            st.write(f"Path: `{nb.path}`")

        with c2:
            st.link_button("Open notebook", REPO_NOTEBOOK_BASE + nb.path)

st.markdown("---")

# ==============================
# Section: Call to action
# ==============================
section("Want to run it?", "Quick start is documented in the repository.", "🚀")

st.markdown(
    """
If you want to reproduce the demo, check the README for the exact commands and expected artifact outputs.
"""
)

c1, c2, c3 = st.columns(3)
with c1:
    st.link_button("Open README", REPO_README)
with c2:
    st.link_button("Open Repo", REPO_URL)
with c3:
    st.link_button("CI evidence", REPO_ACTIONS)
