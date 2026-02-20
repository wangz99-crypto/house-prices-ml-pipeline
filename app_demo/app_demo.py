from __future__ import annotations

import sys
from pathlib import Path
import importlib
import streamlit as st

# ---- required path bootstrap ----
APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- route registry IO to demo implementation ----
_demo_mod = importlib.import_module("registry_io_demo")
sys.modules["registry_io"] = _demo_mod

from registry_io_demo import RegistryLayout


def build_layout(artifacts_dir: Path) -> RegistryLayout:
    return RegistryLayout(artifacts_dir=artifacts_dir)


def main() -> None:
    st.set_page_config(
        page_title="House Prices ML System",
        layout="wide",
    )

    # Session state
    artifacts_dir = REPO_ROOT / "artifacts_demo"
    st.session_state["REGISTRY_LAYOUT"] = build_layout(artifacts_dir)
    st.session_state["DEMO_MODE"] = True

    # ---------------- Sidebar ----------------
    st.sidebar.markdown("## System Navigation")
    st.sidebar.caption(
        "Use the page menu to explore different layers of the ML lifecycle."
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Suggested exploration order")
    st.sidebar.markdown(
        """
1. **Live Prediction** — experience the system output  
2. **Model Behavior** — understand feature impact and performance  
3. **Model Registry** — inspect versioned runs and evaluations  
4. **Reliability Controls** — drift checks and data contracts
"""
    )

    # ---------------- Home Content ----------------
    st.title("🏠 House Prices ML System")
    st.caption("An end-to-end machine learning system designed with production engineering principles.")
    st.info("This demo showcases a production-oriented ML system built around the Ames Housing dataset.")

    st.markdown(
        """
This project demonstrates how a predictive model can be engineered into a structured, 
reproducible machine learning system.

The business objective is straightforward:

**Estimate residential property values using structured housing features.**

However, the emphasis of this project is not leaderboard optimization.  
Instead, it focuses on lifecycle design — evaluation rigor, version management, 
and reliability mechanisms that make ML sustainable in real-world environments.
"""
    )

    st.markdown("---")

    st.subheader("System Perspective")

    st.markdown(
        """
Rather than presenting a single trained model, this system illustrates:

- Reproducible training workflows  
- Cross-validated evaluation  
- Model version tracking  
- Structured experiment management  
- Reliability safeguards (contracts and drift checks)

The goal is to demonstrate how machine learning moves from experimentation 
to system-level thinking.
"""
    )

    st.subheader("System Components")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
**Modeling Layer**
- Ridge
- ExtraTrees
- LightGBM
- Voting Ensemble
- Stacking Ensemble
- Xgb
"""
        )

    with col2:
        st.markdown(
            """
**Engineering Layer**
- Cross-validation evaluation  
- Model registry & artifact tracking  
- Experiment comparisons  
- Drift monitoring & contracts
"""
        )

    st.markdown("---")

    st.markdown(
        """
If you would like a quick system walkthrough, begin with **Live Prediction**.

For deeper engineering design, explore **Model Registry**, **Model Behavior**, 
and **Reliability Controls**.
"""
    )


if __name__ == "__main__":
    main()
