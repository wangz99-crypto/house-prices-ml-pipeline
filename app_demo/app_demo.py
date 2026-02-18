# app/app_demo.py
# Demo entrypoint (online / interview): instant startup, no large model binaries required.

import sys
from pathlib import Path

# ---- path bootstrap (MUST be before local imports) ----
APP_ROOT = Path(__file__).resolve().parent          # .../repo/app
REPO_ROOT = APP_ROOT.parent                         # .../repo

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

# ---- IMPORTANT: route all `import registry_io` in pages to demo IO ----
import importlib
_demo_mod = importlib.import_module("registry_io_demo")
sys.modules["registry_io"] = _demo_mod

from registry_io_demo import RegistryLayout


def build_layout(artifacts_dir: Path) -> RegistryLayout:
    return RegistryLayout(artifacts_dir=artifacts_dir)


def main():
    st.set_page_config(
        page_title="House Prices ML System (Demo)",
        page_icon="🏠",
        layout="wide",
    )

    # -------------------------------
    # Sidebar
    # -------------------------------
    st.sidebar.markdown("## ⚙️ Data Source")
    st.sidebar.caption("Demo-only mode (precomputed assets).")

    artifacts_dir = REPO_ROOT / "artifacts_demo"
    st.sidebar.caption(f"Using: {artifacts_dir}")

    st.session_state["REGISTRY_LAYOUT"] = build_layout(artifacts_dir)
    st.session_state["DEMO_MODE"] = True

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.caption(
        "Use the left page menu to explore Registry, OOF, Contracts, Drift, and Demo Prediction."
    )

    # -------------------------------
    # Home content
    # -------------------------------
    st.title("House Prices ML System (Demo)")
    st.info(
        "Interview demo build: loads **precomputed registry assets** (metrics/OOF/figures/schema) "
        "and does **not** require large model files. "
        "Use the button below to open the demo prediction page.",
        icon="👈",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("▶ Open Live Prediction (Demo)", use_container_width=True):
            st.switch_page("pages/3_Live_Prediction_Demo.py")
    with col2:
        st.caption(
            "Tip: Demo mode avoids shipping model binaries online. Full inference remains available locally via `app.py`."
        )

    st.markdown("---")

    with st.expander("What this demo includes / excludes", expanded=False):
        st.markdown(
            """
- ✅ Model Registry: runs, aliases, metrics, fingerprints, pipeline repr
- ✅ OOF + Error Analysis: uses precomputed `oof.npy` / `test_pred.npy`
- ✅ Contracts + Drift: uses precomputed contract JSON + drift outputs (if present)
- ✅ Live Prediction (Demo): interpretable mock inference (no joblib)
- ❌ Full model inference: available locally (full artifacts), intentionally excluded from online demo
"""
        )


if __name__ == "__main__":
    main()
