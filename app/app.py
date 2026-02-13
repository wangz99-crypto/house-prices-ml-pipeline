import sys
from pathlib import Path

import streamlit as st

from registry_io import RegistryLayout

APP_ROOT = Path(__file__).resolve().parent          # .../repo/app
REPO_ROOT = APP_ROOT.parent                         # .../repo

# 让 "lib", "registry_io" 在 app/ 目录下可导入
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# 让 "src" 在 repo 根目录可导入（joblib load 里会需要）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_layout(artifacts_dir: Path) -> RegistryLayout:
    return RegistryLayout(artifacts_dir=artifacts_dir)


def main():
    st.set_page_config(
        page_title="House Prices ML System",
        page_icon="🏠",
        layout="wide",
    )

    st.sidebar.markdown("## ⚙️ Data Source")
    st.sidebar.caption("Demo-only mode (artifacts_demo).")

    # ✅ 固定只用 demo
    artifacts_dir = REPO_ROOT / "artifacts_demo"
    st.sidebar.caption(f"Using: {artifacts_dir}")

    st.session_state["REGISTRY_LAYOUT"] = build_layout(artifacts_dir)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.caption("Use the left page menu to explore Live Prediction, Model Registry, and Analysis.")

    st.title("House Prices ML System")
    st.info("Use the page menu on the left to start.", icon="👈")


if __name__ == "__main__":
    main()
