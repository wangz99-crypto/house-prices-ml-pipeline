# app/pages/5_Model_Registry.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from lib.ui_style import hero, section
from lib.ui_models import render_quality, render_train_args, render_data_fingerprint
from registry_io import (
    RegistryLayout,
    RunRef,
    get_alias_runref,
    list_model_names,
    list_runs,
    load_run_bundle,
    read_aliases,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _ensure_layout() -> RegistryLayout:
    layout = st.session_state.get("REGISTRY_LAYOUT")
    if layout is None:
        st.warning("Please start from the Home page to initialize the demo assets.")
        with st.expander("Why am I seeing this?", expanded=False):
            st.write(
                "This page reads precomputed artifacts (models, metrics, and metadata). "
                "The Home page initializes the artifact layout for the session."
            )
        st.stop()
    return layout


def _fmt_rmse(x: Any) -> str:
    return f"{float(x):.5f}" if isinstance(x, (int, float)) else "N/A"


def _plot_hist(arr: np.ndarray, title: str, *, xlabel: str, bins: int = 40) -> None:
    fig, ax = plt.subplots()
    ax.hist(arr, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def _safe_path_str(p: Any) -> str:
    try:
        return str(Path(p))
    except Exception:
        return str(p)


def _pick_alias_keys(aliases: Dict[str, Any]) -> list[str]:
    # Prefer stable ordering for demo users
    ordered = ["best", "latest", "production", "staging"]
    return [k for k in ordered if k in aliases]


# --------------------------------------------------
# Page
# --------------------------------------------------
layout: RegistryLayout = _ensure_layout()

hero("📦 Model Registry", "Browse versions, quality, and reproducible run lineage.")

st.caption(
    "This page emphasizes **traceability**: every result maps back to a versioned run with stored metrics, data fingerprint, and training metadata."
)

with st.sidebar:
    st.markdown("## Registry")
    DEBUG = st.toggle("Debug mode", value=False, help="Show additional internal details for troubleshooting.")


model_names = list_model_names(layout)
if not model_names:
    st.warning("No model families were found in the registry for this build.")
    st.stop()

# --------------------------------------------------
# Model picker
# --------------------------------------------------
section("Choose a model", "Select which model family you want to inspect.")
model_name = st.selectbox("Model", model_names, index=0)

aliases = read_aliases(layout, model_name) or {}
alias_keys = _pick_alias_keys(aliases)
runs = list_runs(layout, model_name) or []

# --------------------------------------------------
# Version picker
# --------------------------------------------------
section(
    "Select a version",
    "Use a named version (recommended) or inspect an exact run (advanced).",
)

pick_mode = st.radio(
    "Pick by",
    ["Named version (recommended)", "Run ID (advanced)"],
    horizontal=True,
    index=0,
)

ref: Optional[RunRef] = None
selected_alias: Optional[str] = None

if pick_mode.startswith("Named"):
    if not alias_keys:
        st.info(
            "No named versions (aliases) are defined for this model family yet. "
            "Use **Run ID (advanced)** to browse saved runs."
        )
    else:
        selected_alias = st.selectbox("Named version", alias_keys, index=0)
        ref = get_alias_runref(aliases, selected_alias, default_model_name=model_name)
else:
    if not runs:
        st.info("No runs found for this model family.")
    else:
        run_id = st.selectbox("Run ID", runs, index=0)
        ref = RunRef(model_name=model_name, run_id=run_id)

if ref is None:
    st.stop()

# --------------------------------------------------
# Load bundle
# --------------------------------------------------
try:
    bundle = load_run_bundle(layout, ref)
except Exception as e:
    if DEBUG:
        st.exception(e)
    st.error("This run could not be loaded in the current build.")
    st.stop()

m = bundle.get("metrics", {}) or {}
score = m.get("cv_rmse")

# --------------------------------------------------
# Run header / traceability
# --------------------------------------------------
section("Run snapshot", "A reproducible snapshot: metrics, training context, and stored artifacts.")

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown(f"### `{ref.model_name}` / `{ref.run_id}`")
    run_dir = bundle.get("run_dir")
    if run_dir:
        st.caption(_safe_path_str(run_dir))

    # Traceability: show alias mapping if alias mode was used
    if selected_alias:
        st.info(
            f"Traceability: **`{selected_alias}`** → run **`{ref.run_id}`**  \n"
            "Named versions make it easier to reference and reproduce results."
        )

with right:
    st.metric("Cross-validated RMSE", _fmt_rmse(score))
    st.caption("Lower RMSE generally indicates more stable performance across validation folds.")

st.markdown("---")

# --------------------------------------------------
# Quality + Training context
# --------------------------------------------------
colA, colB = st.columns(2, gap="large")
with colA:
    render_quality(m)
with colB:
    render_train_args(bundle.get("train_args", {}) or {})

st.markdown("---")

# --------------------------------------------------
# Data fingerprint
# --------------------------------------------------
render_data_fingerprint(bundle.get("data_fingerprint", {}) or {})

st.markdown("---")

# --------------------------------------------------
# Prediction distributions
# --------------------------------------------------
section(
    "Prediction distributions",
    "These plots help you sanity-check spread and compare outputs across runs.",
)

oof = bundle.get("oof")
test_pred = bundle.get("test_pred")

# NOTE: Depending on your training design, these may be log-space predictions.
# We avoid over-claiming by wording the axis as "model prediction scale".
xlabel = "Predicted value (model prediction scale)"

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Validation predictions (cross-validation)")
    if oof is None:
        st.info("This run does not include validation predictions.")
    else:
        _plot_hist(np.asarray(oof).ravel(), "OOF prediction distribution", xlabel=xlabel)

with col2:
    st.markdown("#### Batch predictions (batch output)")
    if test_pred is None:
        st.info("This run does not include batch predictions.")
    else:
        _plot_hist(np.asarray(test_pred).ravel(), "Batch prediction distribution", xlabel=xlabel)

st.caption(
    "Tip: If distributions look very different across runs, it can indicate data drift, feature changes, or a model instability issue."
)

# --------------------------------------------------
# Optional technical details
# --------------------------------------------------
with st.expander("Run artifacts (optional)", expanded=False):
    files = bundle.get("files", None)

    if not files:
        st.write("No artifact list was recorded for this run.")
    else:
        st.write("Included artifacts:")

        if isinstance(files, dict):
            st.write(sorted(list(files.keys())))
            with st.expander("Show full artifact mapping", expanded=False):
                st.json(files)

        elif isinstance(files, list):
            st.write(sorted([str(x) for x in files]))

        else:
            st.write(files)

if DEBUG:
    with st.expander("Debug: bundle keys", expanded=False):
        st.write(sorted(list(bundle.keys())))
