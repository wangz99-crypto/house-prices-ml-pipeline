from pathlib import Path

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
    list_runs,          # ✅ 这里用 registry_io.py 里真实存在的 list_runs
    load_run_bundle,
    read_aliases,
)


def _ensure_layout() -> RegistryLayout:
    layout = st.session_state.get("REGISTRY_LAYOUT")
    if layout is None:
        st.error("Registry layout not initialized. Please open the Home page (app.py) first.")
        st.stop()
    return layout


def _plot_hist(arr: np.ndarray, title: str):
    fig, ax = plt.subplots()
    ax.hist(arr, bins=40)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


layout: RegistryLayout = _ensure_layout()

hero("📦 Model Registry", "Browse model versions, quality, and reproducible run lineage.")

model_names = list_model_names(layout)
if not model_names:
    st.warning("No model families found under artifacts registry. Check artifacts path / structure.")
    st.stop()

model_name = st.selectbox("Model family", model_names, index=0)

aliases = read_aliases(layout, model_name)
alias_keys = [k for k in ["best", "latest", "production", "staging"] if k in aliases]

runs = list_runs(layout, model_name)  # ✅ 用 list_runs

section("Select a model version", "You can pick by named version (recommended) or by run ID.", "🧭")
pick_mode = st.radio("Pick run by", ["Alias (recommended)", "Run ID"], horizontal=True, index=0)

ref: RunRef | None = None
if pick_mode.startswith("Alias"):
    if not alias_keys:
        st.warning("No version aliases found for this model family.")
    else:
        alias_key = st.selectbox("Version label", alias_keys, index=0)
        ref = get_alias_runref(aliases, alias_key, default_model_name=model_name)
else:
    if not runs:
        st.warning("No runs found.")
    else:
        run_id = st.selectbox("Run ID", runs, index=0)
        ref = RunRef(model_name=model_name, run_id=run_id)

if ref is None:
    st.warning("Selected version is not set yet.")
    st.stop()

bundle = load_run_bundle(layout, ref)

st.markdown("---")
section("Selected version", "A complete snapshot of this model build.", "✅")

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"### `{ref.model_name}` / `{ref.run_id}`")
    st.caption(str(Path(bundle["run_dir"])))

with c2:
    m = bundle.get("metrics", {})
    score = m.get("cv_rmse")
    st.metric("Quality score (lower is better)", f"{score:.6f}" if isinstance(score, (int, float)) else "N/A")

st.markdown("---")
colA, colB = st.columns(2)

with colA:
    render_quality(bundle.get("metrics", {}))

with colB:
    render_train_args(bundle.get("train_args", {}))

st.markdown("---")
render_data_fingerprint(bundle.get("data_fingerprint", {}))

st.markdown("---")
section("Prediction artifacts", "Distributions help understand stability and typical output ranges.", "📈")

oof = bundle.get("oof")
test_pred = bundle.get("test_pred")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Validation prediction spread")
    if oof is None:
        st.info("No validation predictions found for this run.")
    else:
        _plot_hist(np.asarray(oof).ravel(), "OOF prediction distribution")

with col2:
    st.markdown("#### Batch prediction spread")
    if test_pred is None:
        st.info("No batch predictions found for this run.")
    else:
        _plot_hist(np.asarray(test_pred).ravel(), "Test prediction distribution")

with st.expander("Technical files (optional)", expanded=False):
    st.write("Files captured for this run:")
    st.write(bundle.get("files", {}))
