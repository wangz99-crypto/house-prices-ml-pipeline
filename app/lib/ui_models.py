import json
import streamlit as st
from .ui_text import metric_label_map, explain_quality_score, explain_data_signature

def pretty_json(obj: dict):
    st.json(obj, expanded=False)

def render_quality(metrics: dict):
    mp = metric_label_map()
    st.markdown("### ✅ Model Quality")
    st.caption(explain_quality_score())

    if "cv_rmse" in metrics:
        st.metric(mp["cv_rmse"], f"{metrics['cv_rmse']:.6f}")

    # fold_rmse -> quality checks
    if "fold_rmse" in metrics and isinstance(metrics["fold_rmse"], (list, tuple)):
        st.markdown("**Quality checks (per split)**")
        st.write([float(x) for x in metrics["fold_rmse"]])

def render_train_args(train_args: dict):
    st.markdown("### 🧾 Run Settings")
    # 只展示对外有意义的字段
    keys = ["model_name", "seed", "n_splits"]
    out = {k: train_args.get(k) for k in keys if k in train_args}
    st.json(out, expanded=False)

def render_data_fingerprint(fp: dict):
    mp = metric_label_map()
    st.markdown("### 🧬 Data Snapshot")
    st.caption(explain_data_signature())

    X = fp.get("X", {})
    rows = X.get("rows")
    cols = X.get("cols")
    sig = X.get("head200_sha256")

    c1, c2, c3 = st.columns(3)
    if rows is not None: c1.metric(mp["rows"], rows)
    if cols is not None: c2.metric(mp["cols"], cols)
    if sig is not None: c3.metric("Signature", sig[:12] + "...")

    with st.expander("View full data snapshot (technical)", expanded=False):
        st.json(fp, expanded=False)
