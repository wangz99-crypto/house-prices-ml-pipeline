from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Model Contract", layout="wide")
st.title("🛡 Model Contract Validation")

REPO_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_DIR = REPO_ROOT / "tests" / "contracts"
REGISTRY = REPO_ROOT / "artifacts_demo" / "registry"

contracts = list(CONTRACT_DIR.glob("*_contract.json"))

if not contracts:
    st.error("No contracts found.")
    st.stop()

contract_file = st.selectbox(
    "Select contract",
    contracts,
    format_func=lambda p: p.name,
)

contract = json.loads(contract_file.read_text())

model_name = contract["model_name"]

# load latest model
model_dir = REGISTRY / model_name
runs = sorted(
    [d for d in model_dir.iterdir() if d.is_dir()],
    key=lambda x: x.stat().st_mtime,
    reverse=True,
)

if not runs:
    st.error("No model run found.")
    st.stop()

model = joblib.load(runs[0] / "model.joblib")

golden = pd.DataFrame(contract["golden_X"])
pred = model.predict(golden)

lo = np.array(contract["expected"]["pred_lo"])
hi = np.array(contract["expected"]["pred_hi"])

pass_mask = (pred >= lo) & (pred <= hi)

result_df = pd.DataFrame({
    "prediction": pred,
    "expected_lo": lo,
    "expected_hi": hi,
    "pass": pass_mask,
})

st.subheader("Contract Check Results")

st.dataframe(result_df, use_container_width=True)

pass_rate = pass_mask.mean()

if pass_rate == 1.0:
    st.success("✅ Model contract PASSED")
elif pass_rate > 0.7:
    st.warning("⚠ Partial drift detected")
else:
    st.error("❌ Contract FAILED")

st.metric("Pass rate", f"{pass_rate*100:.1f}%")

st.info("""
This validates model reproducibility.

If predictions fall outside expected tolerance,
the model behavior has changed.
""")
