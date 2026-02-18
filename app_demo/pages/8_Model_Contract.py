from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Model Contract (Demo)", layout="wide")
st.title("🛡 Model Contract Validation (Demo)")
st.caption("Demo build validates reproducibility for lightweight models only.")

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

ALLOWED_MODELS = ["ridge", "xgb", "lgbm"]
MAX_MODEL_MB = 50.0  # safety guard

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = REPO_ROOT / "tests" / "contracts"
REGISTRY = REPO_ROOT / "artifacts_demo" / "registry"

# ------------------------------------------------
# Select model FIRST (clean UX)
# ------------------------------------------------

model_name = st.selectbox("Select model", ALLOWED_MODELS)

# ------------------------------------------------
# Filter contracts for that model
# ------------------------------------------------

contracts = sorted(
    [p for p in CONTRACT_DIR.glob("*_contract.json")
     if model_name in p.name]
)

if not contracts:
    st.error(f"No contract found for model '{model_name}'.")
    st.stop()

contract_file = st.selectbox(
    "Select contract version",
    contracts,
    format_func=lambda p: p.name,
)

contract = json.loads(contract_file.read_text(encoding="utf-8"))

# ------------------------------------------------
# Load model (lightweight only)
# ------------------------------------------------

model_dir = REGISTRY / model_name
if not model_dir.exists():
    st.error(f"Missing demo registry dir: {model_dir}")
    st.stop()

runs = sorted(
    [d for d in model_dir.iterdir() if d.is_dir()],
    key=lambda x: x.stat().st_mtime,
    reverse=True,
)

picked = None
picked_mb = None

for r in runs:
    mp = r / "model.joblib"
    if mp.exists():
        mb = mp.stat().st_size / 1024 / 1024
        if mb <= MAX_MODEL_MB:
            picked = r
            picked_mb = mb
            break

if picked is None:
    st.error(
        f"No eligible lightweight run found for '{model_name}'. "
        f"(model > {MAX_MODEL_MB} MB or missing joblib)"
    )
    st.stop()

model_path = picked / "model.joblib"
st.info(f"Using run: {picked.name}  |  model size: {picked_mb:.2f} MB")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

model = load_model(str(model_path))

# ------------------------------------------------
# Contract validation
# ------------------------------------------------

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

pass_rate = float(pass_mask.mean())

if pass_rate == 1.0:
    st.success("✅ Model contract PASSED")
elif pass_rate > 0.7:
    st.warning("⚠ Partial drift detected")
else:
    st.error("❌ Contract FAILED")

st.metric("Pass rate", f"{pass_rate*100:.1f}%")

st.info(
    "This validates reproducibility. If predictions fall outside the expected tolerance, "
    "the model behavior or preprocessing has changed."
)
