# app_demo/pages/9_Model_Contract.py  (or app/pages/9_Model_Contract.py)
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib


# ------------------------------------------------
# Page
# ------------------------------------------------
st.set_page_config(page_title="Model Contract", layout="wide")
st.title("🛡️ Model Contract Validation")
st.caption(
    "A lightweight regression-style check that verifies model outputs stay within an expected tolerance band "
    "after updates."
)

# ------------------------------------------------
# Config
# ------------------------------------------------
ALLOWED_MODELS = ["ridge", "xgb", "lgbm"]
MAX_MODEL_MB = 50.0  # demo safety guard (kept internal)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = REPO_ROOT / "tests" / "contracts"
REGISTRY_DIR = REPO_ROOT / "artifacts_demo" / "registry"


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _bytes_to_mb(n: int) -> float:
    return float(n) / 1024 / 1024


def _detect_log1p_like(pred: np.ndarray) -> bool:
    """Heuristic fallback: log1p(SalePrice) is usually ~10–13 for Ames."""
    p = np.asarray(pred).reshape(-1)
    if len(p) == 0:
        return True
    med = float(np.nanmedian(p))
    return 6.0 <= med <= 16.0


@st.cache_resource(show_spinner=False)
def _load_model(path: str):
    return joblib.load(path)


def _pick_run_dir(model_dir: Path, prefer: str = "best") -> Path:
    """
    Prefer alias-driven run if available:
      artifacts_demo/registry/<model>/_global/aliases.json
    Fallback: newest run with model.joblib <= MAX_MODEL_MB.
    """
    alias_path = model_dir / "_global" / "aliases.json"
    if alias_path.exists():
        try:
            aliases = _load_json(alias_path)
            rid = aliases.get(prefer)
            if isinstance(rid, str):
                cand = model_dir / rid
                mp = cand / "model.joblib"
                if cand.exists() and mp.exists() and _bytes_to_mb(mp.stat().st_size) <= MAX_MODEL_MB:
                    return cand
        except Exception:
            pass

    runs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name != "_global"],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for r in runs:
        mp = r / "model.joblib"
        if mp.exists() and _bytes_to_mb(mp.stat().st_size) <= MAX_MODEL_MB:
            return r

    raise FileNotFoundError("No eligible runnable version found (missing model.joblib or file too large).")


def _pick_contract_file(model_name: str, prefer_alias: str) -> Path | None:
    """
    Prefer contracts that match the selected alias in filename, e.g.
      ridge__best__contract.json
      xgb__latest__contract.json
    Falls back to any contract for that model.
    """
    if not CONTRACT_DIR.exists():
        return None

    all_cands = sorted(CONTRACT_DIR.glob(f"{model_name}*contract.json"))
    if not all_cands:
        return None

    preferred = [p for p in all_cands if prefer_alias in p.name]
    return (preferred[0] if preferred else all_cands[0])


def _safe_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return []


# ------------------------------------------------
# 1) Select model + version
# ------------------------------------------------
c1, c2 = st.columns([1, 1])
with c1:
    model_name = st.selectbox("Model family", ALLOWED_MODELS, index=0)
with c2:
    prefer_alias = st.selectbox("Version", ["best", "latest"], index=0)

# ------------------------------------------------
# 2) Pick contract (alias-first, but still inspectable)
# ------------------------------------------------
default_contract = _pick_contract_file(model_name, prefer_alias)
contracts_all = sorted(list(CONTRACT_DIR.glob(f"{model_name}*contract.json"))) if CONTRACT_DIR.exists() else []
if not contracts_all:
    st.error(f"No contract file found for `{model_name}` in: {CONTRACT_DIR}")
    st.stop()

# keep UI simple: show contracts, but default is alias-match
default_index = 0
if default_contract and default_contract in contracts_all:
    default_index = contracts_all.index(default_contract)

contract_file = st.selectbox("Contract file", contracts_all, index=default_index, format_func=lambda p: p.name)
contract = _load_json(contract_file)

golden_X = pd.DataFrame(contract.get("golden_X", []))
exp = contract.get("expected", {}) or {}
pred_lo = np.asarray(exp.get("pred_lo", []), dtype=float)
pred_hi = np.asarray(exp.get("pred_hi", []), dtype=float)

meta = contract.get("meta", {}) or {}
target_transform = meta.get("target_transform")  # e.g., "log1p"

st.markdown("---")

# ------------------------------------------------
# 3) Contract summary (product view)
# ------------------------------------------------
a, b, c = st.columns(3)
with a:
    st.metric("Golden rows", f"{len(golden_X):,}")
with b:
    if len(pred_lo) and len(pred_hi):
        band_log = float(np.nanmedian(pred_hi - pred_lo))
        st.metric("Tolerance band (raw)", f"{band_log:.4f}")
    else:
        st.metric("Tolerance band (raw)", "N/A")
with c:
    st.metric("Target scale", target_transform if target_transform else "auto-detect")

st.info(
    "**What are Golden Rows?**\n\n"
    "Golden rows are a small, fixed set of representative examples saved from the training schema. "
    "We run the *same* inputs through the model every time we rebuild or deploy.\n\n"
    "If predictions fall outside the expected tolerance band, it signals that model behavior "
    "(or preprocessing) has changed — similar to a regression test in software."
)

# ------------------------------------------------
# 4) Load model (demo-safe)
# ------------------------------------------------
model_dir = REGISTRY_DIR / model_name
if not model_dir.exists():
    st.error(f"Missing demo registry directory: {model_dir}")
    st.stop()

try:
    run_dir = _pick_run_dir(model_dir, prefer=prefer_alias)
except Exception as e:
    st.error(f"Could not select a runnable version: {e}")
    st.stop()

model_path = run_dir / "model.joblib"
st.caption(f"Selected run: `{run_dir.name}`")

try:
    model = _load_model(str(model_path))
except Exception as e:
    st.error(f"Failed to load model artifact: {e}")
    st.stop()

# ------------------------------------------------
# 5) Validate contract
# ------------------------------------------------
st.subheader("Contract check results")

if golden_X.empty:
    st.error("Contract is missing `golden_X` rows.")
    st.stop()

if len(pred_lo) != len(golden_X) or len(pred_hi) != len(golden_X):
    st.error(
        "Contract shape mismatch.\n"
        f"- golden rows: {len(golden_X)}\n"
        f"- pred_lo len: {len(pred_lo)}\n"
        f"- pred_hi len: {len(pred_hi)}"
    )
    st.stop()

# Optional schema diagnostics (if provided by contract)
schema_cols = _safe_list((contract.get("schema", {}) or {}).get("columns"))

try:
    pred = np.asarray(model.predict(golden_X)).reshape(-1).astype(float)
except Exception as e:
    st.error(
        "Prediction failed. This usually means the contract inputs no longer match the model’s expected schema.\n\n"
        f"Error: {e}"
    )
    with st.expander("Schema diagnostics", expanded=False):
        if schema_cols:
            st.write("Contract schema columns:", schema_cols)
        if hasattr(model, "feature_names_in_"):
            st.write("Model expects:", list(getattr(model, "feature_names_in_")))
        st.write("Golden row columns:", list(golden_X.columns))
    st.stop()

pass_mask = (pred >= pred_lo) & (pred <= pred_hi)
pass_rate = float(np.mean(pass_mask))

# Decide scale for readability
if isinstance(target_transform, str) and target_transform.lower() == "log1p":
    is_log = True
elif isinstance(target_transform, str) and target_transform.lower() in {"none", "identity"}:
    is_log = False
else:
    is_log = _detect_log1p_like(pred)

result_df = pd.DataFrame(
    {"pred": pred, "expected_lo": pred_lo, "expected_hi": pred_hi, "pass": pass_mask}
)

# Add readable view
if is_log:
    result_df["pred_price"] = np.expm1(result_df["pred"])
    result_df["lo_price"] = np.expm1(result_df["expected_lo"])
    result_df["hi_price"] = np.expm1(result_df["expected_hi"])

# Show summary KPIs first
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Pass rate", f"{pass_rate*100:.1f}%")
with k2:
    st.metric("Failed rows", int((~pass_mask).sum()))
with k3:
    st.metric("Checked version", prefer_alias.upper())
with k4:
    st.metric("Checked run", run_dir.name)

if pass_rate == 1.0:
    st.success("✅ Contract PASSED — model behavior matches the expected tolerance.")
elif pass_rate >= 0.8:
    st.warning("⚠️ Partial mismatch — small behavior shift detected. Worth reviewing before deployment.")
else:
    st.error("❌ Contract FAILED — behavior is outside the expected tolerance.")

# Add an intuition line for log-band if we can
if len(pred_lo) and len(pred_hi):
    band_log = float(np.nanmedian(pred_hi - pred_lo))
    if is_log and band_log > 0:
        factor = float(np.exp(band_log))
        st.caption(
            f"Interpretation (log scale): typical tolerance is about a **×{factor:.3f}** multiplicative band "
            "(for intuition only)."
        )

if is_log:
    st.caption(
        "Note: predictions are produced in a transformed scale (log1p). "
        "The table also shows the equivalent price-space ranges for readability."
    )

# Table UX: default to failures only
show_all = st.toggle("Show all golden rows", value=False)
view_df = result_df if show_all else result_df.loc[~result_df["pass"]]

st.dataframe(view_df, use_container_width=True)

# Optional: schema drift view if contract stores schema columns
if schema_cols:
    missing = [c for c in schema_cols if c not in golden_X.columns]
    extra = [c for c in golden_X.columns if c not in schema_cols]
    if missing or extra:
        st.warning("Schema mismatch detected between contract schema and golden rows.")
        with st.expander("Schema mismatch details", expanded=False):
            st.write("Missing in golden_X:", missing if missing else "None")
            st.write("Extra in golden_X:", extra if extra else "None")

with st.expander("What to do if it fails", expanded=False):
    st.markdown(
        """
- Confirm the **same preprocessing pipeline** is being used (feature engineering / encoding).
- If the model was intentionally updated, **regenerate the contract** using the new reference run.
- If the change was unintentional, investigate recent code changes and rerun training for reproducibility.
"""
    )
