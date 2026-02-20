from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Data Drift Monitor", layout="wide")
st.title("📡 Data Drift Monitor")
st.caption("A lightweight monitoring view that compares incoming data to a reference baseline.")

# ------------------------------------------------
# Paths (demo-friendly)
# ------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_TRAIN = REPO_ROOT / "tests" / "data" / "sample_train.csv"
SAMPLE_TEST  = REPO_ROOT / "tests" / "data" / "sample_test.csv"
RAW_TRAIN    = REPO_ROOT / "data" / "raw" / "train.csv"
RAW_TEST     = REPO_ROOT / "data" / "raw" / "test.csv"

# ------------------------------------------------
# Core metrics
# ------------------------------------------------
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index (PSI)
    - Breakpoints built from expected distribution.
    - Robust fallback if quantile cutpoints collapse.
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Quantile breakpoints
    q = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, q)
    breakpoints = np.unique(breakpoints)

    # If too few unique breakpoints (common for discrete columns), fall back:
    if len(breakpoints) < 3:
        uniq = np.unique(expected)
        if len(uniq) >= 3:
            # use sorted unique values as bin edges (cap at bins+1)
            edges = np.unique(np.percentile(uniq, np.linspace(0, 100, min(bins + 1, len(uniq)))))
            breakpoints = np.unique(edges)

    if len(breakpoints) < 3:
        return np.nan

    # Add tiny padding to include min/max safely
    eps = 1e-9
    breakpoints[0] = breakpoints[0] - eps
    breakpoints[-1] = breakpoints[-1] + eps

    e_hist, _ = np.histogram(expected, bins=breakpoints)
    a_hist, _ = np.histogram(actual, bins=breakpoints)

    e_perc = e_hist / max(e_hist.sum(), 1)
    a_perc = a_hist / max(a_hist.sum(), 1)

    e_perc = np.where(e_perc == 0, 1e-6, e_perc)
    a_perc = np.where(a_perc == 0, 1e-6, a_perc)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def drift_band(score: float) -> str:
    if not isinstance(score, (int, float)) or np.isnan(score):
        return "—"
    if score < 0.10:
        return "🟢 Stable"
    if score <= 0.25:
        return "🟡 Watch"
    return "🔴 Action"


def cat_shift(base: pd.Series, new: pd.Series, topk: int = 8) -> pd.DataFrame:
    base_dist = base.value_counts(normalize=True, dropna=False)
    new_dist  = new.value_counts(normalize=True, dropna=False)
    cats = set(base_dist.index) | set(new_dist.index)

    rows = []
    for c in cats:
        b = float(base_dist.get(c, 0.0))
        n = float(new_dist.get(c, 0.0))
        rows.append({"category": c, "reference_pct": b, "incoming_pct": n, "shift": abs(b - n)})

    out = pd.DataFrame(rows).sort_values("shift", ascending=False).head(topk)
    out["reference_pct"] = (out["reference_pct"] * 100).round(2)
    out["incoming_pct"] = (out["incoming_pct"] * 100).round(2)
    out["shift"] = (out["shift"] * 100).round(2)
    return out


def missing_drift(reference: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common = [c for c in reference.columns if c in incoming.columns]
    for col in common:
        r = float(reference[col].isna().mean())
        n = float(incoming[col].isna().mean())
        rows.append({"feature": col, "reference_missing": r, "incoming_missing": n, "shift": abs(r - n)})

    out = pd.DataFrame(rows).sort_values("shift", ascending=False)
    out["reference_missing"] = (out["reference_missing"] * 100).round(2)
    out["incoming_missing"] = (out["incoming_missing"] * 100).round(2)
    out["shift"] = (out["shift"] * 100).round(2)
    return out


def schema_changes(reference: pd.DataFrame, incoming: pd.DataFrame) -> tuple[list[str], list[str]]:
    ref_cols = set(reference.drop(columns=["SalePrice"], errors="ignore").columns)
    inc_cols = set(incoming.columns)
    missing = sorted(list(ref_cols - inc_cols))
    new = sorted(list(inc_cols - ref_cols))
    return missing, new


# ------------------------------------------------
# Loaders (demo-first)
# ------------------------------------------------
@st.cache_data(show_spinner=False)
def load_reference() -> tuple[pd.DataFrame, str]:
    if SAMPLE_TRAIN.exists():
        return pd.read_csv(SAMPLE_TRAIN), "tests/data/sample_train.csv"
    if RAW_TRAIN.exists():
        return pd.read_csv(RAW_TRAIN), "data/raw/train.csv"
    raise FileNotFoundError("Reference dataset not found.")


@st.cache_data(show_spinner=False)
def load_incoming(reference: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if SAMPLE_TEST.exists():
        return pd.read_csv(SAMPLE_TEST), "tests/data/sample_test.csv"
    if RAW_TEST.exists():
        return pd.read_csv(RAW_TEST), "data/raw/test.csv"

    df = reference.drop(columns=["SalePrice"], errors="ignore").copy()
    return df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True), "resampled from reference"


def _safe_numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for drop in ["SalePrice", "Id"]:
        if drop in cols:
            cols.remove(drop)
    return cols


def simulate_incoming_shift(
    incoming: pd.DataFrame,
    *,
    shift_strength: float = 0.25,
    extra_missing_rate: float = 0.12,
    seed: int = 42,
    simulate_schema_breaks: bool = False,
) -> pd.DataFrame:
    """
    Demo-only: create a shifted batch so the monitor has something to react to.
    - By default, it shifts distributions + adds missingness.
    - Optional toggle simulates schema-breaking changes (renames/drops/adds columns).
    """
    rng = np.random.default_rng(seed)
    B = incoming.copy()

    num_cols = _safe_numeric_cols(B)
    for c in num_cols:
        x = pd.to_numeric(B[c], errors="coerce").to_numpy()
        sd = np.nanstd(x)
        mu = np.nanmean(x)
        if sd == 0 or np.isnan(sd):
            continue

        # shift mean + widen spread, with light noise
        x2 = x + (shift_strength * sd)
        x2 = (x2 - (mu + shift_strength * sd)) * (1.0 + shift_strength) + (mu + shift_strength * sd)
        x2 = x2 + rng.normal(0, 0.05 * sd, size=len(x2))

        # keep obviously non-negative features non-negative
        if any(k in c.lower() for k in ["sf", "area", "cars", "bath", "room", "lot", "front", "porch", "deck", "year"]):
            x2 = np.where(np.isnan(x2), x2, np.maximum(x2, 0))

        B[c] = x2

    # extra missingness on numeric fields
    for c in num_cols:
        mask = rng.random(len(B)) < extra_missing_rate
        B.loc[mask, c] = np.nan

    # Optional schema noise (separate toggle so it doesn't confuse the main story)
    if simulate_schema_breaks:
        # rename a couple of non-critical columns (avoid Id)
        rename_candidates = [c for c in B.columns if c not in {"Id", "SalePrice"}][:2]
        B = B.rename(columns={c: f"{c}_NEW" for c in rename_candidates})

        # drop one numeric col if possible
        if len(num_cols) >= 1:
            B = B.drop(columns=[num_cols[-1]], errors="ignore")

        # add a new feature
        B["NewFeature_Noise"] = rng.normal(0, 1, size=len(B))

    return B


def _barh_psi(df_top: pd.DataFrame):
    fig, ax = plt.subplots()
    x = df_top.sort_values("psi", ascending=True)
    ax.barh(x["feature"].astype(str), x["psi"].astype(float))
    ax.set_xlabel("PSI")
    ax.set_title("Top drifted numeric features")
    return fig


# ------------------------------------------------
# Data
# ------------------------------------------------
reference, ref_src = load_reference()
incoming_base, inc_src = load_incoming(reference)

DEFAULT_BINS = 10

# ------------------------------------------------
# A/B Drift Simulation (visible section)
# ------------------------------------------------
st.subheader("🧪 A/B Drift Simulation")
st.caption("Use this section to demonstrate how the monitoring system reacts when incoming data shifts.")

c1, c2 = st.columns([2, 2])
with c1:
    bins = st.slider("PSI bins", 5, 20, DEFAULT_BINS)
with c2:
    incoming_mode = st.selectbox(
        "Incoming dataset",
        ["Baseline (normal incoming batch)", "Simulated shift (stress test demo)"],
        index=0,
    )

simulate_schema_breaks = False
if incoming_mode.startswith("Simulated"):
    st.markdown("##### Stress test controls")
    c3, c4, c5, c6 = st.columns(4)
    with c3:
        shift_strength = st.slider("Shift strength", 0.00, 0.80, 0.25, 0.05)
    with c4:
        extra_missing_rate = st.slider("Extra missing rate", 0.00, 0.40, 0.12, 0.02)
    with c5:
        seed = st.number_input("Random seed", value=42, step=1)
    with c6:
        simulate_schema_breaks = st.checkbox("Also simulate schema breaking changes", value=False)
else:
    shift_strength = 0.25
    extra_missing_rate = 0.12
    seed = 42

# ------------------------------------------------
# Resolve incoming dataset
# ------------------------------------------------
incoming = incoming_base
incoming_label = inc_src
if incoming_mode.startswith("Simulated"):
    incoming = simulate_incoming_shift(
        incoming_base,
        shift_strength=float(shift_strength),
        extra_missing_rate=float(extra_missing_rate),
        seed=int(seed),
        simulate_schema_breaks=bool(simulate_schema_breaks),
    )
    incoming_label = "Simulated shift (demo)"

# ------------------------------------------------
# Executive summary (product-first)
# ------------------------------------------------
st.info(f"Reference: **{ref_src}**  |  Incoming: **{incoming_label}**")

missing_cols, new_cols = schema_changes(reference, incoming)

# PSI table (numeric only)
num_cols = _safe_numeric_cols(reference)
psi_rows = []
for col in num_cols:
    if col not in incoming.columns:
        psi_rows.append({"feature": col, "psi": np.nan, "status": "—", "missing_in_incoming": True})
        continue
    score = psi(reference[col], incoming[col], bins=int(bins))
    psi_rows.append({"feature": col, "psi": score, "status": drift_band(score), "missing_in_incoming": False})

psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)

x = psi_df["psi"].dropna()
severe = int((x > 0.25).sum())
moderate = int(((x >= 0.10) & (x <= 0.25)).sum())
max_psi = float(x.max()) if len(x) else np.nan

# Overall drift score (simple, explainable)
overall_drift = float(x.mean()) if len(x) else np.nan

if incoming_mode.startswith("Simulated"):
    st.warning("Demo mode: Incoming data is intentionally shifted to showcase how monitoring reacts.")

if len(x) == 0:
    st.warning("Not enough numeric features to compute drift scores.")
elif severe > 0:
    st.error(f"🚨 Significant drift detected: {severe} feature(s) exceed PSI > 0.25 (max PSI = {max_psi:.3f}).")
elif moderate > 0:
    st.warning(f"⚠️ Moderate drift detected: {moderate} feature(s) are in PSI 0.10–0.25 (max PSI = {max_psi:.3f}).")
else:
    st.success(f"✅ No meaningful drift detected (max PSI = {max_psi:.3f}).")

# Summary metrics
st.subheader("At-a-glance summary")
m1, m2, m3, m4, m5 = st.columns([1.3, 0.8, 1.0, 1.6, 1.0])
m1.metric("Scored", f"{len(x)}/{len(psi_df)}")
m2.metric("PSI>0.25", severe)
m3.metric("PSI 0.10–0.25", moderate)
m4.metric("Schema changes", f"{len(missing_cols)} missing • {len(new_cols)} new")
m5.metric("Overall drift", f"{overall_drift:.3f}" if not np.isnan(overall_drift) else "N/A")
st.caption("PSI bands: 🟢 Stable < 0.10 • 🟡 Watch 0.10–0.25 • 🔴 Action > 0.25")

# ------------------------------------------------
# Data health
# ------------------------------------------------
st.divider()
st.subheader("Data health")

h1, h2, h3, h4 = st.columns(4)
with h1: st.metric("Reference rows", f"{len(reference):,}")
with h2: st.metric("Incoming rows", f"{len(incoming):,}")
with h3:
    denom = max(1, incoming.shape[0] * incoming.shape[1])
    st.metric("Incoming missing rate", f"{incoming.isna().sum().sum() / denom * 100:.1f}%")
with h4: st.metric("Fields in common", f"{len(set(reference.columns) & set(incoming.columns)):,}")

colL, colR = st.columns(2, gap="large")
with colL:
    st.write("**Missing fields in incoming**")
    st.write(", ".join(missing_cols[:30]) + (" ..." if len(missing_cols) > 30 else "") if missing_cols else "None")
with colR:
    st.write("**New fields in incoming**")
    st.write(", ".join(new_cols[:30]) + (" ..." if len(new_cols) > 30 else "") if new_cols else "None")

# ------------------------------------------------
# 1) Numeric drift
# ------------------------------------------------
st.divider()
st.subheader("1) Numeric drift (PSI)")

top = psi_df.dropna(subset=["psi"]).head(12).copy()
top["psi"] = top["psi"].round(4)

st.dataframe(top[["feature", "psi", "status", "missing_in_incoming"]], use_container_width=True)

if len(top):
    fig = _barh_psi(top[["feature", "psi"]].copy())
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with st.expander("View full numeric PSI table", expanded=False):
    st.dataframe(psi_df, use_container_width=True)

# ------------------------------------------------
# 2) Missing drift
# ------------------------------------------------
st.divider()
st.subheader("2) Missing value drift")

ref_m = reference.drop(columns=["SalePrice", "Id"], errors="ignore")
inc_m = incoming.drop(columns=["Id"], errors="ignore")
md = missing_drift(ref_m, inc_m)

st.dataframe(md.head(20), use_container_width=True)
st.caption("Shows which fields changed most in missing rate (percentage points).")

# ------------------------------------------------
# 3) Categorical drift
# ------------------------------------------------
st.divider()
st.subheader("3) Categorical drift")

ref_cat = reference.select_dtypes(include="object")
inc_cat = incoming.select_dtypes(include="object")

cat_cols = sorted(list(set(ref_cat.columns) & set(inc_cat.columns)))
if not cat_cols:
    st.info("No shared categorical columns available in this build.")
else:
    col_choice = st.selectbox("Choose a categorical feature", cat_cols, index=0)
    drift_table = cat_shift(reference[col_choice], incoming[col_choice], topk=10)
    st.dataframe(drift_table, use_container_width=True)
    st.caption("Shift is absolute change in percentage points between reference and incoming.")

# ------------------------------------------------
# Interpretation
# ------------------------------------------------
with st.expander("How to interpret this monitor", expanded=False):
    st.markdown(
        """
- **Stable (PSI < 0.10):** distributions are similar; model inputs look familiar  
- **Watch (0.10–0.25):** some movement; keep an eye on data quality and performance  
- **Action (> 0.25):** meaningful shift; investigate pipeline changes or consider retraining  
"""
    )
