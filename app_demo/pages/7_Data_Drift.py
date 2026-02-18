from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Data Drift Monitor (A/B Demo)", layout="wide")
st.title("📡 Data Drift Monitor (A/B Demo)")
st.caption(
    "Demo build uses sample datasets. This page runs two drift tests: "
    "A = baseline, B = intentionally shifted (stress test). No raw Kaggle required."
)

# ------------------------------------------------
# Paths
# ------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_TRAIN = REPO_ROOT / "tests" / "data" / "sample_train.csv"
SAMPLE_TEST  = REPO_ROOT / "tests" / "data" / "sample_test.csv"   # optional
RAW_TRAIN    = REPO_ROOT / "data" / "raw" / "train.csv"
RAW_TEST     = REPO_ROOT / "data" / "raw" / "test.csv"


# ------------------------------------------------
# PSI
# ------------------------------------------------
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # use expected percentiles as breakpoints
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return np.nan

    e_hist, _ = np.histogram(expected, bins=breakpoints)
    a_hist, _ = np.histogram(actual, bins=breakpoints)

    e_perc = e_hist / max(e_hist.sum(), 1)
    a_perc = a_hist / max(a_hist.sum(), 1)

    e_perc = np.where(e_perc == 0, 1e-6, e_perc)
    a_perc = np.where(a_perc == 0, 1e-6, a_perc)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


# ------------------------------------------------
# Loaders (demo-friendly)
# ------------------------------------------------
@st.cache_data(show_spinner=False)
def load_train() -> pd.DataFrame:
    if SAMPLE_TRAIN.exists():
        return pd.read_csv(SAMPLE_TRAIN)
    if RAW_TRAIN.exists():
        return pd.read_csv(RAW_TRAIN)
    raise FileNotFoundError(
        "Missing train dataset.\n"
        f"Checked:\n- {SAMPLE_TRAIN}\n- {RAW_TRAIN}"
    )


@st.cache_data(show_spinner=False)
def load_test_A(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    A = baseline test:
      1) tests/data/sample_test.csv if exists
      2) data/raw/test.csv if exists
      3) fallback: resample from train (drop target)
    """
    if SAMPLE_TEST.exists():
        return pd.read_csv(SAMPLE_TEST)
    if RAW_TEST.exists():
        return pd.read_csv(RAW_TEST)

    df = train_df.copy()
    df = df.drop(columns=["SalePrice"], errors="ignore")
    return df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)


def _safe_numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "SalePrice" in cols:
        cols.remove("SalePrice")
    return cols


def make_test_B_from_A(
    A: pd.DataFrame,
    *,
    shift_strength: float = 0.25,
    missing_rate: float = 0.12,
    rename_some: bool = True,
    drop_some: bool = True,
    add_extra_cols: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    B = intentionally shifted dataset (stress test):
    - numeric distributions are shifted/scaled
    - more missingness
    - optionally rename a few columns (schema drift)
    - optionally drop a few columns
    - optionally add a few extra columns
    """
    rng = np.random.default_rng(seed)
    B = A.copy()

    num_cols = _safe_numeric_cols(B)
    if not num_cols:
        return B

    # 1) distribution shift: for each numeric col, apply shift + scale + noise
    for c in num_cols:
        x = pd.to_numeric(B[c], errors="coerce").to_numpy()

        if np.nanstd(x) == 0 or np.isnan(np.nanstd(x)):
            continue

        mu = np.nanmean(x)
        sd = np.nanstd(x)

        # shift mean and scale variance a bit
        x2 = x.copy()
        x2 = x2 + (shift_strength * sd)  # mean shift
        x2 = (x2 - (mu + shift_strength * sd)) * (1.0 + shift_strength) + (mu + shift_strength * sd)  # variance
        x2 = x2 + rng.normal(0, 0.05 * sd, size=len(x2))  # noise

        # keep non-negativity for some obvious area/count fields
        if any(k in c.lower() for k in ["sf", "area", "cars", "bath", "room", "lot", "front", "porch", "deck", "year"]):
            x2 = np.where(np.isnan(x2), x2, np.maximum(x2, 0))

        B[c] = x2

    # 2) increase missingness in numeric columns
    for c in num_cols:
        mask = rng.random(len(B)) < missing_rate
        B.loc[mask, c] = np.nan

    # 3) drop some columns (simulate missing fields)
    if drop_some:
        # drop up to 3 numeric columns with lowest importance to keep demo stable
        drop_k = min(3, max(0, len(num_cols) // 10))
        if drop_k > 0:
            drop_cols = list(rng.choice(num_cols, size=drop_k, replace=False))
            B = B.drop(columns=drop_cols, errors="ignore")

    # 4) rename a few columns (schema drift)
    if rename_some:
        cols = list(B.columns)
        cand = [c for c in cols if c not in {"Id", "SalePrice"}]
        rename_k = min(3, len(cand))
        if rename_k > 0:
            picked = list(rng.choice(cand, size=rename_k, replace=False))
            ren_map = {c: f"{c}_NEW" for c in picked}
            B = B.rename(columns=ren_map)

    # 5) add extra columns (new fields appear)
    if add_extra_cols:
        B["NewFeature_Noise"] = rng.normal(0, 1, size=len(B))
        B["NewFeature_Flag"] = (rng.random(len(B)) > 0.7).astype(int)
        # a "derived" column if core cols exist (won't break PSI because expected doesn't have it)
        if "GrLivArea" in A.columns and "OverallQual" in A.columns:
            gl = pd.to_numeric(A["GrLivArea"], errors="coerce")
            oq = pd.to_numeric(A["OverallQual"], errors="coerce")
            B["QualGrLiv_Alt"] = (gl.fillna(0) * oq.fillna(0)) * (1.0 + shift_strength)

    return B


def compute_psi_table(train_df: pd.DataFrame, test_df: pd.DataFrame, bins: int) -> pd.DataFrame:
    # PSI on numeric cols from TRAIN; if a col missing in test -> PSI = NaN + mark missing
    num_cols = _safe_numeric_cols(train_df)
    rows = []
    for c in num_cols:
        e = train_df[c]
        if c in test_df.columns:
            a = test_df[c]
            missing_in_test = False
        else:
            a = pd.Series([], dtype=float)
            missing_in_test = True

        rows.append(
            {
                "feature": c,
                "psi": psi(pd.to_numeric(e, errors="coerce"), pd.to_numeric(a, errors="coerce"), bins=bins),
                "missing_in_test": missing_in_test,
            }
        )
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return out


def summarize_thresholds(df: pd.DataFrame) -> dict:
    x = df["psi"].dropna()
    return {
        "n_features": int(df.shape[0]),
        "n_scored": int(x.shape[0]),
        "psi_gt_025": int((x > 0.25).sum()),
        "psi_010_025": int(((x >= 0.10) & (x <= 0.25)).sum()),
        "psi_lt_010": int((x < 0.10).sum()),
    }


# ------------------------------------------------
# Data
# ------------------------------------------------
train_df = load_train()
test_A = load_test_A(train_df)

# UI controls for B generation
st.subheader("A/B setup")
c1, c2, c3, c4 = st.columns(4)
with c1:
    bins = st.slider("PSI bins", 5, 20, 10)
with c2:
    shift_strength = st.slider("B: shift strength", 0.00, 0.80, 0.25, 0.05)
with c3:
    missing_rate = st.slider("B: extra missing rate", 0.00, 0.40, 0.12, 0.02)
with c4:
    seed = st.number_input("Random seed", value=42, step=1)

test_B = make_test_B_from_A(
    test_A,
    shift_strength=float(shift_strength),
    missing_rate=float(missing_rate),
    rename_some=True,
    drop_some=True,
    add_extra_cols=True,
    seed=int(seed),
)

# Sources info
a_src = "tests/data/sample_test.csv" if SAMPLE_TEST.exists() else ("data/raw/test.csv" if RAW_TEST.exists() else "resampled from train")
t_src = "tests/data/sample_train.csv" if SAMPLE_TRAIN.exists() else "data/raw/train.csv"
st.info(
    f"Train source: **{t_src}**  |  "
    f"A (baseline) source: **{a_src}**  |  "
    f"B (stress test): **synthetic shifted copy of A**"
)

# ------------------------------------------------
# Drift compute
# ------------------------------------------------
psi_A = compute_psi_table(train_df, test_A, bins=bins)
psi_B = compute_psi_table(train_df, test_B, bins=bins)

sumA = summarize_thresholds(psi_A)
sumB = summarize_thresholds(psi_B)

st.caption("Rule of thumb: PSI < 0.10 = small, 0.10–0.25 = moderate, > 0.25 = significant.")

# ------------------------------------------------
# Summary cards
# ------------------------------------------------
st.subheader("Summary (A vs B)")
sa1, sa2, sa3, sa4 = st.columns(4)
sb1, sb2, sb3, sb4 = st.columns(4)

with sa1: st.metric("A: scored features", f"{sumA['n_scored']}/{sumA['n_features']}")
with sa2: st.metric("A: PSI > 0.25", sumA["psi_gt_025"])
with sa3: st.metric("A: PSI 0.10–0.25", sumA["psi_010_025"])
with sa4: st.metric("A: PSI < 0.10", sumA["psi_lt_010"])

with sb1: st.metric("B: scored features", f"{sumB['n_scored']}/{sumB['n_features']}")
with sb2: st.metric("B: PSI > 0.25", sumB["psi_gt_025"])
with sb3: st.metric("B: PSI 0.10–0.25", sumB["psi_010_025"])
with sb4: st.metric("B: PSI < 0.10", sumB["psi_lt_010"])

# ------------------------------------------------
# Tables + comparison
# ------------------------------------------------
st.divider()
tabA, tabB, tabC = st.tabs(["A: Baseline drift", "B: Stress-test drift", "A vs B comparison"])

with tabA:
    st.subheader("A drift scores (PSI)")
    st.dataframe(psi_A, use_container_width=True)

    topA = psi_A.dropna(subset=["psi"]).head(15)
    st.subheader("Top drift features (A)")
    st.bar_chart(topA.set_index("feature")["psi"])

with tabB:
    st.subheader("B drift scores (PSI)")
    st.dataframe(psi_B, use_container_width=True)

    topB = psi_B.dropna(subset=["psi"]).head(15)
    st.subheader("Top drift features (B)")
    st.bar_chart(topB.set_index("feature")["psi"])

with tabC:
    st.subheader("Compare PSI: B - A (same feature)")
    merged = psi_A[["feature", "psi", "missing_in_test"]].merge(
        psi_B[["feature", "psi", "missing_in_test"]],
        on="feature",
        how="outer",
        suffixes=("_A", "_B"),
    )
    merged["delta_B_minus_A"] = merged["psi_B"] - merged["psi_A"]
    merged = merged.sort_values("delta_B_minus_A", ascending=False)

    st.dataframe(merged, use_container_width=True)

    top_delta = merged.dropna(subset=["delta_B_minus_A"]).head(15)
    st.subheader("Top increases in drift (B vs A)")
    st.bar_chart(top_delta.set_index("feature")["delta_B_minus_A"])

st.divider()

with st.expander("What is B doing (talk track)", expanded=False):
    st.markdown(
        """
**B is a synthetic 'stress test'** designed to demonstrate monitoring behavior:

- **Distribution shift:** numeric columns are shifted/scaled + noise
- **Higher missingness:** additional NaNs injected
- **Schema drift:** some columns renamed / some dropped
- **New columns:** extra fields appear in B

This makes drift monitoring more vivid than a single static report.
"""
    )
