from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART_DEMO = ROOT / "artifacts_demo"
RUN_DIR = ART_DEMO / "registry" / "voting_mean" / "2026-02-06_171206"
MODEL_PATH = ART_DEMO / "current" / "voting_mean.joblib"

# 你需要提供一个“能代表训练列结构”的数据源：
# 优先用 tests/data 里的 sample，其次用 data/raw/train.csv（本地）
CANDIDATES = [
    ROOT / "tests" / "data" / "train_sample.csv",
    ROOT / "tests" / "data" / "sample_train.csv",
    ROOT / "data" / "raw" / "train.csv",
]

def find_data_path() -> Path:
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find a sample dataset. Put a small train sample under tests/data/ "
        "or keep data/raw/train.csv locally to generate defaults."
    )

def get_expected_columns(model) -> list[str]:
    # 最优：很多 sklearn pipeline/estimator 会有 feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # 次优：尝试从 pipeline 的第一层 transformer 里找
    # (不同实现差异较大，所以这里只做兜底)
    for attr in ["named_steps", "steps"]:
        if hasattr(model, attr):
            break

    # 最后兜底：用“故意预测失败”解析报错（你的模型正好会抛 columns are missing）
    try:
        import pandas as pd
        dummy = pd.DataFrame([{}])
        model.predict(dummy)
    except Exception as e:
        msg = str(e)
        key = "columns are missing:"
        if key in msg:
            tail = msg.split(key, 1)[1].strip()
            # tail 通常长得像 "{'A','B',...}"
            tail = tail.strip()
            if tail.startswith("{") and tail.endswith("}"):
                tail = tail[1:-1]
            cols = [c.strip().strip("'").strip('"') for c in tail.split(",") if c.strip()]
            return sorted(set(cols))
        raise

    raise RuntimeError("Could not infer expected columns from the model.")

def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)

    expected_cols = get_expected_columns(model)
    (RUN_DIR / "feature_columns.json").write_text(json.dumps(expected_cols, indent=2), encoding="utf-8")
    print(f"[OK] feature_columns.json saved ({len(expected_cols)} cols)")

    data_path = find_data_path()
    df = pd.read_csv(data_path)

    # Kaggle train.csv 里会有 target SalePrice；把它删掉
    if "SalePrice" in df.columns:
        df = df.drop(columns=["SalePrice"])

    # B 策略：数值 median，类别 mode；对“模型需要但df没有的列”，先给 None（后续在 app 内派生）
    defaults = {}
    for c in expected_cols:
        if c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                defaults[c] = float(np.nanmedian(s.to_numpy(dtype=float)))
            else:
                # mode 可能为空，兜底 NA
                m = s.mode(dropna=True)
                defaults[c] = (m.iloc[0] if len(m) else "NA")
        else:
            defaults[c] = None

    (RUN_DIR / "defaults.json").write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    print("[OK] defaults.json saved")

if __name__ == "__main__":
    main()
