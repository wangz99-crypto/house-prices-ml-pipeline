from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunRef:
    model_name: str
    run_id: str
    run_dir: Path

    @property
    def model_id(self) -> str:
        return f"{self.model_name}/{self.run_id}"


def make_run_dir(registry_root: Path, model_name: str) -> RunRef:
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = registry_root / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunRef(model_name=model_name, run_id=run_id, run_dir=run_dir)


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def fingerprint_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Stable-enough fingerprint for lineage:
    shape + columns + dtypes + deterministic head sample hash.
    """
    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}

    sample = df.head(200).copy()
    sample_bytes = sample.to_csv(index=False).encode("utf-8")
    sample_hash = _sha256_bytes(sample_bytes)

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": cols,
        "dtypes": dtypes,
        "head200_sha256": sample_hash,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
