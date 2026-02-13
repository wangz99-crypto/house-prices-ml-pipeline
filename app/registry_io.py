# app/registry_io.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def safe_read_json(p: Path) -> Dict[str, Any]:
    return read_json(p) if p.exists() else {}

def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

@dataclass(frozen=True)
class RunRef:
    model_name: str
    run_id: str

    @property
    def run_dirname(self) -> str:
        return self.run_id

@dataclass(frozen=True)
class RegistryLayout:
    artifacts_dir: Path

    @property
    def current_dir(self) -> Path:
        return self.artifacts_dir / "current"

    @property
    def registry_dir(self) -> Path:
        return self.artifacts_dir / "registry"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    def aliases_path(self, model_name: str) -> Path:
        return self.registry_dir / model_name / "aliases.json"

    def run_dir(self, model_name: str, run_id: str) -> Path:
        return self.registry_dir / model_name / run_id

    def current_model_path(self, model_name: str) -> Path:
        return self.current_dir / f"{model_name}.joblib"

def list_model_names(layout: RegistryLayout) -> List[str]:
    if not layout.registry_dir.exists():
        return []
    out = []
    for d in layout.registry_dir.iterdir():
        if d.is_dir() and not d.name.startswith("_"):
            out.append(d.name)
    return sorted(out)

def read_aliases(layout: RegistryLayout, model_name: str) -> Dict[str, Any]:
    """
    Supports two formats:
    - Family aliases: {"production": null, "staging": null, "best": "<run_id>", "latest": "<run_id>"}
    - Global aliases (optional): {"best": {"model_name": "...", "run_id": "...", ...}, ...}
    """
    return safe_read_json(layout.aliases_path(model_name))

def get_alias_runref(aliases: Dict[str, Any], alias_key: str, default_model_name: str) -> Optional[RunRef]:
    v = aliases.get(alias_key)

    if isinstance(v, dict):
        mn = v.get("model_name")
        rid = v.get("run_id")
        if isinstance(mn, str) and isinstance(rid, str):
            return RunRef(model_name=mn, run_id=rid)
        return None

    if isinstance(v, str) and v.strip():
        return RunRef(model_name=default_model_name, run_id=v.strip())

    return None

def list_runs(layout: RegistryLayout, model_name: str) -> List[str]:
    md = layout.registry_dir / model_name
    if not md.exists():
        return []
    runs = []
    for d in md.iterdir():
        if d.is_dir() and d.name[:4].isdigit():
            runs.append(d.name)
    return sorted(runs, reverse=True)

def load_run_bundle(layout: RegistryLayout, ref: RunRef) -> Dict[str, Any]:
    rd = layout.run_dir(ref.model_name, ref.run_id)
    files = sorted([p.name for p in rd.iterdir()]) if rd.exists() else []

    return {
        "run_dir": str(rd),
        "metrics": safe_read_json(rd / "metrics.json"),
        "data_fingerprint": safe_read_json(rd / "data_fingerprint.json"),
        "train_args": safe_read_json(rd / "train_args.json"),
        "pipeline_repr": safe_read_text(rd / "pipeline_repr.txt"),
        "model_path": str(rd / "model.joblib") if (rd / "model.joblib").exists() else None,
        "oof": np.load(rd / "oof.npy") if (rd / "oof.npy").exists() else None,
        "test_pred": np.load(rd / "test_pred.npy") if (rd / "test_pred.npy").exists() else None,
        "files": files,
    }

def load_cv_summary(layout: RegistryLayout) -> Dict[str, Any]:
    return safe_read_json(layout.reports_dir / "cv_summary.json")

def load_metrics_csv_text(layout: RegistryLayout) -> str:
    return safe_read_text(layout.reports_dir / "metrics.csv")
