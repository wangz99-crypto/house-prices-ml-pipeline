#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import pandas as pd

from .config import default_paths


# ============================================================
# Run reference
# ============================================================
@dataclass(frozen=True)
class RunRef:
    model_name: str
    run_id: str
    run_dir: Path

    @property
    def model_id(self) -> str:
        return f"{self.model_name}/{self.run_id}"


def make_run_dir(registry_root: Path, model_name: str) -> RunRef:
    """
    Create a versioned run folder:
      <registry_root>/<model_name>/<YYYY-MM-DD_HHMMSS>/
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = registry_root / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunRef(model_name=model_name, run_id=run_id, run_dir=run_dir)


# ============================================================
# Fingerprint (lineage)
# ============================================================
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


# ============================================================
# Atomic JSON helpers
# ============================================================
def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)  # atomic on same filesystem


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


# ============================================================
# Registry root (single source of truth)
# ============================================================
def get_registry_root(registry_root: Path | None = None) -> Path:
    """
    Single source of truth for registry root.
    Defaults to: default_paths().models_registry == artifacts/registry
    """
    return registry_root or default_paths().models_registry


# ============================================================
# Model-family aliases: <registry_root>/<model>/aliases.json
# ============================================================
_ALIAS_KEYS = ("production", "staging", "best", "latest")


def aliases_path(model_name: str, registry_root: Path | None = None) -> Path:
    root = get_registry_root(registry_root)
    return root / model_name / "aliases.json"


def ensure_aliases(model_name: str, registry_root: Path | None = None) -> dict[str, Optional[str]]:
    """
    Ensure aliases.json exists and is valid.
    Always returns a dict with keys:
      production, staging, best, latest
    """
    root = get_registry_root(registry_root)
    p = aliases_path(model_name, root)

    default: dict[str, Optional[str]] = {
        "production": None,
        "staging": None,
        "best": None,
        "latest": None,
    }

    if not p.exists():
        write_json(p, default)
        return default

    # handle empty/corrupt json gracefully
    try:
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            write_json(p, default)
            return default
        payload = json.loads(text)
        if not isinstance(payload, dict):
            write_json(p, default)
            return default
    except Exception:
        write_json(p, default)
        return default

    # patch missing keys
    changed = False
    for k in default:
        if k not in payload:
            payload[k] = None
            changed = True
    if changed:
        write_json(p, cast(dict[str, Any], payload))

    return cast(dict[str, Optional[str]], payload)


def set_alias(
    model_name: str,
    alias: str,
    run_id: Optional[str],
    registry_root: Path | None = None,
) -> None:
    if alias not in _ALIAS_KEYS:
        raise ValueError(f"Unknown alias: {alias}. Must be one of: {_ALIAS_KEYS}")
    root = get_registry_root(registry_root)
    payload = ensure_aliases(model_name, root)
    payload[alias] = run_id
    write_json(aliases_path(model_name, root), cast(dict[str, Any], payload))


def get_alias(model_name: str, alias: str, registry_root: Path | None = None) -> Optional[str]:
    if alias not in _ALIAS_KEYS:
        raise ValueError(f"Unknown alias: {alias}. Must be one of: {_ALIAS_KEYS}")
    root = get_registry_root(registry_root)
    payload = ensure_aliases(model_name, root)
    return payload.get(alias)


# ============================================================
# Run resolving (model-family)
# ============================================================
def run_dir(model_name: str, run_id: str, registry_root: Path | None = None) -> Path:
    root = get_registry_root(registry_root)
    return root / model_name / run_id


def _looks_like_run_id(name: str) -> bool:
    # Expected: YYYY-MM-DD_HHMMSS  => length 17, with '-' at 4,7 and '_' at 10
    return (
        len(name) == 17
        and name[4] == "-"
        and name[7] == "-"
        and name[10] == "_"
        and name[:4].isdigit()
        and name[11:].isdigit()
    )


def list_runs(model_name: str, registry_root: Path | None = None) -> list[str]:
    root = get_registry_root(registry_root)
    d = root / model_name
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if p.is_dir() and _looks_like_run_id(p.name)])


def resolve_run_id(model_name: str, selector: str, registry_root: Path | None = None) -> str:
    """
    selector can be:
      - an alias: production/staging/best/latest
      - an explicit run_id like 2026-01-10_203441
    """
    if selector in _ALIAS_KEYS:
        rid = get_alias(model_name, selector, registry_root=registry_root)
        if not rid:
            raise ValueError(f"Alias '{selector}' is not set for model '{model_name}'")
        return rid
    return selector


def load_model_artifact_path(
    model_name: str,
    selector: str = "production",
    registry_root: Path | None = None,
) -> Path:
    """
    Return path to: <registry_root>/<model>/<run_id>/model.joblib
    selector can be run_id or alias (default: production).
    """
    rid = resolve_run_id(model_name, selector, registry_root=registry_root)
    p = run_dir(model_name, rid, registry_root=registry_root) / "model.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Missing model artifact: {p}")
    return p


# ============================================================
# Global aliases: <registry_root>/_global/aliases.json
# ============================================================
class GlobalRef(TypedDict, total=False):
    model_name: str
    run_id: str
    cv_rmse: float


def global_aliases_path(registry_root: Path | None = None) -> Path:
    root = get_registry_root(registry_root)
    return root / "_global" / "aliases.json"


def read_global_aliases(registry_root: Path | None = None) -> dict[str, GlobalRef]:
    """
    Returns:
      {
        "best": {"model_name": "...", "run_id": "...", "cv_rmse": ...} | {},
        "latest": {...} | {}
      }
    Always patches missing keys and repairs corrupted/empty files.
    """
    p = global_aliases_path(registry_root)
    default: dict[str, GlobalRef] = {"best": {}, "latest": {}}

    if not p.exists():
        write_json(p, cast(dict[str, Any], default))
        return default

    try:
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            write_json(p, cast(dict[str, Any], default))
            return default
        payload = json.loads(text)
        if not isinstance(payload, dict):
            write_json(p, cast(dict[str, Any], default))
            return default
    except Exception:
        write_json(p, cast(dict[str, Any], default))
        return default

    # patch missing keys / wrong types
    changed = False
    for k in ("best", "latest"):
        if k not in payload or not isinstance(payload[k], dict):
            payload[k] = {}
            changed = True

    if changed:
        write_json(p, cast(dict[str, Any], payload))

    return cast(dict[str, GlobalRef], payload)


def set_global_alias(
    alias: str,
    model_name: str,
    run_id: str,
    cv_rmse: float,
    registry_root: Path | None = None,
) -> None:
    """
    Set global alias to point to a specific model/run_id + cv_rmse.
    alias must be 'best' or 'latest'.
    """
    if alias not in {"best", "latest"}:
        raise ValueError(f"Unknown global alias: {alias}. Must be 'best' or 'latest'.")

    payload = read_global_aliases(registry_root)
    payload[alias] = {
        "model_name": model_name,
        "run_id": run_id,
        "cv_rmse": float(cv_rmse),
    }
    write_json(global_aliases_path(registry_root), cast(dict[str, Any], payload))


def resolve_global_model_id(selector: str, registry_root: Path | None = None) -> str:
    """
    selector: 'best' or 'latest'
    returns '<model>/<run_id>'
    """
    if selector not in {"best", "latest"}:
        raise ValueError("Global selector must be 'best' or 'latest'.")

    payload = read_global_aliases(registry_root)
    ref = payload.get(selector) or {}
    mn = ref.get("model_name")
    rid = ref.get("run_id")
    if not mn or not rid:
        raise ValueError(f"Global alias '{selector}' is not set.")
    return f"{mn}/{rid}"

# -------------------------
# Human-friendly status
# -------------------------
def _safe_read_metrics_cv_rmse(run_dir: Path) -> Optional[float]:
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        m = read_json(mp)
        return float(m.get("cv_rmse"))
    except Exception:
        return None


def show_registry_status(registry_root: Path | None = None) -> str:
    """
    Return a pretty text summary of:
    - global aliases: best/latest
    - per-family aliases: best/latest/production/staging
    - if possible, shows cv_rmse from metrics.json for referenced runs
    """
    root = get_registry_root(registry_root)

    lines: list[str] = []
    lines.append(f"Registry root: {root}")

    # ---- global ----
    g = read_global_aliases(registry_root=root)
    lines.append("")
    lines.append("== Global ==")
    for k in ["latest", "best"]:
        ref = g.get(k) or {}
        mn = ref.get("model_name")
        rid = ref.get("run_id")
        rmse = ref.get("cv_rmse")
        if mn and rid:
            lines.append(f"- global/{k}: {mn}/{rid}  (cv_rmse={rmse})")
        else:
            lines.append(f"- global/{k}: (not set)")

    # ---- families ----
    lines.append("")
    lines.append("== Families ==")

    # list model families = folders under registry root excluding _global
    if not root.exists():
        lines.append("(registry root does not exist)")
        return "\n".join(lines)

    families = sorted(
        [
            p.name
            for p in root.iterdir()
            if p.is_dir() and p.name != "_global"
        ]
    )
    if not families:
        lines.append("(no model families found)")
        return "\n".join(lines)

    for model_name in families:
        a = ensure_aliases(model_name, registry_root=root)
        lines.append(f"\n[{model_name}]")

        for alias in ["latest", "best", "production", "staging"]:
            rid = a.get(alias)
            if not rid:
                lines.append(f"  - {alias}: (not set)")
                continue

            rd = run_dir(model_name, rid, registry_root=root)
            rmse = _safe_read_metrics_cv_rmse(rd)
            if rmse is None:
                lines.append(f"  - {alias}: {model_name}/{rid} (cv_rmse=? missing metrics.json)")
            else:
                lines.append(f"  - {alias}: {model_name}/{rid} (cv_rmse={rmse})")

    return "\n".join(lines)
