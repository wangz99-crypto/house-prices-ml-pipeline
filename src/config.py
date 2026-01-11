from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path
    models_registry: Path

    # Backward-compatible aliases
    @property
    def models_dir(self) -> Path:
        return self.models

    @property
    def reports_dir(self) -> Path:
        return self.reports


def default_paths(project_root: Path | None = None) -> Paths:
    root = project_root or Path(__file__).resolve().parents[1]
    return Paths(
        project_root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        models=root / "models",
        reports=root / "reports",
        models_registry=root / "models" / "registry",
    )
