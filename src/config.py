from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class Paths:
    project_root: Path

    # data
    data_raw: Path
    data_processed: Path

    # artifacts root (kept name "models" for backward-compat)
    models: Path  # == artifacts/
    models_current: Path  # artifacts/current
    models_registry: Path  # artifacts/registry

    # outputs
    reports: Path  # artifacts/reports
    predictions: Path  # artifacts/predictions
    submissions: Path  # artifacts/submissions

    # -------------------------
    # Backward-compatible aliases
    # -------------------------
    @property
    def models_dir(self) -> Path:
        # ✅ compat: old code expects "models/<model>.joblib" to mean "current model"
        return self.models_current

    @property
    def reports_dir(self) -> Path:
        return self.reports

    @property
    def predictions_dir(self) -> Path:
        return self.predictions

    @property
    def submissions_dir(self) -> Path:
        return self.submissions


def default_paths(project_root: Path | None = None) -> Paths:
    """
    Centralized path config.

    New layout (single source of truth):
      artifacts/
        current/       # latest "production" model artifacts per model family
        registry/      # versioned training runs
        reports/       # metrics / oof / summaries
        predictions/   # saved prediction CSV + meta
        submissions/   # kaggle submission CSVs
    """
    root = project_root or Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"

    return Paths(
        project_root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        models=artifacts,
        models_current=artifacts / "current",
        models_registry=artifacts / "registry",
        reports=artifacts / "reports",
        predictions=artifacts / "predictions",
        submissions=artifacts / "submissions",
    )


def ensure_dirs(*args: Union[Path, "Paths"]) -> None:
    """
    Create directories safely.

    Supports:
    - ensure_dirs(path1, path2, ...)
    - ensure_dirs(paths) where paths is a Paths object
    """
    for arg in args:
        # Case 1: a single Path
        if isinstance(arg, Path):
            arg.mkdir(parents=True, exist_ok=True)
            continue

        # Case 2: a Paths object
        for p in [
            arg.data_raw,
            arg.data_processed,
            arg.models_current,
            arg.models_registry,
            arg.reports,
            arg.predictions,
            arg.submissions,
        ]:
            p.mkdir(parents=True, exist_ok=True)
