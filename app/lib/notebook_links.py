from dataclasses import dataclass

@dataclass(frozen=True)
class NotebookLink:
    title: str
    desc: str
    path: str  # notebooks/...
    kind: str  # "EDA" | "Diagnostics" | "Experiments"

def default_notebooks():
    return [
        NotebookLink(
            title="Exploratory Data Analysis",
            desc="Understand the dataset, target distribution, and key patterns that influence price.",
            path="notebooks/01_eda/House_EDA.ipynb",
            kind="EDA",
        ),
        NotebookLink(
            title="Error Analysis (Validation Diagnostics)",
            desc="Inspect where the model tends to under/over-estimate and explore high-error segments.",
            path="notebooks/02_model_analysis/error_analysis_oof_interactive.ipynb",
            kind="Diagnostics",
        ),
        NotebookLink(
            title="Feature Importance Viewer",
            desc="Review which signals the model relies on (aggregated for ensemble models).",
            path="notebooks/02_model_analysis/feature_importance_viewer.ipynb",
            kind="Diagnostics",
        ),
        NotebookLink(
            title="Early Experiments (Raw)",
            desc="Track early baselines and exploration before finalizing the pipeline approach.",
            path="notebooks/00_experiments_raw/House_Prices_Experiments.ipynb",
            kind="Experiments",
        ),
    ]
