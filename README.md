# House Prices ML Pipeline

**End-to-End Machine Learning Pipeline with Cross-Validation, Model Registry, Ensembling, Feature Importance, Error Analysis, and Reproducibility**

> **Goal:** demonstrate how a Kaggle-style regression problem can be engineered into a reproducible, testable, production-inspired machine learning system.

---

## Project Overview

This repository implements a **fully reproducible, end-to-end machine learning system** for the Kaggle competition **House Prices – Advanced Regression Techniques**.

Rather than optimizing purely for leaderboard performance, the project focuses on **ML engineering best practices**, including:

- Modular, reusable training pipelines
- K-Fold cross-validation with Out-of-Fold (OOF) predictions
- A centralized **model registry** with aliases (`latest`, `best`, `production`)
- Ensemble methods (blending & stacking)
- Post-hoc error analysis and feature importance
- Clean-environment reproducibility
- CI-friendly unit tests and regression safeguards

The intent is to treat model training as a **system**, not a one-off experiment.

---

## Key Features

### Unified Pipeline Design

- Shared preprocessing with model-specific estimators
- Clear separation between linear and tree-based modeling strategies
- Deterministic behavior via fixed random seeds and controlled CV splits

### Supported Models

The system supports multiple model families under a unified interface:

- **Linear models:** Ridge Regression (baseline & interpretable reference)
- **Tree ensembles:** ExtraTrees, LightGBM, XGBoost
- **Ensembles:**
  - Voting (mean / RMSE-weighted)
  - Stacking (Ridge meta-learner trained on OOF predictions)

### Robust Evaluation

- K-Fold Cross-Validation
- Out-of-Fold (OOF) predictions for unbiased evaluation
- RMSE computed in **log-space** (`log1p(SalePrice)`), consistent with Kaggle standards

### Model Registry (Production-Inspired)

- Versioned training runs stored by model family
- Family-level aliases: `latest`, `best`, `production`, `staging`
- Global aliases across all models: `global/latest`, `global/best`
- Full lineage tracking for each run:
  - training arguments
  - data fingerprints
  - metrics
  - pipeline representation

The registry acts as a lightweight alternative to tools like MLflow,
providing versioning, lineage, and safe model promotion without external services.

### Interpretability & Analysis

- Automated feature importance extraction
- Support for:
  - linear models (`coef_`)
  - tree-based models (`feature_importances_`)
  - ensemble models (Voting / Stacking)
- Interactive error analysis notebooks for residual inspection

### Reproducibility

- Verified from a clean environment
- Deterministic CV splits and seeds
- Explicit artifact structure
- Unit tests validating model artifacts and prediction consistency

---

## Project Structure

```
house-prices-ml-pipeline/
├── data/
│ └── raw/ # Kaggle train.csv / test.csv (gitignored)
├── src/
│ ├── train.py # Training entrypoint + registry integration
│ ├── predict.py # Kaggle + production-style prediction CLI
│ ├── pipelines.py # Model + preprocessing pipelines
│ ├── transformers.py # Custom feature engineering & missing handlers
│ ├── evaluate.py # KFold OOF evaluation logic
│ ├── ensemble.py # Blending / stacking utilities
│ ├── registry.py # Model registry, aliases, fingerprints
│ ├── registry_status.py # Registry inspection helpers
│ ├── data.py # Dataset loading utilities
│ └── config.py # Centralized path configuration
├── analysis/
│ └── feature_importance.py # Registry-aware feature importance extraction
├── notebooks/
│ ├── 00_experiments_raw/ # Early exploratory experiments
│ ├── 01_eda/
│ │ └── House_EDA.ipynb
│ └── 02_model_analysis/
│ ├── error_analysis_oof_interactive.ipynb
│ └── feature_importance_viewer.ipynb
├── tests/
│ ├── unit/ # Unit, regression & contract tests (pytest)
│ ├── data/ # Small sampled datasets for CI
│ ├── contracts/ # Model prediction contracts (golden tests)
│ └── baselines/ # Performance baselines (RMSE)
├── tools/
│ ├── make_sample_data.py # Generate CI-safe sample datasets
│ ├── make_perf_baseline.py # Generate performance baseline
│ ├── make_contract.py # Generate model prediction contracts
│ ├── check_drift.py # Offline data drift detection
│ └── promote.py # Registry alias promotion helper
├── .github/workflows/ci.yml # GitHub Actions CI
├── Makefile.mak # Optional Make interface (non-Windows friendly)
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Create a clean environment (recommended)

conda create -n hp_clean python=3.10 -y
conda activate hp_clean


### 2. Install dependencies

pip install -r requirements.txt


**Windows note:**  
LightGBM / XGBoost are often easier to install via conda:
conda install -c conda-forge lightgbm xgboost -y


---

## Training Models

### Train a single model

python -m src.train --model ridge

### Train all models
python -m src.train --model all

### Common options

--seed <int> # random seed (default: 42)
--folds <int> # number of CV folds (default: 5)
--data-dir <path> # custom directory containing train.csv/test.csv
--no-export-compat-model # disable artifacts/current/<model>.joblib


---

## Training Outputs & Artifacts

After training, the following artifacts are generated:

### artifacts/reports/

- `<model>_metrics.json` — CV metrics
- `<model>_oof.npy` — out-of-fold predictions
- `<model>_test_pred.npy` — averaged test predictions
- `metrics.csv` — model comparison table
- `cv_summary.json` — full CV summary

### artifacts/registry/<model>/<run_id>/

Each training run contains:

- `model.joblib`
- `metrics.json`
- `oof.npy`
- `test_pred.npy`
- `data_fingerprint.json`
- `train_args.json`
- `pipeline_repr.txt`

### artifacts/current/

- `<model>.joblib` — latest snapshot (optional, backward-compatible)

---

## Running Tests

pytest -q


Tests verify:

- model artifact completeness
- prediction reproducibility across save/load
- regression safety (RMSE not degrading)
- registry consistency and contracts

These checks ensure that changes to preprocessing, pipelines, or dependencies
cannot silently alter model behavior.

---

## Kaggle Submission Mode

Generate Kaggle submission files using saved predictions.

### Single model

python -m src.predict kaggle --model lgbm

### Ensemble methods

python -m src.predict kaggle --ensemble blend_mean
python -m src.predict kaggle --ensemble blend_weighted
python -m src.predict kaggle --ensemble stack


Outputs are saved to: `artifacts/submissions/`

---

## Production / Registry Prediction Mode

Batch scoring using registry models and aliases.

### Family-level selectors

python -m src.predict prod --model-id ridge/latest --input data/new_data.csv
python -m src.predict prod --model-id ridge/best --input data/new_data.csv
python -m src.predict prod --model-id ridge/production --input data/new_data.csv


### Global selectors (across all models)

python -m src.predict prod --model-id global/latest --input data/new_data.csv
python -m src.predict prod --model-id global/best --input data/new_data.csv


Each run produces:

- predictions CSV
- metadata JSON (resolved model id, data fingerprint, lineage)

---

## Feature Importance Analysis

Extract feature importance from registry models.

### Default: global best model


Each run produces:

- predictions CSV
- metadata JSON (resolved model id, data fingerprint, lineage)

---

## Feature Importance Analysis

Extract feature importance from registry models.

### Default: global best model

python analysis/feature_importance.py


### Specific model & run

python analysis/feature_importance.py --model lgbm --run-id <run_id> --topk 30


### Outputs

artifacts/reports/feature_importance/
├── <model>__<run_id>top30.csv
├── <model><run_id>top30.png
└── <model><run_id>__meta.json


---

## Error Analysis

Location: `notebooks/02_model_analysis/error_analysis_oof_interactive.ipynb`

Includes:

- residual diagnostics
- bias & segment-level errors
- worst-case prediction inspection

---

## Exploratory Data Analysis (EDA)

EDA is intentionally separated from the training pipeline.

Location: `notebooks/01_eda/House_EDA.ipynb`

Notebooks can be viewed directly on GitHub or run locally.

---

## Feature Importance Viewer

For exploratory analysis and presentation, the repository also includes:

`notebooks/02_model_analysis/feature_importance_viewer.ipynb`

The visualization is ensemble-aware and reflects feature importance
aggregated at the model level rather than from a single estimator.

---

## Dataset

Kaggle: House Prices – Advanced Regression Techniques

- Target variable: `SalePrice`
- Training performed in log-space: `log1p(SalePrice)`

---

## Notes

- This project prioritizes ML system design, correctness, and reproducibility
  over leaderboard-driven optimization.
- All notebooks are optional and included for transparency.
- Large artifacts (`.joblib`, `.npy`, registry runs) are intentionally gitignored.

---

## Contact

This repository is part of a personal ML engineering portfolio.

Feedback and discussion are welcome via GitHub Issues.

