# 🏠 House Prices ML Pipeline

**End-to-End Machine Learning Pipeline with Cross-Validation, Model Registry, Ensembling, Feature Importance, Error Analysis, and Reproducibility**

---

## 📌 Project Overview

This repository implements a **fully reproducible, end-to-end machine learning system** for the Kaggle competition **House Prices – Advanced Regression Techniques**.

The project goes beyond leaderboard optimization and focuses on **ML engineering best practices**, including:
- Modular, reusable training pipelines
- K-Fold cross-validation with Out-of-Fold (OOF) predictions
- Centralized **model registry** with aliases (`latest`, `best`, `production`)
- Ensemble methods (blending & stacking)
- Post-hoc error analysis and feature importance
- Clean-environment reproducibility
- CI-friendly unit tests

---

## 🧠 Key Features

### Unified Pipeline Design
- Shared preprocessing + model-specific pipelines
- Separate strategies for linear vs tree-based models
- Deterministic behavior via fixed random seeds

### Supported Models
- Ridge Regression
- ExtraTrees Regressor
- XGBoost
- LightGBM
- Voting Regressor (mean / weighted)
- Stacking Regressor (Ridge meta-learner)

### Robust Evaluation
- K-Fold Cross-Validation
- Out-of-Fold (OOF) predictions
- RMSE in **log-space** (`log1p(SalePrice)`, Kaggle standard)

### Model Registry (Production-Ready)
- Versioned training runs
- Family-level aliases: `latest`, `best`
- Global aliases across all models: `global/latest`, `global/best`
- Full lineage tracking:
  - training arguments
  - data fingerprints
  - metrics
  - pipeline representation

### Interpretability & Analysis
- Automated feature importance extraction
- Support for:
  - linear models (`coef_`)
  - tree models (`feature_importances_`)
  - ensembles (Voting / Stacking)
- Interactive error analysis notebooks

### Reproducibility
- Verified from a clean environment
- Deterministic CV splits
- Explicit artifact structure
- Unit tests validating model artifacts and prediction consistency

---

## 📁 Repository Structure

```
house-prices-ml-pipeline/
├── data/
│   └── raw/                         # Kaggle train.csv / test.csv (gitignored)
├── src/
│   ├── train.py                     # Training entrypoint + registry integration
│   ├── predict.py                   # Kaggle + production prediction CLI
│   ├── pipelines.py                 # Model & preprocessing pipelines
│   ├── evaluate.py                  # KFold OOF evaluation logic
│   ├── registry.py                  # Model registry, aliases, fingerprints
│   ├── data.py                      # Dataset loading utilities
│   └── config.py                    # Centralized path configuration
├── analysis/
│   └── feature_importance.py        # Registry-aware feature importance extraction
├── notebooks/
│   ├── 00_experiments_raw/          # Early exploratory experiments
│   ├── 01_eda/
│   │   └── House_EDA.ipynb
│   └── 02_model_analysis/
│       ├── error_analysis_oof_interactive.ipynb
│       └── feature_importance_viewer.ipynb
├── tests/
│   └── unit/                        # Unit & regression tests (pytest)
├── tools/
│   ├── promote.py                   # Registry alias promotion helper
│   └── check_drift.py               # Data drift checks (optional)
├── artifacts/                       # Generated at runtime (mostly gitignored)
│   ├── current/
│   ├── predictions/
│   ├── registry/
│   ├── reports/
│   └── submissions/
├── requirements.txt
└── README.md

---

---

## ⚙️ Setup & Installation

### 1️⃣ Create a clean environment (recommended)
conda create -n hp_clean python=3.10 -y
conda activate hp_clean


### 2️⃣ Install dependencies
pip install -r requirements.txt


**Windows note:**  
LightGBM / XGBoost are often easier to install via conda:
conda install -c conda-forge lightgbm xgboost -y


---

## 🚀 Training Models

### Train a single model

python -m src.train --model ridge

### Train all models
python -m src.train --model all

### Common options
--seed <int> # random seed (default: 42)
--folds <int> # number of CV folds (default: 5)
--data-dir <path> # custom directory containing train.csv/test.csv
--no-export-compat-model
# disable artifacts/current/<model>.joblib


---

## 📦 Training Outputs & Artifacts

After training, the following artifacts are generated:

### `artifacts/reports/`
- `<model>_metrics.json` — CV metrics
- `<model>_oof.npy` — out-of-fold predictions
- `<model>_test_pred.npy` — averaged test predictions
- `metrics.csv` — model comparison table
- `cv_summary.json` — full CV summary

### `artifacts/registry/<model>/<run_id>/`
Each training run contains:
- `model.joblib`
- `metrics.json`
- `oof.npy`
- `test_pred.npy`
- `data_fingerprint.json`
- `train_args.json`
- `pipeline_repr.txt`

### `artifacts/current/`
- `<model>.joblib` — latest snapshot (optional, backward-compatible)

---

## 🧪 Running Tests

pytest -q


Tests verify:
- model artifacts completeness
- prediction reproducibility
- regression safety (RMSE not degrading)
- registry consistency

---

## 🧮 Kaggle Submission Mode

Generate Kaggle submission files using saved predictions.

### Single model

python -m src.predict kaggle --model lgbm

### Ensemble methods
python -m src.predict kaggle --ensemble blend_mean
python -m src.predict kaggle --ensemble blend_weighted
python -m src.predict kaggle --ensemble stack


Outputs are saved to: `artifacts/submissions/`

---

## 🏭 Production / Registry Prediction Mode

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

## 🔍 Feature Importance Analysis

Extract feature importance from registry models.

### Default: global best model

python analysis/feature_importance.py

### Specific model & run

python analysis/feature_importance.py --model lgbm --run-id <run_id> --topk 30

### Outputs
artifacts/reports/feature_importance/
├── <model>__<run_id>__top30.csv
├── <model>__<run_id>__top30.png
└── <model>__<run_id>__meta.json

---

## 📉 Error Analysis (Interactive)

Open Jupyter:
jupyter notebook


Then explore:
`notebooks/02_model_analysis/error_analysis_oof_interactive.ipynb`

Includes:
- residual diagnostics
- bias & segment-level errors
- worst-case prediction inspection

---

## 📈 Exploratory Data Analysis (EDA)

EDA is intentionally separated from the training pipeline.

Location: `notebooks/01_eda/House_EDA.ipynb`

Can be viewed directly on GitHub or run locally.

---

## 🏆 Dataset

Kaggle: House Prices – Advanced Regression Techniques

- Target variable: `SalePrice`
- Training performed in log-space: `log1p(SalePrice)`

---

## ✨ Notes

- This project prioritizes ML system design, correctness, and reproducibility over leaderboard tuning.
- All notebooks are optional for running the pipeline and included for transparency.
- Large artifacts (`.joblib`, `.npy`, registry runs) are intentionally gitignored.

---

## 📬 Contact

Questions, suggestions, or improvements are welcome — feel free to open an issue.
