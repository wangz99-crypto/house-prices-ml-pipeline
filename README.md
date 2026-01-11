# 🏠 House Prices ML Pipeline

# 

# End-to-End Machine Learning Pipeline with Ensemble Models, Error Analysis, and Reproducibility

# 

# 📌 Project Overview

# 

# This project implements a fully reproducible, end-to-end machine learning pipeline for the Kaggle House Prices – Advanced Regression Techniques dataset.

# 

# The goal is not only to achieve strong predictive performance, but also to demonstrate good ML engineering practices, including:

# 

# Modular pipelines

# 

# Cross-validation with OOF predictions

# 

# Ensemble models (Voting \& Stacking)

# 

# Feature importance analysis

# 

# Post-hoc error analysis

# 

# Clean-environment reproducibility

# 

# 🧠 Key Features

# 

# Unified Pipeline Design

# 

# Shared preprocessing + model-specific pipelines

# 

# Separate preprocessing strategies for linear vs tree-based models

# 

# Multiple Models Supported

# 

# Ridge Regression

# 

# ExtraTrees Regressor

# 

# XGBoost

# 

# LightGBM

# 

# Voting Regressor (mean)

# 

# Stacking Regressor (meta-learner)

# 

# Robust Evaluation

# 

# K-Fold Cross-Validation

# 

# Out-of-Fold (OOF) predictions

# 

# RMSE in log-space (Kaggle standard)

# 

# Interpretability \& Analysis

# 

# Automated feature importance extraction

# 

# Interactive error analysis notebooks (residuals, bias patterns)

# 

# Reproducibility

# 

# Verified from a clean Conda environment

# 

# All dependencies captured in requirements.txt

# 

# 📁 Repository Structure

# house-prices-ml-pipeline/

# ├── data/                # Raw Kaggle train/test CSV files

# ├── src/                 # Training, pipelines, evaluation logic

# │   ├── train.py

# │   ├── pipelines.py

# │   ├── evaluate.py

# │   └── predict.py

# ├── analysis/             # Automated post-training analysis

# │   └── feature\_importance.py

# ├── notebooks/

# │   ├── 01\_eda/

# │   │   └── house\_eda.ipynb

# │   └── 02\_model\_analysis/

# │       └── error\_analysis\_oof\_interactive.ipynb

# ├── reports/              # CV metrics, OOF predictions, summaries

# ├── models/               # Saved trained pipelines

# ├── tests/                # (Optional) unit / smoke tests

# ├── requirements.txt

# └── README.md

# 

# ⚙️ Setup \& Installation

# 1️⃣ Create a clean environment (recommended)

# conda create -n hp\_clean python=3.10 -y

# conda activate hp\_clean

# 

# 2️⃣ Install dependencies

# pip install -r requirements.txt

# 

# 

# ⚠️ On Windows, LightGBM/XGBoost are best installed via conda-forge:

# 

# conda install -c conda-forge lightgbm xgboost -y

# 

# 🚀 Training Models

# 

# Train a single model:

# 

# python -m src.train --model ridge

# 

# 

# Available models:

# 

# ridge

# extratrees

# xgb

# lgbm

# voting\_mean

# stacking

# 

# 

# Train all models:

# 

# python -m src.train --model all

# 

# 📊 Outputs \& Artifacts

# 

# After training, the reports/ directory will contain:

# 

# <model>\_metrics.json — CV performance metrics

# 

# <model>\_oof.npy — out-of-fold predictions

# 

# <model>\_test\_pred.npy — test-set predictions

# 

# cv\_summary.json — aggregated CV summary

# 

# metrics.csv — comparison across models

# 

# 🔍 Feature Importance Analysis

# 

# Generate feature importance for a trained model:

# 

# python analysis/feature\_importance.py --model extratrees --topk 30

# 

# 

# Outputs:

# 

# CSV with ranked features

# 

# Bar plot visualization

# 

# Metadata describing extraction method

# 

# 📉 Error Analysis (Interactive)

# 

# To explore model errors and bias patterns:

# 

# jupyter notebook

# 

# 

# Then open:

# 

# notebooks/02\_model\_analysis/error\_analysis\_oof\_interactive.ipynb

# 

# 

# This notebook provides:

# 

# Residual vs prediction plots

# 

# Error distribution analysis

# 

# Segment-level error breakdowns

# 

# Identification of worst-case predictions

# 

# 📈 Exploratory Data Analysis (EDA)

# 

# EDA is provided as a separate, exploratory notebook and is not part of the training pipeline.

# 

# Location:

# 

# notebooks/01\_eda/house\_eda.ipynb

# 

# 

# You can:

# 

# View it directly on GitHub (no execution needed)

# 

# Run it locally for full interactivity

# 

# 🔁 Reproducibility Verification

# 

# This project has been successfully validated from a clean environment.

# 

# Minimal reproduction:

# 

# conda create -n hp\_clean python=3.10 -y

# conda activate hp\_clean

# pip install -r requirements.txt

# python -m src.train --model ridge

# 

# 🏆 Dataset

# 

# Kaggle: House Prices – Advanced Regression Techniques

# 

# Target variable: SalePrice

# 

# Training performed in log-space using log1p(SalePrice)

# 

# ✨ Notes

# 

# The focus of this project is ML system design and analysis, not leaderboard optimization.

# 

# All notebooks are optional for running the pipeline but included for transparency and interpretability.

# 

# 📬 Contact

# 

# If you have questions or suggestions, feel free to open an issue or reach out.

