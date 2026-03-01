# MTC-ML: Machine Learning Analysis for Medullary Thyroid Carcinoma

A comprehensive machine learning and statistical analysis pipeline for predicting **recurrence** in Medullary Thyroid Carcinoma (MTC) patients. The project includes a full analysis script, a robust validation framework, and an interactive Shiny dashboard.

---

## Overview

This repository contains three complementary R scripts that together form an end-to-end clinical ML research workflow:

| File | Purpose |
|------|---------|
| `MTC_Full_Analysis.R` | Complete statistical analysis + ML modeling pipeline |
| `MTC_Robust_ML.R` | Advanced validation with nested CV, bootstrap CI & multi-seed stability |
| `MTC_Shiny_App.R` | Interactive Shiny dashboard for exploring results and tuning models |

---

## Requirements

- **R** ≥ 4.0
- **Data file:** `DATA_2.csv` must be present in the working directory

### R Packages

All scripts auto-install missing packages. Key dependencies include:

- **Data & Tables:** tidyverse, tableone
- **Survival:** survival, survminer
- **ML / Modeling:** caret, randomForest, glmnet, xgboost, e1071, nnet, gbm, rpart, naivebayes
- **Class Balancing:** ROSE
- **Evaluation:** pROC, DALEX, DALEXtra
- **Visualization:** ggplot2, ggpubr, cowplot, corrplot, viridis, plotly
- **Shiny (dashboard only):** shiny, shinydashboard, shinyWidgets, shinyjs, DT

---

## Scripts

### 1. `MTC_Full_Analysis.R`

The main analysis pipeline that runs 15 sequential steps and produces all figures and tables for a research manuscript.

**What it does:**

- **Baseline characteristics** — Table 1 stratified by disease type (Sporadic vs Hereditary)
- **Exploratory figures** — Age distribution, tumor size by stage, calcitonin pre/post comparison, treatment frequency
- **Correlation matrix** — Spearman correlation heatmap of continuous clinical variables
- **Survival analysis** — Kaplan-Meier curves (overall, by disease type, stage, recurrence) and multivariable Cox proportional hazards model with forest plot
- **ML pipeline** — Trains 9 models (4 white-box + 5 black-box) for recurrence prediction:
  - *White-box:* LASSO, CART, Naive Bayes, KNN
  - *Black-box:* Random Forest, SVM, Neural Network, GBM, XGBoost
- **Evaluation** — ROC curves, confusion matrices, performance comparison (AUC, sensitivity, specificity, F1)
- **Explainability (XAI)** — SHAP values (XGBoost), Gini importance (RF), relative influence (GBM), DALEX permutation importance
- **Calcitonin reduction analysis** — Biochemical cure assessment

**Outputs:** 12 figures (600 DPI PNG) and 11 tables (CSV).

### 2. `MTC_Robust_ML.R`

Extends the ML analysis with rigorous validation methods to assess model stability and generalizability.

**What it does:**

- **Nested cross-validation** — 10-fold outer × 5-fold inner CV to obtain unbiased performance estimates
- **Bootstrap confidence intervals** — 500 bootstrap iterations per model for AUC confidence intervals
- **Multi-seed stability** — Repeats the full train/test pipeline across 30 different random seeds to assess variance
- **Ensemble stacking** — Stacked generalization combining all base model predictions
- **Comprehensive comparison** — Aggregated figures comparing nested CV, bootstrap, and multi-seed results

**Key settings (configurable):**

```r
N_BOOT  <- 500   # Bootstrap iterations
N_SEEDS <- 30    # Number of random seeds
OUTER_K <- 10    # Outer CV folds
INNER_K <- 5     # Inner CV folds
```

### 3. `MTC_Shiny_App.R`

An interactive Shiny dashboard (`MTC-ML v2.0`) for real-time exploration and experimentation.

**Tabs:**

- **Overview** — Summary statistics and key results
- **Settings** — Tune all hyperparameters, CV strategy, train/test split, and class balancing method
- **Data Explorer** — Browse and filter the raw dataset with Table 1
- **Survival** — Interactive Kaplan-Meier plots and Cox model results
- **Training Times** — Model training benchmarks
- **ROC & Performance** — Interactive ROC curves and performance comparison tables
- **Feature Importance** — Variable importance across all models
- **XAI / SHAP** — SHAP-based explanations for black-box models
- **Predictions** — Individual patient-level predictions
- **Export Center** — Download all figures (600 DPI) and tables (CSV) in one place

**Run the app:**

```r
shiny::runApp("MTC_Shiny_App.R")
```

---

## Data

The analysis expects a CSV file named `DATA_2.csv` in the working directory. The dataset should contain the following key variables:

- **Demographics:** `age_at_diagnosis`, `sex`
- **Disease characteristics:** `disease_type`, `ret_mutation`, `tumor_size_mm`, `stage`, `multifocal`
- **Invasion & metastasis:** `lymph_node_invasion`, `capsular_invasion`, `soft_tissue_invasion`, `metastasis_at_diagnosis`, `lymph_node_metastasis`, `lung_metastasis_present`, `bone_metastasis_present`, `liver_metastasis_present`
- **Biomarkers:** `calcitonin_preop`, `calcitonin_postop`, `cea_preop`, `cea_postop`
- **Outcomes:** `recurrence`, `death_mtc`, `follow_up_year`, `vital_status`
- **Treatment:** `tki_therapy`, `radiotherapy`, `lutetium_therapy`, `chemotherapy`, `mibg_therapy`

> **Note:** The dataset is not included in this repository due to patient privacy.

---

## Quick Start

```r
# 1. Place DATA_2.csv in your working directory

# 2. Run the full analysis
source("MTC_Full_Analysis.R")

# 3. Run robust validation (takes longer)
source("MTC_Robust_ML.R")

# 4. Launch the interactive dashboard
shiny::runApp("MTC_Shiny_App.R")
```

---

## Models

All models are trained for **binary classification** of recurrence (Rec0 vs Rec1) using ROSE-balanced training data and evaluated on a stratified held-out test set (70/30 split).

| Category | Model | Method |
|----------|-------|--------|
| White-box | LASSO | `glmnet` (L1-regularized logistic regression) |
| White-box | CART | `rpart` (Decision tree) |
| White-box | Naive Bayes | `naive_bayes` |
| White-box | KNN | `knn` (k-Nearest Neighbors) |
| Black-box | Random Forest | `rf` (500 trees) |
| Black-box | SVM | `svmRadial` (RBF kernel) |
| Black-box | Neural Network | `nnet` (Single hidden layer) |
| Black-box | GBM | `gbm` (Gradient Boosting Machine) |
| Black-box | XGBoost | `xgboost` (native API with early stopping) |

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to use, share, and adapt this work **as long as you provide appropriate attribution**.

---

## Citation

If you use this code in your research, please cite this repository. Click the **"Cite this repository"** button on GitHub or use:

```bibtex
@software{MTC_RecurrencePrediction,
  author    = {Cakir, Mustafa},
  title     = {MTC-RecurrencePrediction: Machine Learning Pipeline for
               Predicting Recurrence in Medullary Thyroid Carcinoma},
  year      = {2026},
  url       = {https://github.com/ckrmustafa/MTC-RecurrencePrediction},
  license   = {CC-BY-4.0}
}
```

> **Note:** When the associated journal article is published, this section will be updated with the full paper citation.
