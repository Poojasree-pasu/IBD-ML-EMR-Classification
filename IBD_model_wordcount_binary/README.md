# IBD Model Development (Binary Word Count Features)

This repository contains code and notebooks for the development, training, and evaluation of multi-class machine learning models to classify Inflammatory Bowel Disease (IBD) using structured clinical data. In this version, clinical word count features are converted from continuous counts to binary representations, while all other features remain unchanged.

## Overview

The objective of this project is to classify patients into:
- Crohn’s Disease (CD)
- Ulcerative Colitis (UC)
- No IBD

Models are trained using a standard train–test split and evaluated using repeated stratified cross-validation to ensure robustness and reproducibility.

## Installation
pip install -r requirements.txt

## Project Structure
```
IBD_model_development/
├── data/                     # IBD registry data
└── src/
    ├── notebooks/            # Jupyter notebooks for model development and evaluation
    │   ├── IBD_logistic_regression_wordcountbinary.ipynb
    │   ├── IBD_random_forest_wordcountbinary.ipynb
    │   ├── IBD_xgboost_wordcountbinary.ipynb
    │   ├── IBD_logistic_regression_rfecv_wordcountbinary.ipynb
    │   ├── IBD_random_forest_rfecv_wordcountbinary.ipynb
    │   └── IBD_xgboost_rfecv_wordcountbinary.ipynb
    ├── preprocessing.py      # Feature preprocessing pipelines
    ├── utils.py              # Configuration and hyperparameter grids
    ├── evaluation_plots.py   # Model evaluation and visualization functions
    └── repeated_stratified_kfold.py  # Cross-validation implementation
```

## Main Workflow:

## 1. Data

The dataset is located in `data/`:
- 'IBD_1200patients_wordcounttobinary.csv' - Synthetic de-identified sample

The same SQL data extraction criteria, data cleaning procedures, cohort definitions, and feature engineering workflow used in the primary analysis were applied in this version. 
The only change was that clinical word count features were converted from continuous counts to binary indicators, while all other features remained unchanged.

**Note:** Model training and evaluation results in the manuscript were obtained using the original EMR dataset, not the synthetic data included here.

### Target Classes
| Label | Class |
|-------|-------|
| 0 | Crohn's Disease (CD) |
| 1 | No IBD |
| 2 | Ulcerative Colitis (UC) |


## 2. Models

The following models are implemented:

| Model | Description |
|-------|-------------|
| Logistic Regression | L2-regularized multi-class logistic regression |
| Random Forest | Ensemble tree-based classifier |
| XGBoost | Gradient boosting classifier |

Run any notebook:
jupyter notebook src/notebooks/IBD_logistic_regression.ipynb

## 3. Model Training Pipeline

Each notebook performs:

1. Train–test split (70/30)
    Training: 840 patients
    Test: 360 patients
2. Preprocessing
     - Ordinal encoding (OrdinalEncoder with predefined ordering)
     - One-hot encoding (categorical features)
     - Scaling (StandardScaler for logistic regression)
     - All preprocessing transformations were fit exclusively on the training dataset and subsequently applied to the test dataset to prevent data leakage.
3. Model training and hyperparameter tuning
      - Grid search with 5-fold cross-validation is performed on the training dataset.
      - Optimal hyperparameters are selected
4. Model Evaluation (Repeated Stratified K-Fold Cross-Validation)
    - **5 splits × 50 repeats** = 250 model training iterations per model
    - Stratification ensures class distribution is preserved across all folds
    - Cross-validation is performed on training data **only** to prevent data leakage
    **Performance Metrics with 95% Confidence Intervals:**
      - **AUC-ROC**
      - **Sensitivity** (per-class recall)
      - **Specificity** (per-class)
      - **Positive Predictive Value (PPV)**
      - **Accuracy**
      - Note: For feature selection workflows, the selected feature set is used for repeated stratified cross-validation
5. Final model training
6. Final evaluation
   Performance is evaluated on the independent held-out test set


### Feature Selection
Feature Selection Notebooks (RFECV)
RFECV notebooks follow the same overall modeling pipeline as the other notebooks (train–test split, preprocessing, hyperparameter tuning, and evaluation), with the addition of a feature selection step.

- Feature selection is performed on the training dataset only
- An optimal subset of features is identified using:
  RFECV (recursive feature elimination with cross-validation)
       - Iteratively removes less important features based on model importance
       - Optimal subset is selected using stratified cross-validation on the training data.
The selected feature set is then fixed

Using this reduced feature set, the model is:
- Retrained using the selected features
- Evaluated using repeated stratified k-fold cross-validation (5 × 50) on the training data
- Finally evaluated on the independent held-out test datasetß

## Output Files

### Performance Metrics (CSV)
- `*repeatedstratifiedkfold_results.csv` - 95% CI for AUC, sensitivity, specificity, PPV, accuracy

### Plots 
- `*_confusionmatrix.pdf` - Confusion matrix
- `*_ROC_allclasses.pdf` - ROC curves for all classes with macro-average
- `*_ROC_{class}.pdf` - Individual class ROC curves
- `*_ROC_macro_only.pdf` - Macro-average ROC only
- `*_shapsummary_top20_{class}.pdf` - Top 20 SHAP feature importance
- `*_shapsummary_allfeatures_{class}.pdf` - Full SHAP feature importance

## Output Directories

All notebooks save results to ../plots/ and ../saved_models/:

- plots/ — Evaluation figures (ROC curves, confusion matrices, SHAP plots)
- saved_models/ — Trained models and test datasets


