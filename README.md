# Beyond ICD-10 Codes: A Multi-Class Machine Learning Approach to Identify True IBD Cases in Electronic Medical Records

This repository contains code and documentation for electronic medical record data extraction criteria, post extraction data cleaning, cohort construction, feature engineering, and the development and evaluation of multi class machine learning models to classify Inflammatory Bowel Disease (IBD) using structured and unstructured electronic medical record (EMR) data.


The study aims to improve identification of true IBD cases (Crohn’s Disease [CD], Ulcerative Colitis [UC], and No IBD) using machine learning models applied to EMR data.

---

## Project Structure
```
IBD-ML-EMR-Classification/
│
├── IBD_model_development/
│   ├── data/                         # Synthetic sample dataset
│   │
│   ├── src/
│   │   ├── notebooks/
│   │   │   ├── IBD_post_extraction_cleaning_cohort_feature_engineering.ipynb        #Post-extraction cleaning, cohort construction and feature engineering
│   │   │   ├── IBD_logistic_regression.ipynb
│   │   │   ├── IBD_random_forest.ipynb
│   │   │   ├── IBD_xgboost.ipynb
│   │   │   ├── IBD_lasso_logistic_regression.ipynb
│   │   │   ├── IBD_logistic_regression_rfecv.ipynb
│   │   │   ├── IBD_random_forest_rfecv.ipynb
│   │   │   ├── IBD_xgboost_rfecv.ipynb
│   │   │   ├── IBD_model_comparison.ipynb
│   │   │   ├── 3codes&IBDmed.ipynb
│   │   │   └── 5codes.ipynb
│   │   │
│   │   ├── preprocessing.py
│   │   ├── evaluation_plots.py
│   │   ├── repeated_stratified_kfold.py
│   │   └── utils.py
│   │
│   └── README.md
│
├── IBD_model_wordcount_binary/
│ Models using binary representation of clinical word count features, while all other features remain unchanged.
│
├── IBD_temporal_analysis/
│ Temporal validation (training on pre-2022 data, testing on data from 2022 onwards)
│
├── requirements.txt
│ Python dependencies required to reproduce the analysis
│
└── README.md
```

Each subdirectory contains its own README with detailed instructions and descriptions.

---
## Data Preparation and Feature Engineering

The notebook 
`IBD_model_development/src/notebooks/IBD_post_extraction_cleaning_cohort_feature_engineering.ipynb`
documents the institutional data extraction criteria and the Python workflow used for post extraction data cleaning, cohort construction, and feature engineering.

The notebook covers diagnosis definitions, medication variables, labs and BMI preprocessing, demographic variables, encounter variables, clinical text features and construction of the final patient level analytic dataset.

The original institutional SQL queries are not included because they depend on the local Epic Caboodle and OnBase database structures and operate on protected patient level data.

## Methods Overview

- **Models Used**
  - Logistic Regression (LR)
  - Random Forest (RF)
  - XGBoost (XGB)

## Model Training Pipeline

Each model follows a standardized workflow:

- **Train–test split**
   - Standard random split (70/30) is used for model development
   - For temporal validation, models are trained on pre-2022 data and evaluated on data from 2022 onwards
  
- **Preprocessing**
  - Pipeline-based approach to prevent data leakage
  - Ordinal encoding (ordered categorical variables)
  - One-hot encoding (nominal variables)
  - Standard scaling (for LR)
  - All preprocessing steps are fitted exclusively on the training dataset and applied to validation and test data  

- **Hyperparameter tuning**
  - Grid search with 5-fold cross-validation on the training dataset

- **Model Evaluation (training phase)**
  - Repeated Stratified K-Fold Cross-Validation (5 folds × 50 repetitions)
  - Metrics:
    - AUROC (macro and per-class)
    - Sensitivity
    - Specificity
    - Positive Predictive Value (PPV)
    - Accuracy
    Performance is reported with **95% confidence intervals** derived from repeated cross-validation.

- **Final model training**
   - Models are retrained on the full training dataset 

- **Final Evaluation**
  - Independent test dataset
  - Confusion matrices and classification metrics

---

## Performance Metrics

Models are evaluated using:

- Area Under the Receiver Operating Characteristic Curve (AUROC)
- Sensitivity (recall)
- Specificity
- Accuracy

## Feature Selection

Feature selection was performed using two complementary approaches:

### 1. RFECV (Recursive Feature Elimination with Cross-Validation)
- Iteratively removes less important features based on model-specific importance  
- The optimal feature subset is determined using stratified cross-validation on the training dataset  
- Once selected, the feature set is fixed and used for all downstream analysis  

### 2. LASSO (L1-Regularized Logistic Regression)
- Feature selection performed using a model-based approach (`SelectFromModel`)  
- Features with non-zero coefficients are retained  
- **Note:** LASSO-based feature selection was implemented only in the `IBD_model_development` module  

### Feature Selection Workflow
For both approaches:
- Feature selection is performed **only on the training dataset**  
- The selected feature set is then:
  - Used for repeated stratified cross-validation (5 × 50)  
  - Used to retrain the final model  
  - Evaluated on the independent test dataset  

## How to Run

### Install dependencies
pip install -r requirements.txt

### Run notebooks
jupyter notebook IBD_model_development/src/notebooks/IBD_post_extraction_cleaning_cohort_feature_engineering.ipynb

This notebook expects dataframes generated from the institutional extraction steps. Patient level source data are not included in the public repository.

Navigate to desired module:
- IBD_model_development
- Launch: jupyter notebook

##  Reproducibility

- All models use fixed random seeds  
- Preprocessing is embedded within pipelines to prevent data leakage  
- Hyperparameters are selected using cross-validation and fixed during evaluation  
- Repeated stratified cross-validation ensures stable performance estimates  

## Outputs

### Performance Metrics
- `*repeatedstratifiedkfold_results.csv`  
  - AUROC, sensitivity, specificity, PPV, accuracy (with 95% CI)

### Plots
- Confusion matrices  
- ROC curves (per-class and macro-average)  
- SHAP feature importance plots  

### Output Directories
- `plots/` — Evaluation figures  
- `saved_models/` — Trained models and test datasets  

---


## Data Availability

Due to patient privacy and institutional restrictions, raw EMR data cannot be shared.

This repository includes:

- Documentation of institutional data extraction criteria
- Post-extraction data-cleaning and feature-engineering code
- Cohort and variable definitions
- A synthetic sample dataset for demonstration purposes
- Model-development and evaluation pipelines

**Note:** Model training and evaluation results reported in the manuscript were obtained using the original EMR dataset and not the synthetic data included in this repository.
