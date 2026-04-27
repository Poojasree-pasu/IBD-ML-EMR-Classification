"""
Preprocessing and Feature Transformation Utilities for IBD Classification

This module defines preprocessing pipelines and helper functions for transforming
structured EMR-derived features for machine learning models.

Features:
- Column definitions for ordinal, nominal, binary, and numerical variables
- Preprocessing pipelines for:
  - Logistic regression (with scaling)
  - Tree-based models (without scaling)
- Encoding strategies:
  - Ordinal encoding with predefined category ordering
  - One-hot encoding for nominal variables
- Back-transformation of logistic regression coefficients to original scale
  for interpretability (including odds ratios)

Notes:
- Preprocessing pipelines are fitted on training data and applied to both training and test data to prevent data leakage
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import os


ordinal_cols = [
    'Current_Age',
    'ageatIBDDiagnosis',
    'bmi_Classification',
    'maxCRP_Classification',
    'maxESR_Classification',
    'maxFecal calprotectin_Classification',
    'maxFecal lactoferrin_classification'
]

ordinal_categories = [
    ['Missing', 'Less than 18','Greater than or equal to 18 and less than 35','Greater than or equal to 35 and less than 50','Greater than or equal to 50 and less than 65','Greater than or equal to 65'],
    ['Missing', 'Less than or equal to 6','Greater than 6 and less than or equal to 18','Greater than 18 and less than or equal to 50','Greater than 50'],
    ['Missing','Underweight','Normal','Severely obese'],
    ['Missing', 'Normal','High','Very high','Extremely high',],
    ['Missing','Normal','High','Very high','Extremely high'],
    ['Missing', 'Normal','Abnormal','High','Extremely high'],
    ['Missing','Not Detected','Detected']
]

nominal_cols = [
    'Gender',
    'Ethnicity_Race'
]

binary_cols = ['IBD_OP','seenbyDeititian','Adalimumab','Certolizumabpegol','Etrasimod','Golimumab','Infliximab','Mirikizumab','Natalizumab','Olsalazine','Ozanimod','Risankizumab','Tofacitinib','Upadacitinib','Ustekinumab','Vedolizumab','Azathioprine','Balsalazide','Mercaptopurine','Mesalamine','Methotrexate','Sulfasalazine',
               'Budesonide','Hydrocortisone','Methylprednisolone','Prednisone',
               'Aminosalicylates','Immunomodulators','Anti-TNF','Anti-Interleukin','Anti-Integrin','JAKi','S1PRM','Steroid',
               'Endoscopy_Aphthous ulcer', 'Endoscopy_Anastomosis', 'Endoscopy_Pouchitis', 'Endoscopy_ulcerated',
               'Endoscopy_erythema', 'Endoscopy_Loss of vascularity', 'Endoscopy_Friable mucosa', 'Endoscopy_mayo',
               'Endoscopy_contact bleeding', 'Endoscopy_rutgeerts', 'Endoscopy_Ileo-Colonic', 'Endoscopy_SES-CD',
               'Gastroenterology_Ileitis', 'Gastroenterology_Abscess', 'Gastroenterology_Fistula',
               'Gastroenterology_Stricture', 'Gastroenterology_Proctocolectomy', 'Gastroenterology_Colectomy',
               'Gastroenterology_Ileostomy', 'Gastroenterology_Colitis', 'Gastroenterology_Wall Thickening',
               'Gastroenterology_Ischemia', 'Gastroenterology_Ischemic', 'Gastroenterology_Ischaemia',
               'Gastroenterology_Ischaemic', 'Gastroenterology_Ileocecal resection', 'Gastroenterology_Hemicolectomy',
               'Gastroenterology_Diverting ileostomy', 'Gastroenterology_Proctosigmoiditis',
               'Gastroenterology_Total colectomy', 'Gastroenterology_Total proctocolectomy',
               'Gastroenterology_Ileorectal', 'Gastroenterology_Fibrostenotic', 'Gastroenterology_J-pouch',
               'Gastroenterology_Pouchitis', 'Gastroenterology_Ileocolonic Anastomosis',
               'Gastroenterology_Ileal Pouch Anal Anastomosis', 'Gastroenterology_Ileoanal',
               'Imaging_Ileitis', 'Imaging_Abscess', 'Imaging_Fistula', 'Imaging_Stricture', 'Imaging_Proctocolectomy',
               'Imaging_Colectomy', 'Imaging_Ileostomy', 'Imaging_Colitis', 'Imaging_Wall Thickening',
               'Imaging_Ischemia', 'Imaging_Ischemic', 'Imaging_Ischaemia', 'Imaging_Ischaemic',
               'Imaging_Ileocecal resection', 'Imaging_Hemicolectomy', 'Imaging_Diverting ileostomy',
               'Imaging_Proctosigmoiditis', 'Imaging_Total colectomy', 'Imaging_Total proctocolectomy',
               'Imaging_Ileorectal', 'Imaging_Fibrostenotic', 'Imaging_J-pouch', 'Imaging_Pouchitis',
               'Imaging_Ileocolonic Anastomosis', 'Imaging_Ileal Pouch Anal Anastomosis', 'Imaging_Ileoanal',
               'pathology_Chronic', 'pathology_Ileitis', 'pathology_Pouchitis', 'pathology_Colitis',
               'pathology_Granuloma', 'pathology_Granulomatous', 'pathology_Crypt distortion', 'pathology_Crypt Abscess'
               ]


def get_logistic_regression_preprocessor(X_train, return_cols=False):
    numerical_cols = [
        col for col in X_train.columns
        if col not in ordinal_cols + nominal_cols + binary_cols
    ]

    preprocessor = ColumnTransformer([
        ("ordinal", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ), ordinal_cols),
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
        ("num", StandardScaler(), numerical_cols),
        ("binary", "passthrough", binary_cols)
    ])
    if return_cols:
        return preprocessor, numerical_cols
    return preprocessor


def get_tree_preprocessor(X_train):
    numerical_cols = [
        col for col in X_train.columns
        if col not in ordinal_cols + nominal_cols + binary_cols
    ]

    preprocessor = ColumnTransformer([
        ("ordinal", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ), ordinal_cols),
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
        ("num", "passthrough", numerical_cols),
        ("binary", "passthrough", binary_cols)
    ])
    return preprocessor


def extract_lr_coefficients(final_model, label_encoder, numerical_cols, model, plots_dir=None):
    preprocessor = final_model.named_steps['preprocess']
    lr_model = final_model.named_steps['model']
    feature_names = preprocessor.get_feature_names_out()
    coefficients = lr_model.coef_
    intercept = lr_model.intercept_
    classes = label_encoder.classes_

    # NUMERIC SCALER
    num_transformer = preprocessor.named_transformers_['num']
    means = num_transformer.mean_
    stds = num_transformer.scale_

    if len(numerical_cols) != len(stds):
        raise ValueError("Mismatch between numerical_cols and scaler features")

    # Map num__feature → index
    num_feature_map = {
        f"num__{col}": i for i, col in enumerate(numerical_cols)
    }
    # BACK-TRANSFORM COEFFICIENTS
    beta_original = coefficients.copy()
    for j, fname in enumerate(feature_names):
        if fname in num_feature_map:
            idx = num_feature_map[fname]
            std = stds[idx]
            beta_original[:, j] = coefficients[:, j] / std

    # BACK-TRANSFORM INTERCEPT
    intercept_original = intercept.copy()
    for i in range(len(classes)):
        adjustment = 0
        for j, fname in enumerate(feature_names):
            if fname in num_feature_map:
                idx = num_feature_map[fname]
                mean = means[idx]
                std = stds[idx]

                adjustment += (coefficients[i, j] * mean) / std
        intercept_original[i] = intercept[i] - adjustment

    clean_features = [f.split("__")[-1] for f in feature_names]

    results = []
    for i, class_label in enumerate(classes):
        for j, feature in enumerate(clean_features):
            beta = beta_original[i, j]
            odds_ratio = np.exp(beta)

            results.append({
                "Class": class_label,
                "Feature": feature,
                "Coefficient": beta,
                "Odds Ratio": odds_ratio
            })

    coef_df = pd.DataFrame(results)

    coef_df["Abs_Coeff"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by=["Class", "Abs_Coeff"], ascending=[True, False])
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        coef_df.to_csv(os.path.join(plots_dir, f"IBDwordscountBinary_{model}_coefficients_unscalednumericalfeatures.csv"),index=False)

    return coef_df, intercept_original

import numpy as np
import pandas as pd
import os


def Rfecv_extract_lr_coefficients(final_model,feature_names,label_encoder,numerical_cols,means,stds,model,plots_dir=None):
    classes = label_encoder.classes_
    coefficients = final_model.coef_
    intercept = final_model.intercept_

    print(coefficients.shape)
    print(len(feature_names))

    if len(numerical_cols) != len(stds):
        raise ValueError("Mismatch between numerical_cols and scaler features")

    num_feature_map = {
        f"num__{col}": i for i, col in enumerate(numerical_cols)
    }
    # BACK-TRANSFORM COEFFICIENTS
    beta_original = coefficients.copy()

    for j, fname in enumerate(feature_names):
        if fname in num_feature_map:
            idx = num_feature_map[fname]
            beta_original[:, j] = coefficients[:, j] / stds[idx]
    # BACK-TRANSFORM INTERCEPT
    intercept_original = intercept.copy()

    for i in range(len(classes)):
        adjustment = 0
        for j, fname in enumerate(feature_names):
            if fname in num_feature_map:
                idx = num_feature_map[fname]
                adjustment += (coefficients[i, j] * means[idx]) / stds[idx]

        intercept_original[i] = intercept[i] - adjustment

    clean_features = [f.split("__")[-1] for f in feature_names]
    results = []
    for i, class_label in enumerate(classes):
        for j, feature in enumerate(clean_features):
            beta = beta_original[i, j]
            odds_ratio = np.exp(beta)

            results.append({
                "Class": class_label,
                "Feature": feature,
                "Coefficient": beta,
                "Odds Ratio": odds_ratio
            })

    coef_df = pd.DataFrame(results)

    coef_df["Abs_Coeff"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by=["Class", "Abs_Coeff"], ascending=[True, False])
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        coef_df.to_csv(os.path.join(plots_dir, f"IBDwordscountBinary_{model}_coefficients_unscalednumericalfeatures.csv"),index=False)
    return coef_df, intercept_original


