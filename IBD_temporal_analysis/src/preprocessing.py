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

Notes:
- Preprocessing pipelines are applied within training data to prevent data leakage
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer



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
    ['Less than 18','Greater than or equal to 18 and less than 35','Greater than or equal to 35 and less than 50','Greater than or equal to 50 and less than 65','Greater than or equal to 65'],
    ['Less than or equal to 6','Greater than 6 and less than or equal to 18','Greater than 18 and less than or equal to 50','Greater than 50'],
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
               'Aminosalicylates','Immunomodulators','Anti_TNF','Anti_Interleukin','Anti_Integrin','JAKi','S1PRM','Steroid']


def get_logistic_regression_preprocessor(X_train):
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