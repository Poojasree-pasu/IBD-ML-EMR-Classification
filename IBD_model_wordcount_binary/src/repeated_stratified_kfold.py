"""
Repeated Stratified K-Fold Evaluation for IBD Multi-Class Models

This module implements repeated stratified k-fold cross-validation for evaluating model performance on the training data,
including both baseline models and models with RFECV-based feature selection.

Features:
- Stratified k-fold cross-validation with multiple repetitions
- Evaluation of multi-class classification performance (CD, UC, No IBD)
- Computation of AUC, sensitivity, specificity, PPV, and accuracy
- Estimation of 95% confidence intervals for all metrics

Notes:
- Cross-validation is performed on training data only to prevent data leakage
- Results are saved as CSV files
"""

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import pandas as pd
import os

def repeated_stratified_kfold(model_name, X_train, y_train, preprocessor, params, le, plots_dir):
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=42)
    n_classes = len(le.classes_)
    auc_scores = []
    acc_scores = []

    auc_per_class = [[] for _ in range(n_classes)]
    sens_all = [[] for _ in range(n_classes)]
    spec_all = [[] for _ in range(n_classes)]
    ppv_all  = [[] for _ in range(n_classes)]

    for train_idx, val_idx in rkf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        preprocessor_fold = clone(preprocessor)
        if model_name == 'LogisticRegression':
            mdl = LogisticRegression(**params, random_state=42)

        elif model_name == 'RandomForest':
            mdl = RandomForestClassifier(**params, random_state=42)

        elif model_name == 'XGBoost':
            mdl = XGBClassifier(**params,objective='multi:softprob',num_class=n_classes,random_state=42)
        else:
            raise ValueError("Invalid model name")

        model = Pipeline([("preprocess", preprocessor_fold),("model", mdl)])
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        # metrics
        acc_scores.append(accuracy_score(y_val, y_pred))
        auc_scores.append(
            roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro"))

        # Per-class AUC
        y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
        for i in range(n_classes):
            auc_i = roc_auc_score(y_val_bin[:, i], y_prob[:, i])
            auc_per_class[i].append(auc_i)

        # Confusion matrix metrics
        cm = confusion_matrix(y_val, y_pred, labels=np.arange(n_classes))
        for i in range(n_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FN + FP)

            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            ppv  = TP / (TP + FP) if (TP + FP) > 0 else 0

            sens_all[i].append(sens)
            spec_all[i].append(spec)
            ppv_all[i].append(ppv)

    def get_ci(arr):
        return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    def fmt(x):
        return f"{x[0]:.2f} ({x[1]:.2f}-{x[2]:.2f})"

    rows = []

    for i, cls in enumerate(le.classes_):
        rows.append({
            "Class": cls,
            "AUC (95% CI)": fmt(get_ci(auc_per_class[i])),
            "Sensitivity (95% CI)": fmt(get_ci(sens_all[i])),
            "Specificity (95% CI)": fmt(get_ci(spec_all[i])),
            "PPV (95% CI)": fmt(get_ci(ppv_all[i])),
            "Accuracy (95% CI)": ""  # not applicable
        })

    # Overall row
    rows.append({
        "Class": "Overall",
        "AUC (95% CI)": fmt(get_ci(auc_scores)),
        "Sensitivity (95% CI)": "",
        "Specificity (95% CI)": "",
        "PPV (95% CI)": "",
        "Accuracy (95% CI)": fmt(get_ci(acc_scores))
    })

    df_final_table = pd.DataFrame(rows)[[
        "Class",
        "AUC (95% CI)",
        "Sensitivity (95% CI)",
        "Specificity (95% CI)",
        "PPV (95% CI)",
        "Accuracy (95% CI)"
    ]]
    df_final_table.to_csv(os.path.join(plots_dir, f"IBDwordscountBinary_{model_name}_repeatedstratifiedkfold_results.csv"), index=False)
    # print(df_final_table)
    return df_final_table



def rfecv_repeated_stratified_kfold(model_name, selector_final, X_train, y_train, preprocessor, params, le, plots_dir):

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=42)
    n_classes = len(le.classes_)
    auc_scores = []
    acc_scores = []

    auc_per_class = [[] for _ in range(n_classes)]
    sens_all = [[] for _ in range(n_classes)]
    spec_all = [[] for _ in range(n_classes)]
    ppv_all  = [[] for _ in range(n_classes)]

    for train_idx, val_idx in rkf.split(X_train, y_train):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        preprocessor_fold = clone(preprocessor)
        X_tr_processed = preprocessor_fold.fit_transform(X_tr)
        X_val_processed = preprocessor_fold.transform(X_val)

        selected_mask = selector_final.support_
        X_tr_selected = X_tr_processed[:, selected_mask]
        X_val_selected = X_val_processed[:, selected_mask]

        # Model
        if model_name == 'LogisticRegression_rfecv':
            mdl = LogisticRegression(**params, random_state=42)

        elif model_name == 'RandomForest_rfecv':
            mdl = RandomForestClassifier(**params, random_state=42)

        elif model_name == 'XGBoost_rfecv':
            mdl = XGBClassifier(**params,
                                objective='multi:softprob',
                                num_class=n_classes,
                                random_state=42)
        else:
            raise ValueError("Invalid model name")

        mdl.fit(X_tr_selected, y_tr)
        # Predict
        y_pred = mdl.predict(X_val_selected)
        y_prob = mdl.predict_proba(X_val_selected)

        # Overall metrics
        acc_scores.append(accuracy_score(y_val, y_pred))
        auc_scores.append(
            roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")
        )


        # Per-class AUC
        y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))

        for i in range(n_classes):
            auc_i = roc_auc_score(y_val_bin[:, i], y_prob[:, i])
            auc_per_class[i].append(auc_i)

        # Confusion matrix metrics
        cm = confusion_matrix(y_val, y_pred, labels=np.arange(n_classes))
        for i in range(n_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FN + FP)

            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            ppv  = TP / (TP + FP) if (TP + FP) > 0 else 0

            sens_all[i].append(sens)
            spec_all[i].append(spec)
            ppv_all[i].append(ppv)
    def get_ci(arr):
        return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    def fmt(x):
        return f"{x[0]:.2f} ({x[1]:.2f}-{x[2]:.2f})"
    rows = []

    for i, cls in enumerate(le.classes_):
        rows.append({
            "Class": cls,
            "AUC (95% CI)": fmt(get_ci(auc_per_class[i])),
            "Sensitivity (95% CI)": fmt(get_ci(sens_all[i])),
            "Specificity (95% CI)": fmt(get_ci(spec_all[i])),
            "PPV (95% CI)": fmt(get_ci(ppv_all[i])),
            "Accuracy (95% CI)": "-"
        })

    # Overall row
    rows.append({
        "Class": "Overall",
        "AUC (95% CI)": fmt(get_ci(auc_scores)),
        "Sensitivity (95% CI)": "-",
        "Specificity (95% CI)": "-",
        "PPV (95% CI)": "-",
        "Accuracy (95% CI)": fmt(get_ci(acc_scores))
    })

    df_final_table = pd.DataFrame(rows)[[
        "Class",
        "AUC (95% CI)",
        "Sensitivity (95% CI)",
        "Specificity (95% CI)",
        "PPV (95% CI)",
        "Accuracy (95% CI)"
    ]]

    df_final_table.to_csv(
        os.path.join(plots_dir, f"IBDwordscountBinary_{model_name}_repeatedstratifiedkfold_results.csv"),
        index=False
    )
    # print(df_final_table.to_string(index=False))
    return df_final_table



def lasso_repeated_stratified_kfold(model_name, selector, X_train, y_train, preprocessor, params, le, plots_dir):

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=42)
    n_classes = len(le.classes_)

    auc_scores = []
    acc_scores = []

    auc_per_class = [[] for _ in range(n_classes)]
    sens_all = [[] for _ in range(n_classes)]
    spec_all = [[] for _ in range(n_classes)]
    ppv_all  = [[] for _ in range(n_classes)]

    for train_idx, val_idx in rkf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        # Preprocessing
        preprocessor_fold = clone(preprocessor)
        X_tr_processed = preprocessor_fold.fit_transform(X_tr)
        X_val_processed = preprocessor_fold.transform(X_val)

        mask = selector.get_support()
        X_tr_selected = X_tr_processed[:, mask]
        X_val_selected = X_val_processed[:, mask]
        # Model
        if model_name == "LassologisticRegression":
            mdl = LogisticRegression(**params, random_state=42)
        else:
            raise ValueError("Only LassologisticRegression supported here")

        mdl.fit(X_tr_selected, y_tr)
        y_pred = mdl.predict(X_val_selected)
        y_prob = mdl.predict_proba(X_val_selected)
        acc_scores.append(accuracy_score(y_val, y_pred))
        auc_scores.append(
            roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro"))

        y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))

        for i in range(n_classes):
            auc_i = roc_auc_score(y_val_bin[:, i], y_prob[:, i])
            auc_per_class[i].append(auc_i)

        cm = confusion_matrix(y_val, y_pred, labels=np.arange(n_classes))

        for i in range(n_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FN + FP)

            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            ppv  = TP / (TP + FP) if (TP + FP) > 0 else 0

            sens_all[i].append(sens)
            spec_all[i].append(spec)
            ppv_all[i].append(ppv)

    def get_ci(arr):
        return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)
    def fmt(x):
        return f"{x[0]:.2f} ({x[1]:.2f}-{x[2]:.2f})"   # fixed dash issue
    rows = []

    for i, cls in enumerate(le.classes_):
        rows.append({
            "Class": cls,
            "AUC (95% CI)": fmt(get_ci(auc_per_class[i])),
            "Sensitivity (95% CI)": fmt(get_ci(sens_all[i])),
            "Specificity (95% CI)": fmt(get_ci(spec_all[i])),
            "PPV (95% CI)": fmt(get_ci(ppv_all[i])),
            "Accuracy (95% CI)": "-"
        })
    rows.append({
        "Class": "Overall",
        "AUC (95% CI)": fmt(get_ci(auc_scores)),
        "Sensitivity (95% CI)": "-",
        "Specificity (95% CI)": "-",
        "PPV (95% CI)": "-",
        "Accuracy (95% CI)": fmt(get_ci(acc_scores))
    })

    df_final_table = pd.DataFrame(rows)[[
        "Class",
        "AUC (95% CI)",
        "Sensitivity (95% CI)",
        "Specificity (95% CI)",
        "PPV (95% CI)",
        "Accuracy (95% CI)"
    ]]

    df_final_table.to_csv(
        os.path.join(plots_dir, f"IBDwordscountBinary_{model_name}_repeatedstratifiedkfold_results.csv"), index=False)
    return df_final_table