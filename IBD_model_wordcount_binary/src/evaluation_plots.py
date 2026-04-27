"""
Model Evaluation and Visualization for IBD Classification

This module provides functions for evaluating multi-class classification models
and generating plots.

Features:
- Confusion matrix
- ROC curve generation (per-class and macro-average)
- Model comparison using ROC curves
- Computation of sensitivity, specificity, and precision
- SHAP-based feature importance visualization
"""

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import shap


# Confusion matrix
def display_confusion_matrix(model, y_test, y_pred, le, plots_dir):
    cm = confusion_matrix(y_test, y_pred)
    # print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cbar=False
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(model)
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(
        os.path.join(plots_dir, f"IBDwordscountBinary_{model}_confusionmatrix.pdf"), bbox_inches="tight", dpi=300)
    # plt.close()
    plt.show()
    return cm

def precision_per_class(cm, le):
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP

    precision = np.divide(
        TP,
        TP + FP,
        out=np.zeros_like(TP, dtype=float),
        where=(TP + FP) != 0
    )

    precision_df = pd.DataFrame({
        'Class': le.classes_,
        'Precision': precision
    })

    print("\nPer-class Precision:\n", precision_df.round(2))

def safe_divide(a, b):
    return a / b if b != 0 else 0.0

def sensitivity_specificity_per_class(model, cm, le, plots_dir):
    classes = le.classes_
    n_classes = len(classes)

    # Per-class Sensitivity & Specificity
    metrics = []

    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = safe_divide(TP, TP + FN)
        specificity = safe_divide(TN, TN + FP)

        metrics.append({
            "Class": cls,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(plots_dir, f"IBDwordscountBinary_{model}_sensitivityspecificityresults.csv"), index=False)
    print("\nPer-class Sensitivity & Specificity:\n")
    print(metrics_df.round(4))

def sensitivity_specificity_cduc_noibd(cm, le):
    # IBD(CD+UC) and No IBD
    classes = le.classes_
    neg_idx = np.where(classes == "No IBD")[0][0]
    pos_idx = [i for i in range(len(classes)) if i != neg_idx]

    TP = cm[np.ix_(pos_idx, pos_idx)].sum()
    FP = cm[neg_idx, pos_idx].sum()
    FN = cm[pos_idx, neg_idx].sum()
    TN = cm[neg_idx, neg_idx]

    print("\nIBD vs No IBD:")
    print(f"Sensitivity: {safe_divide(TP, TP + FN):.2f}")
    print(f"Specificity: {safe_divide(TN, TN + FP):.2f}")

def display_roc_curves(model, y_test, y_prob, plots_dir, le):
    # ROC CURVES
    classes_idx = np.arange(len(le.classes_))
    y_test_bin = label_binarize(y_test, classes=classes_idx)

    fpr, tpr, roc_auc = {}, {}, {}

    # PER-CLASS ROC
    for i in range(len(classes_idx)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # MACRO-AVERAGE ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes_idx))]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes_idx)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes_idx)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # COMBINED PLOT
    plt.figure(figsize=(6, 5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    for i in range(len(classes_idx)):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f'{le.classes_[i]} (AUC={roc_auc[i]:.2f})'
        )

    plt.plot(
        all_fpr,
        mean_tpr,
        linestyle='--',
        linewidth=2,
        label=f'Macro-average (AUC={roc_auc_macro:.2f})'
    )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve (All Classes)")
    plt.legend(loc="lower right")
    plt.grid(False)

    plt.savefig(
        os.path.join(plots_dir, f"IBDwordscountBinary_{model}_ROC_allclasses.pdf"),
        bbox_inches="tight",
        dpi=300
    )

    plt.show()

    # INDIVIDUAL CLASS PLOTS

    for i in range(len(classes_idx)):
        plt.figure(figsize=(5, 4), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        plt.plot(
            fpr[i],
            tpr[i],
            linewidth=2,
            label=f'{le.classes_[i]} (AUC={roc_auc[i]:.2f})'
        )

        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {le.classes_[i]}')
        plt.legend(loc="lower right")
        plt.grid(False)

        plt.savefig(
            os.path.join(plots_dir, f"IBDwordscountBinary_{model}_ROC_{le.classes_[i]}.pdf"),
            bbox_inches="tight",
            dpi=300
        )

        plt.show()

    # MACRO-ONLY PLOT

    plt.figure(figsize=(5, 4), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.plot(
        all_fpr,
        mean_tpr,
        linestyle='--',
        linewidth=2,
        color='black',
        label=f'Macro-average (AUC={roc_auc_macro:.2f})'
    )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-average ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(False)

    plt.savefig(
        os.path.join(plots_dir, f"IBDwordscountBinary_{model}_ROC_macro_only.pdf"),
        bbox_inches="tight",
        dpi=300
    )

    plt.show()


def shap_plot(model, final_model, X_train_selected, clean_features, le, plots_dir):
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=clean_features)

    if model == "LogisticRegression_rfecv":
        explainer = shap.LinearExplainer(final_model, X_train_selected_df)
    else:
        explainer = shap.TreeExplainer(final_model)

    shap_values = explainer.shap_values(X_train_selected_df)
    class_labels = le.classes_
    # print(f"Number of classes: {len(class_labels)}")
    if isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    # PLOTS
    for i, class_name in enumerate(class_labels):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_list[i],
            X_train_selected_df,
            show=False)
        safe_name = class_name.replace(" ", "_")
        plt.savefig(
            os.path.join(plots_dir, f"{model}_shapsummary_{safe_name}.pdf"),
            bbox_inches="tight",
            dpi=300)
        plt.close()

    # GLOBAL FEATURE IMPORTANCE
    shap_mean_per_class = {}
    for i, class_name in enumerate(class_labels):
        shap_mean_per_class[class_name] = np.mean(
            np.abs(shap_list[i]), axis=0)

    shap_df = pd.DataFrame(shap_mean_per_class, index=clean_features)
    shap_df.reset_index(inplace=True)
    shap_df.rename(columns={'index': 'Feature'}, inplace=True)
    # print("\n=== SHAP FEATURE IMPORTANCE ===")
    # print(shap_df.head(10))
    shap_df.to_csv(os.path.join(plots_dir, f"IBDwordscountBinary_{model}_shapimportanceresults.csv"), index=False)

def get_base_model_name(model_name):
    if "Logistic" in model_name or model_name.startswith("LR"):
        return "Logistic Regression"
    elif "Random Forest" in model_name or model_name.startswith("RF"):
        return "Random Forest"
    elif "XGB" in model_name or "XGBoost" in model_name:
        return "XGBoost"
    return model_name

def plot_roc_comparison(roc_data_dict, le, plots_dir, prefix=""):
    model_colors = {
        "XGBoost": "#1f77b4",
        "Random Forest": "orange",
        "Logistic Regression": "green"
    }

    os.makedirs(plots_dir, exist_ok=True)

    # ================= MACRO ROC =================
    # Sort models by macro AUC (descending)
    sorted_models = sorted(
        roc_data_dict.items(),
        key=lambda x: x[1]["macro_auc"],
        reverse=True)
    fig = plt.figure(figsize=(6,5))
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')

    for model_name, roc_data in sorted_models:
        base_name = get_base_model_name(model_name),
        plt.plot(
            roc_data["macro_fpr"],
            roc_data["macro_tpr"],
            # color=model_colors.get(model_name, None),
            color = model_colors.get(base_name, None),
            linewidth=2,
            label=f'{model_name} (Macro AUROC={roc_data["macro_auc"]:.2f})'
        )

    plt.plot([0,1],[0,1],'k--', linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro Average")
    plt.legend(
        loc="lower right",
        frameon=False,
        facecolor='white',
        edgecolor='black'
    )
    plt.grid(False)

    plt.savefig(
        os.path.join(plots_dir, f"IBDwordscountBinary_{prefix}allmodels_ROC_macro_comparison.pdf"),
        bbox_inches="tight",
        dpi=300,
        facecolor='white'
    )
    plt.close()

    # ================= PER CLASS =================
    for i, class_name in enumerate(le.classes_):

        # Sort models per class AUC
        sorted_models_class = sorted(
            roc_data_dict.items(),
            key=lambda x: x[1]["roc_auc"][i],
            reverse=True
        )

        fig = plt.figure(figsize=(6,5))
        fig.patch.set_facecolor('white')
        ax = plt.gca()
        ax.set_facecolor('white')

        for model_name, roc_data in sorted_models_class:
            base_name = get_base_model_name(model_name),
            plt.plot(
                roc_data["fpr"][i],
                roc_data["tpr"][i],
                # color=model_colors.get(model_name, None),
                color=model_colors.get(base_name, None),
                linewidth=2,
                label=f'{model_name} (AUROC={roc_data["roc_auc"][i]:.2f})'
            )

        plt.plot([0,1],[0,1],'k--', linewidth=1)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {class_name}")
        plt.legend(
            loc="lower right",
            frameon=False,
            facecolor='white',
            edgecolor='black'
        )
        plt.grid(False)

        plt.savefig(
            os.path.join(plots_dir, f"IBDwordscountBinary_{prefix}allmodels_ROC_compare_{class_name}.pdf"),
            bbox_inches="tight",
            dpi=300,
            facecolor='white'
        )
        plt.close()


def get_roc_data(y_test, y_prob, le):
    classes_idx = np.arange(len(le.classes_))
    y_test_bin = label_binarize(y_test, classes=classes_idx)

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(classes_idx)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes_idx))]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes_idx)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes_idx)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "macro_fpr": all_fpr,
        "macro_tpr": mean_tpr,
        "macro_auc": roc_auc_macro
    }