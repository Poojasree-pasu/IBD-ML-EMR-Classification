"""
Utility configuration file for IBD multi-class classification models.

Contains:
- Column definitions (e.g., drop_cols, LABEL_COL)
- Train/test split parameters
- Hyperparameter grids for model tuning
"""

drop_cols = ['PatientDurableKey']

LABEL_COL = "IBD"

test_size = 0.3
random_state = 42

# Grid search parameters
LR_grid = {
    'model__C': [0.01, 0.1, 1],
    'model__solver': ['lbfgs', 'saga'],
    'model__penalty': ['l2'],
    'model__max_iter': [1000, 1500],
}

lassoLRgrid_Params = {
    'model__C': [0.001, 0.01, 0.1, 1],
    'model__solver': ['saga'],
    'model__penalty': ['l1'],
    'model__max_iter': [2000, 3000],
    'model__n_jobs': [-1]
}

RFgrid_Params = {
    'model__n_estimators': [1000, 1200],
    'model__max_depth': [None, 4, 6]
}

RFrfecvgrid_Params = {
    'model__n_estimators': [1200],
    'model__max_depth': [None, 4, 6]
}

# XGgrid_Params = {
#     'model__n_estimators': [1000],
#     'model__max_depth': [6, 8],
#     'model__learning_rate': [0.01, 0.05, 0.1]
# }

XGgrid_Params = {
    'model__n_estimators': [1000,1200],
    'model__max_depth': [6, 8],
    'model__learning_rate': [0.01, 0.05, 0.1]
}
