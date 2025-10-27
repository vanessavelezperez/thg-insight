import numpy as np
import pandas as pd
from collections import defaultdict
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import randint

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.2

def preprocess(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # log-transform
    X_train_log = np.log1p(X_train_imp)
    X_test_log = np.log1p(X_test_imp)

    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)

    return X_train_scaled, X_test_scaled

def select_features_rf(X_train, y_train, X_test, max_features=100):
    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = vt.fit_transform(X_train)
    X_test_vt = vt.transform(X_test)

    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_vt, y_train)

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:max_features]

    return X_train_vt[:, top_idx], X_test_vt[:, top_idx], vt.get_support(indices=True)[top_idx]

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        from scipy.special import expit
        y_prob = expit(model.decision_function(X_test))

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

def random_forest_model(cv=None):
    param_dist = {
        "n_estimators": randint(200, 800),
        "max_depth": [None] + list(range(5, 51, 5)),
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
        "class_weight": [None, "balanced"]
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    return RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=30,
                              cv=cv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1)

def linear_svm_model(cv=None):
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"]}
    svm = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    return GridSearchCV(svm, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

def rbf_svm_model(cv=None):
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100],
                  "gamma": ["scale", "auto", 0.01, 0.1, 1, 10],
                  "class_weight": [None, "balanced"]}
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    return GridSearchCV(svm, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

def logistic_l2_model(cv=None):
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"], "solver": ["sag", "saga", "liblinear"]}
    logreg = LogisticRegression(penalty="l2", max_iter=4000, random_state=RANDOM_STATE)
    return GridSearchCV(logreg, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

def logistic_l1_model(cv=None):
    param_grid = {"C": [0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"], "solver": ["saga", "liblinear"]}
    lasso = LogisticRegression(penalty="l1", max_iter=4000, random_state=RANDOM_STATE)
    return GridSearchCV(lasso, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

def logistic_enet_model(cv=None):
    param_grid = {"C": [0.01, 0.1, 0.5, 1, 10, 100],
                  "l1_ratio": [0.1, 0.5, 0.9, 1.0],
                  "class_weight": [None, "balanced"],
                  "solver": ["saga"]}
    enet = LogisticRegression(penalty="elasticnet", max_iter=4000, random_state=RANDOM_STATE)
    return GridSearchCV(enet, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

def feature_table(feature_importance_by_model, feature_names):
    importance_data = []
    for feat in feature_names:
        row = {"feature": feat}
        
        total_selections = 0
        all_importances = []
        
        # Get stats for each model
        for model_name in ["RandomForest", "LogisticL1", "LogisticElasticNet"]:
            values = feature_importance_by_model[model_name].get(feat, [])
            
            if values:
                row[f"{model_name}_mean_importance"] = np.mean(values)
                row[f"{model_name}_selection_freq"] = len(values) / N_FOLDS
                total_selections += len(values)
                all_importances.extend(values)
            else:
                row[f"{model_name}_mean_importance"] = 0.0
                row[f"{model_name}_selection_freq"] = 0.0
        
        # Overall stats
        row["overall_mean_importance"] = np.mean(all_importances) if all_importances else 0.0
        row["overall_selection_freq"] = total_selections / (N_FOLDS * 3)
        row["selected_by_n_models"] = sum(1 for model in ["RandomForest", "LogisticL1", "LogisticElasticNet"] 
                                        if row[f"{model}_selection_freq"] > 0)
        
        importance_data.append(row)

    # Create and save enhanced feature importance table
    feature_importance_enhanced = pd.DataFrame(importance_data)
    feature_importance_enhanced.sort_values(["overall_selection_freq", "overall_mean_importance"], 
                                        ascending=False, inplace=True)
    #feature_importance_enhanced.to_csv("feature_importance_enhanced.csv", index=False)

    # 4. Keep your original format for backwards compatibility
    feature_importance_records = defaultdict(list)
    for model_name, model_features in feature_importance_by_model.items():
        for feat, values in model_features.items():
            feature_importance_records[feat].extend(values)

    # Your original feature_stability code still works
    feature_stability = pd.DataFrame([
        {"feature": feat,
        "mean_importance": np.mean(vals),
        "selection_freq": len(vals) / (N_FOLDS * 3)}
        for feat, vals in feature_importance_records.items()
    ])

    return feature_importance_enhanced

if __name__ == '__main__':


    pano = pd.read_excel("C:\\Users\\vvelezpe\\OneDrive - Imperial College London\\Projects\\PANORAMA\\PANORAMA Data Summary.xlsx", sheet_name='Breath Only', index_col=0)
    pano.dropna(inplace=True, how='all')

    meta = ['F', 'Group', 'System']
    pano = pano[pano['Group'] != 'DN']
    pano['Cat'] = pano["Group"].apply(lambda x: 0 if x == "HV" else 1)
    dataset = pano[pano.columns.difference(meta)]

    X = dataset.drop('Cat', axis=1)
    y = dataset['Cat']

    #feature_names = [f"compound_{i}" for i in range(X.shape[1])]
    feature_names = pano.index.to_list()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    X_train_prep, X_test_prep = preprocess(X_train, X_test)

    X_train_sel, X_test_sel, sel_idx = select_features_rf(X_train_prep, y_train, X_test_prep, max_features=100)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "RandomForest": random_forest_model(cv=cv),
        "LinearSVM": linear_svm_model(cv=cv),
        "RBFSVM": rbf_svm_model(cv=cv),
        "LogisticL2": logistic_l2_model(cv=cv),
        "LogisticL1": logistic_l1_model(cv=cv),
        "LogisticElasticNet": logistic_enet_model(cv=cv)
    }
    results = []
    
    """
    #feature_importance_records = defaultdict(list)
    feature_importance_by_model = {
    "RandomForest": defaultdict(list),
    "LogisticL1": defaultdict(list), 
    "LogisticElasticNet": defaultdict(list)
    }

    best_params = {
    "RandomForest": defaultdict(list),
    "LogisticL1": defaultdict(list), 
    "LogisticElasticNet": defaultdict(list)
    }

    for model_name in ["RandomForest", "LogisticL1", "LogisticElasticNet"]:
        #evaluate how consistently different features are selected/deemed important across multiple cross-validation folds for 3 different models

        # model loop
        print(f"Running stability analysis for {model_name}...")
        model = models[model_name]

        #for each model, cross-validation split on the training data. Each iteration creates different train/validation subsets
        for train_idx, val_idx in cv.split(X_train_prep, y_train):
            #data splitting
            X_tr, X_val = X_train_prep[train_idx], X_train_prep[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            model.fit(X_tr, y_tr)

            #gets the best model (for hyperparameter-tuned models that use GridSearchCV)
            best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
            best_params[model_name] = {
            "best_score": float(model.best_score_) if hasattr(model, 'best_score_') else None,
            "best_params": model.best_params_ if hasattr(model, 'best_params_') else {},
            "model_params": best_model.get_params(),
            "model_type": type(best_model).__name__
            }
            ## Feature importance extraction
            if isinstance(best_model, RandomForestClassifier):
                #extracts random forest feature importances and records any non-zero values
                importances = best_model.feature_importances_
                for feat, imp in zip(feature_names, importances):
                    if imp > 0:
                        feature_importance_by_model[model_name][feat].append(imp)

            elif isinstance(best_model, LogisticRegression) and best_model.penalty in ["l1", "elasticnet"]:
                #extracts absolute coefficient values and records any above a small threshold - wasn't negligible during regularisation
                coefs = np.abs(best_model.coef_).ravel()
                for feat, coef in zip(feature_names, coefs):
                    if coef > 1e-6:
                        feature_importance_by_model[model_name][feat].append(coef)

    """
    for name, model in models.items():
        res_all = evaluate_model(name + "_All", model, X_train_prep, y_train, X_test_prep, y_test)
        results.append(res_all)

        res_sel = evaluate_model(name + "_Selected", model, X_train_sel, y_train, X_test_sel, y_test)
        results.append(res_sel)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_df.csv", index=False)
    

    # Aggregate feature importance
    feature_stability = feature_table(feature_importance_by_model, feature_names)
    feature_stability.to_csv("feature_stability_BA.csv", index=False)

    with open("best_params_BA.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("✅ Done! Results saved to results_df.csv and feature_stability.csv")
