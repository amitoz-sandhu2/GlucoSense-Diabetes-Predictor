"""Train the diabetes prediction model on the Pima Indians Diabetes dataset.

Run:  python train.py
Data: tries data/diabetes.csv first; if missing, downloads it from public mirrors
      and saves it to data/diabetes.csv.
Output: model/model.joblib and model/metrics.json
"""
import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "diabetes.csv")
MODEL_PATH = os.path.join(BASE, "model", "model.joblib")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
# In this dataset a value of 0 here really means "not measured"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Accept files with or without a header row."""
    if "Outcome" not in df.columns:
        df = pd.read_csv(io.StringIO(df.to_csv(index=False)), header=None)
        df.columns = COLUMNS
    return df[COLUMNS]


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        print(f"Using local dataset: {DATA_PATH}")
        return _normalize(pd.read_csv(DATA_PATH))

    import requests
    for url in URLS:
        try:
            print(f"Downloading dataset from {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df = _normalize(df)
            if len(df) < 700:
                continue
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            print(f"Saved dataset to {DATA_PATH} ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"  failed: {e}")
    sys.exit(
        "\nCould not download the dataset.\n"
        "Download 'diabetes.csv' manually from Kaggle (search: Pima Indians Diabetes Database)\n"
        f"and place it at: {DATA_PATH}\nThen run: python train.py"
    )


# Plausible clinical ranges. Anything outside is a recording error -> treated as missing.
VALID_RANGE = {
    "Glucose": (40, 300), "BloodPressure": (40, 140), "SkinThickness": (5, 60),
    "Insulin": (10, 600), "BMI": (12, 60),
}


def clean_data(df: pd.DataFrame):
    """Clean the raw dataset and return (clean_df, report)."""
    report = {"rows_raw": int(len(df))}
    df = df.drop_duplicates().reset_index(drop=True)
    report["rows_after_duplicates_removed"] = int(len(df))

    missing, outliers = {}, {}
    for col, (lo, hi) in VALID_RANGE.items():
        zeros = int((df[col] == 0).sum())
        bad = int(((df[col] != 0) & ((df[col] < lo) | (df[col] > hi))).sum())
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        missing[col], outliers[col] = zeros, bad
    report["zeros_treated_as_missing"] = missing
    report["out_of_range_set_to_missing"] = outliers
    report["missing_after_cleaning"] = {c: int(df[c].isna().sum()) for c in VALID_RANGE}
    return df, report


def make_pipeline(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def main():
    raw = load_data()
    df, clean_report = clean_data(raw)
    print("\nCleaning report:")
    print(json.dumps(clean_report, indent=2))
    df.to_csv(os.path.join(BASE, "data", "diabetes_cleaned.csv"), index=False)
    X, y = df.drop(columns="Outcome"), df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": make_pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42)),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)),
    }

    grids = {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
        "Random Forest": {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [3, 8]},
        "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
                              "clf__learning_rate": [0.03, 0.05]},
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results, tuned = {}, {}
    print("\nHyperparameter tuning, 5-fold CV (ROC-AUC):")
    for name, pipe in candidates.items():
        gs = GridSearchCV(pipe, grids[name], cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        cv_results[name] = float(gs.best_score_)
        tuned[name] = gs.best_estimator_
        print(f"  {name:20s} {gs.best_score_:.3f}  best params: {gs.best_params_}")

    best_name = max(cv_results, key=cv_results.get)
    best = tuned[best_name]
    print(f"\nBest model: {best_name}")

    # More reliable estimate than one small test split: 5-fold CV repeated 5 times on all data
    rcv = cross_validate(best, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1),
                         scoring=["accuracy", "recall", "precision", "roc_auc"])
    repeated_cv = {k.replace("test_", ""): float(v.mean()) for k, v in rcv.items() if k.startswith("test_")}
    print("\nRepeated 5x5 CV on full data:", {k: round(v, 3) for k, v in repeated_cv.items()})

    prob = best.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "best_model": best_name,
        "dataset_rows": int(len(df)),
        "cleaning_report": clean_report,
        "repeated_cv": repeated_cv,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_roc_auc": cv_results,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "features": list(X.columns),
    }
    clf = best.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = np.abs(clf.coef_[0]) / np.abs(clf.coef_[0]).sum()
    metrics["feature_importance"] = {c: float(v) for c, v in zip(X.columns, imp)}

    print(f"\nTest accuracy : {metrics['accuracy']:.3f}")
    print(f"Test precision: {metrics['precision']:.3f}")
    print(f"Test recall   : {metrics['recall']:.3f}")
    print(f"Test F1       : {metrics['f1']:.3f}")
    print(f"Test ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
