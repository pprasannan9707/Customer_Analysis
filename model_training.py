"""
model_training.py
Trains Logistic Regression, Random Forest, and XGBoost.
Handles class imbalance, compares metrics, persists the best model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from services.data_loader import get_raw_data
from services.feature_engineering import engineer_features, get_feature_columns
from services.preprocess import fit_and_save_preprocessor, get_feature_names_out

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_IMP_PATH = os.path.join(MODEL_DIR, "feature_importance.json")

os.makedirs(MODEL_DIR, exist_ok=True)


def _compute_metrics(name: str, clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": [round(x, 4) for x in fpr.tolist()],
            "tpr": [round(x, 4) for x in tpr.tolist()],
        },
    }


def train_all_models():
    """
    Full training pipeline:
    1. Load + engineer features
    2. Fit preprocessor
    3. SMOTE for imbalance
    4. Train 3 models
    5. Compare and persist best
    """
    print("[Training] Loading data...")
    df = get_raw_data()
    df = engineer_features(df)

    features = get_feature_columns()
    X = df[features]
    y = df["Response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[Training] Fitting preprocessor...")
    preprocessor = fit_and_save_preprocessor(X_train)
    feature_names = get_feature_names_out(preprocessor)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print("[Training] Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_t, y_train)

    # --- Define models ---
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        pos = int(y_train_res.sum())
        neg = len(y_train_res) - pos
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=neg / pos,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )

    print("[Training] Training models...")
    all_metrics = []
    trained = {}

    for name, clf in models.items():
        print(f"  → {name}")
        clf.fit(X_train_res, y_train_res)
        metrics = _compute_metrics(name, clf, X_test_t, y_test)
        all_metrics.append(metrics)
        trained[name] = clf
        print(f"     AUC={metrics['roc_auc']}  F1={metrics['f1']}")

    # --- Pick best by ROC-AUC ---
    best_name = max(all_metrics, key=lambda m: m["roc_auc"])["model"]
    best_clf = trained[best_name]
    print(f"[Training] Best model: {best_name}")

    # --- Feature importance ---
    fi = _extract_feature_importance(best_clf, feature_names, best_name)

    # --- Persist ---
    joblib.dump({"model": best_clf, "name": best_name, "preprocessor": preprocessor}, BEST_MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"models": all_metrics, "best": best_name}, f, indent=2)
    with open(FEATURE_IMP_PATH, "w") as f:
        json.dump(fi, f, indent=2)

    print("[Training] All artifacts saved.")
    return all_metrics, best_name


def _extract_feature_importance(clf, feature_names: list, name: str) -> list:
    """Extract feature importances from tree-based models or coefficients."""
    try:
        if hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imps = np.abs(clf.coef_[0])
        else:
            return []
        pairs = sorted(
            zip(feature_names, imps.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return [{"feature": f, "importance": round(v, 6)} for f, v in pairs[:30]]
    except Exception:
        return []


def load_model_bundle() -> dict:
    """Load the best model bundle {model, name, preprocessor}."""
    if not os.path.exists(BEST_MODEL_PATH):
        train_all_models()
    return joblib.load(BEST_MODEL_PATH)


def load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        train_all_models()
    with open(METRICS_PATH) as f:
        return json.load(f)


def load_feature_importance() -> list:
    if not os.path.exists(FEATURE_IMP_PATH):
        train_all_models()
    with open(FEATURE_IMP_PATH) as f:
        return json.load(f)
