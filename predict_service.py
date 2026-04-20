"""
predict_service.py
Handles single-customer and batch predictions.
Returns probability, binary prediction, and top feature drivers.
"""

import numpy as np
import pandas as pd
from typing import Any

from services.model_training import load_model_bundle, load_feature_importance
from services.feature_engineering import (
    engineer_features, get_feature_columns, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from services.preprocess import get_feature_names_out

_bundle = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_model_bundle()
    return _bundle


def _prepare_single(customer: dict) -> pd.DataFrame:
    """Convert a raw customer dict → engineered DataFrame row."""
    df = pd.DataFrame([customer])
    # Ensure date column exists
    if "Dt_Customer" not in df.columns:
        df["Dt_Customer"] = pd.NaT
    else:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    df = engineer_features(df)
    return df[get_feature_columns()]


def predict_customer(customer: dict) -> dict:
    """
    Predict response probability for a single customer.
    Returns: probability, prediction, top_features.
    """
    bundle = _get_bundle()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    X = _prepare_single(customer)
    X_t = preprocessor.transform(X)
    prob = float(model.predict_proba(X_t)[0, 1])
    prediction = int(prob >= 0.5)

    top_features = _get_top_features(X_t, preprocessor, model, n=8)

    return {
        "probability": round(prob, 4),
        "prediction": prediction,
        "risk_label": _risk_label(prob),
        "top_features": top_features,
    }


def predict_batch(customers: list[dict]) -> list[dict]:
    """Predict for a list of customer dicts."""
    bundle = _get_bundle()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    rows = []
    for c in customers:
        df = pd.DataFrame([c])
        if "Dt_Customer" not in df.columns:
            df["Dt_Customer"] = pd.NaT
        else:
            df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
        df = engineer_features(df)
        rows.append(df[get_feature_columns()].iloc[0])

    X = pd.DataFrame(rows)
    X_t = preprocessor.transform(X)
    probs = model.predict_proba(X_t)[:, 1]

    results = []
    for i, (c, prob) in enumerate(zip(customers, probs)):
        results.append({
            "id": c.get("ID", i),
            "probability": round(float(prob), 4),
            "prediction": int(prob >= 0.5),
            "risk_label": _risk_label(prob),
        })
    return results


def _risk_label(prob: float) -> str:
    if prob >= 0.75:
        return "High Potential"
    elif prob >= 0.50:
        return "Moderate Potential"
    elif prob >= 0.25:
        return "Low Potential"
    else:
        return "Unlikely"


def _get_top_features(X_t: np.ndarray, preprocessor, model, n: int = 8) -> list[dict]:
    """
    Return top-N feature contributions for a single transformed row.
    Uses feature importance × feature value magnitude as a proxy.
    """
    try:
        feature_names = get_feature_names_out(preprocessor)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return []

        row = X_t[0]
        scores = importances * np.abs(row)
        total = scores.sum() + 1e-9
        top_idx = np.argsort(scores)[::-1][:n]

        return [
            {
                "feature": feature_names[i],
                "importance": round(float(importances[i]), 6),
                "value": round(float(row[i]), 4),
                "contribution_pct": round(float(scores[i] / total) * 100, 2),
                "direction": "positive" if row[i] > 0 else "negative",
            }
            for i in top_idx
        ]
    except Exception:
        return []
