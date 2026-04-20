"""
preprocess.py
Builds and fits a scikit-learn ColumnTransformer preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from services.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_feature_columns

PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer (unfitted)."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return preprocessor


def fit_and_save_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on training data and persist it."""
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    preprocessor = build_preprocessor()
    preprocessor.fit(X[get_feature_columns()])
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    return preprocessor


def load_preprocessor() -> ColumnTransformer:
    """Load a persisted preprocessor."""
    return joblib.load(PREPROCESSOR_PATH)


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Extract human-readable feature names after one-hot encoding."""
    num_names = NUMERIC_FEATURES.copy()
    cat_names = list(
        preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names
