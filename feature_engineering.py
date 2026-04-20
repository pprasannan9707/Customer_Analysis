"""
feature_engineering.py
Derives all business-relevant features from raw marketing data.
"""

import pandas as pd
import numpy as np
from datetime import datetime

REFERENCE_YEAR = 2024  # treat dataset as circa 2024 for age calculation
REFERENCE_DATE = datetime(2024, 6, 1)

SPEND_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering on a copy of the dataframe.
    Returns enriched dataframe. Does NOT drop any original columns yet —
    the preprocessing pipeline handles selection.
    """
    df = df.copy()

    # --- Impute Income median ---
    median_income = df["Income"].median()
    df["Income"] = df["Income"].fillna(median_income)

    # --- Age ---
    df["Age"] = REFERENCE_YEAR - df["Year_Birth"]
    # Clip extreme outliers (anyone over 100 is likely a data error)
    df["Age"] = df["Age"].clip(18, 100)

    # --- Customer Tenure ---
    df["Customer_Tenure_Days"] = (REFERENCE_DATE - df["Dt_Customer"]).dt.days.fillna(0).clip(lower=0)
    df["Customer_Tenure_Years"] = (df["Customer_Tenure_Days"] / 365.25).round(2)

    # --- Spend aggregates ---
    df["Total_Spent"] = df[SPEND_COLS].sum(axis=1)
    df["Spend_Wines_Pct"] = (df["MntWines"] / (df["Total_Spent"] + 1)).round(4)
    df["Spend_Meat_Pct"] = (df["MntMeatProducts"] / (df["Total_Spent"] + 1)).round(4)

    # --- Purchase aggregates ---
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    df["Web_Purchase_Pct"] = (df["NumWebPurchases"] / (df["Total_Purchases"] + 1)).round(4)

    # --- Children ---
    df["Children"] = df["Kidhome"] + df["Teenhome"]

    # --- Campaign engagement ---
    df["AcceptedCampaignsTotal"] = df[CAMPAIGN_COLS].sum(axis=1)
    df["CampaignEngagementRate"] = (df["AcceptedCampaignsTotal"] / 5).round(4)

    # --- Income bands ---
    df["Income_Band"] = pd.cut(
        df["Income"],
        bins=[0, 30000, 60000, 90000, 120000, 1e9],
        labels=["<30k", "30-60k", "60-90k", "90-120k", "120k+"]
    ).astype(str)

    # --- Tenure band ---
    df["Tenure_Band"] = pd.cut(
        df["Customer_Tenure_Years"],
        bins=[0, 1, 2, 3, 4, 100],
        labels=["<1yr", "1-2yr", "2-3yr", "3-4yr", "4yr+"]
    ).astype(str)

    # --- Spend per purchase (efficiency) ---
    df["Avg_Spend_Per_Purchase"] = (df["Total_Spent"] / (df["Total_Purchases"] + 1)).round(2)

    # --- Recency bucket (lower recency = more recent) ---
    df["Recency_Bucket"] = pd.cut(
        df["Recency"],
        bins=[-1, 15, 30, 60, 100],
        labels=["Very Recent", "Recent", "Moderate", "Lapsed"]
    ).astype(str)

    return df


def get_feature_columns() -> list:
    """
    Returns the list of model features (excludes ID, target, date, constants,
    and any leakage columns).
    """
    return [
        "Age",
        "Customer_Tenure_Days",
        "Income",
        "Recency",
        "Total_Spent",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCampaignsTotal",
        "Children",
        "Total_Purchases",
        "Avg_Spend_Per_Purchase",
        "CampaignEngagementRate",
        "Web_Purchase_Pct",
        "Complain",
        "Education",
        "Marital_Status",
    ]


NUMERIC_FEATURES = [
    "Age", "Customer_Tenure_Days", "Income", "Recency",
    "Total_Spent", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "AcceptedCampaignsTotal",
    "Children", "Total_Purchases", "Avg_Spend_Per_Purchase",
    "CampaignEngagementRate", "Web_Purchase_Pct", "Complain",
]

CATEGORICAL_FEATURES = ["Education", "Marital_Status"]
