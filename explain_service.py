"""
explain_service.py
Segment-level analytics: response rates, spend breakdowns, tenure analysis.
Provides data for the Segment Explorer and Overview pages.
"""

import pandas as pd
import numpy as np
from functools import lru_cache

from services.data_loader import get_raw_data
from services.feature_engineering import engineer_features, SPEND_COLS, CAMPAIGN_COLS


def _load_enriched() -> pd.DataFrame:
    df = get_raw_data()
    return engineer_features(df)


def get_overview_summary() -> dict:
    df = _load_enriched()
    total = len(df)
    response_rate = float(df["Response"].mean())
    avg_income = float(df["Income"].median())
    avg_spend = float(df["Total_Spent"].mean())
    avg_recency = float(df["Recency"].mean())
    responders = int(df["Response"].sum())
    non_responders = int(total - responders)
    high_income = int((df["Income"] > 70000).sum())

    spend_by_cat = {
        col.replace("Mnt", ""): round(float(df[col].mean()), 2)
        for col in SPEND_COLS
    }

    campaign_accept = {
        col: round(float(df[col].mean() * 100), 2)
        for col in CAMPAIGN_COLS
    }

    age_distribution = _bucket_distribution(df, "Age", [18, 30, 40, 50, 60, 70, 100],
                                             ["18-30", "30-40", "40-50", "50-60", "60-70", "70+"])
    tenure_dist = df["Tenure_Band"].value_counts().to_dict()
    income_dist = df["Income_Band"].value_counts().to_dict()

    education_dist = df.groupby("Education")["Response"].agg(
        count="count", response_rate="mean"
    ).reset_index().rename(columns={"response_rate": "rate"})
    education_dist["rate"] = education_dist["rate"].round(4)
    education_dist = education_dist.to_dict("records")

    marital_dist = df.groupby("Marital_Status")["Response"].agg(
        count="count", response_rate="mean"
    ).reset_index().rename(columns={"response_rate": "rate"})
    marital_dist["rate"] = marital_dist["rate"].round(4)
    marital_dist = marital_dist.to_dict("records")

    return {
        "total_customers": total,
        "response_rate": round(response_rate, 4),
        "responders": responders,
        "non_responders": non_responders,
        "avg_income": round(avg_income, 2),
        "avg_total_spent": round(avg_spend, 2),
        "avg_recency": round(avg_recency, 2),
        "high_income_count": high_income,
        "spend_by_category": spend_by_cat,
        "campaign_acceptance_pct": campaign_accept,
        "age_distribution": age_distribution,
        "tenure_distribution": tenure_dist,
        "income_band_distribution": income_dist,
        "education_response": education_dist,
        "marital_response": marital_dist,
    }


def get_segment_stats(
    education: str = None,
    marital_status: str = None,
    income_band: str = None,
    tenure_band: str = None,
    children: int = None,
) -> dict:
    df = _load_enriched()

    if education:
        df = df[df["Education"] == education]
    if marital_status:
        df = df[df["Marital_Status"] == marital_status]
    if income_band:
        df = df[df["Income_Band"] == income_band]
    if tenure_band:
        df = df[df["Tenure_Band"] == tenure_band]
    if children is not None:
        df = df[df["Children"] == children]

    if df.empty:
        return {"error": "No customers match the selected filters", "count": 0}

    segment_label = " | ".join(filter(None, [education, marital_status, income_band, tenure_band,
                                              f"{children} child(ren)" if children is not None else None]))

    return {
        "segment_label": segment_label or "All Customers",
        "count": len(df),
        "response_rate": round(float(df["Response"].mean()), 4),
        "avg_income": round(float(df["Income"].mean()), 2),
        "avg_spend": round(float(df["Total_Spent"].mean()), 2),
        "avg_recency": round(float(df["Recency"].mean()), 2),
        "avg_web_purchases": round(float(df["NumWebPurchases"].mean()), 2),
        "avg_store_purchases": round(float(df["NumStorePurchases"].mean()), 2),
        "avg_catalog_purchases": round(float(df["NumCatalogPurchases"].mean()), 2),
        "campaign_rate": round(float(df["CampaignEngagementRate"].mean()), 4),
        "avg_children": round(float(df["Children"].mean()), 2),
        "spend_breakdown": {
            col.replace("Mnt", ""): round(float(df[col].mean()), 2)
            for col in SPEND_COLS
        },
        "channel_breakdown": {
            "Web": round(float(df["NumWebPurchases"].mean()), 2),
            "Store": round(float(df["NumStorePurchases"].mean()), 2),
            "Catalog": round(float(df["NumCatalogPurchases"].mean()), 2),
            "Deals": round(float(df["NumDealsPurchases"].mean()), 2),
        },
    }


def get_all_customers(limit: int = 500) -> list:
    df = _load_enriched()
    cols = [
        "ID", "Age", "Education", "Marital_Status", "Income",
        "Total_Spent", "Recency", "Children", "AcceptedCampaignsTotal",
        "Total_Purchases", "Customer_Tenure_Years", "Income_Band",
        "Response", "NumWebPurchases", "NumStorePurchases",
        "NumCatalogPurchases", "NumWebVisitsMonth", "Complain",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].head(limit).fillna(0).to_dict("records")


def _bucket_distribution(df, col, bins, labels):
    df = df.copy()
    df["_bucket"] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df["_bucket"].value_counts().sort_index().to_dict()
