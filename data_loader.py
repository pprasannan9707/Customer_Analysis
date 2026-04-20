"""
data_loader.py
Loads and caches the marketing campaign dataset (tab-separated).
"""

import os
import pandas as pd
from functools import lru_cache

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "marketing_campaign.csv")


@lru_cache(maxsize=1)
def load_raw_data() -> pd.DataFrame:
    """Load the raw tab-separated marketing CSV. Cached after first load."""
    df = pd.read_csv(DATA_PATH, sep="\t")
    # Parse the join date
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    return df


def get_raw_data() -> pd.DataFrame:
    """Public accessor — returns a copy so callers can't mutate the cache."""
    return load_raw_data().copy()
