"""
Preprocessing helpers: coercion, missing handling, dedup, ensure revenue/profit.
"""
from typing import Optional, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def coerce_types(df: pd.DataFrame, date_cols: Optional[List[str]] = None, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                except Exception:
                    logger.warning("Failed to coerce %s to datetime", c)
    if numeric_cols:
        for c in numeric_cols:
            if c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    logger.warning("Failed to coerce %s to numeric", c)
    return df


def handle_missing_values(df: pd.DataFrame, numeric_strategy: str = "median", fill_categorical: str = "missing") -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include="object").columns.tolist()
    for c in num:
        if df[c].isna().any():
            if numeric_strategy == "median":
                val = df[c].median()
            elif numeric_strategy == "mean":
                val = df[c].mean()
            elif numeric_strategy == "zero":
                val = 0
            else:
                val = df[c].median()
            df[c] = df[c].fillna(val)
    for c in cat:
        if df[c].isna().any():
            df[c] = df[c].fillna(fill_categorical)
    return df


def drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if subset:
        return df.drop_duplicates(subset=subset)
    else:
        if "order_id" in df.columns:
            return df.drop_duplicates(subset=["order_id"])
        return df.drop_duplicates()


def ensure_revenue_profit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "revenue" not in df.columns:
        if "quantity" in df.columns and "unit_price" in df.columns:
            try:
                df["revenue"] = pd.to_numeric(df["quantity"], errors="coerce") * pd.to_numeric(df["unit_price"], errors="coerce")
            except Exception:
                df["revenue"] = np.nan
    if "profit" not in df.columns:
        if "revenue" in df.columns and "cost" in df.columns:
            try:
                df["profit"] = pd.to_numeric(df["revenue"], errors="coerce") - pd.to_numeric(df["cost"], errors="coerce")
            except Exception:
                df["profit"] = np.nan
    if "revenue" in df.columns and df["revenue"].isna().any():
        df["revenue"] = df["revenue"].fillna(0)
    if "profit" in df.columns and df["profit"].isna().any():
        df["profit"] = df["profit"].fillna(0)
    return df