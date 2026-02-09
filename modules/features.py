"""
Feature engineering helpers: date parts, AOV, lags and rolling features.
"""
from typing import List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_date_features(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["order_year"] = df[date_col].dt.year
        df["order_month"] = df[date_col].dt.month
        df["order_day"] = df[date_col].dt.day
        df["order_dow"] = df[date_col].dt.dayofweek
        try:
            df["order_week"] = df[date_col].dt.isocalendar().week.astype(int)
        except Exception:
            df["order_week"] = df[date_col].dt.week
    return df


def create_aov_margin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "revenue" in df.columns and "order_id" in df.columns:
        try:
            aov = df.groupby("order_id")["revenue"].sum().rename("order_revenue")
            df = df.merge(aov, left_on="order_id", right_index=True, how="left")
            df["aov"] = df["order_revenue"]
        except Exception:
            logger.exception("Failed to compute AOV")
    if "revenue" in df.columns and "cost" in df.columns:
        try:
            df["margin"] = pd.to_numeric(df["revenue"], errors="coerce") - pd.to_numeric(df["cost"], errors="coerce")
            df["margin_pct"] = df["margin"] / df["revenue"].replace({0: np.nan})
        except Exception:
            logger.exception("Failed to compute margin")
    return df


def create_lag_features(df: pd.DataFrame, group_col: str = "product_name", date_col: str = "order_date", value_col: str = "quantity", lags: List[int] = [7, 14]) -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns or group_col not in df.columns or value_col not in df.columns:
        return df
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        agg = df.groupby([group_col, date_col])[value_col].sum().reset_index().sort_values([group_col, date_col])
        for lag in lags:
            agg[f"lag_{lag}"] = agg.groupby(group_col)[value_col].shift(lag)
        cols_to_merge = [group_col, date_col] + [f"lag_{l}" for l in lags]
        df = df.merge(agg[cols_to_merge], on=[group_col, date_col], how="left")
    except Exception:
        logger.exception("create_lag_features failed")
    return df


def create_rolling_features(df: pd.DataFrame, group_col: str = "product_name", date_col: str = "order_date", value_col: str = "quantity", windows: List[int] = [7, 14]) -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns or group_col not in df.columns or value_col not in df.columns:
        return df
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        agg = df.groupby([group_col, date_col])[value_col].sum().reset_index().sort_values([group_col, date_col])
        for w in windows:
            rolling_mean = agg.groupby(group_col)[value_col].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            rolling_std = agg.groupby(group_col)[value_col].rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
            agg[f"rolling_mean_{w}"] = rolling_mean.values
            agg[f"rolling_std_{w}"] = rolling_std.values
        cols_to_merge = [group_col, date_col] + [c for c in agg.columns if c.startswith("rolling_")]
        df = df.merge(agg[cols_to_merge], on=[group_col, date_col], how="left")
    except Exception:
        logger.exception("create_rolling_features failed")
    return df