"""
Cleaning utilities (improved cleaner functions).
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        num = df.select_dtypes(include="number")
        df = df.copy()
        df[num.columns] = num.fillna(num.mean())
        # categorical fill with mode
        for c in df.select_dtypes(include="object").columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "missing")
        return df
    elif strategy == "median":
        num = df.select_dtypes(include="number")
        df = df.copy()
        df[num.columns] = num.fillna(num.median())
        for c in df.select_dtypes(include="object").columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "missing")
        return df
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Invalid strategy. Choose from 'drop','mean','median','mode'.")


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return df
    # handle zero-variance columns by excluding them
    stds = numeric_df.std()
    zero_var = stds[stds == 0].index.tolist()
    numeric_cols = [c for c in numeric_df.columns if c not in zero_var]
    if not numeric_cols:
        return df
    z_scores = abs(zscore(numeric_df[numeric_cols].fillna(0)))
    mask = (z_scores < threshold).all(axis=1)
    return df.loc[mask].reset_index(drop=True)


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return df
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df.loc[mask].reset_index(drop=True)