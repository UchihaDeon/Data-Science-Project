"""
Customer segmentation using KMeans clustering on total spend and frequency.

This module is defensive: if the input DataFrame does not have a 'revenue'
column we try to compute it from 'quantity' * 'unit_price' (or call the
preprocessing.create_profit helper if available). This makes segmentation
usable on raw transactional data without requiring prior preprocessing.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try to import create_profit from preprocessing for a fallback method
try:
    from modules.preprocessing import create_profit  # type: ignore
except Exception:
    create_profit = None  # type: ignore


def compute_rfm(df: pd.DataFrame, customer_col: str = "customer_id", date_col: str = "order_date", amount_col: str = "revenue") -> pd.DataFrame:
    """
    Compute simple RFM-like features: frequency (count of orders) and total_spend (sum of revenue).
    If the amount_col (default 'revenue') is not present, attempt to compute it from
    quantity * unit_price or by calling preprocessing.create_profit if available.
    """
    try:
        df = df.copy()

        # Ensure date_col is datetime if present
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # If revenue/amount_col missing try to create it
        if amount_col not in df.columns:
            logger.warning("Amount column '%s' not found. Attempting to compute revenue.", amount_col)
            # Prefer using provided create_profit helper if available
            if create_profit is not None:
                try:
                    df = create_profit(df)
                except Exception as e:
                    logger.warning("create_profit failed: %s", e)

            # If still missing, try quantity * unit_price
            if "revenue" not in df.columns and "quantity" in df.columns and "unit_price" in df.columns:
                df["revenue"] = df["quantity"].astype(float) * df["unit_price"].astype(float)
                logger.info("Computed revenue as quantity * unit_price")
            elif "revenue" not in df.columns:
                raise KeyError("No revenue column and unable to compute revenue (need 'quantity' and 'unit_price')")

        # Aggregate: frequency = count of orders, total_spend = sum of revenue
        agg = df.groupby(customer_col).agg(frequency=("order_id", "count"), total_spend=(amount_col, "sum"))
        agg = agg.reset_index()
        logger.info("Computed RFM for %d customers", agg.shape[0])
        return agg
    except Exception:
        logger.exception("compute_rfm failed")
        raise


def kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans]:
    """
    Runs KMeans on frequency & total_spend and returns customer segments with labels.

    Returns:
        (rfm_df_with_segments, trained_kmeans_model)
    """
    try:
        rfm = compute_rfm(df)
        # Ensure numeric types and handle missing/inf
        features = rfm[["frequency", "total_spend"]].fillna(0).replace([np.inf, -np.inf], 0.0)
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        # If customer population < n_clusters, reduce n_clusters
        n_clusters_adj = min(n_clusters, max(1, len(rfm)))
        if n_clusters_adj != n_clusters:
            logger.warning("Reducing n_clusters from %d to %d because only %d customers present", n_clusters, n_clusters_adj, len(rfm))

        kmeans = KMeans(n_clusters=n_clusters_adj, random_state=random_state)
        labels = kmeans.fit_predict(X)
        rfm["segment"] = labels

        # Map segments to 'high/medium/low' by descending mean total_spend
        seg_order = rfm.groupby("segment")["total_spend"].mean().sort_values(ascending=False).index.tolist()
        # Build label names up to number of clusters
        label_names = ["high-value", "medium-value", "low-value"]
        mapping = {seg_order[i]: label_names[i] if i < len(label_names) else f"segment_{i}" for i in range(len(seg_order))}
        rfm["segment_label"] = rfm["segment"].map(mapping)

        logger.info("KMeans segmentation produced %d clusters", n_clusters_adj)
        return rfm, kmeans
    except Exception:
        logger.exception("kmeans_segmentation failed")
        raise