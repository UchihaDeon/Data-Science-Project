"""
Profitability analysis: group by product and region, compute margins.
"""

from typing import Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def compute_profitability(df: pd.DataFrame, by: str = "product_name") -> pd.DataFrame:
    """
    Return revenue, cost, profit, margin aggregated by 'by' column.
    """
    try:
        agg = df.groupby(by).agg(revenue=("revenue", "sum"), cost=("cost", "sum"), profit=("profit", "sum"))
        agg["margin"] = agg["profit"] / agg["revenue"].replace(0, 1)
        agg = agg.sort_values("profit", ascending=False)
        logger.info("Computed profitability by %s", by)
        return agg.reset_index()
    except Exception:
        logger.exception("compute_profitability failed")
        raise


def top_bottom_products(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns top n and bottom n products by profit.
    """
    try:
        agg = compute_profitability(df, by="product_name")
        top = agg.head(n)
        bottom = agg.tail(n)
        return top, bottom
    except Exception:
        logger.exception("top_bottom_products failed")
        raise