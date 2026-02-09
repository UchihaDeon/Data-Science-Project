"""
EDA helpers: summaries and common plots (saves static images to dashboards/plots).
"""
from typing import Optional, List, Dict, Any
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils import ensure_dir
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PLOTS_DIR = "dashboards/plots"
ensure_dir(PLOTS_DIR)


def summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
    }
    try:
        stats["numeric_summary"] = df.select_dtypes(include="number").describe().to_dict()
    except Exception:
        stats["numeric_summary"] = {}
    return stats


def univariate_numeric(df: pd.DataFrame, col: str, outpath: Optional[str] = None) -> Optional[str]:
    try:
        ser = pd.to_numeric(df[col], errors="coerce").dropna()
        if ser.empty:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(ser, ax=axes[0], kde=True)
        axes[0].set_title(f"Distribution: {col}")
        sns.boxplot(x=ser, ax=axes[1])
        axes[1].set_title(f"Boxplot: {col}")
        plt.tight_layout()
        if outpath:
            fig.savefig(outpath)
            plt.close(fig)
            return outpath
        plt.close(fig)
        return None
    except Exception:
        logger.exception("univariate_numeric failed for %s", col)
        return None


def univariate_categorical(df: pd.DataFrame, col: str, outpath: Optional[str] = None, top_n: int = 20) -> Optional[str]:
    try:
        counts = df[col].fillna("NULL").value_counts().nlargest(top_n)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.2 * len(counts))))
        sns.barplot(x=counts.values, y=counts.index, ax=ax)
        ax.set_title(f"Top {top_n} categories: {col}")
        plt.tight_layout()
        if outpath:
            fig.savefig(outpath)
            plt.close(fig)
            return outpath
        plt.close(fig)
        return None
    except Exception:
        logger.exception("univariate_categorical failed for %s", col)
        return None


def trend_by_date(df: pd.DataFrame, date_col: str, value_col: str, outpath: Optional[str] = None, freq: str = "D") -> Optional[str]:
    try:
        dfc = df.copy()
        dfc[date_col] = pd.to_datetime(dfc[date_col], errors="coerce")
        s = dfc.set_index(date_col)[value_col].resample(freq).sum().fillna(0)
        if s.empty:
            return None
        fig, ax = plt.subplots(figsize=(10, 4))
        s.plot(ax=ax)
        ax.set_title(f"{value_col} trend ({freq})")
        ax.set_ylabel(value_col)
        plt.tight_layout()
        if outpath:
            fig.savefig(outpath)
            plt.close(fig)
            return outpath
        plt.close(fig)
        return None
    except Exception:
        logger.exception("trend_by_date failed")
        return None