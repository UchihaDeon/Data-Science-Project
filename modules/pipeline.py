"""
Phase 1 pipeline orchestrator (filesystem-only).
"""
from typing import Dict, Any, Optional
import os
import logging

import pandas as pd

from modules.mapping_dynamic import suggest_candidates
from modules.cleaning import handle_missing_values, remove_duplicates
from modules.preprocessing import coerce_types, handle_missing_values as pre_handle_missing, drop_duplicates, ensure_revenue_profit
from modules.eda import summary_stats, univariate_numeric, univariate_categorical, trend_by_date
from modules.features import create_date_features, create_aov_margin, create_lag_features, create_rolling_features
from modules.reporting import create_html_report
from modules.utils import ensure_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PLOTS_DIR = "dashboards/plots"
REPORTS_DIR = "dashboards/reports"
ensure_dir(PLOTS_DIR)
ensure_dir(REPORTS_DIR)


def run_pipeline(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
    dataset_path: Optional[str] = None,
    do_eda: bool = True,
    do_features: bool = True,
    auto_fix: bool = True,
    forecast_horizon: int = 14,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"diagnostics": {}, "kpis": {}, "plots": {}, "features_info": {}}
    try:
        if mapping:
            inv = {v: k for k, v in mapping.items() if v}
            df = df.rename(columns=inv)
        # Validation-lite: schema suggestion available
        try:
            candidates = suggest_candidates(df)
            results["diagnostics"]["mapping_suggestions"] = {k: [c["column"] for c in v[:3]] for k, v in candidates.items()}
        except Exception:
            results["diagnostics"]["mapping_suggestions"] = {}

        # Basic coercion: detect date-like cols and numeric cols
        date_cols = [c for c in df.columns if "date" in c.lower()]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df = coerce_types(df, date_cols=date_cols, numeric_cols=numeric_cols)

        # Missing handling and duplicates
        df = pre_handle_missing(df, numeric_strategy="median")
        df = drop_duplicates(df)

        # Ensure revenue/profit if possible
        df = ensure_revenue_profit(df)

        # KPIs
        total_revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
        total_profit = float(df["profit"].sum()) if "profit" in df.columns else 0.0
        orders = int(df["order_id"].nunique()) if "order_id" in df.columns else int(len(df))
        avg_order_value = total_revenue / orders if orders > 0 else 0.0
        results["kpis"] = {
            "total_revenue": total_revenue,
            "total_profit": total_profit,
            "orders": orders,
            "avg_order_value": avg_order_value,
        }

        # EDA
        if do_eda:
            results["summary_stats"] = summary_stats(df)
            numeric_candidates = [c for c in df.select_dtypes(include="number").columns if c not in ("order_id",)]
            for col in numeric_candidates[:6]:
                outpath = os.path.join(PLOTS_DIR, f"uni_num_{col}.png")
                p = univariate_numeric(df, col, outpath=outpath)
                if p:
                    results["plots"].setdefault("univariate_numeric", []).append(p)
            cat_candidates = [c for c in df.select_dtypes(include="object").columns]
            for col in cat_candidates[:6]:
                outpath = os.path.join(PLOTS_DIR, f"uni_cat_{col}.png")
                p = univariate_categorical(df, col, outpath=outpath)
                if p:
                    results["plots"].setdefault("univariate_categorical", []).append(p)
            if "order_date" in df.columns and "revenue" in df.columns:
                trend_path = os.path.join(PLOTS_DIR, "revenue_trend.png")
                p = trend_by_date(df, date_col="order_date", value_col="revenue", outpath=trend_path, freq="D")
                if p:
                    results["plots"]["trend"] = p

        # Features
        if do_features:
            df = create_date_features(df, date_col="order_date")
            df = create_aov_margin(df)
            df = create_lag_features(df, group_col="product_name", date_col="order_date", value_col="quantity", lags=[7, 14])
            df = create_rolling_features(df, group_col="product_name", date_col="order_date", value_col="quantity", windows=[7, 14])
            results["features_info"]["created"] = ["date_parts", "aov_margin", "lag_7_14", "rolling_7_14"]

        # Report
        try:
            report_path = os.path.join(REPORTS_DIR, "phase1_report.html")
            create_html_report(
                outpath=report_path,
                kpis=results.get("kpis", {}),
                plots=[p for p in sum([v if isinstance(v, list) else [v] for v in results.get("plots", {}).values()], []) if p],
                top_products=[],
                bottom_products=[],
                recommendations={},
                forecast_summary={},
            )
            results["report_path"] = report_path
        except Exception:
            results["report_path"] = None

        results["cleaned_df"] = df
    except Exception as e:
        logger.exception("run_pipeline failed")
        results["error"] = str(e)
    return results