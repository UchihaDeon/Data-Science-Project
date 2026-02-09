"""
modules/one_click.py

Orchestrates a one-click analysis for retail/e-commerce datasets.
This module is kept small and returns a dictionary of results that the
Streamlit UI can display, and a report path.

Function:
- run_one_click_analysis(data_path, parse_dates=None, forecast_horizon=14, auto_apply=True)
"""
from typing import Dict, Any, List
import os
import logging

import pandas as pd

from modules.io import load_csv
from modules.validation import validate_dataframe, apply_suggested_fixes, get_default_schema
from modules.preprocessing import drop_duplicates, handle_missing_values, create_profit
from modules.eda import plot_sales_trend, plot_product_comparison, plot_region_performance
from modules.profitability import compute_profitability, top_bottom_products
from modules.segmentation import kmeans_segmentation
from modules.recommendations import generate_recommendations
from modules.forecasting import sarima_forecast
from modules.reporting import create_html_report
from modules.utils import ensure_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PLOTS_DIR = "dashboards/plots"
REPORTS_DIR = "dashboards/reports"
ensure_dir(PLOTS_DIR)
ensure_dir(REPORTS_DIR)


def _compute_simple_forecast_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic measure: for each product_name compute average quantity in the last 7 days
    vs the previous 7 days and compute percent change. Returns DataFrame with columns:
    product_id (or product_name), forecast_change_pct
    """
    try:
        df = df.copy()
        if "order_date" not in df.columns or "quantity" not in df.columns:
            # Can't compute; return zero changes
            products = df.get("product_name", pd.Series(dtype=str)).unique().tolist()
            return pd.DataFrame({"product_id": products, "forecast_change_pct": [0.0] * len(products)})
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        last_date = df["order_date"].max()
        if pd.isna(last_date):
            return pd.DataFrame({"product_id": [], "forecast_change_pct": []})
        recent_mask = df["order_date"] > (last_date - pd.Timedelta(days=7))
        prev_mask = (df["order_date"] <= (last_date - pd.Timedelta(days=7))) & (df["order_date"] > (last_date - pd.Timedelta(days=14)))
        recent = df[recent_mask].groupby("product_name")["quantity"].mean()
        prev = df[prev_mask].groupby("product_name")["quantity"].mean()
        products = sorted(list(set(df["product_name"].dropna().unique().tolist())))
        rows = []
        for p in products:
            r = recent.get(p, 0.0)
            pv = prev.get(p, 0.0)
            if pv == 0 and r == 0:
                pct = 0.0
            elif pv == 0:
                pct = 1.0  # arbitrarily treat as 100% increase
            else:
                pct = (r - pv) / pv
            rows.append({"product_id": p, "forecast_change_pct": float(pct)})
        return pd.DataFrame(rows)
    except Exception:
        logger.exception("Failed computing simple forecast change")
        return pd.DataFrame({"product_id": [], "forecast_change_pct": []})


def run_one_click_analysis(data_path: str, parse_dates: list | None = None, forecast_horizon: int = 14, auto_apply: bool = True) -> Dict[str, Any]:
    """
    Run the entire one-click analysis and return a results dictionary.

    Returns structure:
    {
      "kpis": {...},
      "plots": {"sales_trend": path, "product_comp": path, "region_perf": path},
      "profit_df": profit_df (as dict or small table),
      "top_products": top (as dict),
      "bottom_products": bottom (as dict),
      "segments_preview": seg_df.head(...).to_dict(...),
      "recommendations": recs,
      "forecast_summary": maybe dict,
      "report_path": path_to_html
    }
    """
    results: Dict[str, Any] = {}
    try:
        df = load_csv(data_path, parse_dates=parse_dates)
    except Exception as e:
        logger.exception("Failed to load data for one-click: %s", e)
        raise

    # Validate and optionally apply fixes
    schema = get_default_schema()
    diag = validate_dataframe(df, schema=schema)
    results["validation"] = diag
    if not diag.get("valid", False) and auto_apply:
        try:
            df_fixed, summary = apply_suggested_fixes(df, diag)
            df = df_fixed
            results["fix_summary"] = summary.get("applied", [])
        except Exception:
            logger.exception("Auto-apply fixes failed; continuing with original df")

    # Preprocessing
    df = drop_duplicates(df)
    df = handle_missing_values(df, numeric_strategy="median")
    df = create_profit(df)

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

    # EDA plots (saved)
    sales_trend_path = os.path.join(PLOTS_DIR, "oneclick_sales_trend.png")
    product_comp_path = os.path.join(PLOTS_DIR, "oneclick_product_comp.png")
    region_perf_path = os.path.join(PLOTS_DIR, "oneclick_region_perf.png")

    try:
        plot_sales_trend(df, date_col="order_date", value_col="revenue", outpath=sales_trend_path)
    except Exception:
        logger.exception("plot_sales_trend failed in one-click")

    try:
        plot_product_comparison(df, product_col="product_name", value_col="revenue", outpath=product_comp_path)
    except Exception:
        logger.exception("plot_product_comparison failed in one-click")

    try:
        plot_region_performance(df, region_col="region", value_col="revenue", outpath=region_perf_path)
    except Exception:
        logger.exception("plot_region_performance failed in one-click")

    results["plots"] = {
        "sales_trend": sales_trend_path if os.path.exists(sales_trend_path) else None,
        "product_comp": product_comp_path if os.path.exists(product_comp_path) else None,
        "region_perf": region_perf_path if os.path.exists(region_perf_path) else None,
    }

    # Profitability
    try:
        profit_df = compute_profitability(df)
        results["profit_df"] = profit_df.head(20).to_dict(orient="records")
    except Exception:
        logger.exception("compute_profitability failed in one-click")
        results["profit_df"] = []

    # Top/bottom products
    try:
        top, bottom = top_bottom_products(df, n=5)
        results["top_products"] = top.to_dict(orient="records")
        results["bottom_products"] = bottom.to_dict(orient="records")
    except Exception:
        logger.exception("top_bottom_products failed")
        results["top_products"] = []
        results["bottom_products"] = []

    # Segmentation (preview)
    try:
        seg_df, kmodel = kmeans_segmentation(df)
        results["segments_preview"] = seg_df.head(20).to_dict(orient="records")
    except Exception:
        logger.exception("kmeans_segmentation failed")
        results["segments_preview"] = []

    # Simple forecast change estimates per product for recommendations
    try:
        df_forecast_est = _compute_simple_forecast_change(df)
    except Exception:
        df_forecast_est = pd.DataFrame({"product_id": [], "forecast_change_pct": []})

    # Recommendations
    try:
        # profit_df may have column 'product_name' or index; ensure correct formatting
        profit_for_recs = None
        try:
            profit_for_recs = compute_profitability(df).reset_index().rename(columns={"product_name": "product"})
        except Exception:
            # fallback: prepare minimal profit-like table
            profit_for_recs = pd.DataFrame({"product": df.get("product_name", pd.Series(dtype=str)).unique(), "profit": 0.0, "margin": 0.0})
        recs = generate_recommendations(df_forecast=df_forecast_est.rename(columns={"product_id": "product_id"}), df_profit=profit_for_recs)
        results["recommendations"] = recs
    except Exception:
        logger.exception("generate_recommendations failed")
        results["recommendations"] = {"recommendations": []}

    # Forecast a top product using SARIMA as a quick example (non-blocking)
    forecast_summary = {}
    try:
        if results.get("top_products"):
            top_prod = results["top_products"][0].get("product_name") if isinstance(results["top_products"][0], dict) else results["top_products"][0]
            if not top_prod:
                top_prod = df["product_name"].mode().iloc[0] if "product_name" in df.columns else None
            if top_prod is not None:
                df_prod = df[df["product_name"] == top_prod][["order_date", "quantity"]].groupby("order_date").sum().reset_index()
                try:
                    fc = sarima_forecast(df_prod, date_col="order_date", value_col="quantity", periods=forecast_horizon)
                    # convert forecast to list for JSON-friendly return
                    forecast_summary = {"product": top_prod, "sarima_forecast": fc["forecast"].tolist() if hasattr(fc["forecast"], "tolist") else [], "model_path": fc.get("model_path")}
                except Exception:
                    logger.exception("SARIMA forecast failed for top product")
    except Exception:
        logger.exception("Forecast summary branch failed")
    results["forecast_summary"] = forecast_summary

    # Create HTML report
    try:
        report_path = os.path.join(REPORTS_DIR, "one_click_report.html")
        create_html_report(
            outpath=report_path,
            kpis=results.get("kpis", {}),
            plots=[p for p in results["plots"].values() if p],
            top_products=results.get("top_products", []),
            bottom_products=results.get("bottom_products", []),
            recommendations=results.get("recommendations", {}),
            forecast_summary=results.get("forecast_summary", {}),
        )
        results["report_path"] = report_path
    except Exception:
        logger.exception("Failed to create report")
        results["report_path"] = None

    return results