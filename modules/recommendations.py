"""
Rule-based engine for prescriptive recommendations.
"""

from typing import Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generate_recommendations(df_forecast: pd.DataFrame, df_profit: pd.DataFrame, threshold_increase: float = 0.2, margin_threshold: float = 0.1) -> Dict[str, Any]:
    """
    df_forecast: DataFrame with columns ['product_id','forecast_change_pct'] or product-level forecast summary.
    df_profit: profitability df produced by modules.profitability.compute_profitability

    Rules:
    - Increase stock for products forecasted to increase more than threshold_increase (pct)
    - Adjust price (recommend review) for products with margin < margin_threshold
    - Suggest bundling: basic heuristic - low-margin items commonly bought with high-value items (not implemented fully; return placeholder)
    """
    recs: List[Dict[str, Any]] = []
    try:
        # Increase stock
        inc_mask = df_forecast["forecast_change_pct"] > threshold_increase
        for _, row in df_forecast[inc_mask].iterrows():
            recs.append({"product_id": row["product_id"], "action": "increase_stock", "reason": f"Forecasted increase {row['forecast_change_pct']:.2%}"})

        # Price adjustments
        low_margin = df_profit[df_profit["margin"] < margin_threshold]
        for _, r in low_margin.iterrows():
            recs.append({"product": r.get("product_name", r.get("product")), "action": "review_pricing", "reason": f"Low margin {r['margin']:.2%}"})

        # Bundling (placeholder heuristic)
        recs.append({"action": "bundle_suggestion", "details": "Consider bundling low-price accessory items with high-value products."})

        logger.info("Generated %d recommendations", len(recs))
        return {"recommendations": recs}
    except Exception:
        logger.exception("generate_recommendations failed")
        raise