"""
modules/validation.py

Validate uploaded ecommerce datasets for required columns, types, and simple data quality checks.
Provides:
- get_default_schema()
- validate_dataframe(df, schema=None)
- apply_suggested_fixes(df, diagnostics)  # attempts coercions described in diagnostics
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_default_schema() -> Dict[str, List[str]]:
    """
    Returns the default expected schema for ecommerce retail datasets.
    Adjust this list if your data has different column names.
    """
    return {
        "required": [
            "order_id",
            "order_date",
            "product_id",
            "product_name",
            "category",
            "region",
            "quantity",
            "unit_price",
            "cost",
            "customer_id",
        ],
        "date_cols": ["order_date"],
        "numeric_cols": ["quantity", "unit_price", "cost"],
        # optionally useful categorical cols to check existence/uniqueness
        "categorical_cols": ["product_name", "category", "region", "customer_id"],
    }


def _sample_values(series: pd.Series, n: int = 5) -> List[Any]:
    return series.dropna().unique().tolist()[:n]


def validate_dataframe(df: pd.DataFrame, schema: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Validate a DataFrame against the provided schema (or default).
    Returns a diagnostics dict describing issues and suggested fixes.

    Diagnostics structure:
    {
      "valid": bool,
      "missing_columns": [...],
      "date_problems": {col: "not_parseable" or "ok"},
      "numeric_problems": {col: {"non_numeric_count": int, "examples": [...]}},
      "negative_value_columns": {col: count},
      "duplicates": { "rows": int, "example": ... },
      "suggested_fixes": [ { "action": "coerce_numeric", "column": col }, ... ]
    }
    """
    if schema is None:
        schema = get_default_schema()

    diagnostics: Dict[str, Any] = {
        "valid": True,
        "missing_columns": [],
        "date_problems": {},
        "numeric_problems": {},
        "negative_value_columns": {},
        "duplicates": {},
        "suggested_fixes": [],
    }

    # Check required columns
    required = schema.get("required", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        diagnostics["missing_columns"] = missing
        diagnostics["valid"] = False
        diagnostics["suggested_fixes"].append({"action": "add_missing_columns_info", "columns": missing})

    # Date columns: check parseability
    for dcol in schema.get("date_cols", []):
        if dcol in df.columns:
            try:
                parsed = pd.to_datetime(df[dcol], errors="coerce")
                n_invalid = int(parsed.isna().sum())
                if n_invalid > 0:
                    diagnostics["date_problems"][dcol] = {"invalid_count": n_invalid, "examples": _sample_values(df[dcol].astype(str))}
                    diagnostics["valid"] = False
                    diagnostics["suggested_fixes"].append({"action": "coerce_date", "column": dcol})
                else:
                    diagnostics["date_problems"][dcol] = {"invalid_count": 0}
            except Exception as e:
                diagnostics["date_problems"][dcol] = {"error": str(e)}
                diagnostics["valid"] = False
                diagnostics["suggested_fixes"].append({"action": "coerce_date", "column": dcol})
        else:
            diagnostics["date_problems"][dcol] = {"missing": True}

    # Numeric columns: check numericness
    for ncol in schema.get("numeric_cols", []):
        if ncol in df.columns:
            ser = df[ncol]
            # count non-numeric entries when coerced
            coerced = pd.to_numeric(ser, errors="coerce")
            non_numeric_count = int(coerced.isna().sum())
            if non_numeric_count > 0:
                diagnostics["numeric_problems"][ncol] = {"non_numeric_count": non_numeric_count, "examples": _sample_values(ser)}
                diagnostics["valid"] = False
                diagnostics["suggested_fixes"].append({"action": "coerce_numeric", "column": ncol})
            # negative checks (for quantity/cost/unit_price negative may be invalid)
            if np.issubdtype(coerced.dtype, np.number):
                neg_count = int((coerced < 0).sum())
                if neg_count > 0:
                    diagnostics["negative_value_columns"][ncol] = neg_count
                    diagnostics["suggested_fixes"].append({"action": "review_negative_values", "column": ncol, "count": neg_count})
        else:
            diagnostics["numeric_problems"][ncol] = {"missing": True}
            diagnostics["valid"] = False
            diagnostics["suggested_fixes"].append({"action": "add_missing_numeric", "column": ncol})

    # Duplicate orders check (simple)
    if "order_id" in df.columns:
        dup_mask = df.duplicated(subset=["order_id"])
        dup_count = int(dup_mask.sum())
        diagnostics["duplicates"] = {"rows": dup_count}
        if dup_count > 0:
            diagnostics["valid"] = False
            diagnostics["suggested_fixes"].append({"action": "drop_duplicates", "column": "order_id", "count": dup_count})

    return diagnostics


def apply_suggested_fixes(df: pd.DataFrame, diagnostics: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Attempts to apply simple suggested fixes from diagnostics:
    - coerce_date: pd.to_datetime with errors='coerce'
    - coerce_numeric: pd.to_numeric with errors='coerce'
    - drop_duplicates: df.drop_duplicates(subset=['order_id'])
    Returns (fixed_df, applied_actions_summary)
    """
    df = df.copy()
    applied = []
    for s in diagnostics.get("suggested_fixes", []):
        action = s.get("action")
        col = s.get("column")
        try:
            if action == "coerce_date" and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                applied.append({"action": action, "column": col})
            elif action == "coerce_numeric" and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                applied.append({"action": action, "column": col})
            elif action == "drop_duplicates" and s.get("column") in df.columns:
                df = df.drop_duplicates(subset=[s.get("column")])
                applied.append({"action": action, "column": s.get("column")})
            # No-op for other actions (informational) — they need human review
        except Exception as e:
            logger.warning("Failed to apply fix %s on column %s: %s", action, col, e)
    # After applying coercions, recompute diagnostics for transparency
    new_diag = validate_dataframe(df, schema=None)
    return df, {"applied": applied, "new_diagnostics": new_diag}