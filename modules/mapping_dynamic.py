"""
Dynamic mapping helpers that inspect an uploaded DataFrame and suggest candidates,
and persist mappings keyed by file checksum (filesystem-only).
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import hashlib
import re
import difflib

import pandas as pd
from modules.utils import compute_file_checksum, ensure_dir, read_json, write_json

STORE = Path("data/uploads/mappings_by_checksum.json")
ensure_dir(str(STORE.parent))
if not STORE.exists():
    STORE.write_text(json.dumps({}))


SYNONYMS = {
    "order_id": ["orderid", "order_id", "order", "id", "saleid", "transaction_id", "txn_id"],
    "order_date": ["orderdate", "order_date", "date", "transaction_date", "sale_date", "tx_date"],
    "product_id": ["productid", "product_id", "sku", "item_id", "productcode"],
    "product_name": ["productname", "product_name", "item", "item_name", "title", "name"],
    "category": ["category", "cat", "product_category", "dept", "department"],
    "region": ["region", "state", "country", "area", "location"],
    "quantity": ["quantity", "qty", "units", "amount", "qty_sold"],
    "unit_price": ["unit_price", "unitprice", "price", "sale_price"],
    "cost": ["cost", "unit_cost", "cogs", "cost_price"],
    "customer_id": ["customerid", "customer_id", "cust_id", "client_id", "buyer_id"],
    "currency": ["currency", "curr"],
    "revenue": ["revenue", "total", "total_amount", "amount", "sales"],
    "profit": ["profit", "margin", "gross_profit"],
}


def canonical_fields() -> List[str]:
    return list(SYNONYMS.keys())


def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^\w]", " ", s)
    return s


def _is_date_like(series: pd.Series, sample_n: int = 50) -> float:
    s = series.dropna().astype(str).head(sample_n)
    if s.empty:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce")
    return float(parsed.notna().sum()) / max(1, len(s))


def _is_numeric_like(series: pd.Series, sample_n: int = 50) -> float:
    s = series.dropna().head(sample_n)
    if s.empty:
        return 0.0
    coerced = pd.to_numeric(s, errors="coerce")
    return float(coerced.notna().sum()) / max(1, len(s))


def _uniqueness_score(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    uniq = series.nunique(dropna=True)
    return float(uniq) / max(1, n)


def _token_overlap_score(col_name: str, synonyms: List[str]) -> float:
    cn = _normalize(col_name)
    c_tokens = set([t for t in cn.split() if t])
    if not c_tokens:
        return 0.0
    best = 0.0
    for syn in synonyms:
        stoks = set([t for t in _normalize(syn).split() if t])
        if not stoks:
            continue
        overlap = len(c_tokens & stoks) / max(1, len(stoks))
        best = max(best, overlap)
    return float(best)


def _fuzzy_name_score(col_name: str, target_field: str) -> float:
    ncol = _normalize(col_name).replace(" ", "")
    ntarget = _normalize(target_field).replace(" ", "")
    if not ncol or not ntarget:
        return 0.0
    seq = difflib.SequenceMatcher(None, ncol, ntarget)
    return seq.ratio()


def _score_candidate(col: str, df: pd.DataFrame, field: str) -> Dict[str, Any]:
    heur_date = _is_date_like(df[col])
    heur_num = _is_numeric_like(df[col])
    uniq = _uniqueness_score(df[col])
    token = _token_overlap_score(col, SYNONYMS.get(field, []))
    fuzzy = _fuzzy_name_score(col, field)

    score = 0.0
    reasons = []

    name_signal = max(token, fuzzy)
    if name_signal > 0.6:
        score += 0.45
        reasons.append(f"name match ({name_signal:.2f})")
    elif name_signal > 0.3:
        score += 0.2
        reasons.append(f"weak name ({name_signal:.2f})")

    if "date" in field:
        score += 0.4 * heur_date
        if heur_date > 0.5:
            reasons.append(f"date-like ({heur_date:.2f})")
    elif field in ("quantity", "unit_price", "cost", "revenue", "profit"):
        score += 0.45 * heur_num
        if heur_num > 0.5:
            reasons.append(f"numeric-like ({heur_num:.2f})")
    elif field.endswith("id"):
        score += 0.35 * min(1.0, uniq * 1.2)
        if uniq > 0.5:
            reasons.append(f"unique id-like ({uniq:.2f})")
    else:
        score += 0.2 * (1 - heur_num) + 0.15 * min(1.0, uniq)

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reasons) if reasons else "heuristic fallback"
    return {"column": col, "score": round(float(score), 3), "reason": reason}


def suggest_candidates(df: pd.DataFrame, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    candidates = {}
    if df is None or df.empty:
        for f in canonical_fields():
            candidates[f] = []
        return candidates

    cols = list(df.columns)
    for field in canonical_fields():
        scored = []
        for c in cols:
            try:
                sc = _score_candidate(c, df, field)
                scored.append(sc)
            except Exception:
                continue
        scored_sorted = sorted(scored, key=lambda x: (-x["score"], x["column"]))
        candidates[field] = scored_sorted[:top_k]
    return candidates


def _ensure_store():
    ensure_dir(str(STORE.parent))
    if not STORE.exists():
        STORE.write_text(json.dumps({}))


def persist_mapping_by_checksum(dataset_path: str, mapping: Dict[str, Optional[str]]):
    _ensure_store()
    key = compute_file_checksum(dataset_path)
    store = read_json(str(STORE))
    store[key] = mapping
    write_json(str(STORE), store)


def load_mapping_by_checksum(dataset_path: str) -> Optional[Dict[str, Optional[str]]]:
    if not STORE.exists():
        return None
    key = compute_file_checksum(dataset_path)
    store = read_json(str(STORE))
    return store.get(key)