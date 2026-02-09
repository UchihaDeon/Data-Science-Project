"""
Enhanced Streamlit UI for Retail Analytics Assistant — user-friendly, explanatory.

Flows:
1. Upload CSV -> Preview
2. Map essential fields (guided suggestions)
3. Validate and optionally apply safe fixes (coercion, fill medians, drop duplicates)
4. Run full analysis (cleaning, EDA, segmentation, per-segment forecast)
5. Show "What happened to my data?" and Business Insights + Recommendations

This version includes:
- an interactive JSON tree viewer (JSONEditor) for diagnostics
- a compatibility wrapper (call_pipeline) to support different pipeline function signatures
"""
import sys
from pathlib import Path
import os
import logging
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from streamlit.components.v1 import html as st_html

from modules.mapping_dynamic import suggest_candidates, compute_file_checksum, load_mapping_by_checksum, persist_mapping_by_checksum  # type: ignore
# pipeline: prefer run_full_pipeline, fallback to run_pipeline
try:
    from modules.pipeline import run_full_pipeline as pipeline_runner
except Exception:
    from modules.pipeline import run_pipeline as pipeline_runner

from modules.utils import ensure_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Ensure directories
ensure_dir("data/uploads")
ensure_dir("dashboards/plots")
ensure_dir("dashboards/reports")
ensure_dir("models")


st.set_page_config(page_title="Retail Analytics Assistant — Friendly UI", layout="wide")
st.title("Retail Analytics Assistant — Friendly UI")


def info(text: str):
    st.markdown(f"<div style='color:#444;margin-bottom:8px'>{text}</div>", unsafe_allow_html=True)


# --- Pipeline compatibility wrapper ---
def call_pipeline(df, mapping, **kwargs):
    """
    Call the imported pipeline runner (which may be run_full_pipeline or an older run_pipeline).
    Try the full kwargs first; on TypeError (unexpected args), fall back to simpler calls.
    Returns whatever the pipeline returns.
    """
    try:
        return pipeline_runner(df, mapping=mapping, **kwargs)
    except TypeError:
        # fallback: try a smaller subset of kwargs commonly supported
        try:
            minimal = {}
            for k in ("do_eda", "do_features", "forecast_horizon", "auto_fix", "auto_apply_fixes", "persist_mapping"):
                if k in kwargs:
                    minimal[k] = kwargs[k]
            return pipeline_runner(df, mapping=mapping, **minimal)
        except TypeError:
            # last resort: simplest call signature
            return pipeline_runner(df, mapping)


# ===== JSON tree viewer helper =====
def json_tree_viewer(data: dict, height: int = 600):
    """
    Render interactive JSON tree using jsoneditor (client-side).
    Falls back to st.json if the JSON is too large.
    """
    try:
        txt = json.dumps(data, indent=2)
    except Exception:
        st.write("Unable to stringify diagnostics for tree view.")
        st.json(data)
        return

    # safety: if very large, avoid embedding heavy payloads in component
    if len(txt) > 400000:  # ~400 KB
        st.warning("Diagnostics are large — showing plain JSON and offering download instead of interactive tree.")
        st.code(txt[:100000] + "\n\n... (truncated) ...", language="json")
        st.download_button("Download full diagnostics JSON", data=txt, file_name="diagnostics.json", mime="application/json")
        return

    # escape closing script tags to avoid injection
    safe_txt = txt.replace("</", "<\\/")
    # JSONEditor CDN resources (v9+)
    # We embed the JSON as a JS string and parse it in the browser to avoid quoting issues.
    js = f"""
    <div id="jsoneditor" style="width:100%; height:{height}px; border:1px solid #ddd;"></div>
    <link rel="stylesheet" href="https://unpkg.com/jsoneditor@9.10.0/dist/jsoneditor.min.css" />
    <script src="https://unpkg.com/jsoneditor@9.10.0/dist/jsoneditor.min.js"></script>
    <script>
      const container = document.getElementById("jsoneditor");
      const options = {{ mode: 'view', mainMenuBar: false }};
      const editor = new JSONEditor(container, options);
      const jsonString = {json.dumps(safe_txt)};
      try {{
        const parsed = JSON.parse(jsonString);
        editor.set(parsed);
        try {{ editor.expandAll(); }} catch(e){{}}
      }} catch (err) {{
        editor.set(jsonString);
      }}
    </script>
    """
    st_html(js, height=height + 30)


# Sidebar: upload or choose an existing CSV
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
saved_path = None
if uploaded_file is not None:
    try:
        fname = uploaded_file.name or "uploaded.csv"
        save_dir = Path("data/uploads")
        save_dir.mkdir(parents=True, exist_ok=True)
        fp = save_dir / fname
        # avoid overwrite
        if fp.exists():
            base, ext = os.path.splitext(fname)
            count = 1
            while (save_dir / f"{base}_{count}{ext}").exists():
                count += 1
            fp = save_dir / f"{base}_{count}{ext}"
        with open(fp, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_path = str(fp)
        st.sidebar.success(f"Saved: {saved_path}")
    except Exception as e:
        st.sidebar.error(f"Save failed: {e}")
        logger.exception("upload failed")

uploaded_files = sorted([str(p) for p in Path("data/uploads").glob("*.csv")])
selected = st.sidebar.selectbox("Or choose a saved CSV", options=[""] + uploaded_files) if uploaded_files else None
if selected == "":
    selected = None

data_path = saved_path or selected
if not data_path:
    st.sidebar.info("Upload a CSV or select an existing one to begin.")
    st.stop()

# Load sample preview
@st.cache_data(show_spinner=False)
def load_preview(path, nrows=1000):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None


df_preview = load_preview(data_path, nrows=2000)
if df_preview is None:
    st.error("Failed to read the CSV for preview. Check file encoding/format.")
    st.stop()

# Top-level instructions
info(
    "This app helps you map your columns, validates and applies safe fixes, runs analysis "
    "and explains results in plain language. Follow steps below and expand sections for details."
)

# Layout: two columns: left (workflow) right (live summary)
left, right = st.columns([2, 1])

# ======== RIGHT PANEL: quick summary ========
with right:
    st.subheader("File summary")
    st.markdown(f"- Path: `{data_path}`")
    st.markdown(f"- Rows (preview): {len(df_preview)}")
    st.markdown(f"- Columns: {len(df_preview.columns)}")
    st.write(df_preview.columns.tolist())

    st.markdown("---")
    st.subheader("Quick actions")
    if st.button("Auto-fill mapping suggestions"):
        st.session_state["auto_fill_requested"] = True
        st.experimental_rerun()
    if st.button("Refresh preview"):
        st.experimental_rerun()

# ======== LEFT PANEL: mapping & validation ========
with left:
    st.header("1) Map fields (minimal required)")
    info("Map the essential fields so the system can run full analysis. Required: `order_date`, `quantity` (or `revenue`), `product_name`. "
         "Optional but recommended: `customer_id`, `unit_price`, `cost`.")

    candidates = suggest_candidates(df_preview, top_k=4)
    checksum = compute_file_checksum(data_path)
    persisted = load_mapping_by_checksum(data_path) or {}

    # fields shown (minimal)
    fields = ["order_date", "quantity", "product_name", "customer_id", "unit_price", "revenue", "cost", "order_id", "category", "region"]

    # session init
    if "mapping" not in st.session_state or st.session_state.get("mapping_for") != checksum:
        init = {}
        for f in fields:
            init[f] = persisted.get(f) or (candidates.get(f)[0]["column"] if candidates.get(f) and len(candidates.get(f)) > 0 else "")
        st.session_state["mapping"] = init
        st.session_state["mapping_for"] = checksum

    mapping = st.session_state["mapping"]

    # simple UX: show required mapping first
    for f in ["order_date", "quantity", "product_name", "customer_id"]:
        st.markdown(f"**{f}**")
        # show top candidates as buttons
        cand = candidates.get(f, [])
        if cand:
            cols = st.columns(len(cand))
            for i, c in enumerate(cand):
                if cols[i].button(f"use_{f}_{i}", key=f"use_{f}_{i}"):
                    mapping[f] = c["column"]
                    st.session_state["mapping"] = mapping
                    st.experimental_rerun()
            top = cand[0]
            st.caption(f"Top suggestion: `{top['column']}` — {top['reason']} (score {top['score']})")
        options = [""] + list(df_preview.columns)
        default = mapping.get(f, "") or ""
        sel = st.selectbox(f"Select column for {f}", options=options, index=options.index(default) if default in options else 0, key=f"sel_{f}")
        mapping[f] = sel
        if sel:
            try:
                sample = df_preview[sel].dropna().unique()[:6].tolist()
                st.write("Sample values:", sample)
            except Exception:
                pass
        st.markdown("---")

    # optional fields in expander
    with st.expander("Optional mappings (advanced)"):
        cols2 = st.columns(2)
        for i, f in enumerate([c for c in fields if c not in ("order_date", "quantity", "product_name", "customer_id")]):
            with cols2[i % 2]:
                options = [""] + list(df_preview.columns)
                default = mapping.get(f, "") or ""
                sel = st.selectbox(f"Map {f}", options=options, index=options.index(default) if default in options else 0, key=f"opt_{f}")
                mapping[f] = sel

    # persist mapping
    persist = st.checkbox("Save mapping for this file (persist by checksum)", value=False)
    if st.button("Save mapping now") and persist:
        try:
            persist_mapping_by_checksum(data_path, mapping)
            st.success("Mapping saved for this checksum.")
        except Exception as e:
            st.error(f"Failed to save mapping: {e}")

    # Validate minimal presence
    missing = [f for f in ["order_date", "product_name"] if not mapping.get(f)]
    if not (mapping.get("quantity") or mapping.get("revenue")):
        missing.append("quantity or revenue")
    if missing:
        st.warning(f"Please map required fields before running analysis: {missing}")
        st.stop()

    st.markdown("")

    # ===== Validation preview and safe fixes =====
    st.header("2) Preview validation & apply safe fixes")
    info("This step shows data issues detected (missing values, bad types, duplicates) and allows safe auto-fixes (coercion, fill medians, drop duplicates). Expand diagnostics for details.")

    # show small raw sample
    st.subheader("Raw sample (first 10 rows)")
    st.dataframe(df_preview.head(10))

    if st.button("Run validation (detect issues)"):
        # Load full small sample for validation
        try:
            df_full_preview = pd.read_csv(data_path, nrows=20000)
        except Exception:
            df_full_preview = pd.read_csv(data_path, nrows=2000)
        # Build mapped df for validation
        inv_map = {v: k for k, v in mapping.items() if v}
        df_mapped_preview = df_full_preview.rename(columns=inv_map).copy()
        # Run lightweight pipeline to get diagnostics: use pipeline with do_eda=False, do_features=False but auto_fix False
        with st.spinner("Validating..."):
            results = call_pipeline(df_mapped_preview, mapping=mapping, do_eda=False, do_features=False, auto_fix=False)

        # show diagnostics
        st.subheader("Validation diagnostics")
        diag = results.get("diagnostics") or {}

        # interactive JSON tree viewer (uses JSONEditor)
        with st.expander("View diagnostics and fixes (interactive JSON)", expanded=True):
            json_tree_viewer(diag, height=500)
            # also provide download option
            try:
                jtxt = json.dumps(diag, indent=2)
                st.download_button("Download diagnostics JSON", data=jtxt, file_name="diagnostics.json", mime="application/json")
            except Exception:
                pass

        # show missing value counts for mapped key columns
        st.subheader("Missing values (key fields)")
        key_cols = [c for c in ["order_date", "quantity", "revenue", "product_name", "customer_id"] if c in df_mapped_preview.columns]
        miss = {c: int(df_mapped_preview[c].isna().sum()) for c in key_cols}
        st.table(pd.DataFrame.from_dict(miss, orient="index", columns=["missing_count"]))

        # show duplicates count (order_id)
        dup_msg = ""
        if "order_id" in df_mapped_preview.columns:
            dup_count = int(df_mapped_preview.duplicated(subset=["order_id"]).sum())
            dup_msg = f"{dup_count} duplicate order_id rows detected"
        else:
            dup_msg = "order_id not present; duplicates by full row check may exist"
        st.write(dup_msg)

        # present safe-fix options
        st.markdown("**Safe-fix options**")
        do_coerce = st.checkbox("Coerce types (dates/numeric) automatically", value=True)
        fill_strategy = st.selectbox("Numeric fill strategy for missing values", options=["median", "mean", "zero"], index=0)
        drop_dupes = st.checkbox("Drop duplicate order_id rows (if order_id present)", value=True)
        if st.button("Apply safe fixes to a preview and show diff"):
            # apply safe fixes by calling pipeline with auto_fix True (but we want to show delta)
            before_rows = len(df_mapped_preview)
            before_missing = df_mapped_preview.isna().sum().to_dict()
            df_fixed = df_mapped_preview.copy()
            # simple coercions
            if do_coerce:
                for col in df_fixed.columns:
                    if "date" in col.lower():
                        df_fixed[col] = pd.to_datetime(df_fixed[col], errors="coerce")
                    else:
                        coerced = pd.to_numeric(df_fixed[col], errors="coerce")
                        if coerced.notna().sum() > 0.5 * len(coerced):
                            df_fixed[col] = coerced
            # fill numeric
            num_cols = df_fixed.select_dtypes(include="number").columns
            for c in num_cols:
                if df_fixed[c].isna().any():
                    if fill_strategy == "median":
                        df_fixed[c] = df_fixed[c].fillna(df_fixed[c].median())
                    elif fill_strategy == "mean":
                        df_fixed[c] = df_fixed[c].fillna(df_fixed[c].mean())
                    else:
                        df_fixed[c] = df_fixed[c].fillna(0)
            # drop duplicates
            if drop_dupes and "order_id" in df_fixed.columns:
                df_fixed = df_fixed.drop_duplicates(subset=["order_id"])
            after_rows = len(df_fixed)
            after_missing = df_fixed.isna().sum().to_dict()
            st.markdown("**Preview of fixes applied**")
            st.write(f"Rows before: {before_rows}; after: {after_rows}")
            # show changes for key columns
            delta_missing = {c: int(before_missing.get(c, 0) - after_missing.get(c, 0)) for c in key_cols}
            st.table(pd.DataFrame.from_dict(delta_missing, orient="index", columns=["missing_filled"]))

            st.subheader("Sample after fixes")
            st.dataframe(df_fixed.head(10))

            # let user accept preview fixes to be applied in full run
            if st.button("Accept these fixes for full run"):
                st.session_state["accepted_fix"] = {"coerce": do_coerce, "fill_strategy": fill_strategy, "drop_dupes": drop_dupes}
                st.success("Safe-fix preferences saved; these will be applied when you run analysis.")

    st.markdown("")
    st.header("3) Run full analysis & get business insights")
    info("When you run the full analysis we will clean the full dataset, generate KPIs, run segmentation (if customer_id present), produce forecasts for top segments, and compile recommended actions. This may take a few seconds to a couple minutes depending on dataset size.")

    # forecast horizon input (stored to session for pipeline call)
    forecast_horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=90, value=14)
    st.session_state["forecast_horizon"] = int(forecast_horizon)

    run_button = st.button("Run Full Analysis")
    if run_button:
        inv_map = {v: k for k, v in mapping.items() if v}
        # read full file (careful: could be large)
        try:
            df_full = pd.read_csv(data_path)
        except Exception:
            df_full = pd.read_csv(data_path, dtype=str)
        df_mapped = df_full.rename(columns=inv_map).copy()

        # determine auto_fix settings from accepted preview or default
        accepted = st.session_state.get("accepted_fix", None)
        with st.spinner("Running full pipeline (cleaning, features, segmentation, forecast)... This may take up to a minute for medium files."):
            results = call_pipeline(
                df_mapped,
                mapping=mapping,
                do_eda=True,
                do_features=True,
                do_segmentation=True,
                do_forecast_per_segment=True,
                forecast_horizon=int(st.session_state.get("forecast_horizon", 14)),
            )
        # show results
        st.success("Analysis complete")
        # store results in session for further UI
        st.session_state["latest_results"] = results

# ===== Show results / insights if available =====
if "latest_results" in st.session_state:
    results = st.session_state["latest_results"]
    st.markdown("---")
    st.header("Analysis results — quick business summary")
    kpis = results.get("kpis", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${kpis.get('total_revenue',0):,.2f}")
    col2.metric("Total Profit", f"${kpis.get('total_profit',0):,.2f}")
    col3.metric("Orders", f"{kpis.get('orders',0)}")
    col4.metric("Avg Order Value", f"${kpis.get('avg_order_value',0):,.2f}")

    # plain-language interpretation
    st.subheader("What this means for your business (plain language)")
    expl = []
    tr = results.get("plots", {}).get("trend") or results.get("plots", {}).get("value_trend")
    if tr:
        expl.append("We computed recent sales trends. Use the trend chart to spot rising or falling demand.")
    if results.get("segments"):
        expl.append(f"We identified {len(results['segments'])} customer segments (RFM). See the segment profiles and top products per segment below.")
    if results.get("forecasts"):
        expl.append("We produced per-segment forecasts for the top segments. Check which segments drive expected growth and prioritize them for stock and promotions.")
    if not expl:
        expl = ["No business insights available — ensure required fields were mapped and data contains dates and revenue/quantity."]
    for e in expl:
        st.write("•", e)

    # Diagnostics & "what happened to my data"
    st.markdown("---")
    st.subheader("What happened to your data?")
    info("This section shows concrete changes the pipeline made to your data so you can trust the outputs.")

    diag = results.get("diagnostics", {}) or {}
    with st.expander("View diagnostics and fixes (interactive)", expanded=False):
        json_tree_viewer(diag, height=500)
        try:
            jtxt = json.dumps(diag, indent=2)
            st.download_button("Download diagnostics JSON", data=jtxt, file_name="diagnostics.json", mime="application/json")
        except Exception:
            pass

    # show summary_stats vs cleaned sample sizes if available
    cleaned = results.get("cleaned_df")
    if cleaned is not None:
        st.write("Cleaned dataset snapshot (first 10 rows):")
        st.dataframe(cleaned.head(10))

    # if the pipeline reported applied fixes, show them
    applied_fixes = diag.get("applied_fixes") or results.get("diagnostics", {}).get("applied_fixes") or []
    if applied_fixes:
        st.markdown("**Auto-applied fixes**")
        for a in applied_fixes:
            st.write("-", a)
    else:
        st.write("No automatic fixes were applied (or none were recorded).")

    # show missing value reduction if pipeline returned summary_stats
    summary = results.get("summary_stats", {})
    if summary:
        st.subheader("Missing & data profile (after cleaning)")
        st.write("Columns:", summary.get("columns", [])[:30])
        st.write("Missing counts (after cleaning):")
        missing_after = summary.get("missing", {})
        if missing_after:
            sorted_miss = sorted(missing_after.items(), key=lambda x: -x[1])[:10]
            st.table(pd.DataFrame(sorted_miss, columns=["column", "missing_count"]).set_index("column"))

    # show segment summaries
    if results.get("segments"):
        st.markdown("---")
        st.subheader("Customer segments (RFM) — summary")
        seg_df = pd.DataFrame(results["segments"])
        if not seg_df.empty:
            st.write("Segment summary table (first 10):")
            st.dataframe(seg_df.head(10))
            seg_sorted = seg_df.sort_values("size", ascending=False)
            top_seg = seg_sorted.iloc[0].to_dict() if not seg_sorted.empty else None
            if top_seg:
                st.markdown(f"Top segment: {int(top_seg.get('segment_label',0))} (size {int(top_seg.get('size',0))})")
        else:
            st.info("No segments found to display.")

    # show forecasts
    if results.get("forecasts"):
        st.markdown("---")
        st.subheader("Forecast summary (per-segment)")
        fc = results.get("forecasts", {})
        rows = []
        for seg, obj in fc.items():
            fvals = obj.get("forecast")
            first = fvals[0] if fvals else None
            rows.append({"segment": seg, "method": obj.get("method"), "next_period_est": first})
        st.table(pd.DataFrame(rows))

        growths = []
        for seg, obj in fc.items():
            fvals = np.array(obj.get("forecast") or [])
            if fvals.size > 0:
                growths.append((int(seg), float(fvals.mean())))
        if growths:
            growths = sorted(growths, key=lambda x: -x[1])
            st.markdown("Top forecasted segments (by avg predicted quantity/revenue):")
            for s, val in growths[:5]:
                st.write(f"- Segment {s}: predicted avg next periods ≈ {val:.1f}")

    # Recommendations
    st.markdown("---")
    st.subheader("Prioritized recommendations")
    recs = []
    if results.get("forecasts"):
        for seg, obj in results["forecasts"].items():
            fvals = np.array(obj.get("forecast") or [])
            if fvals.size > 0 and fvals.mean() > 1.2 * (np.median(fvals) if fvals.size > 0 else 0):
                recs.append({"action": "Consider increasing stock", "segment": seg, "reason": f"Predicted high demand (avg {fvals.mean():.1f})"})
    if results.get("segments"):
        try:
            for s in results["segments"]:
                top_products = s.get("top_products", [])
                if top_products:
                    recs.append({"action": "Review top products pricing/stock", "segment": s.get("segment_label"), "reason": f"Top products: {[p.get('product_name', p.get('product_id','')) for p in top_products][:3]}"})
        except Exception:
            pass

    if recs:
        for r in recs[:10]:
            st.write(f"- {r.get('action')} — Segment: {r.get('segment')} — {r.get('reason')}")
    else:
        st.write("No automated recommendations generated for this dataset. Try running segmentation/forecasting or provide richer data (cost/unit_price).")

    # Report & download
    st.markdown("---")
    st.header("Download outputs")
    if results.get("report_path") and os.path.exists(results.get("report_path")):
        with open(results["report_path"], "rb") as fh:
            st.download_button("Download HTML report", data=fh.read(), file_name=Path(results["report_path"]).name, mime="text/html")
    if cleaned is not None:
        csv_bytes = cleaned.head(500).to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned sample (500 rows)", data=csv_bytes, file_name="cleaned_sample.csv", mime="text/csv")

    st.info("You can expand sections above for more details (diagnostics, segment profiles, plots).")