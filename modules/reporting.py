"""
Simple HTML report generator for Phase 1.
"""
from typing import List, Dict, Any
import datetime
import os
import html
from modules.utils import ensure_dir
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _render_kpis(kpis: Dict[str, Any]) -> str:
    items = []
    for k, v in kpis.items():
        if isinstance(v, (int, float)):
            items.append(f"<div class='kpi'><strong>{k.replace('_',' ').title()}:</strong> ${v:,.2f}</div>")
        else:
            items.append(f"<div class='kpi'><strong>{k.replace('_',' ').title()}:</strong> {html.escape(str(v))}</div>")
    return "\n".join(items)


def _render_table(title: str, rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    if not rows:
        return f"<h3>{html.escape(title)}</h3><p>No data</p>"
    headers = list(rows[0].keys())
    header_html = "".join([f"<th>{html.escape(str(h))}</th>" for h in headers])
    rows_html = ""
    for r in rows[:max_rows]:
        row_cells = "".join([f"<td>{html.escape(str(r.get(h,'')))}</td>" for h in headers])
        rows_html += f"<tr>{row_cells}</tr>"
    return f"<h3>{html.escape(title)}</h3><table class='table'><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"


def _render_plots(plots: List[str]) -> str:
    if not plots:
        return "<p>No plots available.</p>"
    parts = []
    for p in plots:
        if os.path.exists(p):
            parts.append(f"<div class='plot'><img src='../{p}' alt='{os.path.basename(p)}' style='max-width:600px;'></div>")
        else:
            parts.append(f"<div class='plot'>Plot not found: {html.escape(str(p))}</div>")
    return "\n".join(parts)


def create_html_report(
    outpath: str,
    kpis: Dict[str, Any],
    plots: List[str],
    top_products: List[Dict[str, Any]],
    bottom_products: List[Dict[str, Any]],
    recommendations: Dict[str, Any],
    forecast_summary: Dict[str, Any],
):
    try:
        ensure_dir(os.path.dirname(outpath) or ".")
        now = datetime.datetime.utcnow().isoformat()
        html_kpis = _render_kpis(kpis)
        html_plots = _render_plots(plots)
        html_top = _render_table("Top Products", top_products)
        html_bottom = _render_table("Bottom Products", bottom_products)
        recs = recommendations.get("recommendations", []) if isinstance(recommendations, dict) else []
        recs_html = "<ul>" + "".join([f"<li>{html.escape(str(r))}</li>" for r in recs]) + "</ul>" if recs else "<p>No recommendations</p>"
        fc_html = "<pre>" + html.escape(str(forecast_summary)) + "</pre>" if forecast_summary else "<p>No forecast summary</p>"

        html_content = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Analysis Report</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .kpi {{ display:inline-block; margin-right:20px; padding:10px; background:#f3f3f3; border-radius:6px; }}
            .table {{ width:100%; border-collapse: collapse; margin-bottom: 20px; }}
            .table th, .table td {{ border:1px solid #ddd; padding:6px; text-align:left; }}
            .plot img {{ max-width: 100%; height: auto; border:1px solid #ccc; padding:4px; background:#fff; }}
          </style>
        </head>
        <body>
          <h1>Analysis Report</h1>
          <p>Generated: {now} UTC</p>
          <h2>Key Performance Indicators</h2>
          <div>{html_kpis}</div>
          <h2>Plots</h2>
          <div>{html_plots}</div>
          {html_top}
          {html_bottom}
          <h2>Recommendations</h2>
          {recs_html}
          <h2>Forecast Summary</h2>
          {fc_html}
        </body>
        </html>
        """
        with open(outpath, "w", encoding="utf-8") as fh:
            fh.write(html_content)
        logger.info("Wrote HTML report to %s", outpath)
        return outpath
    except Exception:
        logger.exception("Failed to create HTML report")
        raise