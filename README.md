# Retail Analytics Assistant — DB‑free (Streamlit)

A filesystem-only Streamlit app that ingests arbitrary e‑commerce CSVs, helps you map columns, performs data cleaning & EDA, computes RFM customer segmentation, and produces per‑segment sales forecasts and business recommendations — all without a database. Designed to be friendly for analysts and non‑technical users.

---

## Features

- Upload any CSV (no fixed schema required)
- Dynamic column mapping with conservative auto-suggestions
- Validation diagnostics with an interactive JSON tree viewer
- Safe, optional auto-fixes (type coercion, missing-value fills, duplicate removal)
- Exploratory Data Analysis (univariate, trends, correlation)
- RFM-based customer segmentation (KMeans)
- Per-segment sales forecasting (auto-ARIMA when available)
- File-based persistence for mappings (checksum keyed JSON)
- HTML report generation and downloadable cleaned sample
- DB‑free — all artifacts saved to the local filesystem:
  - `data/uploads/` — uploaded CSVs & mapping store
  - `dashboards/plots/` — generated plots
  - `dashboards/reports/` — HTML reports
  - `models/` — (optional) persisted models

---

## Quick Start

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-root>
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows (PowerShell)
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > Note: `pmdarima` is used for automated ARIMA model selection. If installing `pmdarima` fails, the system will fall back to a naive forecast. For advanced models (Prophet / LSTM) install their packages separately.

4. Run the Streamlit app:
   ```bash
   streamlit run dashboards/streamlit_app.py
   ```

5. Open the displayed URL in your browser (usually `http://localhost:8501`).

---

## Recommended dataset schema (best results)

For best and complete outputs (forecasting, segmentation, product insights), provide a transaction‑level CSV with these columns:

Required
- `order_date` — parseable date or timestamp
- `customer_id` — stable customer identifier (for RFM/segmentation)
- `product_id` or `product_name` — product identifier
- `quantity` or `revenue` — at least one; having both is ideal

Recommended
- `order_id` — deduplication & order-level metrics
- `unit_price`, `cost` — allow margin & profit analysis
- `category`, `region`, `channel` — dimension-level insights
- `is_promo`, `discount_amount` — account for promotional spikes

Time & volume guidance
- At least 6 months of daily data for simple forecasts; 1+ years preferred for seasonality.
- Hundreds of customers for stable segmentation.

---

## How the UI guides the user

1. Upload or select a CSV file in the sidebar.
2. Preview the first rows and use mapping suggestions to map:
   - `order_date`, `quantity` (or `revenue`), `product_name`, and optionally `customer_id`.
3. Run Validation to detect missing values, bad types, duplicates and view diagnostics in an interactive JSON tree.
4. Apply safe fixes in preview mode and accept them if results look good.
5. Run Full Analysis to produce KPIs, segment profiles, per‑segment forecasts, recommendations, and an HTML report.
6. Download cleaned sample and report.

---

## Files & folders

- dashboards/
  - `streamlit_app.py` — the Streamlit UI (main entry)
  - `plots/`, `reports/` — generated artifacts
- modules/
  - `mapping_dynamic.py` — mapping suggestions + persistence
  - `cleaning.py`, `preprocessing.py` — cleaning utilities
  - `eda.py` — exploratory functions & plotting
  - `features.py` — lag/rolling/AOV/RFM helpers
  - `segmentation.py` — RFM + KMeans
  - `timeseries.py` — aggregation + auto_arima wrapper
  - `pipeline.py` — orchestrator for full analysis
  - `reporting.py` — HTML report generator
  - `utils.py` — helpers (checksum, IO)
- `data/uploads/mappings_by_checksum.json` — saved mappings (created on first use)
- `requirements.txt` — core Python dependencies

---

## Diagnostics & explanations

- The app shows a readable diagnostics object (interactive JSON tree) so you can inspect mapping suggestions, validation results and any automatic fixes applied.
- The "What happened to your data?" section summarizes:
  - Rows/columns changed
  - Missing values filled
  - Duplicates dropped
  - Which columns were coerced/derived
- Business insights and recommendations are provided in plain language (e.g., "Consider increasing stock for Segment 2 based on forecasted demand").

---

## Troubleshooting

- SyntaxError / Import issues:
  - Ensure the project root is on PYTHONPATH (Streamlit run from the repo root is recommended).
  - Confirm `modules/__init__.py` exists (can be empty).
- pmdarima installation problems:
  - Try `pip install pmdarima` separately; on Windows you may need a C++ build toolchain or use wheels.
  - If unavailable, the app will still run but forecasting falls back to a simpler method.
- Large CSVs:
  - The app reads the full CSV for full analysis — if your file is very large (>100MB), split or sample the data to test.
  - Consider adding chunked processing or increasing machine resources.

---

## Privacy & storage

- This app stores all uploaded files and generated artifacts on the local filesystem. Do not upload PII or sensitive customer data unless you are authorized to store it locally.
- Mappings are persisted by file checksum to `data/uploads/mappings_by_checksum.json`. Remove files or mappings manually if needed.

---

## Extending the app

Possible improvements:
- Add background tasks (Redis + RQ) for long-running model training (LSTM).
- Add Prophet or other probabilistic forecasting models.
- Add multi-currency normalization and external features (holidays, marketing) for better accuracy.
- Add authentication if deploying to a shared environment.

---

## License & author

- MIT License — feel free to modify and use for personal or internal business projects.
- Author: Deon
- GitHub: https://github.com/UchihaDeon

---

## Want help?

If you want, you can:
- Upload a sample CSV and the app will run diagnostics and tell you readiness steps.
- Ask for a tailored configuration (e.g., include Prophet, add Dockerfile, or add background jobs).

Save this README as `README.md` in the project root so other users immediately understand how to run and use the Retail Analytics Assistant.