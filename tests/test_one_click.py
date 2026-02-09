import os
from modules.one_click import run_one_click_analysis
from modules.io import load_csv

def test_run_one_click(tmp_path):
    # Use the sample data path shipped with the project
    data_path = "data/sample_retail_data.csv"
    assert os.path.exists(data_path)
    results = run_one_click_analysis(data_path=data_path, parse_dates=["order_date"], forecast_horizon=3, auto_apply=True)
    # Basic assertions
    assert "kpis" in results
    assert "plots" in results
    assert "report_path" in results
    report = results.get("report_path")
    if report:
        assert os.path.exists(report)
    # KPIs numeric
    kpis = results.get("kpis", {})
    assert isinstance(kpis.get("total_revenue", 0), float)
    assert isinstance(kpis.get("orders", 0), int)