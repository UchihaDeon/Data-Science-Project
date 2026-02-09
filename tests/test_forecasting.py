import pandas as pd
from modules.forecasting import sarima_forecast
from modules.io import load_csv

def test_sarima():
    df = load_csv("data/sample_retail_data.csv", parse_dates=["order_date"])
    df_prod = df[df["product_name"] == "Widget A"][["order_date","quantity"]].groupby("order_date").sum().reset_index()
    res = sarima_forecast(df_prod, date_col="order_date", value_col="quantity", periods=3)
    assert "forecast" in res