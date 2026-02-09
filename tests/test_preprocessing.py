import pandas as pd
from modules.preprocessing import create_profit, handle_missing_values, drop_duplicates

def test_create_profit():
    df = pd.DataFrame({"quantity":[1,2],"unit_price":[10,20],"cost":[3,4]})
    df2 = create_profit(df)
    assert "revenue" in df2.columns and "profit" in df2.columns
    assert df2["profit"].tolist() == [7,36]

def test_handle_missing_and_duplicates():
    df = pd.DataFrame({"a":[1,None,1],"order_id":[1,2,1]})
    df = drop_duplicates(df, subset=["order_id"])
    df = handle_missing_values(df)
    assert df.isnull().sum().sum() == 0