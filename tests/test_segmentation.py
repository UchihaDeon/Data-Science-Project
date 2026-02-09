from modules.segmentation import kmeans_segmentation
from modules.io import load_csv

def test_kmeans():
    df = load_csv("data/sample_retail_data.csv", parse_dates=["order_date"])
    seg_df, model = kmeans_segmentation(df, n_clusters=2)
    assert "segment" in seg_df.columns