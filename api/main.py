"""
FastAPI app exposing endpoints for preprocessing, forecasting, segmentation, and recommendations.

Run:
    uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, UploadFile, File

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import modules.io as io
import modules.preprocessing as pre
import modules.feature_engineering as fe
import modules.forecasting as fc
import modules.segmentation as seg
import modules.recommendations as rec
import modules.profitability as prof
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

app = FastAPI(title="Retail DS Workflow API")


class LoadRequest(BaseModel):
    csv_path: Optional[str] = None
    api_url: Optional[str] = None


class ForecastRequest(BaseModel):
    product_name: str
    method: str = "sarima"
    periods: int = 14


class SegmentRequest(BaseModel):
    n_clusters: int = 3


@app.post("/preprocess")
def preprocess(req: LoadRequest):
    """
    Preprocess data provided by csv_path or api_url.
    Returns a small summary.
    """
    try:
        if req.csv_path:
            df = io.load_csv(req.csv_path, parse_dates=["order_date"])
        elif req.api_url:
            df = io.fetch_api(req.api_url)
        else:
            raise HTTPException(status_code=400, detail="Provide csv_path or api_url")
        df = pre.drop_duplicates(df)
        df = pre.handle_missing_values(df)
        df = pre.create_profit(df)
        return {"rows": len(df), "columns": df.columns.tolist()}
    except Exception as e:
        logger.exception("preprocess failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast")
def forecast(req: ForecastRequest):
    """
    Forecast for a product using specified method.
    """
    try:
        df = io.load_csv("data/sample_retail_data.csv", parse_dates=["order_date"])
        df_prod = df[df["product_name"] == req.product_name][["order_date", "quantity", "revenue"]].groupby("order_date").sum().reset_index()
        if req.method == "sarima":
            res = fc.sarima_forecast(df_prod, date_col="order_date", value_col="quantity", periods=req.periods)
            return {"forecast": res["forecast"].to_json(), "model_path": res["model_path"]}
        elif req.method == "prophet":
            res = fc.prophet_forecast(df_prod, date_col="order_date", value_col="revenue", periods=req.periods)
            # serialize small piece
            return {"forecast": res["forecast"].tail(req.periods).to_dict(orient="records"), "model_path": res["model_path"]}
        elif req.method == "lstm":
            s = df_prod.set_index("order_date")["quantity"].asfreq('D').fillna(0)
            res = fc.lstm_forecast(s, periods=req.periods)
            return {"forecast": res["forecast"].tolist(), "model_path": res["model_path"]}
        else:
            raise HTTPException(status_code=400, detail="Unknown method")
    except Exception as e:
        logger.exception("forecast endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
def segment(req: SegmentRequest):
    try:
        df = io.load_csv("data/sample_retail_data.csv", parse_dates=["order_date"])
        seg_df, model = seg.kmeans_segmentation(df, n_clusters=req.n_clusters)
        return {"segments": seg_df.to_dict(orient="records")}
    except Exception as e:
        logger.exception("segment endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/recommendations")
def recommendations():
    try:
        df = io.load_csv("data/sample_retail_data.csv", parse_dates=["order_date"])
        df = pre.create_profit(df)
        profit_df = prof.compute_profitability(df).rename(columns={"product_name": "product"})
        # create demo forecast_change_pct for all products as +0.2
        fc_demo = profit_df.copy()
        fc_demo["product_id"] = fc_demo["product"]
        fc_demo["forecast_change_pct"] = 0.2
        recs = rec.generate_recommendations(df_forecast=fc_demo.rename(columns={"product":"product_id"}), df_profit=profit_df)
        return recs
    except Exception as e:
        logger.exception("recommendations endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

    # (Keep existing content, add the following endpoint to the file)

    # ... existing imports ...

    @app.post("/upload")
    async def upload_csv(file: UploadFile = File(...)):
        """
        Upload a CSV file via multipart form-data. Returns saved path and columns.
        """
        try:
            # Save uploaded file to data/uploads using the modules.io helper
            # Note: UploadFile has .file (a SpooledTemporaryFile) or .read()
            # We will read the bytes and pass along to save_uploaded_file
            content = await file.read()
            saved_path = io.save_uploaded_file(content, save_dir="data/uploads", filename=file.filename)
            # Try to read a small sample to report columns
            try:
                df = io.load_csv(saved_path, parse_dates=["order_date"]) if "order_date" in pd.read_csv(saved_path,
                                                                                                        nrows=1).columns else load_csv(
                    saved_path)
                cols = df.columns.tolist()
            except Exception:
                cols = []
            return {"saved_path": saved_path, "columns": cols}
        except Exception as e:
            logger.exception("upload endpoint failed")
            raise HTTPException(status_code=500, detail=str(e))