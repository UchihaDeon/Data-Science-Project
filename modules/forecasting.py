"""
Forecasting: SARIMA (statsmodels), Prophet, simple LSTM (tensorflow).
Provide function interfaces that accept DataFrame grouped by product/series and forecast N periods.
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    tf = None

from modules.utils import save_model, ensure_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MODEL_DIR = "models"
ensure_dir(MODEL_DIR)


def _ensure_ts(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """
    Prepare a daily-frequency time series (pd.Series) from DataFrame.
    Ensures date_col is datetime, sets it as index, enforces daily frequency and forward-fills missing values.
    """
    if date_col not in df.columns or value_col not in df.columns:
        raise KeyError("date_col or value_col not in df")
    s = df[[date_col, value_col]].dropna().copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    s = s.set_index(date_col)[value_col].astype(float)
    # Ensure index is DatetimeIndex
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.DatetimeIndex(s.index)
    # Reindex to daily frequency and forward-fill (use ffill instead of deprecated fillna(method='ffill'))
    s = s.asfreq("D")
    s = s.ffill()
    return s


def sarima_forecast(df: pd.DataFrame, date_col: str, value_col: str, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), periods: int = 7) -> Dict[str, Any]:
    """
    Train SARIMA and forecast next `periods`. Returns {forecast: pd.Series, model_path: str}
    """
    try:
        s = _ensure_ts(df, date_col, value_col)
        model = SARIMAX(s, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=periods)
        f = pred.predicted_mean
        path = os.path.join(MODEL_DIR, f"sarima_{value_col}.pkl")
        save_model(res, path)
        logger.info("SARIMA trained and saved to %s", path)
        return {"forecast": f, "model_path": path}
    except Exception:
        logger.exception("sarima_forecast failed")
        raise


def prophet_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 7) -> Dict[str, Any]:
    """
    Train Prophet and forecast. Expects df with columns date_col and value_col.
    """
    if Prophet is None:
        raise ImportError("prophet package not available")
    try:
        s = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"}).copy()
        s["ds"] = pd.to_datetime(s["ds"], errors="coerce")
        s = s.dropna(subset=["ds"])
        m = Prophet()
        m.fit(s)
        future = m.make_future_dataframe(periods=periods)
        fcst = m.predict(future)
        path = os.path.join(MODEL_DIR, f"prophet_{value_col}.pkl")
        save_model(m, path)
        logger.info("Prophet trained and saved to %s", path)
        return {"forecast": fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]], "model_path": path}
    except Exception:
        logger.exception("prophet_forecast failed")
        raise


def lstm_forecast(series: pd.Series, lookback: int = 7, epochs: int = 20, batch_size: int = 8, periods: int = 7) -> Dict[str, Any]:
    """
    Very small LSTM forecasting example returning numeric predictions for `periods`.
    """
    if tf is None:
        raise ImportError("TensorFlow required for LSTM")
    try:
        arr = series.values.astype(float)
        if len(arr) <= lookback:
            raise ValueError("Series length must be greater than lookback")
        X, y = [], []
        for i in range(len(arr) - lookback):
            X.append(arr[i:i+lookback])
            y.append(arr[i+lookback])
        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        model = Sequential([LSTM(32, input_shape=(lookback,1)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        # generate future
        inp = arr[-lookback:].tolist()
        preds = []
        for _ in range(periods):
            x = np.array(inp[-lookback:]).reshape(1, lookback, 1)
            p = model.predict(x, verbose=0)[0,0]
            preds.append(p)
            inp.append(p)
        path = os.path.join(MODEL_DIR, f"lstm_series.h5")
        model.save(path)
        logger.info("LSTM trained and saved to %s", path)
        return {"forecast": np.array(preds), "model_path": path}
    except Exception:
        logger.exception("lstm_forecast failed")
        raise