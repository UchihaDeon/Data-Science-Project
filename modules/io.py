"""
I/O helpers: load CSV, fetch from API (simple), save DataFrame, and save uploaded files.
"""

from typing import Optional, Union, IO
import pandas as pd
import requests
import logging
import os
from modules.utils import ensure_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_csv(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load CSV from local path.
    """
    if not os.path.exists(path):
        logger.error("CSV file not found: %s", path)
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
        logger.info("Loaded CSV %s shape=%s", path, df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to load CSV: %s", e)
        raise


def fetch_api(url: str, params: Optional[dict] = None, timeout: int = 10) -> pd.DataFrame:
    """
    Fetch JSON data from an API endpoint and convert to DataFrame.
    """
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        logger.info("Fetched API data from %s shape=%s", url, df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to fetch API data: %s", e)
        raise


def save_df_html(df: pd.DataFrame, path: str):
    """
    Save DataFrame as HTML for simple report export.
    """
    try:
        ensure_dir(os.path.dirname(path) or ".")
        df.to_html(path, index=False)
        logger.info("Saved HTML data to %s", path)
    except Exception:
        logger.exception("Failed to save HTML")
        raise


def save_uploaded_file(uploaded: Union[IO, bytes], save_dir: str = "data/uploads", filename: Optional[str] = None) -> str:
    """
    Persist an uploaded file-like object or bytes to disk inside save_dir.
    Works with:
      - Streamlit uploaded file (has .getvalue or .read)
      - FastAPI UploadFile (has .file or .read)
      - raw bytes

    Returns:
        Path to the saved file (string)
    """
    ensure_dir(save_dir)
    try:
        # Determine filename
        if filename is None:
            # Try to extract from file-like object attributes
            fn = None
            if hasattr(uploaded, "name") and isinstance(uploaded.name, str):
                fn = os.path.basename(uploaded.name)
            if hasattr(uploaded, "filename") and isinstance(uploaded.filename, str):
                fn = os.path.basename(uploaded.filename)
            filename = fn or "uploaded_data.csv"
        save_path = os.path.join(save_dir, filename)

        # Read bytes
        content = None
        if isinstance(uploaded, (bytes, bytearray)):
            content = bytes(uploaded)
        elif hasattr(uploaded, "read"):
            # Some file-like objects (FastAPI UploadFile .file)
            try:
                uploaded.seek(0)
            except Exception:
                pass
            content = uploaded.read()
            # If read returned a memoryview or list etc., make bytes
            if isinstance(content, memoryview):
                content = content.tobytes()
        elif hasattr(uploaded, "getvalue"):
            # Streamlit's UploadedFile has getvalue()
            content = uploaded.getvalue()
        else:
            raise TypeError("Unsupported uploaded file type")

        # Write to disk
        mode = "wb"
        with open(save_path, mode) as fh:
            if isinstance(content, str):
                fh.write(content.encode("utf-8"))
            else:
                fh.write(content)
        logger.info("Saved uploaded file to %s", save_path)
        return save_path
    except Exception:
        logger.exception("Failed to save uploaded file")
        raise