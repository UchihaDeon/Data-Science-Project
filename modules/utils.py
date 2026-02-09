"""
Utility helpers for filesystem-only persistence and small helpers.
"""
from pathlib import Path
import json
import hashlib
import os


def ensure_dir(path: str):
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_file_checksum(path: str) -> str:
    """
    Return an MD5 checksum (hex) of the file bytes.
    Falls back to hashing the path string if reading fails.
    """
    try:
        b = Path(path).read_bytes()
        return hashlib.md5(b).hexdigest()
    except Exception:
        return hashlib.md5(str(path).encode("utf-8")).hexdigest()


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def write_json(path: str, data):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)