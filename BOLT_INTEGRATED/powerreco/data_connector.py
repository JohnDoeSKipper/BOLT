"""
Data connector for PowerRECO.
Parses and normalises outputs from the BOLT Manager and Predictor.
"""
from __future__ import annotations
import json
import io
import numpy as np
import pandas as pd


def parse_manager_output(file_obj) -> pd.DataFrame:
    """
    Load BOLT Manager optimisation results into a normalised DataFrame.
    Accepts CSV, Excel, or JSON (list-of-records or dict with a results key).
    """
    name = getattr(file_obj, "name", "").lower()

    if name.endswith(".json"):
        raw = json.load(file_obj)
        df = _json_to_df(raw)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        df = _read_csv_flexible(file_obj)

    df = _normalise_columns(df)
    df = _parse_timestamp(df)
    return df


def _json_to_df(raw) -> pd.DataFrame:
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict):
        for key in ("results", "intervals", "data", "log", "actions", "records"):
            if key in raw and isinstance(raw[key], list):
                return pd.DataFrame(raw[key])
        try:
            return pd.DataFrame(raw)
        except Exception:
            return pd.json_normalize(raw)
    raise ValueError("JSON structure not recognised – expected a list or dict.")


def _read_csv_flexible(file_obj) -> pd.DataFrame:
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(io.StringIO(content))


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
                    .replace("(", "").replace(")", "")
        for c in df.columns
    ]
    return df


def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("timestamp", "datetime", "time", "interval", "period", "date"):
        if col in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df[col])
            except Exception:
                pass
            break
    return df


def parse_predictor_output(file_obj) -> pd.DataFrame:
    """
    Load BOLT Predictor forecast/actuals into a normalised DataFrame.
    Expects columns: timestamp, kw_import, and optionally median/p10/p90.
    """
    name = getattr(file_obj, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        df = _read_csv_flexible(file_obj)
    df = _normalise_columns(df)
    df = _parse_timestamp(df)
    return df


def extract_peak_demand_stats(df: pd.DataFrame) -> dict:
    """Summarise demand statistics from Predictor data."""
    kw_col = None
    for col in ("kw_import", "kw", "power_kw", "demand_kw", "median", "forecast_kw", "load_kw"):
        if col in df.columns:
            kw_col = col
            break

    if kw_col is None:
        return {}

    series = pd.to_numeric(df[kw_col], errors="coerce").dropna()
    if series.empty:
        return {}

    n_top = max(1, int(len(series) * 0.05))
    top_vals = series.nlargest(n_top)

    return {
        "kw_col_used":    kw_col,
        "peak_kw":        round(float(series.max()), 1),
        "avg_top5pct_kw": round(float(top_vals.mean()), 1),
        "p95_kw":         round(float(series.quantile(0.95)), 1),
        "mean_kw":        round(float(series.mean()), 1),
        "n_intervals":    int(len(series)),
    }
