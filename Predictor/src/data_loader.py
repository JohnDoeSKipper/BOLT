"""
Data loader that normalizes all 4 competition Excel formats into a common schema:
    columns = ['timestamp', 'kw_import', 'kw_export', 'kvar_import', 'kvar_export']
    index   = timestamp (30-min frequency, sorted ascending)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


CANONICAL_COLS = ["timestamp", "kw_import", "kw_export", "kvar_import", "kvar_export"]


def _clean_col(c: str) -> str:
    """Normalise a column name: strip BOM, whitespace, lowercase, underscores."""
    return c.strip().lstrip("﻿").strip().lower().replace(" ", "_")


def _standardize(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Lowercase columns, rename, sort, de-dup."""
    df = df.rename(columns={c: _clean_col(c) for c in df.columns})
    # Map common variants
    rename_map = {
        "date_/_end_time": "timestamp",
        "end_time": "timestamp",
        ts_col.lower(): "timestamp",
    }
    df = df.rename(columns=rename_map)
    # Ensure we have the core columns (fill missing with 0)
    for col in ["kw_import", "kw_export", "kvar_import", "kvar_export"]:
        if col not in df.columns:
            df[col] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[CANONICAL_COLS].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    # Coerce numerics
    for col in CANONICAL_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_sol_format(path: str | Path) -> pd.DataFrame:
    """SoL file: two sheets, Sept has a 'Solar Installed' header row, Okt doesn't."""
    parts = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        # Find the row containing 'Date' — that's our header
        header_row = None
        for i in range(min(5, len(raw))):
            if raw.iloc[i].astype(str).str.contains("Date", case=False, na=False).any():
                header_row = i
                break
        if header_row is None:
            continue
        df = pd.read_excel(path, sheet_name=sheet, header=header_row)
        df = _standardize(df, "Date / End Time")
        parts.append(df)
    return pd.concat(parts, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def load_e_format(path: str | Path) -> pd.DataFrame:
    """E file: clean header on row 0, lowercase columns, reverse chronological order."""
    df = pd.read_excel(path, header=0)
    # 'end_time' is the timestamp we want (what kW was measured during the interval ending at that time)
    return _standardize(df, "end_time")


def load_sun_or_mi2_format(path: str | Path) -> pd.DataFrame:
    """SuN and Mi2 files: 'Meter Type' row on top, header on row 1."""
    df = pd.read_excel(path, header=1)
    return _standardize(df, "Date / End Time")


def auto_load(path: str | Path) -> pd.DataFrame:
    """
    Auto-detect format and load. Useful for the Streamlit file uploader:
    the user just drops any of the 4 files and we figure out which format.
    """
    path = Path(path)
    # Peek at sheet names and first few rows
    xl = pd.ExcelFile(path)
    first_sheet = xl.sheet_names[0]
    peek = pd.read_excel(path, sheet_name=first_sheet, header=None, nrows=3)

    # Heuristic 1: has 'start_time' and 'end_time' columns → E format
    header0 = peek.iloc[0].astype(str).str.lower().tolist()
    if "start_time" in header0 and "end_time" in header0:
        return load_e_format(path)

    # Heuristic 2: has multiple sheets matching 'Sept', 'Okt', 'Jan' pattern → SoL format
    if len(xl.sheet_names) > 1:
        return load_sol_format(path)

    # Heuristic 3: first row contains 'Meter Type' → SuN/Mi2 format
    if "meter type" in " ".join(str(v).lower() for v in peek.iloc[0].values):
        return load_sun_or_mi2_format(path)

    # Fallback: try SoL loader (most permissive)
    try:
        return load_sol_format(path)
    except Exception:
        return load_sun_or_mi2_format(path)


def load_csv(path: str | Path) -> pd.DataFrame:
    """For users feeding in a plain CSV of new data during live demo."""
    # utf-8-sig strips the BOM that Excel on Windows adds to the first column name
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Try to find the timestamp column (search before renaming so we match the raw name)
    ts_col = None
    for c in df.columns:
        if _clean_col(c) in {"timestamp", "date", "datetime", "end_time", "date_/_end_time"}:
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]  # best guess: first column
    df = _standardize(df, _clean_col(ts_col))
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Quick stats for display in the UI."""
    return {
        "rows": len(df),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "days": (df["timestamp"].max() - df["timestamp"].min()).days,
        "mean_kw_import": float(df["kw_import"].mean()),
        "max_kw_import": float(df["kw_import"].max()),
        "has_export": bool((df["kw_export"] > 0).any()),
    }