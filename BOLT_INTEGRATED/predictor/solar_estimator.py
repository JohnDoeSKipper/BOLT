"""
Estimate installed solar PV capacity (kWp) from a load profile alone.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NO_SOLAR_MIDDAY_NIGHT_RATIO = 1.10
SOLAR_PEAK_CURVE_FRACTION = 0.50


def detect_has_solar(df: pd.DataFrame) -> tuple[bool, str]:
    if (df["kw_export"] > 0).any():
        return True, "Non-zero kW Export detected (power flowing back to grid)."
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    night = d.loc[(d["hour"] >= 1) & (d["hour"] <= 4), "kw_import"].mean()
    midday = d.loc[(d["hour"] >= 12) & (d["hour"] <= 14), "kw_import"].mean()
    morning = d.loc[(d["hour"] >= 9) & (d["hour"] <= 11), "kw_import"].mean()
    if np.isfinite(midday) and np.isfinite(morning) and midday < morning * 0.9:
        return True, f"Midday dip detected: {midday:.0f} kW vs morning {morning:.0f} kW."
    return False, "No solar signature detected in load profile."


def estimate_solar_capacity_kwp(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    night = d.loc[(d["hour"] >= 1) & (d["hour"] <= 4), "kw_import"].mean()
    midday = d.loc[(d["hour"] >= 11) & (d["hour"] <= 13), "kw_import"].mean()
    morning = d.loc[(d["hour"] >= 9) & (d["hour"] <= 10), "kw_import"].mean()
    max_export = d["kw_export"].max()
    export_based = float(max_export) * 1.3 if max_export > 5 else 0.0
    expected_no_solar_midday = max(morning, night * NO_SOLAR_MIDDAY_NIGHT_RATIO)
    solar_kw = max(0.0, expected_no_solar_midday - midday)
    load_based = solar_kw / SOLAR_PEAK_CURVE_FRACTION if solar_kw > 0 else 0.0
    estimate = max(export_based, load_based)
    estimate = round(estimate / 10) * 10
    return {
        "capacity_kwp": float(estimate),
        "night_baseline_kw": round(float(night), 1),
        "midday_observed_kw": round(float(midday), 1),
        "morning_observed_kw": round(float(morning), 1),
        "max_export_kw": round(float(max_export), 1),
        "export_based_estimate_kwp": round(export_based, 0),
        "load_based_estimate_kwp": round(load_based, 0),
        "method": "combined_load_and_export_heuristic",
    }
