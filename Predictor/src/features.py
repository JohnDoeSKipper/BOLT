"""
Feature engineering for load forecasting.
Input:  normalized DataFrame from data_loader (timestamp, kw_import, kw_export, ...)
Output: DataFrame of features + target ready for LightGBM.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# Lags in units of 30-min steps (the data frequency)
DEFAULT_LAGS = [1, 2, 4, 8, 16, 24, 48, 96, 336]  # 30min, 1h, 2h, 4h, 8h, 12h, 24h, 48h, 1 week


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Cyclical and categorical time-of-day / day-of-week features."""
    out = df.copy()
    ts = out[ts_col]
    out["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_business_hours"] = ((out["hour"] >= 8) & (out["hour"] < 18) & (out["is_weekend"] == 0)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    return out


def add_lag_features(df: pd.DataFrame, target: str = "kw_import", lags: list[int] = None) -> pd.DataFrame:
    """Historical lags — crucial for any time-series forecasting."""
    if lags is None:
        lags = DEFAULT_LAGS
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target].shift(L)
    return out


def add_rolling_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    """Rolling stats give the model a sense of recent trend + volatility."""
    out = df.copy()
    s = out[target].shift(1)
    out["roll_mean_24"]  = s.rolling(24,  min_periods=1).mean()
    out["roll_mean_48"]  = s.rolling(48,  min_periods=1).mean()
    out["roll_std_48"]   = s.rolling(48,  min_periods=1).std().fillna(0)
    out["roll_max_48"]   = s.rolling(48,  min_periods=1).max()
    out["roll_min_48"]   = s.rolling(48,  min_periods=1).min()
    out["roll_mean_336"] = s.rolling(336, min_periods=48).mean()  # 7-day average
    return out


def add_schedule_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    """
    Historical mean and std for each (day-of-week, hour) bucket.
    Gives the model a strong weekly schedule prior — the single most
    predictive signal for long-horizon forecasts on commercial sites.
    """
    out = df.copy()
    hour_bin = out["timestamp"].dt.hour
    dow = out["timestamp"].dt.dayofweek
    # 168 unique keys (7 days × 24 hours)
    out["_sched_key"] = dow * 24 + hour_bin
    out["dow_hour_mean"] = out.groupby("_sched_key")[target].transform("mean")
    out["dow_hour_std"]  = out.groupby("_sched_key")[target].transform("std").fillna(0)
    out.drop(columns=["_sched_key"], inplace=True)
    return out


def add_regime_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    """
    Regime-shift signals: tell the trees how today compares to history.

    Without these, on bimodal sites (some days busy, some quiet), the model's
    median prediction collapses to the historical mean of the dow_hour bucket,
    which is between the two regimes — so it over-predicts on quiet days and
    under-predicts on busy ones. The deltas + recent-vs-schedule give a
    splittable feature that says "we are in a quiet/loud regime right now",
    which the trees can use to pick the appropriate leaf.
    """
    out = df.copy()
    s = out[target]
    out["delta_vs_24h"] = s.shift(1) - s.shift(48)
    out["delta_vs_7d"]  = s.shift(1) - s.shift(336)
    # Recent rolling mean vs the long-run dow_hour baseline.
    # Both inputs are already shifted-1 (rolling) and historical (groupby
    # transform), so no current-row leakage.
    if "roll_mean_24" in out.columns and "dow_hour_mean" in out.columns:
        out["recent_vs_schedule"] = out["roll_mean_24"] - out["dow_hour_mean"]
    return out


# ---- Tropical temperature model (Malaysia) ----------------------------------
# Half-hourly temperature (°C) for a typical Malaysian day.
# Sinusoidal: mean 29 °C, amplitude 4 °C, trough ~06:00 (hh=12), peak ~14:00 (hh=28).
_HH = np.arange(48)
_TROPICAL_TEMP_C = 29.0 + 4.0 * np.sin(2 * np.pi * (_HH - 12) / 48.0)


def estimated_temp_c(ts: pd.Series) -> pd.Series:
    """Synthetic ambient temperature (°C) based on time-of-day for tropical Malaysia."""
    hh = (ts.dt.hour * 2 + ts.dt.minute // 30).astype(int) % 48
    return pd.Series(_TROPICAL_TEMP_C[hh.values], index=ts.index)


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add estimated temperature and a 'feels hot' flag as features."""
    out = df.copy()
    out["est_temp_c"] = estimated_temp_c(out["timestamp"])
    out["is_hot_period"] = (out["est_temp_c"] > 31.0).astype(int)
    return out


# ---- Solar generation model -------------------------------------------------
_SOLAR_CURVE_PER_HH = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.01, 0.03,
    0.06, 0.09, 0.13, 0.17, 0.20, 0.22, 0.24, 0.25,
    0.25, 0.24, 0.23, 0.21,
    0.19, 0.16, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=float)
_SOLAR_CURVE_PER_HH = _SOLAR_CURVE_PER_HH * (4.0 / _SOLAR_CURVE_PER_HH.sum())


def estimated_solar_gen_kw(ts: pd.Series, capacity_kwp: float) -> pd.Series:
    """Estimate instantaneous solar kW output at each timestamp given installed capacity."""
    hh = (ts.dt.hour * 2 + ts.dt.minute // 30).astype(int)
    kwh_per_hh = pd.Series(_SOLAR_CURVE_PER_HH[hh.values], index=ts.index)
    return kwh_per_hh * capacity_kwp * 2.0


def add_solar_features(df: pd.DataFrame, capacity_kwp: float) -> pd.DataFrame:
    """Append solar-related features (zeroes out if capacity_kwp == 0)."""
    out = df.copy()
    out["solar_capacity_kwp"] = float(capacity_kwp)
    out["est_solar_gen_kw"] = estimated_solar_gen_kw(out["timestamp"], capacity_kwp)
    out["has_solar"] = int(capacity_kwp > 0)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    capacity_kwp: float = 0.0,
    target: str = "kw_import",
) -> tuple[pd.DataFrame, list[str]]:
    """One-call pipeline."""
    out = add_time_features(df)
    out = add_lag_features(out, target=target)
    out = add_rolling_features(out, target=target)
    out = add_schedule_features(out, target=target)
    # add_regime_features depends on roll_mean_24 + dow_hour_mean already
    # being present, so it must run after those two.
    out = add_regime_features(out, target=target)
    out = add_solar_features(out, capacity_kwp=capacity_kwp)
    out = add_temperature_features(out)

    feature_cols = [
        "hour", "dow", "month", "is_weekend", "is_business_hours",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        *[f"lag_{L}" for L in DEFAULT_LAGS],
        "roll_mean_24", "roll_mean_48", "roll_std_48", "roll_max_48", "roll_min_48",
        "roll_mean_336",
        "dow_hour_mean", "dow_hour_std",
        "delta_vs_24h", "delta_vs_7d", "recent_vs_schedule",
        "solar_capacity_kwp", "est_solar_gen_kw", "has_solar",
        "est_temp_c", "is_hot_period",
    ]
    out = out.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    return out, feature_cols
