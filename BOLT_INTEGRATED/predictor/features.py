"""
Feature engineering for load forecasting.
Input:  normalized DataFrame from data_loader (timestamp, kw_import, kw_export, ...)
Output: DataFrame of features + target ready for LightGBM.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


DEFAULT_LAGS = [1, 2, 4, 8, 16, 24, 48, 96, 336]


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
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
    if lags is None:
        lags = DEFAULT_LAGS
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target].shift(L)
    return out


def add_rolling_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    out = df.copy()
    s = out[target].shift(1)
    out["roll_mean_24"]  = s.rolling(24,  min_periods=1).mean()
    out["roll_mean_48"]  = s.rolling(48,  min_periods=1).mean()
    out["roll_std_48"]   = s.rolling(48,  min_periods=1).std().fillna(0)
    out["roll_max_48"]   = s.rolling(48,  min_periods=1).max()
    out["roll_min_48"]   = s.rolling(48,  min_periods=1).min()
    out["roll_mean_336"] = s.rolling(336, min_periods=48).mean()
    return out


def add_schedule_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    out = df.copy()
    hour_bin = out["timestamp"].dt.hour
    dow = out["timestamp"].dt.dayofweek
    out["_sched_key"] = dow * 24 + hour_bin
    out["dow_hour_mean"] = out.groupby("_sched_key")[target].transform("mean")
    out["dow_hour_std"]  = out.groupby("_sched_key")[target].transform("std").fillna(0)
    out.drop(columns=["_sched_key"], inplace=True)
    return out


def add_regime_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    out = df.copy()
    s = out[target]
    out["delta_vs_24h"] = s.shift(1) - s.shift(48)
    out["delta_vs_7d"]  = s.shift(1) - s.shift(336)
    if "roll_mean_24" in out.columns and "dow_hour_mean" in out.columns:
        out["recent_vs_schedule"] = out["roll_mean_24"] - out["dow_hour_mean"]
    return out


_HH = np.arange(48)
_TROPICAL_TEMP_C = 29.0 + 4.0 * np.sin(2 * np.pi * (_HH - 12) / 48.0)


def estimated_temp_c(ts: pd.Series) -> pd.Series:
    hh = (ts.dt.hour * 2 + ts.dt.minute // 30).astype(int) % 48
    return pd.Series(_TROPICAL_TEMP_C[hh.values], index=ts.index)


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["est_temp_c"] = estimated_temp_c(out["timestamp"])
    out["is_hot_period"] = (out["est_temp_c"] > 31.0).astype(int)
    return out


_SOLAR_CURVE_PER_HH = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.01, 0.03, 0.06, 0.09, 0.13, 0.17, 0.20, 0.22, 0.24, 0.25,
    0.25, 0.24, 0.23, 0.21, 0.19, 0.16, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=float)
_SOLAR_CURVE_PER_HH = _SOLAR_CURVE_PER_HH * (4.0 / _SOLAR_CURVE_PER_HH.sum())


def estimated_solar_gen_kw(ts: pd.Series, capacity_kwp: float) -> pd.Series:
    hh = (ts.dt.hour * 2 + ts.dt.minute // 30).astype(int)
    kwh_per_hh = pd.Series(_SOLAR_CURVE_PER_HH[hh.values], index=ts.index)
    return kwh_per_hh * capacity_kwp * 2.0


def add_solar_features(df: pd.DataFrame, capacity_kwp: float) -> pd.DataFrame:
    out = df.copy()
    out["solar_capacity_kwp"] = float(capacity_kwp)
    out["est_solar_gen_kw"] = estimated_solar_gen_kw(out["timestamp"], capacity_kwp)
    out["has_solar"] = int(capacity_kwp > 0)
    return out


def build_feature_matrix(df: pd.DataFrame, capacity_kwp: float = 0.0,
                          target: str = "kw_import") -> tuple[pd.DataFrame, list[str]]:
    out = add_time_features(df)
    out = add_lag_features(out, target=target)
    out = add_rolling_features(out, target=target)
    out = add_schedule_features(out, target=target)
    out = add_regime_features(out, target=target)
    out = add_solar_features(out, capacity_kwp=capacity_kwp)
    out = add_temperature_features(out)

    feature_cols = [
        "hour", "dow", "month", "is_weekend", "is_business_hours",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        *[f"lag_{L}" for L in DEFAULT_LAGS],
        "roll_mean_24", "roll_mean_48", "roll_std_48", "roll_max_48", "roll_min_48",
        "roll_mean_336", "dow_hour_mean", "dow_hour_std",
        "delta_vs_24h", "delta_vs_7d", "recent_vs_schedule",
        "solar_capacity_kwp", "est_solar_gen_kw", "has_solar",
        "est_temp_c", "is_hot_period",
    ]
    out = out.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    return out, feature_cols
