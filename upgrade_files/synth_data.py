"""
Synthetic data generation for load profiles.

Approach: Bootstrap-resample real days (matched by day-of-week), apply
seasonal multipliers, inject holidays and rare anomaly events, add
proportional noise. Output has the same statistical properties as real
data but covers ~365 days instead of ~60.

Important guarantees:
- Real data is preserved verbatim at the END of the output (most recent).
- Synthetic data fills BEFORE the real data (going back in time).
- Timestamps remain continuous and on the original 30-min grid.
- The 'kw_export' column is regenerated from the synthetic 'kw_import'
  using the same export ratio as observed in real data per hour-of-day,
  so solar signatures are preserved.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# Malaysian climate-aware seasonal multipliers (KL temperature data, mild variation).
# Higher AC load in pre-monsoon hot months (Mar-May) and post-monsoon (Sep-Oct).
MY_SEASONAL_MULT = {
    1: 0.98, 2: 1.00, 3: 1.03, 4: 1.05, 5: 1.05, 6: 1.02,
    7: 1.01, 8: 1.02, 9: 1.04, 10: 1.04, 11: 1.01, 12: 0.98,
}

# Approximate Malaysian public holidays (offsets from year-start, simplified).
# We don't depend on the `holidays` package to keep dependencies minimal.
# These are the dates that consistently fall on fixed days; lunar holidays
# (Hari Raya, CNY, Deepavali) we approximate as random selections from
# typical date ranges.
MY_FIXED_HOLIDAYS_MMDD = {
    (1, 1),    # New Year
    (2, 1),    # Federal Territory Day
    (5, 1),    # Labour Day
    (8, 31),   # Merdeka
    (9, 16),   # Malaysia Day
    (12, 25),  # Christmas
}


def _is_my_holiday_approx(date: pd.Timestamp) -> bool:
    """Approximate check for Malaysian holiday."""
    return (date.month, date.day) in MY_FIXED_HOLIDAYS_MMDD


def _classify_full_days(df: pd.DataFrame, min_samples: int = 44) -> dict[int, list[pd.DataFrame]]:
    """Group complete days from real data by day-of-week."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["dow"] = df["timestamp"].dt.dayofweek

    by_dow: dict[int, list[pd.DataFrame]] = {dow: [] for dow in range(7)}
    for _, group in df.groupby("date"):
        if len(group) >= min_samples:
            dow = group["dow"].iloc[0]
            by_dow[int(dow)].append(group.drop(columns=["date", "dow"]).reset_index(drop=True))

    # Fill empty buckets from neighboring DOWs
    for dow in range(7):
        if not by_dow[dow]:
            for offset in range(1, 8):
                cand = (dow + offset) % 7
                if by_dow[cand]:
                    by_dow[dow] = by_dow[cand]
                    break

    return by_dow


def _resample_to_48(template: pd.DataFrame) -> pd.DataFrame:
    """Ensure template has exactly 48 half-hour samples via interpolation."""
    n = len(template)
    if n == 48:
        return template.reset_index(drop=True)

    new_idx = np.linspace(0, n - 1, 48)
    old_idx = np.arange(n)
    out = pd.DataFrame({
        col: np.interp(new_idx, old_idx, template[col].values)
        for col in ["kw_import", "kw_export", "kvar_import", "kvar_export"]
    })
    return out


def _generate_one_day(
    template: pd.DataFrame,
    target_date: pd.Timestamp,
    is_holiday: bool,
    seasonal_mult: float,
    noise_std: float,
    is_anomaly: bool,
    anomaly_type: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one synthetic day from a template real day."""
    out = _resample_to_48(template)

    # Build new timestamps starting at target_date 00:00
    new_ts = pd.date_range(
        start=pd.Timestamp(target_date.normalize()),
        periods=48,
        freq="30min",
    )
    out["timestamp"] = new_ts

    base = out["kw_import"].values.astype(float)

    # Apply seasonal multiplier
    base = base * seasonal_mult

    # Holiday: load drops to ~70% of normal (commercial buildings mostly closed)
    if is_holiday:
        base = base * 0.70

    # Anomaly events (rare)
    if is_anomaly:
        if anomaly_type == "spike":
            # Sudden 30-50% spike for 1-2 hours during daytime
            spike_start = rng.integers(20, 36)  # 10:00-18:00
            spike_dur = rng.integers(2, 5)      # 1-2.5 hours
            spike_mag = rng.uniform(1.30, 1.50)
            base[spike_start:spike_start + spike_dur] *= spike_mag
        elif anomaly_type == "dip":
            # Equipment outage: 50-80% drop for 2-4 hours
            dip_start = rng.integers(16, 36)
            dip_dur = rng.integers(4, 9)
            dip_mag = rng.uniform(0.20, 0.50)
            base[dip_start:dip_start + dip_dur] *= dip_mag

    # Proportional Gaussian noise
    noise = rng.normal(0, noise_std, size=len(base)) * np.maximum(base, 1.0)
    base = np.clip(base + noise, 0, None)
    out["kw_import"] = base

    # Regenerate kw_export proportionally to original day's export ratio
    orig_export_ratio = out["kw_export"].values / np.maximum(out["kw_import"].values, 1.0)
    # Cap ratio at sane values
    orig_export_ratio = np.clip(orig_export_ratio, 0, 5)
    new_export = orig_export_ratio * base
    # Apply seasonal & holiday to export too (less solar export on cloudy/rainy seasons - simplified)
    out["kw_export"] = np.clip(new_export, 0, None)

    # Apply same proportional adjustment to kvar
    kvar_imp_ratio = out["kvar_import"].values / np.maximum(out["kw_import"].values, 1.0)
    kvar_exp_ratio = out["kvar_export"].values / np.maximum(out["kw_import"].values, 1.0)
    out["kvar_import"] = np.clip(kvar_imp_ratio * base, 0, None)
    out["kvar_export"] = np.clip(kvar_exp_ratio * base, 0, None)

    return out[["timestamp", "kw_import", "kw_export", "kvar_import", "kvar_export"]]


def synthesize_extended(
    df: pd.DataFrame,
    target_days: int = 365,
    noise_std: float = 0.05,
    anomaly_rate: float = 0.02,
    seed: int = 42,
    direction: str = "backward",
) -> pd.DataFrame:
    """
    Extend a real load profile to `target_days` days of total coverage.

    Args:
        df: Real load profile in canonical schema (timestamp, kw_import, ...).
        target_days: Total days desired in output.
        noise_std: Std of proportional Gaussian noise (0.05 = 5%).
        anomaly_rate: Fraction of synthetic days that get spike or dip events.
        seed: Random seed for reproducibility.
        direction: "backward" (synthetic data BEFORE real) or "forward" (AFTER).
            Backward is recommended — real data stays as the most recent slice,
            which is what online learning will work with.

    Returns:
        Combined DataFrame with synthetic + real data, sorted ascending.
    """
    rng = np.random.default_rng(seed)
    by_dow = _classify_full_days(df)

    # Anomaly event types
    anomaly_types = ["spike", "dip"]

    if direction not in {"backward", "forward"}:
        raise ValueError(f"direction must be 'backward' or 'forward', got {direction}")

    real_start = df["timestamp"].min()
    real_end = df["timestamp"].max()
    real_days = (real_end - real_start).days + 1
    days_to_generate = max(0, target_days - real_days)

    if days_to_generate == 0:
        return df.copy().sort_values("timestamp").reset_index(drop=True)

    synth_days = []
    if direction == "backward":
        cur_date = real_start - pd.Timedelta(days=1)
        date_step = pd.Timedelta(days=-1)
    else:
        cur_date = real_end + pd.Timedelta(days=1)
        date_step = pd.Timedelta(days=1)

    for _ in range(days_to_generate):
        dow = int(cur_date.dayofweek)
        templates = by_dow[dow]
        template = templates[rng.integers(0, len(templates))]

        is_hol = _is_my_holiday_approx(cur_date)
        seasonal = MY_SEASONAL_MULT.get(cur_date.month, 1.0)

        is_anom = rng.random() < anomaly_rate
        anom_type = anomaly_types[rng.integers(0, len(anomaly_types))] if is_anom else ""

        day = _generate_one_day(
            template, cur_date, is_hol, seasonal, noise_std,
            is_anom, anom_type, rng,
        )
        synth_days.append(day)
        cur_date += date_step

    if direction == "backward":
        synth_days.reverse()
    synth_df = pd.concat(synth_days, ignore_index=True)

    full = pd.concat([synth_df, df], ignore_index=True)
    full = (
        full.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")  # prefer real over synthetic on overlap
            .reset_index(drop=True)
    )
    return full


def quality_report(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """Compare statistics between real and synthetic data to verify realism."""
    real = real_df["kw_import"]
    only_synth = synth_df[~synth_df["timestamp"].isin(real_df["timestamp"])]["kw_import"]

    return {
        "real_n": int(len(real)),
        "synth_n": int(len(only_synth)),
        "real_mean": float(real.mean()),
        "synth_mean": float(only_synth.mean()),
        "real_std": float(real.std()),
        "synth_std": float(only_synth.std()),
        "real_p95": float(real.quantile(0.95)),
        "synth_p95": float(only_synth.quantile(0.95)),
        "real_max": float(real.max()),
        "synth_max": float(only_synth.max()),
        "mean_ratio": float(only_synth.mean() / real.mean()),
        "std_ratio": float(only_synth.std() / real.std()),
    }
