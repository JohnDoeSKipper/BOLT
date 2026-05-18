"""
Battery sizing engine for PowerRECO.
Derives minimum battery capacity from the BOLT Manager's optimization output.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# Battery operating parameters (LFP chemistry)
SOC_MIN = 0.20          # 20% lower cut-off
SOC_MAX = 0.80          # 80% upper cut-off
USABLE_DOD = SOC_MAX - SOC_MIN          # 60% usable window
UPSIZE_FACTOR = 1 / USABLE_DOD         # ×1.6667 — always applied
ROUNDTRIP_EFF = 0.90    # 90% round-trip efficiency (charge + discharge losses)

# Spike protection: size to average of top-N% demand days, not the single peak.
# This avoids grossly oversizing for a rare one-off anomaly.
TOP_PCT = 0.05          # top 5% of days → average used for sizing

# Commercial LFP battery bank sizes (kWh) available in Malaysian market
_COMMERCIAL_SIZES = [
    10, 15, 20, 25, 30, 40, 50, 60, 75, 100,
    120, 150, 200, 250, 300, 400, 500,
]


def calculate_battery_sizing(
    manager_df: pd.DataFrame,
    interval_minutes: int = 30,
) -> dict:
    """
    Calculate minimum battery capacity from BOLT Manager optimization output.

    The Manager outputs per-interval results showing original vs optimised kVA
    (or explicit battery discharge amounts).  PowerRECO derives the daily energy
    demand the battery must cover, applies the 60% DOD upsize factor, and rounds
    to the nearest commercial bank size.

    Spike protection: instead of sizing to the single worst day, the engine
    takes the average of the top 5% of highest-demand days, which eliminates
    the effect of rare anomalous spikes.

    Args:
        manager_df:        DataFrame produced by parse_manager_output().
        interval_minutes:  Duration of each interval (30 min for TNB MD window).

    Returns:
        dict with all sizing parameters, methodology breakdown, and recommendation.
    """
    interval_h = interval_minutes / 60.0

    # ── 1. Resolve the energy discharged per interval ─────────────────────────
    discharge_series, discharge_col = _resolve_discharge(manager_df, interval_h)

    # ── 2. Group by date → daily totals ───────────────────────────────────────
    ts_col = _find_ts_col(manager_df)
    if ts_col:
        df_work = manager_df.copy()
        df_work["_date"] = pd.to_datetime(df_work[ts_col]).dt.date
        df_work["_discharge_kwh"] = discharge_series.values
        daily = df_work.groupby("_date")["_discharge_kwh"].sum()
    else:
        # No timestamp column — treat the entire upload as one representative day
        daily = pd.Series([float(discharge_series.sum())])

    n_days = int(len(daily))

    # ── 3. Spike protection: average of top-N days ────────────────────────────
    n_top = max(1, math.ceil(n_days * TOP_PCT))
    top_days = daily.nlargest(max(n_top, min(3, n_days)))
    avg_top_kwh = float(top_days.mean())
    peak_kwh = float(daily.max())

    # ── 4. Account for round-trip losses and SOC window ──────────────────────
    # We need `avg_top_kwh` delivered to the load; the battery must store more.
    required_usable_kwh = avg_top_kwh / ROUNDTRIP_EFF
    min_capacity_kwh = required_usable_kwh * UPSIZE_FACTOR
    commercial_kwh = _round_to_commercial(min_capacity_kwh)

    # ── 5. MD reduction achieved (p95 of per-interval reduction) ─────────────
    md_kw = _estimate_md_reduction(manager_df, interval_h)

    return {
        "n_days_analyzed": n_days,
        "peak_daily_discharge_kwh": round(peak_kwh, 2),
        "n_top_days_used": int(len(top_days)),
        "avg_top_discharge_kwh": round(avg_top_kwh, 2),
        "roundtrip_eff_pct": ROUNDTRIP_EFF * 100,
        "required_usable_kwh": round(required_usable_kwh, 2),
        "usable_dod_pct": USABLE_DOD * 100,
        "soc_range": f"{int(SOC_MIN*100)}%–{int(SOC_MAX*100)}%",
        "upsize_factor": round(UPSIZE_FACTOR, 4),
        "min_capacity_kwh": round(min_capacity_kwh, 2),
        "min_capacity_kwh_commercial": commercial_kwh,
        "md_reduction_kw": round(md_kw, 1),
        "discharge_col_used": discharge_col,
        "spike_note": (
            f"Sized to average of top {len(top_days)} highest-discharge days "
            f"({avg_top_kwh:.1f} kWh), not to the single peak ({peak_kwh:.1f} kWh)."
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_discharge(
    df: pd.DataFrame, interval_h: float
) -> tuple[pd.Series, str]:
    """
    Return (series of kWh discharged per interval, column name used).
    Tries explicit discharge columns first, then infers from before/after kVA.
    """
    for col in ["battery_discharged_kwh", "discharge_kwh", "energy_discharged_kwh",
                "batt_discharge_kwh", "kwh_discharged"]:
        if col in df.columns:
            return df[col].clip(lower=0).fillna(0), col

    # Infer kW reduction from before/after columns, then convert to kWh
    for before, after in [
        ("original_kva", "optimized_kva"),
        ("kva_before", "kva_after"),
        ("kw_before", "kw_after"),
        ("demand_before_kw", "demand_after_kw"),
    ]:
        if before in df.columns and after in df.columns:
            # kVA difference → kW (assume PF ≈ 0.95 for industrial)
            kva_reduction = (df[before] - df[after]).clip(lower=0).fillna(0)
            kw_series = kva_reduction * 0.95
            kwh_series = kw_series * interval_h
            return kwh_series, f"derived({before}–{after})"

    # Direct kW reduction column
    for col in ["kw_reduction", "load_reduced_kw", "reduction_kw",
                "delta_kw", "action_kw", "kva_reduction"]:
        if col in df.columns:
            kwh_series = df[col].clip(lower=0).fillna(0) * interval_h
            return kwh_series, col

    raise ValueError(
        "PowerRECO cannot find discharge data in the Manager output. "
        "Expected columns such as: battery_discharged_kwh, original_kva + optimized_kva, "
        "kw_reduction, etc. Check that you are uploading the Manager's results export."
    )


def _find_ts_col(df: pd.DataFrame) -> str | None:
    for col in ["timestamp", "datetime", "time", "interval", "period", "date"]:
        if col in df.columns:
            return col
    return None


def _estimate_md_reduction(df: pd.DataFrame, interval_h: float) -> float:
    """Best estimate of peak demand reduction (kW) achieved by the battery."""
    for before, after in [
        ("original_kva", "optimized_kva"),
        ("kva_before", "kva_after"),
        ("kw_before", "kw_after"),
    ]:
        if before in df.columns and after in df.columns:
            reduction = (df[before] - df[after]).clip(lower=0)
            # Use 95th percentile of non-zero reductions as the MD saving
            nonzero = reduction[reduction > 0]
            if len(nonzero):
                return float(nonzero.quantile(0.95)) * 0.95  # convert kVA→kW
    for col in ["kw_reduction", "reduction_kw", "md_reduction_kw"]:
        if col in df.columns:
            nonzero = df[col][df[col] > 0]
            if len(nonzero):
                return float(nonzero.quantile(0.95))
    return 0.0


def _round_to_commercial(kwh: float) -> float:
    """Round up to the nearest commercially available LFP battery bank size."""
    for size in _COMMERCIAL_SIZES:
        if kwh <= size:
            return float(size)
    return float(math.ceil(kwh / 50) * 50)
