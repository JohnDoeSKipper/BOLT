"""
Data bridge — converts outputs from each BOLT module into inputs for the next.

Pipeline flow:
  Predictor (MultiStepForecastResult)
    → forecast_to_manager_df()
    → Manager (run_ai_manager)
    → manager_results_to_sam_df()      → Calculator (calculate_bill)
    → manager_results_to_powerreco_df() → PowerRECO (calculate_battery_sizing, calculate_roi)

ASSUMPTIONS / FABRICATIONS (documented):
  1. Power factor: kVAR estimated as kW * tan(arccos(PF)) where PF=0.85 is the
     default for commercial/industrial loads when kVAR is not available from history.
     If historical data is provided, PF is derived from the actual mean(kVAR)/mean(kW).
  2. kw_export in forecast output is 0 unless solar capacity is known. Solar generation
     is NOT simulated in the bridge; pass solar-corrected history to get export values.
  3. battery_discharged_kwh uses 30-min intervals (interval_h=0.5).
"""
from __future__ import annotations
import math
import io
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictor.forecaster import MultiStepForecastResult

DEFAULT_PF = 0.85  # assumed commercial/industrial power factor


def _pf_to_tan(pf: float) -> float:
    """tan(arccos(PF)) — ratio of kVAR to kW."""
    pf = max(0.01, min(0.9999, pf))
    return math.tan(math.acos(pf))


def _derive_pf_from_history(historical_df: pd.DataFrame) -> float:
    """
    Estimate power factor from historical kW and kVAR columns.
    Falls back to DEFAULT_PF if data is missing or degenerate.
    """
    if historical_df is None:
        return DEFAULT_PF
    kw_col   = next((c for c in ("kw_import", "kw_net") if c in historical_df.columns), None)
    kvar_col = next((c for c in ("kvar_import", "kvar_net") if c in historical_df.columns), None)
    if kw_col is None or kvar_col is None:
        return DEFAULT_PF
    kw   = pd.to_numeric(historical_df[kw_col],   errors="coerce").dropna()
    kvar = pd.to_numeric(historical_df[kvar_col], errors="coerce").dropna()
    if len(kw) == 0 or kw.mean() < 1.0:
        return DEFAULT_PF
    # Apparent power from kW and kVAR → PF = mean(kW / kVA)
    kva  = np.sqrt(kw ** 2 + kvar.reindex(kw.index).fillna(0) ** 2)
    mask = kva > 1.0
    if mask.sum() == 0:
        return DEFAULT_PF
    pf = float((kw[mask] / kva[mask]).mean())
    return max(0.5, min(0.99, pf))


def forecast_to_manager_df(
    forecast_result: "MultiStepForecastResult",
    historical_df: pd.DataFrame | None = None,
    assumed_pf: float | None = None,
) -> pd.DataFrame:
    """
    Convert a MultiStepForecastResult into the DataFrame format expected by
    run_ai_manager() (columns: timestamp, kw_import, kw_export, kvar_import,
    kvar_export, kw_net, kvar_net, kva).

    Power factor handling:
      - If historical_df is provided, PF is derived from actual kW/kVAR readings.
      - Otherwise uses assumed_pf (default 0.85) — see module-level docstring.
    """
    if assumed_pf is None:
        assumed_pf = _derive_pf_from_history(historical_df) if historical_df is not None else DEFAULT_PF

    tan_phi = _pf_to_tan(assumed_pf)
    kw_vals = forecast_result.median
    ts_vals = forecast_result.timestamps

    kvar_vals = kw_vals * tan_phi
    kva_vals  = np.sqrt(kw_vals ** 2 + kvar_vals ** 2)

    df = pd.DataFrame({
        "timestamp":   ts_vals,
        "kw_import":   kw_vals,
        "kw_export":   np.zeros(len(kw_vals)),
        "kvar_import": kvar_vals,
        "kvar_export": np.zeros(len(kw_vals)),
        "kw_net":      kw_vals,
        "kvar_net":    kvar_vals,
        "kva":         kva_vals,
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def historical_to_manager_df(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a historical load DataFrame has all columns required by run_ai_manager().
    Adds kw_net, kvar_net, kva if absent; estimates kvar from default PF if missing.
    """
    df = historical_df.copy()

    # Ensure kw_export / kvar_export exist
    for col in ("kw_export", "kvar_export"):
        if col not in df.columns:
            df[col] = 0.0

    # kw_net / kvar_net
    if "kw_net" not in df.columns:
        df["kw_net"] = df.get("kw_import", df.get("kw_net", 0.0)) - df["kw_export"]
    if "kvar_net" not in df.columns:
        kvar_imp = df.get("kvar_import", None)
        if kvar_imp is None:
            tan_phi = _pf_to_tan(DEFAULT_PF)
            df["kvar_import"] = df["kw_net"].clip(lower=0) * tan_phi
        df["kvar_net"] = df["kvar_import"] - df["kvar_export"]

    if "kva" not in df.columns:
        df["kva"] = np.sqrt(df["kw_net"] ** 2 + df["kvar_net"] ** 2)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def manager_results_to_sam_df(results: list[dict]) -> pd.DataFrame:
    """
    Convert Manager results list → DataFrame in predictor/data_loader canonical format
    (timestamp, kw_import, kw_export, kvar_import, kvar_export) using the MANAGED values.

    Use this to calculate the bill AFTER optimization.
    """
    rows = []
    for r in results:
        rows.append({
            "timestamp":   pd.to_datetime(r["timestamp"]),
            "kw_import":   float(r.get("kw_managed",   r.get("kw_original", 0))),
            "kw_export":   0.0,  # Manager does not model solar export — add externally if needed
            "kvar_import": float(r.get("kvar_managed", r.get("kvar_original", 0))),
            "kvar_export": 0.0,
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def manager_results_to_original_df(results: list[dict]) -> pd.DataFrame:
    """
    Convert Manager results → canonical DataFrame using the ORIGINAL (pre-optimization) values.

    Use this to calculate the bill BEFORE optimization for savings comparison.
    """
    rows = []
    for r in results:
        rows.append({
            "timestamp":   pd.to_datetime(r["timestamp"]),
            "kw_import":   float(r.get("kw_original", 0)),
            "kw_export":   0.0,
            "kvar_import": float(r.get("kvar_original", 0)),
            "kvar_export": 0.0,
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def manager_results_to_powerreco_df(results: list[dict]) -> pd.DataFrame:
    """
    Convert Manager results → DataFrame expected by PowerRECO battery_sizing.

    Produces:
      original_kva, optimized_kva — for _resolve_discharge() kVA-diff path
      battery_discharged_kwh      — direct discharge energy (preferred path)
      timestamp
    """
    INTERVAL_H = 0.5
    rows = []
    for r in results:
        dis_kw = float(r.get("battery_discharge_kw", 0.0))
        rows.append({
            "timestamp":              pd.to_datetime(r["timestamp"]),
            "original_kva":           float(r.get("kva_original", 0.0)),
            "optimized_kva":          float(r.get("kva_managed",  0.0)),
            "battery_discharged_kwh": dis_kw * INTERVAL_H,
            "battery_charge_kw":      float(r.get("battery_charge_kw", 0.0)),
            "battery_soc_kwh":        float(r.get("battery_soc_kwh", 0.0)),
            "battery_soc_pct":        float(r.get("battery_soc_pct", 0.0)),
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def manager_results_to_csv(results: list[dict]) -> str:
    """Serialize Manager results list to a CSV string (excludes nested 'actions' list)."""
    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "actions"}
        flat.append(row)
    df = pd.DataFrame(flat)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def forecast_result_to_csv(forecast_result: "MultiStepForecastResult") -> str:
    """Serialize a MultiStepForecastResult to a CSV string."""
    df = pd.DataFrame({
        "timestamp": forecast_result.timestamps,
        "median_kw": forecast_result.median,
        "p10_kw":    forecast_result.p10,
        "p90_kw":    forecast_result.p90,
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
