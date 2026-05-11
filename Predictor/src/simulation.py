"""
Live simulation engine for the forecaster demo.

Manages the state needed to auto-replay a 'future' data stream against a
trained model, one 30-min reading at a time. Tracks forecast-vs-actual
accuracy as real data arrives, and triggers warm-start retrains periodically.

Key design:
- Simulation state lives in a single SimulationState object (pickle-able
  for st.session_state).
- Each "tick" advances one step: reveal the next actual reading, compare
  against the prior forecast, append to history, optionally retrain.
- Forecasts are stored historically so we can plot "what we predicted"
  against "what actually happened" at the same timestamp.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type hints; not needed at runtime
    from src.forecaster import DirectMultiStepForecaster, MultiStepForecastResult


@dataclass
class SimulationState:
    """Everything we need to drive a live replay."""

    # The full timeline of readings the simulation will eventually reveal
    # (columns: timestamp, kw_import, kw_export, kvar_import, kvar_export)
    future_data: pd.DataFrame

    # Current position in future_data (how many readings have been revealed)
    tick: int = 0

    # When the sim last triggered a warm-start retrain
    last_retrain_tick: int = 0

    # Frequency of retraining (every N revealed readings)
    retrain_every_n: int = 12  # default: every 6 hours of data

    # Seconds to wait between auto-replay ticks
    tick_interval_s: float = 1.0

    # Warm-start rounds per retrain
    warm_start_rounds: int = 50

    # Is the simulation currently auto-playing?
    is_playing: bool = False

    # Is the simulation finished (all ticks revealed)?
    is_finished: bool = False

    # Per-tick forecast records. Each entry:
    #   {"origin_ts": ts when forecast was made,
    #    "target_ts": ts being predicted,
    #    "horizon_steps": int,
    #    "median": float, "p10": float, "p90": float,
    #    "actual": float | None (filled in when that ts is revealed)}
    forecast_log: list[dict] = field(default_factory=list)

    # Retrain event log
    retrain_log: list[dict] = field(default_factory=list)

    # Latest full forecast (for chart display)
    current_forecast: Optional["MultiStepForecastResult"] = None

    # Per-horizon bias correction (kW), one entry per output step. Positive
    # entry = model was under-predicting at that horizon. The forecaster
    # applies the array as-is; on regime-shifting sites the gap is roughly
    # constant across horizons so per-horizon correction beats scalar +
    # linear decay (which would zero out the correction at long horizons,
    # exactly where it's needed for next-day noon predictions).
    current_bias_profile: Optional[np.ndarray] = None

    # Backwards-compat: scalar h=1 bias correction. Set to bias_profile[0]
    # so the existing UI metric panel keeps working without changes.
    current_bias: float = 0.0

    # Per-horizon conformal half-width (kW): empirical |residual| quantile
    # used to widen the prediction band. Trained quantile bands cover only
    # ~43% of actuals vs the 80% target — this calibration restores it.
    current_conformal_hw: Optional[np.ndarray] = None

    # =================================================================
    #                         LIFECYCLE
    # =================================================================
    @property
    def total_ticks(self) -> int:
        return len(self.future_data)

    @property
    def progress(self) -> float:
        if self.total_ticks == 0:
            return 0.0
        return min(1.0, self.tick / self.total_ticks)

    def revealed_data(self) -> pd.DataFrame:
        """Rows that have already been 'seen' by the simulation."""
        return self.future_data.iloc[: self.tick].copy()

    def next_reading(self) -> Optional[pd.Series]:
        """Peek at the next reading without advancing."""
        if self.tick >= self.total_ticks:
            return None
        return self.future_data.iloc[self.tick]


def initialize_simulation(
    forecaster: "DirectMultiStepForecaster",
    future_data: pd.DataFrame,
    retrain_every_n: int = 12,
    tick_interval_s: float = 1.0,
    warm_start_rounds: int = 50,
) -> SimulationState:
    """
    Create a fresh simulation state, with the forecaster already trained
    on 'past' data and `future_data` being the stream to replay.
    """
    # Sort future_data ascending by timestamp
    future_data = (
        future_data.sort_values("timestamp")
                   .drop_duplicates("timestamp")
                   .reset_index(drop=True)
    )

    state = SimulationState(
        future_data=future_data,
        retrain_every_n=retrain_every_n,
        tick_interval_s=tick_interval_s,
        warm_start_rounds=warm_start_rounds,
    )

    # Generate the first forecast from the frozen trained model
    state.current_forecast = forecaster.forecast(output_steps=48)
    _log_forecast(state, forecaster.history["timestamp"].max(), state.current_forecast)

    return state


def _log_forecast(
    state: SimulationState,
    origin_ts: pd.Timestamp,
    fr: "MultiStepForecastResult",
) -> None:
    """Record a forecast for later comparison against actuals."""
    for h_idx, ts in enumerate(fr.timestamps):
        state.forecast_log.append({
            "forecast_made_at": origin_ts,
            "target_ts": ts,
            "horizon_steps": h_idx + 1,
            "median": float(fr.median[h_idx]),
            "p10": float(fr.p10[h_idx]),
            "p90": float(fr.p90[h_idx]),
            "actual": None,
        })


def _fill_actuals(state: SimulationState, actual_ts: pd.Timestamp, actual_kw: float) -> None:
    """When a new actual arrives, fill it into all forecast rows predicting that ts."""
    for row in state.forecast_log:
        if row["target_ts"] == actual_ts and row["actual"] is None:
            row["actual"] = float(actual_kw)


def _compute_calibration(
    state: SimulationState,
    output_steps: int,
    bias_window: int = 6,
    conformal_window: int = 96,
    conformal_alpha: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (bias_profile, conformal_half_width), each length output_steps.

    bias_profile[i]: mean signed error (actual - median) at horizon i+1 over
        the last `bias_window` verified rows for that horizon. Why
        per-horizon and not a scalar with linear decay: on stable commercial
        sites a scalar h=1 bias works, but on regime-shifting sites the gap
        between forecast and actual is roughly *constant* across horizons
        (e.g. 'we're in a quiet week, all 48 horizons over-predict by ~30
        kW'). Linear decay then fades the correction to zero exactly where
        it's most needed — long-horizon noon predictions. A per-horizon
        profile catches each horizon's own gap. Falls back to neighbouring
        horizons when a horizon hasn't accumulated enough samples yet.

    conformal_half_width[i]: empirical `conformal_alpha`-quantile of
        |residual| at horizon i+1, computed over the last `conformal_window`
        verified rows. Widens the band post-hoc to hit the 80% coverage
        target (raw LightGBM quantile bands cover only ~43%).
    """
    verified = [r for r in state.forecast_log if r["actual"] is not None]
    bias_profile = np.zeros(output_steps)
    hw = np.zeros(output_steps)
    if not verified:
        return bias_profile, hw

    by_h: dict[int, list[dict]] = {}
    for r in verified:
        by_h.setdefault(int(r["horizon_steps"]), []).append(r)

    all_residuals = [abs(r["actual"] - r["median"]) for r in verified[-conformal_window:]]
    global_hw = float(np.quantile(all_residuals, conformal_alpha)) if all_residuals else 0.0

    # Cap scaling: use whichever of recent actual / recent median is larger,
    # so on a regime shift the cap follows the gap (smaller actual, larger
    # forecast → cap should still be big enough to close the gap).
    recent = verified[-conformal_window:]
    recent_actual_mean = float(np.mean([r["actual"] for r in recent]))
    recent_median_mean = float(np.mean([r["median"] for r in recent]))
    cap = 0.80 * max(recent_actual_mean, recent_median_mean, 1.0)

    # Compute per-horizon bias from this horizon's own recent verified rows.
    # Use exponential weighting (most recent count ~2x as much as oldest in
    # the window) so the bias tracks the current model's residual rather
    # than averaging with stale forecasts the model has since corrected.
    # Skip horizons with < 3 samples and fill those in via interpolation
    # against the populated horizons.
    raw_bias = np.full(output_steps, np.nan)
    for h in range(1, output_steps + 1):
        rows = by_h.get(h, [])[-bias_window:]
        if len(rows) >= 3:
            errs = np.array([r["actual"] - r["median"] for r in rows], dtype=float)
            n = len(errs)
            # Half-life of n/2 samples → newest sample weighted ~2x oldest.
            ages = np.arange(n - 1, -1, -1)
            w = 0.5 ** (ages / max(1.0, n / 2))
            est = float(np.sum(w * errs) / np.sum(w))
            raw_bias[h - 1] = float(np.clip(est, -cap, cap))
    if np.isfinite(raw_bias).any():
        x = np.arange(output_steps)
        valid = np.isfinite(raw_bias)
        bias_profile = np.interp(x, x[valid], raw_bias[valid])
    # Light smoothing so the per-horizon profile is monotone enough not to
    # introduce its own kinks; v3 backtest showed unsmoothed profiles caused
    # mid-horizon variance regressions on stable sites.
    if output_steps >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        bias_profile = np.convolve(bias_profile, kernel, mode="same")

    for h in range(1, output_steps + 1):
        rows = by_h.get(h, [])[-conformal_window:]
        if len(rows) >= 8:
            resid = [abs(r["actual"] - r["median"]) for r in rows]
            hw[h - 1] = float(np.quantile(resid, conformal_alpha))
        else:
            hw[h - 1] = global_hw

    if output_steps >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        hw = np.convolve(hw, kernel, mode="same")

    return bias_profile, hw


def advance_one_tick(
    forecaster: "DirectMultiStepForecaster",
    state: SimulationState,
) -> dict:
    """
    Advance the simulation by one tick:
      1. Reveal the next reading (append to forecaster.history).
      2. Fill the actual into any prior forecast rows targeting this ts.
      3. If enough time has passed since last retrain, warm-start retrain.
      4. Regenerate a fresh 24h forecast from the updated model.

    Returns a dict with a summary of what happened this tick.
    """
    if state.tick >= state.total_ticks:
        state.is_finished = True
        state.is_playing = False
        return {"status": "finished", "tick": state.tick}

    # --- Step 1: reveal next reading ---
    new_row = state.future_data.iloc[[state.tick]].copy()
    reveal_ts = new_row["timestamp"].iloc[0]
    reveal_kw = float(new_row["kw_import"].iloc[0])

    # Append to forecaster's history (but don't retrain yet)
    forecaster.history = pd.concat(
        [forecaster.history, new_row], ignore_index=True,
    ).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # --- Step 2: fill actuals into prior forecasts ---
    _fill_actuals(state, reveal_ts, reveal_kw)

    state.tick += 1

    # --- Step 3: decide whether to retrain ---
    did_retrain = False
    retrain_info = None
    if state.tick - state.last_retrain_tick >= state.retrain_every_n:
        # Get just the rows added since last retrain - warm-start on the new slice only
        window_start = state.last_retrain_tick
        window_data = state.future_data.iloc[window_start:state.tick].copy()
        if len(window_data) > 0:
            metrics = forecaster.update(
                window_data,
                n_rounds=state.warm_start_rounds,
            )
            did_retrain = True
            retrain_info = {
                "at_tick": state.tick,
                "reveal_ts": reveal_ts,
                "samples_added": len(window_data),
                "mean_mape": metrics["mean_mape"],
                "mape_at_h24": metrics.get("mape_at_h24"),
            }
            state.retrain_log.append(retrain_info)
            state.last_retrain_tick = state.tick

    # --- Step 4: per-horizon bias + conformal half-width, then forecast ---
    bias_profile, conformal_hw = _compute_calibration(state, output_steps=48)
    state.current_bias_profile = bias_profile
    # Backwards-compat scalar (the UI displays this); use the h=1 entry.
    state.current_bias = float(bias_profile[0]) if len(bias_profile) else 0.0
    state.current_conformal_hw = conformal_hw

    state.current_forecast = forecaster.forecast(
        output_steps=48,
        bias_correction=bias_profile,
        conformal_half_width=conformal_hw,
    )
    _log_forecast(state, reveal_ts, state.current_forecast)

    if state.tick >= state.total_ticks:
        state.is_finished = True
        state.is_playing = False

    return {
        "status": "tick_ok",
        "tick": state.tick,
        "reveal_ts": reveal_ts,
        "reveal_kw": reveal_kw,
        "did_retrain": did_retrain,
        "retrain_info": retrain_info,
    }


# =====================================================================
#                         ACCURACY TRACKING
# =====================================================================

def compute_running_accuracy(state: SimulationState) -> dict:
    """
    Compute live accuracy metrics across all forecast rows that have
    been 'verified' (i.e., the target_ts has now been revealed).
    """
    verified = [r for r in state.forecast_log if r["actual"] is not None]
    if not verified:
        return {
            "n_verified": 0,
            "overall_mae": None,
            "overall_mape": None,
            "by_horizon": {},
            "within_80ci_pct": None,
        }

    mae_errors = []
    mape_errors = []
    within_ci = 0
    by_horizon: dict[int, list] = {}

    for r in verified:
        err = abs(r["median"] - r["actual"])
        mae_errors.append(err)
        mape_errors.append(err / max(r["actual"], 1.0))
        if r["p10"] <= r["actual"] <= r["p90"]:
            within_ci += 1
        by_horizon.setdefault(r["horizon_steps"], []).append((err, r["actual"]))

    out = {
        "n_verified": len(verified),
        "overall_mae": float(np.mean(mae_errors)),
        "overall_mape": float(np.mean(mape_errors) * 100),
        "within_80ci_pct": 100.0 * within_ci / len(verified),
        "by_horizon": {},
    }
    for h, pairs in by_horizon.items():
        errs = [p[0] for p in pairs]
        acts = [p[1] for p in pairs]
        out["by_horizon"][h] = {
            "n": len(pairs),
            "mae": float(np.mean(errs)),
            "mape": float(np.mean([e / max(a, 1.0) for e, a in zip(errs, acts)]) * 100),
        }
    return out


def build_forecast_vs_actual_df(state: SimulationState) -> pd.DataFrame:
    """For plotting: time series of recent forecasts alongside their actuals."""
    if not state.forecast_log:
        return pd.DataFrame(columns=["target_ts", "median", "p10", "p90", "actual", "horizon_steps"])

    df = pd.DataFrame(state.forecast_log)
    # For each target_ts, keep only the MOST RECENT forecast (smallest horizon)
    # - that's what would be displayed as "the forecast for now" in a real system.
    df = (
        df.sort_values(["target_ts", "horizon_steps"], ascending=[True, True])
          .drop_duplicates("target_ts", keep="first")
          .reset_index(drop=True)
    )
    return df


def split_historical_for_simulation(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience: chronologically split a single historical dataset into
    (train_df, future_df) so the user doesn't need two separate files to demo.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * train_fraction)
    train = df.iloc[:split].reset_index(drop=True)
    future = df.iloc[split:].reset_index(drop=True)
    return train, future
