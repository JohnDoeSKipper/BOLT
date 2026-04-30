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

    # Current rolling bias correction (kW). Positive = model was under-predicting.
    # Recomputed each tick from recent h=1 verified errors.
    current_bias: float = 0.0

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

    # --- Step 4: compute rolling bias then regenerate forecast ---
    # Use the last 8 verified h=1 predictions (covers 4 hours of data).
    # h=1 is the immediate next-step prediction — any systematic offset
    # there reveals a level shift the model hasn't yet learned.
    recent_h1 = [
        r for r in state.forecast_log
        if r["actual"] is not None and r["horizon_steps"] == 1
    ][-8:]

    if len(recent_h1) >= 3:
        errors = [r["actual"] - r["median"] for r in recent_h1]
        raw_bias = float(np.mean(errors))
        # Cap at ±20 % of recent mean actual so we don't over-correct
        # during temporary spikes (e.g. a single anomalous reading).
        mean_actual = float(np.mean([r["actual"] for r in recent_h1]))
        cap = 0.20 * max(mean_actual, 1.0)
        state.current_bias = float(np.clip(raw_bias, -cap, cap))
    else:
        state.current_bias = 0.0

    state.current_forecast = forecaster.forecast(
        output_steps=48, bias_correction=state.current_bias
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
