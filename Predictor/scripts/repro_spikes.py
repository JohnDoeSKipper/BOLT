"""
Reproduce the 'midnight spike hallucination' the user reported on the LIVE
TEST DATA file. Trains on the first 35% of the file (matching app default),
then advances 220 ticks and dumps what the median forecast looks like.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loader import load_csv
from src.forecaster import DirectMultiStepForecaster
from src.simulation import (
    initialize_simulation, advance_one_tick, split_historical_for_simulation,
)


def main():
    df = load_csv("data/_sim_LIVE TEST DATA.csv")
    print(f"data: rows={len(df)}  range={df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"      mean={df['kw_import'].mean():.1f}  max={df['kw_import'].max():.1f}")

    train, future = split_historical_for_simulation(df, train_fraction=0.35)
    print(f"\ntrain: {len(train)}  range {train['timestamp'].min()} → {train['timestamp'].max()}  mean={train['kw_import'].mean():.1f}  max={train['kw_import'].max():.1f}")
    print(f"future: {len(future)}  range {future['timestamp'].min()} → {future['timestamp'].max()}  mean={future['kw_import'].mean():.1f}  max={future['kw_import'].max():.1f}")

    fc = DirectMultiStepForecaster(capacity_kwp=0.0)
    r = fc.fit(train, verbose=False)
    print(f"\nfit: mean_mape={r['mean_mape']:.2f}%  mape@h1={r['mape_at_h1']:.2f}%  mape@h24={r['mape_at_h24']:.2f}%  mape@h48={r['mape_at_h48']:.2f}%")

    state = initialize_simulation(fc, future, retrain_every_n=4,
                                  tick_interval_s=0.0, warm_start_rounds=50)
    fr = state.current_forecast
    print(f"\ninitial forecast (next 48 steps from {state.future_data['timestamp'].iloc[0]}):")
    print(f"  median: min={fr.median.min():.1f} max={fr.median.max():.1f} mean={fr.median.mean():.1f}")
    print(f"  next 12: {fr.median[:12].round(0).tolist()}")
    print(f"  next 24: {fr.median[12:24].round(0).tolist()}")

    for i in range(220):
        advance_one_tick(fc, state)
    fr = state.current_forecast
    print(f"\nafter 220 ticks  bias={state.current_bias:+.1f}  hw[h=1]={state.current_conformal_hw[0]:.0f}")
    print(f"forecast horizons:")
    print(f"  median: min={fr.median.min():.1f} max={fr.median.max():.1f} mean={fr.median.mean():.1f}")
    print(f"  raw_horizons (trained):")
    for h in fc.horizons[:12]:
        ts = fr.last_history_ts + pd.Timedelta(minutes=30 * h)
        med = fr.raw_predictions[h][0.5]
        print(f"    h={h:2d}  → ts={ts}  median={med:8.1f}  p10={fr.raw_predictions[h][0.1]:8.1f}  p90={fr.raw_predictions[h][0.9]:8.1f}")
    print(f"  next 12 actuals : {future['kw_import'].iloc[220:232].round(0).tolist()}")
    print(f"  next 12 forecast: {fr.median[:12].round(0).tolist()}")

    # Sample forecasts across the daily cycle
    print("\nverification: forecast median across the next 48 horizons (h=1..48)")
    for h in range(1, 49):
        ts = fr.last_history_ts + pd.Timedelta(minutes=30 * h)
        print(f"  h={h:2d}  hh={ts.hour:02d}:{ts.minute:02d}  median={fr.median[h-1]:7.1f}  p10={fr.p10[h-1]:7.1f}  p90={fr.p90[h-1]:7.1f}")


if __name__ == "__main__":
    main()
