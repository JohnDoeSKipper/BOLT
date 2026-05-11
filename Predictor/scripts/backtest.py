"""
Walk-forward backtest that mirrors the live simulation so we can analyze
sources of inaccuracy in the trained forecaster.

It trains on the first `train_fraction` of a dataset, then advances one
30-min tick at a time through the rest, exactly like
`src.simulation.advance_one_tick`. Every (origin_ts, target_ts, horizon,
p10, median, p90, actual) tuple gets recorded so we can later slice the
errors by horizon, hour-of-day, day-of-week, regime, etc.

Output: CSV at <out_path> with one row per (origin, target) pair plus a
small JSON of run metadata.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python scripts/backtest.py` from the Predictor/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loader import load_csv
from src.forecaster import DirectMultiStepForecaster
from src.solar_estimator import detect_has_solar, estimate_solar_capacity_kwp
from src.simulation import (
    initialize_simulation,
    advance_one_tick,
    build_forecast_vs_actual_df,
    compute_running_accuracy,
    split_historical_for_simulation,
)


def run_backtest(
    data_path: Path,
    out_path: Path,
    train_fraction: float = 0.7,
    retrain_every_n: int = 12,
    warm_start_rounds: int = 50,
    max_ticks: int | None = None,
) -> dict:
    print(f"[load] {data_path}")
    df = load_csv(data_path)
    print(f"  rows={len(df)}  range={df['timestamp'].min()} → {df['timestamp'].max()}")

    has_solar, reason = detect_has_solar(df)
    capacity = float(estimate_solar_capacity_kwp(df)["capacity_kwp"]) if has_solar else 0.0
    print(f"  has_solar={has_solar} ({reason})  capacity_kwp={capacity:.1f}")

    train_df, future_df = split_historical_for_simulation(df, train_fraction=train_fraction)
    print(f"[split] train={len(train_df)}  future={len(future_df)}")

    print("[fit] training forecaster from scratch")
    fc = DirectMultiStepForecaster(capacity_kwp=capacity)
    t0 = time.time()
    fit_result = fc.fit(train_df, verbose=False)
    print(f"  done in {time.time()-t0:.1f}s   mean_mape={fit_result['mean_mape']:.2f}%")

    print(f"[sim] initializing  retrain_every_n={retrain_every_n}  rounds={warm_start_rounds}")
    state = initialize_simulation(
        fc,
        future_df,
        retrain_every_n=retrain_every_n,
        tick_interval_s=0.0,
        warm_start_rounds=warm_start_rounds,
    )

    n_total = state.total_ticks if max_ticks is None else min(max_ticks, state.total_ticks)
    print(f"[run] advancing {n_total} ticks")
    t0 = time.time()
    for i in range(n_total):
        info = advance_one_tick(fc, state)
        if info["did_retrain"]:
            ri = info["retrain_info"]
            print(f"  tick {state.tick:4d}/{n_total}: retrain  "
                  f"samples_added={ri['samples_added']}  "
                  f"mean_mape={ri['mean_mape']:.2f}%")
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(f"  tick {state.tick:4d}/{n_total}  rate={rate:.1f} ticks/s  eta={eta:.0f}s")

    print(f"[done] sim finished in {time.time()-t0:.1f}s")

    print("[save] writing forecast log")
    flog = pd.DataFrame(state.forecast_log)
    flog.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({len(flog)} rows)")

    acc = compute_running_accuracy(state)
    meta = {
        "data_path": str(data_path),
        "n_train": len(train_df),
        "n_future": len(future_df),
        "n_ticks_run": n_total,
        "capacity_kwp": capacity,
        "fit_mean_mape": fit_result["mean_mape"],
        "running_overall_mape": acc.get("overall_mape"),
        "running_overall_mae": acc.get("overall_mae"),
        "within_80ci_pct": acc.get("within_80ci_pct"),
        "by_horizon": acc.get("by_horizon"),
        "retrain_log": state.retrain_log,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  → {meta_path}")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mi2_historical.csv")
    p.add_argument("--out", default="data/_backtest_forecast_log.csv")
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--retrain-every-n", type=int, default=12)
    p.add_argument("--warm-start-rounds", type=int, default=50)
    p.add_argument("--max-ticks", type=int, default=None)
    args = p.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_path = (base / args.data).resolve()
    out_path = (base / args.out).resolve()

    run_backtest(
        data_path=data_path,
        out_path=out_path,
        train_fraction=args.train_fraction,
        retrain_every_n=args.retrain_every_n,
        warm_start_rounds=args.warm_start_rounds,
        max_ticks=args.max_ticks,
    )


if __name__ == "__main__":
    main()
