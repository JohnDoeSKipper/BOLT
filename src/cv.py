"""
Time-series cross-validation for forecaster evaluation.

Solves Problem 2 from the audit: replaces the misleading single 80/20 split
with proper expanding-window CV that produces mean ± std MAPE across multiple
folds — the standard for time-series accuracy reporting.

Approach (per fold):
  1. Train on days [0, train_end_day).
  2. Walk through validation period in 12-hour blocks.
  3. At each block start, set forecaster.history = train + val[:block_start],
     generate a fresh forecast, compare against val[block_start:block_start+48].
  4. Record errors per horizon.

This is "rolling-origin" evaluation — the gold standard for time-series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable
from copy import deepcopy


def expanding_window_cv(
    df: pd.DataFrame,
    forecaster_factory: Callable,
    n_splits: int = 4,
    min_train_days: int = 21,
    val_block_days: int = 5,
    forecast_origins_per_fold: int = 5,
    target_horizons: list[int] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run expanding-window time-series CV with rolling forecast origins.

    Args:
        df: Full historical DataFrame (canonical schema).
        forecaster_factory: Callable returning a fresh forecaster instance.
            Example: lambda: DirectMultiStepForecaster(capacity_kwp=500, n_estimators=150)
        n_splits: Number of CV folds (each adds val_block_days to the training set).
        min_train_days: Minimum days in the FIRST fold's training set.
        val_block_days: Days held out as validation in each fold.
        forecast_origins_per_fold: How many fresh forecasts to generate within
            the validation block (more = more honest, slower).
        target_horizons: Horizons (in 30-min steps) to report. Defaults to [1,6,12,24,48].
        verbose: Print fold-by-fold progress.

    Returns:
        Dict with per-fold and aggregate metrics.
    """
    if target_horizons is None:
        target_horizons = [1, 6, 12, 24, 48]

    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["date"] = df["timestamp"].dt.date
    unique_dates = sorted(df["date"].unique())
    total_days = len(unique_dates)

    required = min_train_days + val_block_days * n_splits
    if total_days < required:
        raise ValueError(
            f"Need at least {required} days but have {total_days}. "
            f"Reduce n_splits or use synthetic data."
        )

    fold_results = []
    for fold_idx in range(n_splits):
        train_end_day_idx = min_train_days + fold_idx * val_block_days
        val_start_day_idx = train_end_day_idx
        val_end_day_idx = val_start_day_idx + val_block_days

        if val_end_day_idx > total_days:
            break

        train_dates = set(unique_dates[:train_end_day_idx])
        val_dates = set(unique_dates[val_start_day_idx:val_end_day_idx])

        train_df = df[df["date"].isin(train_dates)].drop(columns=["date"]).reset_index(drop=True)
        val_df = df[df["date"].isin(val_dates)].drop(columns=["date"]).reset_index(drop=True)
        val_df = val_df.sort_values("timestamp").reset_index(drop=True)

        if verbose:
            print(f"\n  Fold {fold_idx + 1}/{n_splits}:")
            print(f"    Train: {min(train_dates)} → {max(train_dates)} ({len(train_dates)} days, {len(train_df)} samples)")
            print(f"    Val:   {min(val_dates)} → {max(val_dates)} ({len(val_dates)} days, {len(val_df)} samples)")

        # Train forecaster on initial training window
        fc = forecaster_factory()
        fc.fit(train_df, verbose=False)

        # Pick `forecast_origins_per_fold` evenly-spaced origins within the val block
        # An "origin" is a timestamp from which we generate a 24-hour forecast.
        n_val_samples = len(val_df)
        if n_val_samples < 50:
            origin_indices = [0]
        else:
            # Space origins so each forecast horizon stays within val data
            max_origin = n_val_samples - max(target_horizons) - 1
            if max_origin < 1:
                origin_indices = [0]
            else:
                origin_indices = np.linspace(0, max_origin, forecast_origins_per_fold).astype(int).tolist()

        per_horizon_errs = {h: {"abs": [], "act": []} for h in target_horizons}

        for origin_i in origin_indices:
            # Build the history that the forecaster "sees" at this origin
            origin_ts = val_df["timestamp"].iloc[origin_i]
            history_for_origin = pd.concat([
                train_df,
                val_df.iloc[:origin_i + 1],  # include up to and including origin
            ], ignore_index=True)
            history_for_origin = (
                history_for_origin.drop_duplicates("timestamp")
                                  .sort_values("timestamp")
                                  .reset_index(drop=True)
            )

            # Reuse the trained model but swap history
            # (avoids the cost of refitting per origin)
            fc.history = history_for_origin
            try:
                result = fc.forecast(output_steps=max(target_horizons))
            except Exception as e:
                if verbose:
                    print(f"    Forecast failed at origin {origin_ts}: {e}")
                continue

            # For each target horizon, find the actual value
            for h in target_horizons:
                pred_idx = h - 1
                if pred_idx >= len(result.median):
                    continue
                pred_ts = result.timestamps[pred_idx]
                pred_val = float(result.median[pred_idx])

                actual_row = val_df[val_df["timestamp"] == pred_ts]
                if len(actual_row) == 0:
                    continue
                actual = float(actual_row["kw_import"].iloc[0])
                per_horizon_errs[h]["abs"].append(abs(pred_val - actual))
                per_horizon_errs[h]["act"].append(actual)

        # Aggregate for this fold
        fold_summary = {
            "fold": fold_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "train_start": str(min(train_dates)),
            "train_end": str(max(train_dates)),
            "val_start": str(min(val_dates)),
            "val_end": str(max(val_dates)),
            "n_origins": len(origin_indices),
            "horizon_metrics": {},
        }
        for h in target_horizons:
            errs = per_horizon_errs[h]["abs"]
            acts = per_horizon_errs[h]["act"]
            if errs:
                mae = float(np.mean(errs))
                mape = float(np.mean([e / max(a, 1.0) for e, a in zip(errs, acts)]) * 100)
                fold_summary["horizon_metrics"][h] = {
                    "mae": mae, "mape": mape, "n": len(errs),
                }
                if verbose:
                    print(f"    h={h:2d}: MAE={mae:6.1f} kW  MAPE={mape:5.2f}%  (n={len(errs)})")

        fold_results.append(fold_summary)

    # Aggregate across all folds
    aggregate = {h: {"mapes": [], "maes": []} for h in target_horizons}
    for fr in fold_results:
        for h in target_horizons:
            if h in fr["horizon_metrics"]:
                aggregate[h]["mapes"].append(fr["horizon_metrics"][h]["mape"])
                aggregate[h]["maes"].append(fr["horizon_metrics"][h]["mae"])

    summary = {
        "n_folds_completed": len(fold_results),
        "fold_results": fold_results,
        "aggregate": {},
    }
    for h in target_horizons:
        if aggregate[h]["mapes"]:
            summary["aggregate"][h] = {
                "mean_mape": float(np.mean(aggregate[h]["mapes"])),
                "std_mape": float(np.std(aggregate[h]["mapes"])),
                "mean_mae": float(np.mean(aggregate[h]["maes"])),
                "std_mae": float(np.std(aggregate[h]["maes"])),
                "n_folds": len(aggregate[h]["mapes"]),
            }
    return summary


def format_cv_report(summary: dict) -> str:
    """Pretty-print a CV summary for inclusion in reports."""
    lines = []
    lines.append("=" * 68)
    lines.append(f"CROSS-VALIDATION REPORT  ({summary['n_folds_completed']} folds completed)")
    lines.append("=" * 68)
    lines.append(f"{'Horizon':>8} | {'Mean MAPE':>10} | {'Std MAPE':>10} | {'Mean MAE':>10} | {'Std MAE':>10}")
    lines.append("-" * 68)
    for h, m in sorted(summary["aggregate"].items()):
        h_label = f"+{h * 30}min" if h * 30 < 60 else f"+{h / 2:.1f}h"
        lines.append(
            f"{h_label:>8} | "
            f"{m['mean_mape']:>9.2f}% | "
            f"{m['std_mape']:>9.2f}% | "
            f"{m['mean_mae']:>9.1f} kW | "
            f"{m['std_mae']:>9.1f} kW"
        )
    lines.append("=" * 68)
    return "\n".join(lines)
