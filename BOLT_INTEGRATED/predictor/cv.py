"""
Time-series cross-validation for DirectMultiStepForecaster.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable


def expanding_window_cv(df: pd.DataFrame, forecaster_factory: Callable,
                         n_splits: int = 4, min_train_days: int = 21,
                         val_block_days: int = 5, forecast_origins_per_fold: int = 5,
                         target_horizons: list[int] | None = None,
                         verbose: bool = False) -> dict:
    if target_horizons is None:
        target_horizons = [1, 6, 12, 24, 48]

    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["date"] = df["timestamp"].dt.date
    unique_dates = sorted(df["date"].unique())
    total_days = len(unique_dates)

    required = min_train_days + val_block_days * n_splits
    if total_days < required:
        raise ValueError(f"Need at least {required} days but have {total_days}.")

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

        fc = forecaster_factory()
        fc.fit(train_df, verbose=False)

        n_val_samples = len(val_df)
        if n_val_samples < 50:
            origin_indices = [0]
        else:
            max_origin = n_val_samples - max(target_horizons) - 1
            if max_origin < 1:
                origin_indices = [0]
            else:
                origin_indices = np.linspace(0, max_origin, forecast_origins_per_fold).astype(int).tolist()

        per_horizon_errs = {h: {"abs": [], "act": []} for h in target_horizons}

        for origin_i in origin_indices:
            origin_ts = val_df["timestamp"].iloc[origin_i]
            history_for_origin = pd.concat([train_df, val_df.iloc[:origin_i + 1]], ignore_index=True)
            history_for_origin = (history_for_origin.drop_duplicates("timestamp")
                                  .sort_values("timestamp").reset_index(drop=True))
            fc.history = history_for_origin
            try:
                result = fc.forecast(output_steps=max(target_horizons))
            except Exception:
                continue

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

        fold_summary = {
            "fold": fold_idx, "train_days": len(train_dates), "val_days": len(val_dates),
            "train_start": str(min(train_dates)), "train_end": str(max(train_dates)),
            "val_start": str(min(val_dates)), "val_end": str(max(val_dates)),
            "n_origins": len(origin_indices), "horizon_metrics": {},
        }
        for h in target_horizons:
            errs = per_horizon_errs[h]["abs"]
            acts = per_horizon_errs[h]["act"]
            if errs:
                mae = float(np.mean(errs))
                mape = float(np.mean([e / max(a, 1.0) for e, a in zip(errs, acts)]) * 100)
                fold_summary["horizon_metrics"][h] = {"mae": mae, "mape": mape, "n": len(errs)}
        fold_results.append(fold_summary)

    aggregate = {h: {"mapes": [], "maes": []} for h in target_horizons}
    for fr in fold_results:
        for h in target_horizons:
            if h in fr["horizon_metrics"]:
                aggregate[h]["mapes"].append(fr["horizon_metrics"][h]["mape"])
                aggregate[h]["maes"].append(fr["horizon_metrics"][h]["mae"])

    summary = {"n_folds_completed": len(fold_results), "fold_results": fold_results, "aggregate": {}}
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
    lines = ["=" * 68, f"CROSS-VALIDATION REPORT  ({summary['n_folds_completed']} folds completed)",
             "=" * 68,
             f"{'Horizon':>8} | {'Mean MAPE':>10} | {'Std MAPE':>10} | {'Mean MAE':>10} | {'Std MAE':>10}",
             "-" * 68]
    for h, m in sorted(summary["aggregate"].items()):
        h_label = f"+{h * 30}min" if h * 30 < 60 else f"+{h / 2:.1f}h"
        lines.append(f"{h_label:>8} | {m['mean_mape']:>9.2f}% | {m['std_mape']:>9.2f}% | "
                     f"{m['mean_mae']:>9.1f} kW | {m['std_mae']:>9.1f} kW")
    lines.append("=" * 68)
    return "\n".join(lines)
