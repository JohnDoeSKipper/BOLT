"""
Slice the backtest forecast log every way that might explain accuracy gaps.

Reads: data/_backtest_*.csv (output of scripts/backtest.py)
Prints sections of analysis to stdout — kept terse so we can eyeball the
biggest offenders quickly.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _add_error_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["actual"]).copy()
    df["err"] = df["median"] - df["actual"]                # signed: + = over-predict
    df["abs_err"] = df["err"].abs()
    df["pct_err"] = df["err"] / df["actual"].clip(lower=1.0) * 100
    df["abs_pct_err"] = df["pct_err"].abs()
    df["in_80ci"] = ((df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"])).astype(int)
    df["band_width"] = df["p90"] - df["p10"]
    df["target_ts"] = pd.to_datetime(df["target_ts"])
    df["forecast_made_at"] = pd.to_datetime(df["forecast_made_at"])
    df["hour"] = df["target_ts"].dt.hour
    df["dow"] = df["target_ts"].dt.dayofweek
    return df


def _fmt_table(name: str, frame: pd.DataFrame, sort_col: str | None = None,
               keep: list[str] | None = None, top: int | None = None) -> None:
    print(f"\n--- {name} ---")
    if sort_col is not None:
        frame = frame.sort_values(sort_col, ascending=False)
    if top is not None:
        frame = frame.head(top)
    if keep is not None:
        frame = frame[keep]
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", "{:.2f}".format):
        print(frame.to_string(index=False))


def by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("horizon_steps").agg(
        n=("actual", "size"),
        mae=("abs_err", "mean"),
        mape=("abs_pct_err", "mean"),
        bias=("err", "mean"),
        coverage_80=("in_80ci", "mean"),
        band_w=("band_width", "mean"),
    ).reset_index()
    g["coverage_80"] = g["coverage_80"] * 100
    return g


def by_hour(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("hour").agg(
        n=("actual", "size"),
        mae=("abs_err", "mean"),
        mape=("abs_pct_err", "mean"),
        bias=("err", "mean"),
        actual_mean=("actual", "mean"),
    ).reset_index()
    g["bias_pct_of_load"] = g["bias"] / g["actual_mean"].clip(lower=1.0) * 100
    return g


def by_dow(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("dow").agg(
        n=("actual", "size"),
        mae=("abs_err", "mean"),
        mape=("abs_pct_err", "mean"),
        bias=("err", "mean"),
    ).reset_index()
    g["dow_name"] = g["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    return g


def by_load_regime(df: pd.DataFrame) -> pd.DataFrame:
    quants = df["actual"].quantile([0.25, 0.75]).values
    def lab(v):
        if v <= quants[0]:
            return "1_low"
        if v >= quants[1]:
            return "3_high"
        return "2_mid"
    df = df.assign(regime=df["actual"].map(lab))
    g = df.groupby("regime").agg(
        n=("actual", "size"),
        mae=("abs_err", "mean"),
        mape=("abs_pct_err", "mean"),
        bias=("err", "mean"),
        actual_mean=("actual", "mean"),
    ).reset_index().sort_values("regime")
    return g


def peak_capture(df: pd.DataFrame, top_pct: float = 0.05) -> dict:
    """How well do we predict the highest 5% of actuals?"""
    cutoff = df["actual"].quantile(1 - top_pct)
    peaks = df[df["actual"] >= cutoff]
    return {
        "peak_cutoff_kw": float(cutoff),
        "n_peaks": int(len(peaks)),
        "median_underpredict_kw": float((peaks["actual"] - peaks["median"]).mean()),
        "p90_underpredict_kw": float((peaks["actual"] - peaks["p90"]).mean()),
        "p90_misses_pct": float(((peaks["actual"] > peaks["p90"]).mean()) * 100),
        "mape_at_peaks": float(peaks["abs_pct_err"].mean()),
    }


def bias_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon coverage and bias — band sanity check."""
    g = df.groupby("horizon_steps").agg(
        coverage_80=("in_80ci", "mean"),
        miss_high_pct=("actual", lambda s: float((s > df.loc[s.index, "p90"]).mean()) * 100),
        miss_low_pct=("actual", lambda s: float((s < df.loc[s.index, "p10"]).mean()) * 100),
    ).reset_index()
    g["coverage_80"] = g["coverage_80"] * 100
    return g


def transition_errors(df: pd.DataFrame) -> dict:
    """Big actual-to-actual jumps — does the model lag transitions?"""
    only_h1 = df[df["horizon_steps"] == 1].sort_values("target_ts").copy()
    only_h1["d_actual"] = only_h1["actual"].diff().abs()
    cutoff = only_h1["d_actual"].quantile(0.9)
    big_moves = only_h1[only_h1["d_actual"] >= cutoff]
    calm = only_h1[only_h1["d_actual"] < only_h1["d_actual"].quantile(0.5)]
    return {
        "transition_cutoff_kw": float(cutoff),
        "mae_during_transitions": float(big_moves["abs_err"].mean()),
        "mae_when_calm": float(calm["abs_err"].mean()),
        "mape_during_transitions": float(big_moves["abs_pct_err"].mean()),
        "mape_when_calm": float(calm["abs_pct_err"].mean()),
        "n_transitions": int(len(big_moves)),
    }


def worst_intervals(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Which target_ts had the largest h=1 errors? Useful sanity look."""
    only_h1 = df[df["horizon_steps"] == 1].copy()
    return only_h1.nlargest(top, "abs_err")[
        ["target_ts", "actual", "median", "p10", "p90", "err"]
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="data/_backtest_mi2.csv")
    args = p.parse_args()

    base = Path(__file__).resolve().parent.parent
    log_path = (base / args.log).resolve()
    print(f"[load] {log_path}")
    df = pd.read_csv(log_path)
    df = _add_error_cols(df)
    print(f"  verified rows: {len(df)}")
    print(f"  overall MAE:   {df['abs_err'].mean():.2f} kW")
    print(f"  overall MAPE:  {df['abs_pct_err'].mean():.2f} %")
    print(f"  overall BIAS:  {df['err'].mean():+.2f} kW   (+ = over-predict)")
    print(f"  80% CI cov:    {df['in_80ci'].mean()*100:.1f}%   (target = 80%)")

    _fmt_table("By horizon", by_horizon(df))
    _fmt_table("By hour-of-day", by_hour(df))
    _fmt_table("By day-of-week", by_dow(df))
    _fmt_table("By load regime", by_load_regime(df))
    _fmt_table("Per-horizon CI calibration", bias_calibration(df))

    print("\n--- Peak capture (top 5% loads) ---")
    for k, v in peak_capture(df).items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n--- Transition vs calm (h=1) ---")
    for k, v in transition_errors(df).items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    _fmt_table("Worst h=1 intervals", worst_intervals(df))


if __name__ == "__main__":
    main()
