"""
DirectMultiStepForecaster

Solves Problems 1 + 3 from the audit:
  - Problem 1: TRUE warm-start updates via lgb.train(init_model=...)
              Adds a few boosting rounds on the new data instead of full retrain.
  - Problem 3: Direct multi-step forecasting (no recursive error compounding)
              + quantile regression bands (P10, P50, P90) for honest uncertainty.

For each horizon h in HORIZONS and each quantile q in QUANTILES, we train one
LightGBM model that DIRECTLY predicts kw_import[t+h] from features at time t.
This avoids feeding predictions back as inputs.

By default 14 horizons × 3 quantiles = 42 models.
Initial training: ~30s for 60 days of data.
Warm-start update: ~10s.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from src.features import build_feature_matrix


# Every step to h=12 (full 6h forecast is exact), every 2 steps to h=24 (12h),
# every 4 steps to h=48. Fixes the sawtooth where odd horizons (h=7, 9, 11, 13)
# inherit linear-interpolation error from sparse training points.
DEFAULT_HORIZONS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    14, 16, 18, 20, 22, 24,
    28, 32, 36, 40, 44, 48,
]
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class MultiStepForecastResult:
    """Result of a direct multi-step forecast."""
    timestamps: pd.DatetimeIndex     # length = number of *interpolated* steps (e.g., 48)
    median: np.ndarray                # P50 predictions
    p10: np.ndarray                   # P10 lower band
    p90: np.ndarray                   # P90 upper band
    last_history_ts: pd.Timestamp
    raw_horizons: list[int]           # the horizons we actually trained for
    raw_predictions: dict[int, dict]  # raw {horizon: {q: value}} before interpolation


class DirectMultiStepForecaster:
    def __init__(
        self,
        capacity_kwp: float = 0.0,
        horizons: list[int] | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.capacity_kwp = capacity_kwp
        self.horizons = list(horizons) if horizons else list(DEFAULT_HORIZONS)
        self.quantiles = tuple(quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        # boosters[(h, q)] = lgb.Booster
        self.boosters: dict[tuple[int, float], lgb.Booster] = {}
        self.feature_cols: list[str] = []
        self.history: Optional[pd.DataFrame] = None
        self.metrics: dict = {}
        self.n_train_rows: int = 0
        self.last_update_method: str = "none"

    # ===================================================================
    #                        DATA PREPARATION
    # ===================================================================
    def _prepare_supervised(self, df: pd.DataFrame, h: int):
        """Build (X_train, y_train, X_val, y_val) for horizon h."""
        feat_df, feature_cols = build_feature_matrix(
            df, capacity_kwp=self.capacity_kwp, target="kw_import"
        )
        self.feature_cols = feature_cols

        y_h = feat_df["kw_import"].shift(-h)
        valid = ~y_h.isna()
        X_full = feat_df[feature_cols][valid].values
        y_full = y_h[valid].values

        split = int(len(X_full) * 0.8)
        return X_full[:split], y_full[:split], X_full[split:], y_full[split:]

    def _make_params(self, q: float, h: int = 1) -> dict:
        # Long-horizon models need more leaves to capture complex weekly patterns.
        # Short-horizon (h≈1) is dominated by lag autocorrelation — simpler is fine.
        num_leaves = min(127, 31 + h * 2)
        # Tail quantiles need looser leaves so peaks aren't compressed.
        # Backtest showed 97% of top-5% peaks landed above the p90 envelope.
        if q <= 0.15 or q >= 0.85:
            min_child = 8
        else:
            min_child = 15
        return {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "learning_rate": self.learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": self.random_state,
        }

    # ===================================================================
    #                         FROM-SCRATCH FIT
    # ===================================================================
    def fit(self, df: pd.DataFrame, verbose: bool = False) -> dict:
        """Train all (horizon × quantile) models from scratch."""
        self.history = df.copy().sort_values("timestamp").reset_index(drop=True)
        self.boosters.clear()

        per_horizon = {}
        for h in self.horizons:
            X_tr, y_tr, X_va, y_va = self._prepare_supervised(self.history, h)
            train_ds = lgb.Dataset(X_tr, label=y_tr)
            val_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)

            for q in self.quantiles:
                booster = lgb.train(
                    params=self._make_params(q, h),
                    train_set=train_ds,
                    num_boost_round=self.n_estimators,
                    valid_sets=[val_ds],
                    callbacks=[lgb.early_stopping(20, verbose=False)],
                )
                self.boosters[(h, q)] = booster

            # Metrics on the median model (q=0.5)
            median_booster = self.boosters[(h, 0.5)]
            y_va_pred = median_booster.predict(X_va)
            mae = float(np.mean(np.abs(y_va - y_va_pred)))
            mape = float(np.mean(np.abs((y_va - y_va_pred) / np.clip(y_va, 1.0, None))) * 100)

            # Pinball loss across quantiles (proper quantile metric)
            pinball_total = 0.0
            for q in self.quantiles:
                yp = self.boosters[(h, q)].predict(X_va)
                err = y_va - yp
                pinball_total += float(np.mean(np.maximum(q * err, (q - 1) * err)))

            per_horizon[h] = {
                "mae": mae, "mape": mape, "pinball": pinball_total / len(self.quantiles),
                "n_val": int(len(y_va)),
            }
            if verbose:
                print(f"  h={h:2d}: MAE={mae:6.1f} kW  MAPE={mape:5.2f}%  Pinball={per_horizon[h]['pinball']:6.2f}")

        self.metrics = {"per_horizon": per_horizon}
        self.n_train_rows = len(self.history)
        self.last_update_method = "full_fit"

        all_mape = [v["mape"] for v in per_horizon.values()]
        return {
            "method": "full_fit",
            "n_train_rows": self.n_train_rows,
            "n_models_trained": len(self.boosters),
            "mean_mape": float(np.mean(all_mape)),
            "mape_at_h1": per_horizon[self.horizons[0]]["mape"],
            "mape_at_h24": per_horizon.get(24, {}).get("mape"),
            "mape_at_h48": per_horizon.get(48, {}).get("mape"),
            "per_horizon": per_horizon,
        }

    # ===================================================================
    #                    WARM-START INCREMENTAL UPDATE
    # ===================================================================
    def update(
        self,
        new_df: pd.DataFrame,
        n_rounds: int = 50,
        learning_rate: float = 0.03,
        lookback_days: float = 30.0,
        recency_half_life_days: float = 7.0,
        verbose: bool = False,
    ) -> dict:
        """
        Recency-weighted warm-start update.

        Two key improvements over a naive warm-start:

        1. SLIDING WINDOW — only the most recent `lookback_days` of history
           is used as the training set. This prevents old patterns from
           drowning out recent regime changes.

        2. EXPONENTIAL SAMPLE WEIGHTS — within the training window, the
           newest rows get weight 1.0 and weight halves every
           `recency_half_life_days`. The new data therefore dominates the
           gradient signal even though it is a small fraction of rows.

        The old boosters are kept as the starting point (init_model), so
        structural knowledge is preserved; only the residuals on recent
        data are corrected by the new rounds.
        """
        if not self.boosters or self.history is None:
            return self.fit(new_df, verbose=verbose)

        # Merge new rows into full history (kept for feature lag computation)
        combined = pd.concat([self.history, new_df], ignore_index=True)
        combined = (
            combined.drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
        )
        self.history = combined

        # Training window: recent lookback_days + 8-day buffer so lag_336
        # (7-day lag) has enough history to be non-NaN.
        max_ts = combined["timestamp"].max()
        window_cutoff = max_ts - pd.Timedelta(days=lookback_days + 8)
        training_df = combined[combined["timestamp"] >= window_cutoff].reset_index(drop=True)

        # Per-row recency weights: exponential decay by age.
        # We align weights to training_df rows (before NaN drops), then
        # slice to match X_tr inside _prepare_supervised.
        n_raw = len(training_df)
        half_life_rows = max(1, recency_half_life_days * 48)  # 48 half-hour steps/day
        # positions: 0 = oldest row, n_raw-1 = newest row
        raw_positions = np.arange(n_raw)
        raw_weights = np.exp(raw_positions * np.log(2) / half_life_rows)
        raw_weights = (raw_weights / raw_weights.mean()).clip(0.05, 20.0)

        per_horizon = {}
        for h in self.horizons:
            X_tr, y_tr, X_va, y_va = self._prepare_supervised(training_df, h)
            if len(X_tr) < 10:
                continue

            # _prepare_supervised returns the first 80% of valid rows as X_tr.
            # Map raw_weights onto the valid (non-NaN) rows, then take the
            # first 80% to align with X_tr.
            feat_df, _ = build_feature_matrix(
                training_df, capacity_kwp=self.capacity_kwp, target="kw_import"
            )
            y_h = feat_df["kw_import"].shift(-h)
            valid_mask = ~y_h.isna()
            valid_indices = np.where(valid_mask)[0]
            split = int(len(valid_indices) * 0.8)
            train_indices = valid_indices[:split]
            # Clip indices to raw_weights length (safety guard)
            safe_indices = np.clip(train_indices, 0, len(raw_weights) - 1)
            w_tr = raw_weights[safe_indices]

            train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)

            for q in self.quantiles:
                params = self._make_params(q, h)
                params["learning_rate"] = learning_rate

                old_booster = self.boosters[(h, q)]
                new_booster = lgb.train(
                    params=params,
                    train_set=train_ds,
                    num_boost_round=n_rounds,
                    init_model=old_booster,
                    keep_training_booster=False,
                )
                self.boosters[(h, q)] = new_booster

            # Validate on the newest data (X_va is the last 20% of the window)
            median = self.boosters[(h, 0.5)]
            y_va_pred = median.predict(X_va)
            mae = float(np.mean(np.abs(y_va - y_va_pred)))
            mape = float(np.mean(np.abs((y_va - y_va_pred) / np.clip(y_va, 1.0, None))) * 100)
            per_horizon[h] = {"mae": mae, "mape": mape, "n_val": int(len(y_va))}
            if verbose:
                print(f"  h={h:2d}: MAE={mae:6.1f}  MAPE={mape:5.2f}%  "
                      f"(recency-weighted +{n_rounds} rounds, window={len(training_df)} rows)")

        self.metrics = {"per_horizon": per_horizon}
        self.n_train_rows = len(combined)
        self.last_update_method = f"recency_weighted_{n_rounds}_rounds"

        all_mape = [v["mape"] for v in per_horizon.values() if "mape" in v]
        return {
            "method": "recency_weighted_warm_start",
            "warm_start_rounds": n_rounds,
            "lookback_days": lookback_days,
            "n_train_rows": self.n_train_rows,
            "mean_mape": float(np.mean(all_mape)) if all_mape else float("nan"),
            "mape_at_h1": per_horizon.get(self.horizons[0], {}).get("mape"),
            "mape_at_h24": per_horizon.get(24, {}).get("mape"),
            "mape_at_h48": per_horizon.get(48, {}).get("mape"),
            "per_horizon": per_horizon,
        }

    # ===================================================================
    #                          FORECASTING
    # ===================================================================
    def forecast(
        self,
        output_steps: int = 48,
        bias_correction: float | np.ndarray = 0.0,
        conformal_half_width: np.ndarray | None = None,
    ) -> MultiStepForecastResult:
        """
        Predict at all trained horizons in one shot, then linearly interpolate
        between them to produce a smooth `output_steps`-point series.

        bias_correction:
          - scalar: additive offset (kW) applied to h=1 with linear decay to
            zero at the longest horizon (legacy behaviour).
          - 1-D array of length output_steps: per-horizon bias offset, applied
            as-is. Use this when you have a per-horizon error profile from
            recent verified predictions — backtest showed bias swings from
            +18 kW at h=20 to −16 kW at h=36, so a single h=1 value with
            linear decay was leaving error on the table.

        conformal_half_width: optional 1-D array of length output_steps,
        widening (p10, p90) to (median ± half_width) per horizon. Set this
        from empirical |residual| quantiles when the LightGBM quantile bands
        are systematically too narrow (43% coverage observed vs 80% target).
        """
        if not self.boosters or self.history is None:
            raise RuntimeError("Call fit() before forecast().")

        feat_df, _ = build_feature_matrix(
            self.history, capacity_kwp=self.capacity_kwp, target="kw_import"
        )
        X_last = feat_df[self.feature_cols].iloc[[-1]].values
        last_ts = self.history["timestamp"].max()

        # Predict at trained horizons
        raw = {}
        for h in self.horizons:
            row = {}
            for q in self.quantiles:
                pred = float(self.boosters[(h, q)].predict(X_last)[0])
                row[q] = max(0.0, pred)
            raw[h] = row

        # Build interpolated dense series (1, 2, ..., output_steps)
        dense_steps = np.arange(1, output_steps + 1)
        median = np.interp(dense_steps, self.horizons, [raw[h][0.5] for h in self.horizons])
        p10 = np.interp(dense_steps, self.horizons, [raw[h][0.1] for h in self.horizons])
        p90 = np.interp(dense_steps, self.horizons, [raw[h][0.9] for h in self.horizons])

        # Bias correction: scalar (legacy) or per-horizon array
        if isinstance(bias_correction, np.ndarray):
            correction = bias_correction[:output_steps]
            if len(correction) < output_steps:
                correction = np.pad(correction, (0, output_steps - len(correction)),
                                    constant_values=correction[-1] if len(correction) else 0.0)
        elif bias_correction != 0.0:
            decay = np.linspace(1.0, 0.0, output_steps)
            correction = float(bias_correction) * decay
        else:
            correction = np.zeros(output_steps)
        median = np.clip(median + correction, 0.0, None)
        p10 = np.clip(p10 + correction, 0.0, None)
        p90 = np.clip(p90 + correction, 0.0, None)

        # Conformal calibration: widen the band from empirical residuals.
        # Replaces (not adds to) the LightGBM band whenever it would be
        # narrower than the empirical one — quantile crossings get fixed below.
        if conformal_half_width is not None:
            hw = np.asarray(conformal_half_width, dtype=float)[:output_steps]
            if len(hw) < output_steps:
                hw = np.pad(hw, (0, output_steps - len(hw)),
                            constant_values=hw[-1] if len(hw) else 0.0)
            p10 = np.minimum(p10, median - hw)
            p90 = np.maximum(p90, median + hw)
            p10 = np.clip(p10, 0.0, None)

        # Enforce p10 ≤ median ≤ p90 (quantile crossings can occur)
        p10 = np.minimum(p10, median)
        p90 = np.maximum(p90, median)

        future_ts = pd.date_range(
            last_ts + pd.Timedelta(minutes=30),
            periods=output_steps,
            freq="30min",
        )

        return MultiStepForecastResult(
            timestamps=future_ts,
            median=median, p10=p10, p90=p90,
            last_history_ts=last_ts,
            raw_horizons=list(self.horizons),
            raw_predictions=raw,
        )

    def detect_peaks(self, fr: MultiStepForecastResult, top_n: int = 3) -> pd.DataFrame:
        order = np.argsort(-fr.median)[:top_n]
        return pd.DataFrame({
            "timestamp": fr.timestamps[order],
            "predicted_kw": fr.median[order],
            "lower_bound_kw": fr.p10[order],
            "upper_bound_kw": fr.p90[order],
        }).sort_values("timestamp").reset_index(drop=True)

    # ===================================================================
    #                          PERSISTENCE
    # ===================================================================
    def save(self, path: str | Path):
        # Boosters need to be serialized as strings then reloaded
        booster_strings = {k: b.model_to_string() for k, b in self.boosters.items()}
        joblib.dump({
            "capacity_kwp": self.capacity_kwp,
            "horizons": self.horizons,
            "quantiles": self.quantiles,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "boosters": booster_strings,
            "feature_cols": self.feature_cols,
            "history": self.history,
            "metrics": self.metrics,
            "n_train_rows": self.n_train_rows,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "DirectMultiStepForecaster":
        d = joblib.load(path)
        fc = cls(
            capacity_kwp=d["capacity_kwp"],
            horizons=d["horizons"],
            quantiles=d["quantiles"],
            n_estimators=d["n_estimators"],
            learning_rate=d["learning_rate"],
            random_state=d["random_state"],
        )
        fc.boosters = {k: lgb.Booster(model_str=s) for k, s in d["boosters"].items()}
        fc.feature_cols = d["feature_cols"]
        fc.history = d["history"]
        fc.metrics = d.get("metrics", {})
        fc.n_train_rows = d.get("n_train_rows", 0)
        return fc
