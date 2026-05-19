"""
DirectMultiStepForecaster
Direct multi-step LightGBM forecasting with quantile bands.
[MODIFIED: import changed from src.features to predictor.features for integrated package]
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from predictor.features import build_feature_matrix


DEFAULT_HORIZONS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    14, 16, 18, 20, 22, 24,
    28, 32, 36, 40, 44, 48,
]
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class MultiStepForecastResult:
    timestamps: pd.DatetimeIndex
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    last_history_ts: pd.Timestamp
    raw_horizons: list[int]
    raw_predictions: dict[int, dict]


class DirectMultiStepForecaster:
    def __init__(self, capacity_kwp: float = 0.0, horizons: list[int] | None = None,
                 quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
                 n_estimators: int = 250, learning_rate: float = 0.05, random_state: int = 42):
        self.capacity_kwp = capacity_kwp
        self.horizons = list(horizons) if horizons else list(DEFAULT_HORIZONS)
        self.quantiles = tuple(quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.boosters: dict[tuple[int, float], lgb.Booster] = {}
        self.feature_cols: list[str] = []
        self.history: Optional[pd.DataFrame] = None
        self.metrics: dict = {}
        self.n_train_rows: int = 0
        self.last_update_method: str = "none"

    def _prepare_supervised(self, df: pd.DataFrame, h: int):
        feat_df, feature_cols = build_feature_matrix(df, capacity_kwp=self.capacity_kwp, target="kw_import")
        self.feature_cols = feature_cols
        y_h = feat_df["kw_import"].shift(-h)
        valid = ~y_h.isna()
        X_full = feat_df[feature_cols][valid].values
        y_full = y_h[valid].values
        split = int(len(X_full) * 0.8)
        return X_full[:split], y_full[:split], X_full[split:], y_full[split:]

    def _make_params(self, q: float, h: int = 1) -> dict:
        num_leaves = min(127, 31 + h * 2)
        min_child = 8 if (q <= 0.15 or q >= 0.85) else 15
        return {
            "objective": "quantile", "alpha": q, "metric": "quantile",
            "learning_rate": self.learning_rate, "num_leaves": num_leaves,
            "min_child_samples": min_child, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "verbose": -1, "seed": self.random_state,
        }

    def fit(self, df: pd.DataFrame, verbose: bool = False) -> dict:
        self.history = df.copy().sort_values("timestamp").reset_index(drop=True)
        self.boosters.clear()
        per_horizon = {}
        for h in self.horizons:
            X_tr, y_tr, X_va, y_va = self._prepare_supervised(self.history, h)
            train_ds = lgb.Dataset(X_tr, label=y_tr)
            val_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)
            for q in self.quantiles:
                booster = lgb.train(
                    params=self._make_params(q, h), train_set=train_ds,
                    num_boost_round=self.n_estimators, valid_sets=[val_ds],
                    callbacks=[lgb.early_stopping(20, verbose=False)],
                )
                self.boosters[(h, q)] = booster
            median_booster = self.boosters[(h, 0.5)]
            y_va_pred = median_booster.predict(X_va)
            mae = float(np.mean(np.abs(y_va - y_va_pred)))
            mape = float(np.mean(np.abs((y_va - y_va_pred) / np.clip(y_va, 1.0, None))) * 100)
            pinball_total = 0.0
            for q in self.quantiles:
                yp = self.boosters[(h, q)].predict(X_va)
                err = y_va - yp
                pinball_total += float(np.mean(np.maximum(q * err, (q - 1) * err)))
            per_horizon[h] = {"mae": mae, "mape": mape,
                               "pinball": pinball_total / len(self.quantiles), "n_val": int(len(y_va))}
            if verbose:
                print(f"  h={h:2d}: MAE={mae:6.1f} kW  MAPE={mape:5.2f}%")

        self.metrics = {"per_horizon": per_horizon}
        self.n_train_rows = len(self.history)
        self.last_update_method = "full_fit"
        all_mape = [v["mape"] for v in per_horizon.values()]
        return {
            "method": "full_fit", "n_train_rows": self.n_train_rows,
            "n_models_trained": len(self.boosters),
            "mean_mape": float(np.mean(all_mape)),
            "mape_at_h1": per_horizon[self.horizons[0]]["mape"],
            "mape_at_h24": per_horizon.get(24, {}).get("mape"),
            "mape_at_h48": per_horizon.get(48, {}).get("mape"),
            "per_horizon": per_horizon,
        }

    def update(self, new_df: pd.DataFrame, n_rounds: int = 50, learning_rate: float = 0.03,
               lookback_days: float = 30.0, recency_half_life_days: float = 7.0,
               verbose: bool = False) -> dict:
        if not self.boosters or self.history is None:
            return self.fit(new_df, verbose=verbose)
        combined = pd.concat([self.history, new_df], ignore_index=True)
        combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        self.history = combined
        max_ts = combined["timestamp"].max()
        window_cutoff = max_ts - pd.Timedelta(days=lookback_days + 8)
        training_df = combined[combined["timestamp"] >= window_cutoff].reset_index(drop=True)
        n_raw = len(training_df)
        half_life_rows = max(1, recency_half_life_days * 48)
        raw_positions = np.arange(n_raw)
        raw_weights = np.exp(raw_positions * np.log(2) / half_life_rows)
        raw_weights = (raw_weights / raw_weights.mean()).clip(0.05, 20.0)

        per_horizon = {}
        for h in self.horizons:
            X_tr, y_tr, X_va, y_va = self._prepare_supervised(training_df, h)
            if len(X_tr) < 10:
                continue
            feat_df, _ = build_feature_matrix(training_df, capacity_kwp=self.capacity_kwp, target="kw_import")
            y_h = feat_df["kw_import"].shift(-h)
            valid_mask = ~y_h.isna()
            valid_indices = np.where(valid_mask)[0]
            split = int(len(valid_indices) * 0.8)
            train_indices = valid_indices[:split]
            safe_indices = np.clip(train_indices, 0, len(raw_weights) - 1)
            w_tr = raw_weights[safe_indices]
            train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
            for q in self.quantiles:
                params = self._make_params(q, h)
                params["learning_rate"] = learning_rate
                new_booster = lgb.train(
                    params=params, train_set=train_ds, num_boost_round=n_rounds,
                    init_model=self.boosters[(h, q)], keep_training_booster=False,
                )
                self.boosters[(h, q)] = new_booster
            median = self.boosters[(h, 0.5)]
            y_va_pred = median.predict(X_va)
            mae = float(np.mean(np.abs(y_va - y_va_pred)))
            mape = float(np.mean(np.abs((y_va - y_va_pred) / np.clip(y_va, 1.0, None))) * 100)
            per_horizon[h] = {"mae": mae, "mape": mape, "n_val": int(len(y_va))}

        self.metrics = {"per_horizon": per_horizon}
        self.n_train_rows = len(combined)
        self.last_update_method = f"recency_weighted_{n_rounds}_rounds"
        all_mape = [v["mape"] for v in per_horizon.values() if "mape" in v]
        return {
            "method": "recency_weighted_warm_start", "warm_start_rounds": n_rounds,
            "lookback_days": lookback_days, "n_train_rows": self.n_train_rows,
            "mean_mape": float(np.mean(all_mape)) if all_mape else float("nan"),
            "mape_at_h1": per_horizon.get(self.horizons[0], {}).get("mape"),
            "mape_at_h24": per_horizon.get(24, {}).get("mape"),
            "mape_at_h48": per_horizon.get(48, {}).get("mape"),
            "per_horizon": per_horizon,
        }

    def forecast(self, output_steps: int = 48, bias_correction: float | np.ndarray = 0.0,
                 conformal_half_width: np.ndarray | None = None) -> MultiStepForecastResult:
        if not self.boosters or self.history is None:
            raise RuntimeError("Call fit() before forecast().")
        feat_df, _ = build_feature_matrix(self.history, capacity_kwp=self.capacity_kwp, target="kw_import")
        X_last = feat_df[self.feature_cols].iloc[[-1]].values
        last_ts = self.history["timestamp"].max()
        raw = {}
        for h in self.horizons:
            row = {}
            for q in self.quantiles:
                pred = float(self.boosters[(h, q)].predict(X_last)[0])
                row[q] = max(0.0, pred)
            raw[h] = row

        dense_steps = np.arange(1, output_steps + 1)
        median = np.interp(dense_steps, self.horizons, [raw[h][0.5] for h in self.horizons])
        p10 = np.interp(dense_steps, self.horizons, [raw[h][0.1] for h in self.horizons])
        p90 = np.interp(dense_steps, self.horizons, [raw[h][0.9] for h in self.horizons])

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

        if conformal_half_width is not None:
            hw = np.asarray(conformal_half_width, dtype=float)[:output_steps]
            if len(hw) < output_steps:
                hw = np.pad(hw, (0, output_steps - len(hw)),
                            constant_values=hw[-1] if len(hw) else 0.0)
            p10 = np.minimum(p10, median - hw)
            p90 = np.maximum(p90, median + hw)
            p10 = np.clip(p10, 0.0, None)

        p10 = np.minimum(p10, median)
        p90 = np.maximum(p90, median)
        future_ts = pd.date_range(last_ts + pd.Timedelta(minutes=30), periods=output_steps, freq="30min")
        return MultiStepForecastResult(
            timestamps=future_ts, median=median, p10=p10, p90=p90,
            last_history_ts=last_ts, raw_horizons=list(self.horizons), raw_predictions=raw,
        )

    def detect_peaks(self, fr: MultiStepForecastResult, top_n: int = 3) -> pd.DataFrame:
        order = np.argsort(-fr.median)[:top_n]
        return pd.DataFrame({
            "timestamp": fr.timestamps[order],
            "predicted_kw": fr.median[order],
            "lower_bound_kw": fr.p10[order],
            "upper_bound_kw": fr.p90[order],
        }).sort_values("timestamp").reset_index(drop=True)

    def save(self, path: str | Path):
        booster_strings = {k: b.model_to_string() for k, b in self.boosters.items()}
        joblib.dump({
            "capacity_kwp": self.capacity_kwp, "horizons": self.horizons,
            "quantiles": self.quantiles, "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate, "random_state": self.random_state,
            "boosters": booster_strings, "feature_cols": self.feature_cols,
            "history": self.history, "metrics": self.metrics, "n_train_rows": self.n_train_rows,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "DirectMultiStepForecaster":
        d = joblib.load(path)
        fc = cls(capacity_kwp=d["capacity_kwp"], horizons=d["horizons"], quantiles=d["quantiles"],
                 n_estimators=d["n_estimators"], learning_rate=d["learning_rate"],
                 random_state=d["random_state"])
        fc.boosters = {k: lgb.Booster(model_str=s) for k, s in d["boosters"].items()}
        fc.feature_cols = d["feature_cols"]
        fc.history = d["history"]
        fc.metrics = d.get("metrics", {})
        fc.n_train_rows = d.get("n_train_rows", 0)
        return fc
