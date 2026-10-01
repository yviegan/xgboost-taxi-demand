from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_time(frame: pd.DataFrame, validation_days: int, test_days: int) -> TimeSplit:
    """Create contiguous train, validation, and test periods."""
    end = frame["timestamp"].max() + pd.Timedelta(hours=1)
    test_start = end - pd.Timedelta(days=test_days)
    validation_start = test_start - pd.Timedelta(days=validation_days)

    train = frame.loc[frame["timestamp"] < validation_start].copy()
    validation = frame.loc[frame["timestamp"].between(validation_start, test_start, inclusive="left")].copy()
    test = frame.loc[frame["timestamp"] >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Configured time windows produce an empty split.")
    return TimeSplit(train=train, validation=validation, test=test)


def fit_xgboost(
    split: TimeSplit,
    features: list[str],
    model_params: dict,
) -> XGBRegressor:
    params = model_params.copy()
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)
    model = XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds)
    model.fit(
        split.train[features],
        split.train["pickups"],
        eval_set=[(split.validation[features], split.validation["pickups"])],
        verbose=False,
    )
    return model


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "previous_hour": frame["pickups_lag_1"].to_numpy(),
        "previous_day": frame["pickups_lag_24"].to_numpy(),
        "previous_week": frame["pickups_lag_168"].to_numpy(),
    }


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.clip(np.asarray(predicted, dtype=float), 0, None)
    denominator = actual_array.sum()
    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(mean_squared_error(actual_array, predicted_array) ** 0.5),
        "wape": float(np.abs(actual_array - predicted_array).sum() / denominator) if denominator else float("nan"),
        "r2": float(r2_score(actual_array, predicted_array)),
    }
