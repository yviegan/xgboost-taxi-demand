from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(
    panel: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Create time and demand-history features without using the current target."""
    featured = panel.sort_values(["zone_id", "timestamp"]).copy()
    timestamp = featured["timestamp"]
    featured["hour"] = timestamp.dt.hour
    featured["day_of_week"] = timestamp.dt.dayofweek
    featured["month"] = timestamp.dt.month
    featured["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")
    featured["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    featured["dow_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    featured["dow_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)

    grouped = featured.groupby("zone_id", sort=False)["pickups"]
    for lag in lags:
        featured[f"pickups_lag_{lag}"] = grouped.shift(lag)

    for window in rolling_windows:
        featured[f"pickups_roll_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).mean()
        )
        featured[f"pickups_roll_std_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).std()
        )

    featured["available_vehicles"] = featured.groupby("zone_id", sort=False)["dropoffs"].shift(1)
    return featured


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "pickups", "dropoffs"}
    return [column for column in frame.columns if column not in excluded]
