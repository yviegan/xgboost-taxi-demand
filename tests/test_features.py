import pandas as pd

from src.data import build_hourly_panel, build_panel_extension
from src.features import make_features


def test_rolling_features_never_include_current_target() -> None:
    panel = pd.DataFrame(
        {
            "zone_id": [1] * 5,
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
            "pickups": [1, 2, 100, 4, 5],
            "dropoffs": [5, 6, 7, 8, 9],
        }
    )

    result = make_features(panel, lags=[1], rolling_windows=[2])

    assert result.loc[2, "pickups_lag_1"] == 2
    assert result.loc[2, "pickups_roll_mean_2"] == 1.5
    assert result.loc[2, "available_vehicles"] == 6


def test_dropoff_after_horizon_does_not_extend_target_panel() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-31 23:50"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-02-01 00:10"]),
            "PULocationID": [1],
            "DOLocationID": [2],
        }
    )

    panel = build_hourly_panel(trips, min_zone_total_pickups=1)

    assert panel["timestamp"].max() == pd.Timestamp("2025-01-31 23:00")


def test_panel_extension_keeps_existing_zone_universe() -> None:
    existing = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "pickups": [1, 1],
            "dropoffs": [1, 1],
        }
    )
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-02-01 00:05"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-02-01 00:20"]),
            "PULocationID": [3],
            "DOLocationID": [3],
        }
    )

    extension = build_panel_extension(existing, trips)

    assert set(extension["zone_id"]) == {1, 2}
    assert extension["pickups"].sum() == 0
