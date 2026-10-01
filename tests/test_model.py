import pandas as pd

from src.model import regression_metrics, split_by_time


def test_time_split_is_contiguous_and_ordered() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=24 * 20, freq="h"),
            "pickups": 1,
        }
    )

    split = split_by_time(frame, validation_days=4, test_days=3)

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    assert split.validation["timestamp"].nunique() == 4 * 24
    assert split.test["timestamp"].nunique() == 3 * 24


def test_regression_metrics_include_r2() -> None:
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])

    assert metrics["r2"] == 1.0
