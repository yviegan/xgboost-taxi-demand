import pandas as pd
import pytest

from src.evaluation import (
    add_segments,
    make_fold_spec,
    split_for_fold,
    summarize_test_metrics,
    training_top_zones,
)


def test_fold_ranges_are_ordered_and_exact() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", "2025-02-01", freq="h", inclusive="left"),
            "zone_id": 1,
            "pickups": 1,
        }
    )
    fold = make_fold_spec("test", "2025-01-20", validation_days=5, test_days=4)

    split = split_for_fold(frame, fold)

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    assert split.validation["timestamp"].nunique() == 5 * 24
    assert split.test["timestamp"].nunique() == 4 * 24


def test_top_zones_use_training_demand_only() -> None:
    train = pd.DataFrame(
        {"zone_id": [1, 1, 2, 2, 3], "pickups": [10, 10, 9, 9, 100]}
    )

    assert training_top_zones(train, count=2) == {1, 3}


def test_segments_are_mutually_exclusive_and_complete() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-06 08:00", "2025-01-06 12:00", "2025-01-11 08:00"]
            ),
            "zone_id": [1, 2, 1],
            "pickups": [10, 5, 8],
            "prediction": [9, 6, 7],
        }
    )

    result = add_segments(predictions, top_zones={1}, peak_hours=[7, 8, 9])

    assert result["peak_segment"].tolist() == ["peak", "off_peak", "off_peak"]
    assert result["zone_segment"].tolist() == ["top_20", "other", "top_20"]
    assert result["day_segment"].tolist() == ["weekday", "weekday", "weekend"]
    assert result[["peak_segment", "zone_segment", "day_segment"]].notna().all().all()


def test_summary_uses_mean_of_fold_metrics() -> None:
    metrics = pd.DataFrame(
        [
            {"fold": "a", "dataset": "test", "model": "xgboost", "mae": 1, "rmse": 2, "wape": 0.1, "r2": 0.9},
            {"fold": "b", "dataset": "test", "model": "xgboost", "mae": 3, "rmse": 4, "wape": 0.2, "r2": 0.8},
            {"fold": "a", "dataset": "test", "model": "baseline", "mae": 4, "rmse": 5, "wape": 0.4, "r2": 0.5},
            {"fold": "b", "dataset": "test", "model": "baseline", "mae": 4, "rmse": 5, "wape": 0.4, "r2": 0.5},
        ]
    )

    summary = summarize_test_metrics(metrics).set_index("model")

    assert summary.loc["xgboost", "wape_mean"] == pytest.approx(0.15)
    assert summary.loc["xgboost", "wape_improvement_vs_best_baseline"] == pytest.approx(0.625)
