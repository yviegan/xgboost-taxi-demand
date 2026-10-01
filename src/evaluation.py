from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import TimeSplit, regression_metrics


@dataclass(frozen=True)
class FoldSpec:
    name: str
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_fold_spec(
    name: str,
    test_start: str | pd.Timestamp,
    validation_days: int,
    test_days: int,
) -> FoldSpec:
    test_start_timestamp = pd.Timestamp(test_start)
    return FoldSpec(
        name=name,
        validation_start=test_start_timestamp - pd.Timedelta(days=validation_days),
        test_start=test_start_timestamp,
        test_end=test_start_timestamp + pd.Timedelta(days=test_days),
    )


def split_for_fold(frame: pd.DataFrame, fold: FoldSpec) -> TimeSplit:
    """Create an expanding training split with fixed validation and test windows."""
    train = frame.loc[frame["timestamp"] < fold.validation_start].copy()
    validation = frame.loc[
        frame["timestamp"].between(fold.validation_start, fold.test_start, inclusive="left")
    ].copy()
    test = frame.loc[
        frame["timestamp"].between(fold.test_start, fold.test_end, inclusive="left")
    ].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Fold {fold.name} produces an empty split.")
    if not (
        train["timestamp"].max() < validation["timestamp"].min()
        and validation["timestamp"].max() < test["timestamp"].min()
    ):
        raise ValueError(f"Fold {fold.name} has overlapping time ranges.")
    return TimeSplit(train=train, validation=validation, test=test)


def metric_row(
    fold: str,
    dataset: str,
    model: str,
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    metrics = regression_metrics(actual, predicted)
    return {
        "fold": fold,
        "dataset": dataset,
        "model": model,
        "observations": int(len(actual)),
        **metrics,
    }


def training_top_zones(train: pd.DataFrame, count: int = 20) -> set[int]:
    """Select high-demand zones using training data only."""
    totals = train.groupby("zone_id", observed=True)["pickups"].sum()
    return set(int(zone_id) for zone_id in totals.nlargest(count).index)


def add_segments(
    predictions: pd.DataFrame,
    top_zones: set[int],
    peak_hours: list[int],
) -> pd.DataFrame:
    segmented = predictions.copy()
    is_weekday = segmented["timestamp"].dt.dayofweek < 5
    segmented["peak_segment"] = np.where(
        is_weekday & segmented["timestamp"].dt.hour.isin(peak_hours), "peak", "off_peak"
    )
    segmented["zone_segment"] = np.where(
        segmented["zone_id"].isin(top_zones), "top_20", "other"
    )
    segmented["day_segment"] = np.where(is_weekday, "weekday", "weekend")
    return segmented


def segment_metric_rows(fold: str, predictions: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    dimensions = {
        "peak": "peak_segment",
        "zone_demand": "zone_segment",
        "day_type": "day_segment",
    }
    for dimension, column in dimensions.items():
        for segment, group in predictions.groupby(column, observed=True):
            rows.append(
                {
                    "fold": fold,
                    "dimension": dimension,
                    "segment": str(segment),
                    "observations": int(len(group)),
                    "actual_pickups": int(group["pickups"].sum()),
                    **regression_metrics(group["pickups"], group["prediction"].to_numpy()),
                }
            )
    return rows


def overfit_diagnostic_row(
    fold: str,
    metrics: pd.DataFrame,
    best_iteration: int,
) -> dict[str, float | int | str]:
    indexed = metrics.set_index("dataset")
    train = indexed.loc["train"]
    validation = indexed.loc["validation"]
    test = indexed.loc["test"]
    return {
        "fold": fold,
        "best_iteration": best_iteration,
        "train_wape": train["wape"],
        "validation_wape": validation["wape"],
        "test_wape": test["wape"],
        "validation_minus_train_wape": validation["wape"] - train["wape"],
        "test_minus_train_wape": test["wape"] - train["wape"],
        "train_r2": train["r2"],
        "validation_r2": validation["r2"],
        "test_r2": test["r2"],
        "train_minus_validation_r2": train["r2"] - validation["r2"],
        "train_minus_test_r2": train["r2"] - test["r2"],
    }


def summarize_test_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    test = fold_metrics.loc[fold_metrics["dataset"] == "test"]
    summary = (
        test.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
            wape_min=("wape", "min"),
            wape_max=("wape", "max"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
    )
    baseline_wape = summary.loc[summary["model"] != "xgboost", "wape_mean"].min()
    summary["wape_improvement_vs_best_baseline"] = 1 - summary["wape_mean"] / baseline_wape
    return summary
