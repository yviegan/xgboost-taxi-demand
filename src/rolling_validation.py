from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    add_segments,
    make_fold_spec,
    metric_row,
    overfit_diagnostic_row,
    segment_metric_rows,
    split_for_fold,
    summarize_test_metrics,
    training_top_zones,
)
from .features import feature_columns, make_features
from .model import baseline_predictions, fit_xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panel = pd.read_parquet(resolve_path(config["hourly_input"]))
    featured = (
        make_features(panel, config["lags"], config["rolling_windows"])
        .dropna()
        .reset_index(drop=True)
    )
    features = feature_columns(featured)

    fold_metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    segment_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_config in config["folds"]:
        fold = make_fold_spec(
            fold_config["name"],
            fold_config["test_start"],
            config["validation_days"],
            config["test_days"],
        )
        split = split_for_fold(featured, fold)
        model = fit_xgboost(split, features, config["model"])
        xgboost_rows: list[dict] = []

        for dataset_name, dataset in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            predicted = np.clip(model.predict(dataset[features]), 0, None)
            row = metric_row(fold.name, dataset_name, "xgboost", dataset["pickups"], predicted)
            fold_metric_rows.append(row)
            xgboost_rows.append(row)

            if dataset_name == "test":
                prediction_frame = dataset[["timestamp", "zone_id", "pickups"]].copy()
                prediction_frame["prediction"] = predicted
                prediction_frame["fold"] = fold.name
                top_zones = training_top_zones(split.train, config["top_zone_count"])
                prediction_frame = add_segments(
                    prediction_frame, top_zones, config["peak_hours"]
                )
                prediction_frames.append(prediction_frame)
                segment_rows.extend(segment_metric_rows(fold.name, prediction_frame))

        for baseline_name, baseline_values in baseline_predictions(split.test).items():
            fold_metric_rows.append(
                metric_row(
                    fold.name,
                    "test",
                    baseline_name,
                    split.test["pickups"],
                    baseline_values,
                )
            )

        fold_xgboost_metrics = pd.DataFrame(xgboost_rows)
        best_iteration = int(model.best_iteration) + 1
        diagnostic_rows.append(
            overfit_diagnostic_row(fold.name, fold_xgboost_metrics, best_iteration)
        )
        print(
            f"{fold.name}: test WAPE={fold_xgboost_metrics.loc[fold_xgboost_metrics['dataset'] == 'test', 'wape'].iloc[0]:.4f}, "
            f"best_iteration={best_iteration}"
        )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    segments = pd.DataFrame(segment_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_test_metrics(fold_metrics)

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "fold_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "overfit_diagnostics.csv", index=False)
    segments.to_csv(output_dir / "segment_metrics.csv", index=False)
    segment_summary = (
        segments.groupby(["dimension", "segment"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
            r2_mean=("r2", "mean"),
        )
    )
    segment_summary.to_csv(output_dir / "segment_summary.csv", index=False)
    predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)

    xgboost_summary = summary.loc[summary["model"] == "xgboost"].iloc[0]
    worst_fold = (
        fold_metrics.loc[
            (fold_metrics["model"] == "xgboost") & (fold_metrics["dataset"] == "test")
        ]
        .nlargest(1, "wape")
        .iloc[0]
    )
    core_summary = {
        "folds": int(xgboost_summary["folds"]),
        "xgboost_wape_mean": float(xgboost_summary["wape_mean"]),
        "xgboost_wape_std": float(xgboost_summary["wape_std"]),
        "xgboost_r2_mean": float(xgboost_summary["r2_mean"]),
        "xgboost_r2_std": float(xgboost_summary["r2_std"]),
        "wape_improvement_vs_best_baseline": float(
            xgboost_summary["wape_improvement_vs_best_baseline"]
        ),
        "worst_fold": str(worst_fold["fold"]),
        "worst_fold_wape": float(worst_fold["wape"]),
        "mean_validation_minus_train_wape": float(
            diagnostics["validation_minus_train_wape"].mean()
        ),
        "mean_test_minus_train_wape": float(
            diagnostics["test_minus_train_wape"].mean()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(core_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(core_summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling time validation and error analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "rolling_validation.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config.resolve())
