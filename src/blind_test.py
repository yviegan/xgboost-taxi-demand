from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_panel_extension, clean_trips, load_trips
from .evaluation import make_fold_spec, metric_row, overfit_diagnostic_row, split_for_fold
from .features import feature_columns, make_features
from .model import baseline_predictions, fit_xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    historical = pd.read_parquet(resolve_path(config["historical_panel"]))
    blind_trips = clean_trips(load_trips([resolve_path(config["blind_month_file"])]))
    extension = build_panel_extension(historical, blind_trips)
    panel = (
        pd.concat([historical, extension], ignore_index=True)
        .drop_duplicates(["zone_id", "timestamp"], keep="last")
        .sort_values(["zone_id", "timestamp"])
        .reset_index(drop=True)
    )
    featured = (
        make_features(panel, config["lags"], config["rolling_windows"])
        .dropna()
        .reset_index(drop=True)
    )
    features = feature_columns(featured)

    selected = json.loads(resolve_path(config["selected_params"]).read_text(encoding="utf-8"))
    model_params = {**config["fixed_model_params"], **selected["params"]}
    fold = make_fold_spec(
        "blind-2025-07-end",
        config["test_start"],
        config["validation_days"],
        config["test_days"],
    )
    split = split_for_fold(featured, fold)
    model = fit_xgboost(split, features, model_params)

    metric_rows: list[dict] = []
    xgboost_rows: list[dict] = []
    test_predictions: np.ndarray | None = None
    for dataset_name, dataset in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        predicted = np.clip(model.predict(dataset[features]), 0, None)
        row = metric_row(fold.name, dataset_name, "xgboost_tuned", dataset["pickups"], predicted)
        metric_rows.append(row)
        xgboost_rows.append(row)
        if dataset_name == "test":
            test_predictions = predicted

    for baseline_name, predicted in baseline_predictions(split.test).items():
        metric_rows.append(
            metric_row(fold.name, "test", baseline_name, split.test["pickups"], predicted)
        )

    metrics = pd.DataFrame(metric_rows)
    test_metrics = metrics.loc[metrics["dataset"] == "test"].copy()
    best_baseline = test_metrics.loc[test_metrics["model"] != "xgboost_tuned"].sort_values("wape").iloc[0]
    tuned = test_metrics.loc[test_metrics["model"] == "xgboost_tuned"].iloc[0]
    summary = {
        "protocol": "parameters locked before downloading and evaluating July data",
        "test_start": split.test["timestamp"].min().isoformat(),
        "test_end": split.test["timestamp"].max().isoformat(),
        "blind_month_valid_trips": int(len(blind_trips)),
        "zones": int(split.test["zone_id"].nunique()),
        "best_iteration": int(model.best_iteration) + 1,
        "xgboost_wape": float(tuned["wape"]),
        "xgboost_mae": float(tuned["mae"]),
        "xgboost_rmse": float(tuned["rmse"]),
        "xgboost_r2": float(tuned["r2"]),
        "best_baseline": str(best_baseline["model"]),
        "best_baseline_wape": float(best_baseline["wape"]),
        "relative_wape_improvement": float(1 - tuned["wape"] / best_baseline["wape"]),
    }

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [overfit_diagnostic_row(fold.name, pd.DataFrame(xgboost_rows), int(model.best_iteration) + 1)]
    ).to_csv(output_dir / "overfit_diagnostic.csv", index=False)
    assert test_predictions is not None
    predictions = split.test[["timestamp", "zone_id", "pickups"]].copy()
    predictions["prediction"] = test_predictions
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    model.save_model(output_dir / "xgboost_tuned.json")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate locked parameters on the July blind test.")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "blind_test.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config.resolve())
