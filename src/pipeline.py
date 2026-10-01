from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_hourly_panel_from_files
from .features import feature_columns, make_features
from .model import baseline_predictions, fit_xgboost, regression_metrics, split_by_time
from .rebalancing import greedy_rebalance, summarize_rebalancing

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_paths = sorted(PROJECT_ROOT.glob(config["raw_glob"]))
    print(f"Loading {len(raw_paths)} trip files...")
    panel, trip_rows = build_hourly_panel_from_files(
        raw_paths, config["min_zone_total_pickups"]
    )

    hourly_output = resolve_path(config["hourly_output"])
    hourly_output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(hourly_output, index=False)

    featured = make_features(panel, config["lags"], config["rolling_windows"]).dropna().reset_index(drop=True)
    split = split_by_time(featured, config["validation_days"], config["test_days"])
    features = feature_columns(featured)
    model = fit_xgboost(split, features, config["model"])

    predictions = np.clip(model.predict(split.test[features]), 0, None)
    metrics = {"xgboost": regression_metrics(split.test["pickups"], predictions)}
    for name, values in baseline_predictions(split.test).items():
        metrics[name] = regression_metrics(split.test["pickups"], values)

    latest_timestamp = split.test["timestamp"].max()
    latest = split.test.loc[split.test["timestamp"] == latest_timestamp, ["zone_id", "available_vehicles"]].copy()
    latest_predictions = predictions[split.test["timestamp"].to_numpy() == np.datetime64(latest_timestamp)]
    latest["predicted_demand"] = np.rint(latest_predictions).astype(int)
    latest["available_vehicles"] = latest["available_vehicles"].round().astype(int)
    moves = greedy_rebalance(latest)
    metrics["rebalancing"] = summarize_rebalancing(latest, moves)
    metrics["data"] = {
        "trip_rows_after_cleaning": int(trip_rows),
        "panel_rows": int(len(panel)),
        "zones": int(panel["zone_id"].nunique()),
        "start": panel["timestamp"].min().isoformat(),
        "end": panel["timestamp"].max().isoformat(),
        "test_start": split.test["timestamp"].min().isoformat(),
    }

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(resolve_path(config["importance_output"]), index=False)

    prediction_output = split.test[["timestamp", "zone_id", "pickups"]].copy()
    prediction_output["prediction"] = predictions
    prediction_output["absolute_error"] = (
        prediction_output["pickups"] - prediction_output["prediction"]
    ).abs()
    prediction_path = resolve_path(config["predictions_output"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.to_parquet(prediction_path, index=False)

    zone_metrics = (
        prediction_output.groupby("zone_id", as_index=False)
        .agg(
            observations=("pickups", "size"),
            actual_pickups=("pickups", "sum"),
            absolute_error=("absolute_error", "sum"),
            mae=("absolute_error", "mean"),
        )
    )
    zone_metrics["wape"] = zone_metrics["absolute_error"] / zone_metrics["actual_pickups"].replace(0, np.nan)
    zone_r2 = {
        int(zone_id): regression_metrics(group["pickups"], group["prediction"])["r2"]
        for zone_id, group in prediction_output.groupby("zone_id")
    }
    zone_metrics["r2"] = zone_metrics["zone_id"].map(zone_r2)
    zone_metrics.to_csv(resolve_path(config["zone_metrics_output"]), index=False)

    top_zone_ids = zone_metrics.nlargest(20, "actual_pickups")["zone_id"]
    top_zone_predictions = prediction_output[prediction_output["zone_id"].isin(top_zone_ids)]
    metrics["xgboost_diagnostics"] = {
        "median_zone_r2": float(zone_metrics["r2"].median()),
        "positive_r2_zone_share": float((zone_metrics["r2"] > 0).mean()),
        "top_20_demand_zones_r2": regression_metrics(
            top_zone_predictions["pickups"], top_zone_predictions["prediction"]
        )["r2"],
    }

    metrics_output = resolve_path(config["metrics_output"])
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = resolve_path(config["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    moves.assign(timestamp=latest_timestamp).to_csv(resolve_path(config["rebalancing_output"]), index=False)

    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NYC taxi demand forecasting baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.json",
        help="Path to the experiment configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config.resolve())
