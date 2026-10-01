from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from .evaluation import make_fold_spec, metric_row, overfit_diagnostic_row, split_for_fold
from .features import feature_columns, make_features
from .model import fit_xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sample_candidates(
    parameter_space: dict[str, list[Any]],
    manual_params: dict[str, Any],
    n_random_candidates: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Return deterministic unique random candidates plus the manual baseline."""
    total_combinations = int(np.prod([len(values) for values in parameter_space.values()]))
    sampled = list(
        ParameterSampler(
            parameter_space,
            n_iter=min(n_random_candidates + 1, total_combinations),
            random_state=random_state,
        )
    )
    manual_key = json.dumps(manual_params, sort_keys=True)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for params in sampled:
        key = json.dumps(params, sort_keys=True)
        if key not in seen and key != manual_key:
            unique.append(params)
            seen.add(key)
    unique = unique[:n_random_candidates]
    unique.append(manual_params.copy())
    return unique


def validate_fold_separation(
    development_folds: list[dict[str, str]],
    holdout_folds: list[dict[str, str]],
    test_days: int,
) -> None:
    latest_development_end = max(
        pd.Timestamp(fold["test_start"]) + pd.Timedelta(days=test_days)
        for fold in development_folds
    )
    earliest_holdout_start = min(
        pd.Timestamp(fold["test_start"]) for fold in holdout_folds
    )
    if latest_development_end > earliest_holdout_start:
        raise ValueError("Development and holdout windows overlap.")


def summarize_candidates(trials: pd.DataFrame) -> pd.DataFrame:
    summary = (
        trials.groupby(["candidate_id", "candidate_type", "params_json"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            validation_wape_mean=("validation_wape", "mean"),
            validation_wape_std=("validation_wape", "std"),
            validation_wape_max=("validation_wape", "max"),
            validation_r2_mean=("validation_r2", "mean"),
            best_iteration_mean=("best_iteration", "mean"),
        )
        .sort_values(
            ["validation_wape_mean", "validation_wape_std", "validation_wape_max"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def select_candidate(candidate_summary: pd.DataFrame) -> dict[str, Any]:
    """Select only from development metrics; holdout results are not accepted."""
    required = {"candidate_id", "params_json", "validation_wape_mean", "validation_wape_std"}
    missing = required - set(candidate_summary.columns)
    if missing:
        raise ValueError(f"Missing development-summary columns: {sorted(missing)}")
    selected = candidate_summary.sort_values(
        ["validation_wape_mean", "validation_wape_std", "validation_wape_max"],
        kind="stable",
    ).iloc[0]
    return {
        "candidate_id": str(selected["candidate_id"]),
        "candidate_type": str(selected["candidate_type"]),
        "params": json.loads(selected["params_json"]),
        "development_validation_wape_mean": float(selected["validation_wape_mean"]),
        "development_validation_wape_std": float(selected["validation_wape_std"]),
    }


def _full_params(fixed_params: dict[str, Any], candidate_params: dict[str, Any]) -> dict[str, Any]:
    return {**fixed_params, **candidate_params}


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_fold_separation(
        config["development_folds"], config["holdout_folds"], config["test_days"]
    )
    panel = pd.read_parquet(resolve_path(config["hourly_input"]))
    featured = (
        make_features(panel, config["lags"], config["rolling_windows"])
        .dropna()
        .reset_index(drop=True)
    )
    features = feature_columns(featured)
    candidates = sample_candidates(
        config["parameter_space"],
        config["manual_params"],
        config["n_random_candidates"],
        config["random_state"],
    )

    trial_rows: list[dict[str, Any]] = []
    for candidate_number, candidate_params in enumerate(candidates, start=1):
        candidate_id = f"candidate_{candidate_number:02d}"
        candidate_type = "manual_baseline" if candidate_number == len(candidates) else "random_search"
        params_json = json.dumps(candidate_params, sort_keys=True)
        for fold_config in config["development_folds"]:
            fold = make_fold_spec(
                fold_config["name"], fold_config["test_start"],
                config["validation_days"], config["test_days"],
            )
            split = split_for_fold(featured, fold)
            model = fit_xgboost(split, features, _full_params(config["fixed_model_params"], candidate_params))
            prediction = np.clip(model.predict(split.test[features]), 0, None)
            metrics = metric_row(
                fold.name, "development_validation", candidate_id,
                split.test["pickups"], prediction,
            )
            trial_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_type": candidate_type,
                    "params_json": params_json,
                    "fold": fold.name,
                    "validation_wape": metrics["wape"],
                    "validation_mae": metrics["mae"],
                    "validation_rmse": metrics["rmse"],
                    "validation_r2": metrics["r2"],
                    "best_iteration": int(model.best_iteration) + 1,
                }
            )
        candidate_trials = trial_rows[-len(config["development_folds"]):]
        mean_wape = np.mean([row["validation_wape"] for row in candidate_trials])
        print(f"{candidate_id}/{len(candidates)}: development WAPE={mean_wape:.4f}")

    trials = pd.DataFrame(trial_rows)
    candidate_summary = summarize_candidates(trials)
    selected = select_candidate(candidate_summary)
    selected["selection_rule"] = "lowest mean development WAPE; then lower standard deviation and worst-fold WAPE"
    selected["development_folds"] = config["development_folds"]

    holdout_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    comparison = {
        "tuned": selected["params"],
        "manual": config["manual_params"],
    }
    for fold_config in config["holdout_folds"]:
        fold = make_fold_spec(
            fold_config["name"], fold_config["test_start"],
            config["validation_days"], config["test_days"],
        )
        split = split_for_fold(featured, fold)
        for model_name, candidate_params in comparison.items():
            model = fit_xgboost(
                split, features,
                _full_params(config["fixed_model_params"], candidate_params),
            )
            model_rows: list[dict[str, Any]] = []
            for dataset_name, dataset in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            ):
                prediction = np.clip(model.predict(dataset[features]), 0, None)
                row = metric_row(
                    fold.name, dataset_name, model_name,
                    dataset["pickups"], prediction,
                )
                holdout_rows.append(row)
                model_rows.append(row)
            if model_name == "tuned":
                diagnostic_rows.append(
                    overfit_diagnostic_row(
                        fold.name,
                        pd.DataFrame(model_rows),
                        int(model.best_iteration) + 1,
                    )
                )
        print(f"Evaluated locked parameters on {fold.name}")

    holdout_metrics = pd.DataFrame(holdout_rows)
    test_metrics = holdout_metrics.loc[holdout_metrics["dataset"] == "test"]
    holdout_summary = (
        test_metrics.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
            wape_max=("wape", "max"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
        )
    )
    tuned_wape = float(holdout_summary.loc[holdout_summary["model"] == "tuned", "wape_mean"].iloc[0])
    manual_wape = float(holdout_summary.loc[holdout_summary["model"] == "manual", "wape_mean"].iloc[0])
    tuned_by_fold = test_metrics.loc[test_metrics["model"] == "tuned"].set_index("fold")["wape"]
    manual_by_fold = test_metrics.loc[test_metrics["model"] == "manual"].set_index("fold")["wape"]
    outcome = {
        "tuned_wape_mean": tuned_wape,
        "manual_wape_mean": manual_wape,
        "relative_wape_improvement": 1 - tuned_wape / manual_wape,
        "tuned_wins": int((tuned_by_fold < manual_by_fold).sum()),
        "holdout_folds": len(config["holdout_folds"]),
    }

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_dir / "trial_results.csv", index=False)
    candidate_summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    (output_dir / "selected_params.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    holdout_metrics.to_csv(output_dir / "holdout_metrics.csv", index=False)
    holdout_summary.to_csv(output_dir / "holdout_summary.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "overfit_diagnostics.csv", index=False)
    (output_dir / "holdout_summary.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "holdout": outcome}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost with time-safe development windows.")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "tuning.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config.resolve())
