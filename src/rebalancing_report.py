from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import clean_trips, load_trips
from .rebalancing import (
    greedy_rebalance,
    historical_travel_time_costs,
    summarize_rebalancing,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_zone_state(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    timestamp: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    predictions = predictions.copy()
    panel = panel.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    snapshot = predictions["timestamp"].max() if timestamp is None else pd.Timestamp(timestamp)
    previous_hour = snapshot - pd.Timedelta(hours=1)

    demand = predictions.loc[
        predictions["timestamp"] == snapshot, ["zone_id", "prediction"]
    ].rename(columns={"prediction": "predicted_demand"})
    supply = panel.loc[
        panel["timestamp"] == previous_hour, ["zone_id", "dropoffs"]
    ].rename(columns={"dropoffs": "available_vehicles"})
    state = demand.merge(supply, on="zone_id", how="inner", validate="one_to_one")
    state["predicted_demand"] = np.rint(state["predicted_demand"]).astype(int)
    state["available_vehicles"] = state["available_vehicles"].round().astype(int)
    if state["zone_id"].nunique() != demand["zone_id"].nunique():
        raise ValueError("Supply history is incomplete for the rebalancing snapshot")
    return snapshot, state.sort_values("zone_id").reset_index(drop=True)


def add_route_times(moves: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    route_times = costs.rename(columns={"cost": "travel_minutes"})
    result = moves.drop(columns=["cost_per_vehicle", "total_cost"]).merge(
        route_times,
        on=["origin_zone", "destination_zone"],
        how="left",
        validate="many_to_one",
    )
    if result["travel_minutes"].isna().any():
        raise ValueError("A rebalancing route is missing its travel-time cost")
    result["vehicle_minutes"] = result["vehicles"] * result["travel_minutes"]
    return result


def apply_moves(zone_state: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    state = zone_state.copy()
    incoming = moves.groupby("destination_zone")["vehicles"].sum() if not moves.empty else pd.Series(dtype=float)
    outgoing = moves.groupby("origin_zone")["vehicles"].sum() if not moves.empty else pd.Series(dtype=float)
    state["initial_deficit"] = (
        state["predicted_demand"] - state["available_vehicles"]
    ).clip(lower=0)
    state["initial_surplus"] = (
        state["available_vehicles"] - state["predicted_demand"]
    ).clip(lower=0)
    state["vehicles_in"] = state["zone_id"].map(incoming).fillna(0).astype(int)
    state["vehicles_out"] = state["zone_id"].map(outgoing).fillna(0).astype(int)
    state["vehicles_after"] = (
        state["available_vehicles"] + state["vehicles_in"] - state["vehicles_out"]
    )
    state["remaining_deficit"] = (
        state["predicted_demand"] - state["vehicles_after"]
    ).clip(lower=0)
    return state


def run(root: Path = PROJECT_ROOT) -> dict[str, object]:
    predictions = pd.read_parquet(root / "reports/test_predictions.parquet")
    panel = pd.read_parquet(root / "data/processed/hourly_demand.parquet")
    lookup = pd.read_csv(root / "data/raw/taxi_zone_lookup.csv")
    timestamp, zone_state = build_zone_state(predictions, panel)

    surplus_zones = zone_state.loc[
        zone_state["available_vehicles"] > zone_state["predicted_demand"], "zone_id"
    ].astype(int).tolist()
    deficit_zones = zone_state.loc[
        zone_state["predicted_demand"] > zone_state["available_vehicles"], "zone_id"
    ].astype(int).tolist()
    historical_trips = clean_trips(
        load_trips([root / "data/raw/yellow_tripdata_2025-06.parquet"])
    )
    costs = historical_travel_time_costs(
        historical_trips,
        surplus_zones,
        deficit_zones,
        cutoff=timestamp,
        min_route_samples=20,
    )

    uniform_moves = greedy_rebalance(zone_state)
    uniform_with_time = add_route_times(uniform_moves, costs)
    optimized_moves = greedy_rebalance(zone_state, costs)
    optimized_with_time = add_route_times(optimized_moves, costs)
    optimized_state = apply_moves(zone_state, optimized_with_time)

    zone_names = lookup[["LocationID", "Borough", "Zone"]].rename(
        columns={"LocationID": "zone_id", "Borough": "borough", "Zone": "zone_name"}
    )
    optimized_state = optimized_state.merge(zone_names, on="zone_id", how="left", validate="one_to_one")
    origin_names = zone_names.rename(
        columns={
            "zone_id": "origin_zone",
            "borough": "origin_borough",
            "zone_name": "origin_name",
        }
    )
    destination_names = zone_names.rename(
        columns={
            "zone_id": "destination_zone",
            "borough": "destination_borough",
            "zone_name": "destination_name",
        }
    )
    optimized_with_time = (
        optimized_with_time.merge(origin_names, on="origin_zone", how="left")
        .merge(destination_names, on="destination_zone", how="left")
        .sort_values(["vehicles", "vehicle_minutes"], ascending=[False, False])
        .reset_index(drop=True)
    )

    optimized_summary = summarize_rebalancing(zone_state, optimized_moves)
    uniform_minutes = float(uniform_with_time["vehicle_minutes"].sum())
    optimized_minutes = float(optimized_with_time["vehicle_minutes"].sum())
    observed_vehicle_share = float(
        optimized_with_time.loc[
            optimized_with_time["cost_source"] == "observed_od", "vehicles"
        ].sum()
        / optimized_with_time["vehicles"].sum()
    )
    summary: dict[str, object] = {
        "timestamp": timestamp.isoformat(),
        **optimized_summary,
        "routes": int(len(optimized_with_time)),
        "origins": int(optimized_with_time["origin_zone"].nunique()),
        "destinations": int(optimized_with_time["destination_zone"].nunique()),
        "uniform_assignment_vehicle_minutes": uniform_minutes,
        "travel_time_aware_vehicle_minutes": optimized_minutes,
        "vehicle_minutes_reduction": uniform_minutes - optimized_minutes,
        "vehicle_minutes_reduction_rate": (
            (uniform_minutes - optimized_minutes) / uniform_minutes if uniform_minutes else 0.0
        ),
        "observed_od_vehicle_share": observed_vehicle_share,
        "cost_history_start": historical_trips["tpep_pickup_datetime"].min().isoformat(),
        "cost_history_end": historical_trips.loc[
            historical_trips["tpep_pickup_datetime"] < timestamp, "tpep_pickup_datetime"
        ].max().isoformat(),
        "min_route_samples": 20,
        "cost_note": "Median observed June trip time by OD pair before the snapshot; sparse routes use an explicit fallback.",
    }

    output_dir = root / "reports/demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    optimized_state.to_csv(output_dir / "rebalancing_zone_state.csv", index=False)
    optimized_with_time.to_csv(output_dir / "rebalancing_moves.csv", index=False)
    (output_dir / "rebalancing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    blind = pd.read_parquet(root / "reports/blind_test/predictions.parquet").merge(
        zone_names, on="zone_id", how="left", validate="many_to_one"
    )
    blind.to_parquet(output_dir / "blind_predictions.parquet", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact travel-time-aware rebalancing demo artifacts."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args().root.resolve()), indent=2))
