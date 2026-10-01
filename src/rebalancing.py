from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DROPOFF_COLUMN, PICKUP_COLUMN


def greedy_rebalance(
    zone_state: pd.DataFrame,
    costs: pd.DataFrame | None = None,
    max_cost: float | None = None,
) -> pd.DataFrame:
    """Match surplus vehicles to forecast deficits, choosing the cheapest route first."""
    required = ["zone_id", "predicted_demand", "available_vehicles"]
    missing = set(required) - set(zone_state.columns)
    if missing:
        raise ValueError(f"Missing zone-state columns: {sorted(missing)}")
    if zone_state["zone_id"].duplicated().any():
        raise ValueError("zone_state must contain one row per zone_id")
    if (zone_state[["predicted_demand", "available_vehicles"]] < 0).any().any():
        raise ValueError("Demand and vehicle counts must be non-negative")

    state = zone_state.loc[:, required].copy()
    state["surplus"] = (state["available_vehicles"] - state["predicted_demand"]).clip(lower=0).astype(int)
    state["deficit"] = (state["predicted_demand"] - state["available_vehicles"]).clip(lower=0).astype(int)
    supply = state.loc[state["surplus"] > 0, ["zone_id", "surplus"]].set_index("zone_id")["surplus"].to_dict()
    demand = state.loc[state["deficit"] > 0, ["zone_id", "deficit"]].set_index("zone_id")["deficit"].to_dict()

    if costs is None:
        routes = pd.DataFrame(
            [(origin, destination, 1.0) for origin in supply for destination in demand],
            columns=["origin_zone", "destination_zone", "cost"],
        )
    else:
        cost_columns = {"origin_zone", "destination_zone", "cost"}
        missing_costs = cost_columns - set(costs.columns)
        if missing_costs:
            raise ValueError(f"Missing cost columns: {sorted(missing_costs)}")
        routes = costs.loc[:, ["origin_zone", "destination_zone", "cost"]].copy()
        if routes[["origin_zone", "destination_zone"]].duplicated().any():
            raise ValueError("costs must contain one row per origin-destination route")
        if not np.isfinite(routes["cost"]).all() or (routes["cost"] < 0).any():
            raise ValueError("Route costs must be finite and non-negative")
        routes = routes[routes["origin_zone"].isin(supply) & routes["destination_zone"].isin(demand)]
    if max_cost is not None:
        if max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        routes = routes[routes["cost"] <= max_cost]

    moves: list[dict[str, float | int]] = []
    for route in routes.sort_values(
        ["cost", "origin_zone", "destination_zone"], kind="stable"
    ).itertuples(index=False):
        origin = int(route.origin_zone)
        destination = int(route.destination_zone)
        vehicles = min(supply.get(origin, 0), demand.get(destination, 0))
        if vehicles <= 0:
            continue
        moves.append(
            {
                "origin_zone": origin,
                "destination_zone": destination,
                "vehicles": vehicles,
                "cost_per_vehicle": float(route.cost),
                "total_cost": vehicles * float(route.cost),
            }
        )
        supply[origin] -= vehicles
        demand[destination] -= vehicles

    return pd.DataFrame(
        moves,
        columns=["origin_zone", "destination_zone", "vehicles", "cost_per_vehicle", "total_cost"],
    )


def historical_travel_time_costs(
    trips: pd.DataFrame,
    origin_zones: list[int],
    destination_zones: list[int],
    cutoff: pd.Timestamp,
    min_route_samples: int = 20,
) -> pd.DataFrame:
    """Estimate OD costs from past median observed trip times, with explicit fallback."""
    required = {PICKUP_COLUMN, DROPOFF_COLUMN, "PULocationID", "DOLocationID"}
    missing = required - set(trips.columns)
    if missing:
        raise ValueError(f"Missing trip columns: {sorted(missing)}")
    if min_route_samples < 1:
        raise ValueError("min_route_samples must be at least one")

    history = trips.loc[
        (trips[PICKUP_COLUMN] < pd.Timestamp(cutoff))
        & trips["PULocationID"].isin(origin_zones)
        & trips["DOLocationID"].isin(destination_zones)
    ].copy()
    history["travel_minutes"] = (
        history[DROPOFF_COLUMN] - history[PICKUP_COLUMN]
    ).dt.total_seconds() / 60.0
    history = history[history["travel_minutes"] > 0]
    if history.empty:
        raise ValueError("No historical trips are available for candidate routes")

    route_stats = (
        history.groupby(["PULocationID", "DOLocationID"], as_index=False)
        .agg(observed_median_minutes=("travel_minutes", "median"), route_samples=("travel_minutes", "size"))
        .rename(columns={"PULocationID": "origin_zone", "DOLocationID": "destination_zone"})
    )
    reliable = route_stats[route_stats["route_samples"] >= min_route_samples]
    origin_fallback = history.groupby("PULocationID")["travel_minutes"].median()
    destination_fallback = history.groupby("DOLocationID")["travel_minutes"].median()
    global_fallback = float(history["travel_minutes"].median())

    pairs = pd.MultiIndex.from_product(
        [sorted(set(origin_zones)), sorted(set(destination_zones))],
        names=["origin_zone", "destination_zone"],
    ).to_frame(index=False)
    costs = pairs.merge(reliable, on=["origin_zone", "destination_zone"], how="left")
    costs["cost_source"] = np.where(costs["observed_median_minutes"].notna(), "observed_od", "fallback")
    fallback = costs["origin_zone"].map(origin_fallback)
    fallback = fallback.fillna(costs["destination_zone"].map(destination_fallback)).fillna(global_fallback)
    costs["cost"] = costs["observed_median_minutes"].fillna(fallback)
    costs["route_samples"] = costs["route_samples"].fillna(0).astype(int)
    return costs[
        ["origin_zone", "destination_zone", "cost", "cost_source", "route_samples"]
    ]


def summarize_rebalancing(zone_state: pd.DataFrame, moves: pd.DataFrame) -> dict[str, float]:
    """Report forecast deficits before and after simulated moves."""
    initial_deficit = (
        zone_state["predicted_demand"] - zone_state["available_vehicles"]
    ).clip(lower=0).sum()
    moved = float(moves["vehicles"].sum()) if not moves.empty else 0.0
    return {
        "initial_forecast_deficit": float(initial_deficit),
        "vehicles_rebalanced": moved,
        "remaining_forecast_deficit": float(max(initial_deficit - moved, 0)),
        "rebalancing_cost": float(moves["total_cost"].sum()) if not moves.empty else 0.0,
    }
