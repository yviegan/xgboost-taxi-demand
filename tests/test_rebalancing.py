import pandas as pd
import pytest

from src.rebalancing import (
    greedy_rebalance,
    historical_travel_time_costs,
    summarize_rebalancing,
)


def test_rebalancing_preserves_supply_and_reduces_deficit() -> None:
    state = pd.DataFrame(
        {
            "zone_id": [1, 2, 3],
            "predicted_demand": [2, 8, 5],
            "available_vehicles": [7, 2, 3],
        }
    )
    costs = pd.DataFrame(
        {
            "origin_zone": [1, 1],
            "destination_zone": [2, 3],
            "cost": [1.0, 2.0],
        }
    )

    moves = greedy_rebalance(state, costs)
    summary = summarize_rebalancing(state, moves)

    assert moves["vehicles"].sum() == 5
    assert moves.groupby("origin_zone")["vehicles"].sum().max() <= 5
    assert summary["initial_forecast_deficit"] == 8
    assert summary["remaining_forecast_deficit"] == 3


def test_rebalancing_uses_cheapest_routes_and_respects_max_cost() -> None:
    state = pd.DataFrame(
        {
            "zone_id": [1, 2, 3],
            "predicted_demand": [0, 3, 3],
            "available_vehicles": [5, 0, 0],
        }
    )
    costs = pd.DataFrame(
        {
            "origin_zone": [1, 1],
            "destination_zone": [2, 3],
            "cost": [3.0, 1.0],
        }
    )

    moves = greedy_rebalance(state, costs, max_cost=2.0)

    assert moves[["origin_zone", "destination_zone", "vehicles"]].to_dict("records") == [
        {"origin_zone": 1, "destination_zone": 3, "vehicles": 3}
    ]


def test_rebalancing_conserves_vehicles_across_multiple_zones() -> None:
    state = pd.DataFrame(
        {
            "zone_id": [1, 2, 3, 4],
            "predicted_demand": [1, 2, 7, 6],
            "available_vehicles": [5, 6, 1, 2],
        }
    )

    moves = greedy_rebalance(state)
    outbound = moves.groupby("origin_zone")["vehicles"].sum()
    inbound = moves.groupby("destination_zone")["vehicles"].sum()

    assert moves["vehicles"].sum() == min(8, 10)
    assert outbound.to_dict() == {1: 4, 2: 4}
    assert inbound.to_dict() == {3: 6, 4: 2}


def test_rebalancing_validates_inputs_and_handles_no_moves() -> None:
    balanced = pd.DataFrame(
        {"zone_id": [1], "predicted_demand": [3], "available_vehicles": [3]}
    )

    assert greedy_rebalance(balanced).empty
    with pytest.raises(ValueError, match="Missing zone-state columns"):
        greedy_rebalance(pd.DataFrame({"zone_id": [1]}))
    with pytest.raises(ValueError, match="one row per zone_id"):
        greedy_rebalance(pd.concat([balanced, balanced], ignore_index=True))


def test_historical_travel_time_costs_use_observed_median_and_fallback() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-06-01 10:00", "2025-06-01 11:00", "2025-06-01 12:00"]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2025-06-01 10:10", "2025-06-01 11:20", "2025-06-01 12:30"]
            ),
            "PULocationID": [1, 1, 1],
            "DOLocationID": [2, 2, 3],
        }
    )

    costs = historical_travel_time_costs(
        trips, [1], [2, 3], pd.Timestamp("2025-06-02"), min_route_samples=2
    ).set_index(["origin_zone", "destination_zone"])

    assert costs.loc[(1, 2), "cost"] == pytest.approx(15.0)
    assert costs.loc[(1, 2), "cost_source"] == "observed_od"
    assert costs.loc[(1, 3), "cost"] == pytest.approx(20.0)
    assert costs.loc[(1, 3), "cost_source"] == "fallback"
