import pandas as pd

from src.rebalancing_report import apply_moves, build_zone_state


def test_build_zone_state_uses_previous_hour_dropoffs() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-06-30 23:00"] * 2),
            "zone_id": [1, 2],
            "prediction": [4.6, 2.2],
        }
    )
    panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-06-30 22:00", "2025-06-30 22:00", "2025-06-30 23:00"]
            ),
            "zone_id": [1, 2, 1],
            "dropoffs": [8, 1, 99],
        }
    )

    timestamp, state = build_zone_state(predictions, panel)

    assert timestamp == pd.Timestamp("2025-06-30 23:00")
    assert state["predicted_demand"].tolist() == [5, 2]
    assert state["available_vehicles"].tolist() == [8, 1]


def test_apply_moves_reports_before_and_after_deficits() -> None:
    state = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "predicted_demand": [2, 6],
            "available_vehicles": [5, 2],
        }
    )
    moves = pd.DataFrame(
        {
            "origin_zone": [1],
            "destination_zone": [2],
            "vehicles": [3],
            "distance_miles": [1.5],
            "vehicle_miles": [4.5],
        }
    )

    result = apply_moves(state, moves).set_index("zone_id")

    assert result["vehicles_after"].to_dict() == {1: 2, 2: 5}
    assert result["remaining_deficit"].to_dict() == {1: 0, 2: 1}
