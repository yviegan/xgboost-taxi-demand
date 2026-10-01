from pathlib import Path

import pandas as pd
import pytest

from src.visualize import generate_all, select_top_zone


ROOT = Path(__file__).resolve().parents[1]
FULL_REPORT_INPUTS = [
    "reports/blind_test/metrics.csv",
    "reports/blind_test/predictions.parquet",
    "reports/rolling/fold_metrics.csv",
    "reports/rolling/segment_summary.csv",
    "reports/feature_importance.csv",
]
PUBLISHED_ARTIFACTS = [
    "data/raw/taxi_zone_lookup.csv",
    "reports/demo/blind_predictions.parquet",
    "reports/demo/rebalancing_moves.csv",
    "reports/demo/rebalancing_summary.json",
    "reports/demo/rebalancing_zone_state.csv",
    "reports/figures/blind_test_comparison.png",
    "reports/figures/july_top_zone_forecast.png",
    "reports/figures/rolling_wape.png",
    "reports/figures/segment_wape.png",
    "reports/figures/feature_importance.png",
    "reports/figures/rebalancing_snapshot.png",
]


def test_select_top_zone_uses_total_actual_demand_with_stable_tie_break() -> None:
    predictions = pd.DataFrame(
        {
            "zone_id": [3, 2, 3, 2, 1],
            "pickups": [4, 5, 6, 5, 9],
        }
    )

    assert select_top_zone(predictions) == 2


def test_published_demo_and_figure_artifacts_exist() -> None:
    missing = [path for path in PUBLISHED_ARTIFACTS if not (ROOT / path).is_file()]

    assert not missing, f"Missing published artifacts: {missing}"
    assert all((ROOT / path).stat().st_size > 0 for path in PUBLISHED_ARTIFACTS)


@pytest.mark.skipif(
    not all((ROOT / path).is_file() for path in FULL_REPORT_INPUTS),
    reason="Full local experiment reports are not included in the public repository.",
)
def test_generate_all_creates_six_nonempty_pngs() -> None:
    outputs = generate_all(ROOT)

    assert len(outputs) == 6
    assert all(path.suffix == ".png" for path in outputs)
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
