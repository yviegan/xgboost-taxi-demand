import json

import pandas as pd
import pytest

from src.tuning import (
    sample_candidates,
    select_candidate,
    summarize_candidates,
    validate_fold_separation,
)


PARAMETER_SPACE = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05],
    "reg_lambda": [1.0, 5.0],
}
MANUAL_PARAMS = {"max_depth": 8, "learning_rate": 0.05, "reg_lambda": 5.0}


def test_candidate_sampling_is_deterministic_and_keeps_manual_last() -> None:
    first = sample_candidates(PARAMETER_SPACE, MANUAL_PARAMS, 4, random_state=42)
    second = sample_candidates(PARAMETER_SPACE, MANUAL_PARAMS, 4, random_state=42)

    assert first == second
    assert len(first) == 5
    assert first[-1] == MANUAL_PARAMS
    assert len({json.dumps(candidate, sort_keys=True) for candidate in first}) == 5


def test_development_and_holdout_windows_must_not_overlap() -> None:
    development = [{"test_start": "2025-04-07"}]
    valid_holdout = [{"test_start": "2025-05-01"}]
    overlapping_holdout = [{"test_start": "2025-04-10"}]

    validate_fold_separation(development, valid_holdout, test_days=7)
    with pytest.raises(ValueError):
        validate_fold_separation(development, overlapping_holdout, test_days=7)


def test_selection_uses_lowest_development_wape() -> None:
    trials = pd.DataFrame(
        [
            {"candidate_id": "a", "candidate_type": "random", "params_json": '{"max_depth": 4}', "fold": "one", "validation_wape": 0.20, "validation_r2": 0.8, "best_iteration": 20},
            {"candidate_id": "a", "candidate_type": "random", "params_json": '{"max_depth": 4}', "fold": "two", "validation_wape": 0.18, "validation_r2": 0.8, "best_iteration": 20},
            {"candidate_id": "b", "candidate_type": "random", "params_json": '{"max_depth": 6}', "fold": "one", "validation_wape": 0.17, "validation_r2": 0.9, "best_iteration": 30},
            {"candidate_id": "b", "candidate_type": "random", "params_json": '{"max_depth": 6}', "fold": "two", "validation_wape": 0.16, "validation_r2": 0.9, "best_iteration": 30},
        ]
    )

    selected = select_candidate(summarize_candidates(trials))

    assert selected["candidate_id"] == "b"
    assert selected["params"] == {"max_depth": 6}
