from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

PICKUP_COLUMN = "tpep_pickup_datetime"
DROPOFF_COLUMN = "tpep_dropoff_datetime"


def load_trips(paths: list[Path]) -> pd.DataFrame:
    """Load needed columns and reject rows outside each file's named month."""
    columns = [PICKUP_COLUMN, DROPOFF_COLUMN, "PULocationID", "DOLocationID"]
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        match = re.search(r"(\d{4})-(\d{2})", path.stem)
        if match:
            period = pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M")
            pickup_time = pd.to_datetime(frame[PICKUP_COLUMN], errors="coerce")
            frame = frame.loc[pickup_time.dt.to_period("M") == period]
        frames.append(frame)
    if not frames:
        raise ValueError("No trip files matched the configured path.")
    return pd.concat(frames, ignore_index=True)


def clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid timestamps and zone identifiers."""
    cleaned = trips.copy()
    cleaned[PICKUP_COLUMN] = pd.to_datetime(cleaned[PICKUP_COLUMN], errors="coerce")
    cleaned[DROPOFF_COLUMN] = pd.to_datetime(cleaned[DROPOFF_COLUMN], errors="coerce")
    valid = (
        cleaned[PICKUP_COLUMN].notna()
        & cleaned[DROPOFF_COLUMN].notna()
        & cleaned["PULocationID"].between(1, 263)
        & cleaned["DOLocationID"].between(1, 263)
        & (cleaned[DROPOFF_COLUMN] >= cleaned[PICKUP_COLUMN])
        & ((cleaned[DROPOFF_COLUMN] - cleaned[PICKUP_COLUMN]).dt.total_seconds() <= 6 * 3600)
    )
    return cleaned.loc[valid].copy()


def _hourly_counts(trips: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pickups = (
        trips.assign(timestamp=trips[PICKUP_COLUMN].dt.floor("h"))
        .groupby(["timestamp", "PULocationID"], observed=True)
        .size()
        .rename("pickups")
    )
    dropoffs = (
        trips.assign(timestamp=trips[DROPOFF_COLUMN].dt.floor("h"))
        .groupby(["timestamp", "DOLocationID"], observed=True)
        .size()
        .rename("dropoffs")
    )
    return pickups, dropoffs


def _balanced_panel(
    pickups: pd.Series,
    dropoffs: pd.Series,
    min_zone_total_pickups: int,
) -> pd.DataFrame:
    zone_totals = pickups.groupby(level=1).sum()
    zones = zone_totals[zone_totals >= min_zone_total_pickups].index.sort_values()
    # Pickup timestamps define the prediction horizon. A trip may drop off just
    # outside the source month and must not extend the target panel.
    start = pickups.index.get_level_values(0).min()
    end = pickups.index.get_level_values(0).max()
    hours = pd.date_range(start, end, freq="h")
    index = pd.MultiIndex.from_product([zones, hours], names=["zone_id", "timestamp"])

    panel = pd.DataFrame(index=index)
    panel["pickups"] = pickups.reorder_levels([1, 0]).reindex(index, fill_value=0)
    panel["dropoffs"] = dropoffs.reorder_levels([1, 0]).reindex(index, fill_value=0)
    return panel.reset_index().sort_values(["zone_id", "timestamp"]).reset_index(drop=True)


def build_hourly_panel(trips: pd.DataFrame, min_zone_total_pickups: int = 500) -> pd.DataFrame:
    """Build a balanced zone-hour panel from an in-memory trip table."""
    pickups, dropoffs = _hourly_counts(trips)
    return _balanced_panel(pickups, dropoffs, min_zone_total_pickups)


def build_panel_extension(existing_panel: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Build a new time block using the zones fixed by the existing panel."""
    pickups, dropoffs = _hourly_counts(trips)
    zones = pd.Index(sorted(existing_panel["zone_id"].unique()))
    start = pickups.index.get_level_values(0).min()
    end = pickups.index.get_level_values(0).max()
    hours = pd.date_range(start, end, freq="h")
    index = pd.MultiIndex.from_product([zones, hours], names=["zone_id", "timestamp"])
    extension = pd.DataFrame(index=index)
    extension["pickups"] = pickups.reorder_levels([1, 0]).reindex(index, fill_value=0)
    extension["dropoffs"] = dropoffs.reorder_levels([1, 0]).reindex(index, fill_value=0)
    return extension.reset_index().sort_values(["zone_id", "timestamp"]).reset_index(drop=True)


def build_hourly_panel_from_files(
    paths: list[Path],
    min_zone_total_pickups: int = 500,
) -> tuple[pd.DataFrame, int]:
    """Aggregate one source file at a time to keep peak memory bounded."""
    pickup_parts: list[pd.Series] = []
    dropoff_parts: list[pd.Series] = []
    trip_rows = 0
    for path in paths:
        trips = clean_trips(load_trips([path]))
        pickups, dropoffs = _hourly_counts(trips)
        pickup_parts.append(pickups)
        dropoff_parts.append(dropoffs)
        trip_rows += len(trips)
        print(f"Aggregated {path.name}: {len(trips):,} valid trips")

    if not pickup_parts:
        raise ValueError("No trip files matched the configured path.")
    pickups = pd.concat(pickup_parts).groupby(level=[0, 1]).sum()
    dropoffs = pd.concat(dropoff_parts).groupby(level=[0, 1]).sum()
    return _balanced_panel(pickups, dropoffs, min_zone_total_pickups), trip_rows
