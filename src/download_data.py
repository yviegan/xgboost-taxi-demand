from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_MAP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def download(url: str, destination: Path) -> None:
    """Download atomically so interrupted transfers never look complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "xgboost-taxi-demand/1.0"})
    with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def valid_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as file:
        header = file.read(4)
        file.seek(-4, 2)
        footer = file.read(4)
    return header == footer == b"PAR1"


def valid_zip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except OSError:
        return False


def main(year: int, start_month: int, end_month: int) -> None:
    raw = PROJECT_ROOT / "data" / "raw"
    for month in range(start_month, end_month + 1):
        path = raw / f"yellow_tripdata_{year}-{month:02d}.parquet"
        if not valid_parquet(path):
            print(f"Downloading {path.name}...")
            download(TRIP_URL.format(year=year, month=month), path)
        if not valid_parquet(path):
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Parquet integrity check failed: {path}")
        print(f"Ready: {path.name} ({path.stat().st_size:,} bytes)")

    lookup = raw / "taxi_zone_lookup.csv"
    if not lookup.exists():
        download(ZONE_LOOKUP_URL, lookup)

    zone_map = raw / "taxi_zones.zip"
    if not valid_zip(zone_map):
        download(ZONE_MAP_URL, zone_map)
    if not valid_zip(zone_map):
        zone_map.unlink(missing_ok=True)
        raise RuntimeError(f"ZIP integrity check failed: {zone_map}")
    print("Zone lookup and boundaries are ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and validate NYC TLC source data.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=6)
    arguments = parser.parse_args()
    main(arguments.year, arguments.start_month, arguments.end_month)
