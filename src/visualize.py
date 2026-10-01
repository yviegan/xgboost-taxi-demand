from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"
LIGHT_GRAY = "#c3c2b7"
DARK_GRAY = "#52514e"


MODEL_LABELS = {
    "xgboost_tuned": "XGBoost",
    "xgboost": "XGBoost",
    "previous_hour": "Previous hour",
    "previous_day": "Previous day",
    "previous_week": "Previous week",
}
FEATURE_LABELS = {
    "pickups_lag_168": "Demand one week ago",
    "pickups_lag_24": "Demand one day ago",
    "pickups_lag_3": "Demand 3 hours ago",
    "pickups_lag_2": "Demand 2 hours ago",
    "pickups_lag_1": "Demand 1 hour ago",
    "pickups_roll_mean_3": "3-hour rolling mean",
    "pickups_roll_mean_6": "6-hour rolling mean",
    "pickups_roll_mean_24": "24-hour rolling mean",
    "pickups_roll_mean_168": "168-hour rolling mean",
    "pickups_roll_std_3": "3-hour rolling std.",
    "pickups_roll_std_6": "6-hour rolling std.",
    "pickups_roll_std_24": "24-hour rolling std.",
    "pickups_roll_std_168": "168-hour rolling std.",
    "available_vehicles": "Available-vehicle proxy",
    "hour_cos": "Hour (cosine)",
    "hour_sin": "Hour (sine)",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelcolor": SECONDARY,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": SECONDARY,
            "axes.edgecolor": LIGHT_GRAY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def select_top_zone(predictions: pd.DataFrame) -> int:
    """Select the highest-demand zone deterministically from blind-test actuals."""
    totals = predictions.groupby("zone_id", observed=True, as_index=False)["pickups"].sum()
    ranked = totals.sort_values(
        ["pickups", "zone_id"], ascending=[False, True], kind="stable"
    )
    return int(ranked.iloc[0]["zone_id"])


def plot_blind_comparison(root: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(root / "reports/blind_test/metrics.csv")
    data = metrics.loc[metrics["dataset"] == "test"].copy()
    order = ["xgboost_tuned", "previous_week", "previous_hour", "previous_day"]
    data["model"] = pd.Categorical(data["model"], categories=order, ordered=True)
    data = data.sort_values("model", ascending=False)
    colors = [BLUE if model == "xgboost_tuned" else LIGHT_GRAY for model in data["model"]]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(data["model"].map(MODEL_LABELS), data["wape"], color=colors, height=0.58)
    ax.set_title("July blind test: XGBoost beats seasonal baselines", loc="left", pad=16)
    ax.text(
        0, 1.02, "WAPE · July 25–31, 2025 · lower is better",
        transform=ax.transAxes, color=SECONDARY, va="bottom",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, data["wape"].max() * 1.22)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, data["wape"]):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.2%}", va="center", fontweight="bold")
    save_figure(fig, output_dir / "blind_test_comparison.png")


def plot_top_zone_forecast(root: Path, output_dir: Path) -> None:
    predictions = pd.read_parquet(root / "reports/blind_test/predictions.parquet")
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    zone_id = select_top_zone(predictions)
    zone = predictions.loc[predictions["zone_id"] == zone_id].sort_values("timestamp")
    lookup = pd.read_csv(root / "data/raw/taxi_zone_lookup.csv")
    match = lookup.loc[lookup["LocationID"] == zone_id]
    zone_name = match["Zone"].iloc[0] if not match.empty else f"Zone {zone_id}"
    borough = match["Borough"].iloc[0] if not match.empty else "NYC"

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(zone["timestamp"], zone["pickups"], color=ORANGE, linewidth=2.0, label="Actual")
    ax.plot(zone["timestamp"], zone["prediction"], color=BLUE, linewidth=2.0, linestyle="--", label="XGBoost")
    ax.set_title(f"Hourly demand in {zone_name}, {borough}", loc="left", pad=16)
    ax.text(
        0, 1.02, f"Highest-demand zone in the July blind-test week · Zone {zone_id}",
        transform=ax.transAxes, color=SECONDARY, va="bottom",
    )
    ax.set_ylabel("Pickups per hour")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save_figure(fig, output_dir / "july_top_zone_forecast.png")


def plot_rolling_wape(root: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(root / "reports/rolling/fold_metrics.csv")
    data = metrics.loc[metrics["dataset"] == "test"].copy()
    folds = [
        ("2025-04-end", "Apr 24–30"),
        ("2025-05-end", "May 25–31"),
        ("2025-06-mid", "Jun 10–16"),
        ("2025-06-end", "Jun 24–30"),
    ]
    model_order = ["xgboost", "previous_week", "previous_hour", "previous_day"]
    model_labels = [MODEL_LABELS[model] for model in model_order]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for ax, (fold, fold_label) in zip(axes.flat, folds):
        subset = data.loc[data["fold"] == fold].set_index("model").reindex(model_order)
        values = subset["wape"].to_numpy()
        bars = ax.barh(
            model_labels[::-1],
            values[::-1],
            color=[LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, BLUE],
            height=0.56,
        )
        ax.set_title(fold_label, loc="left", fontsize=12, pad=8)
        ax.set_xlim(0, 0.38)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values[::-1]):
            ax.text(
                value + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}",
                va="center",
                fontsize=9,
                fontweight="bold" if bar.get_facecolor()[:3] == matplotlib.colors.to_rgb(BLUE) else "normal",
            )

    fig.suptitle(
        "XGBoost leads in every rolling test window",
        x=0.08,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.92,
        "Four non-overlapping 7-day windows · WAPE · lower is better",
        color=SECONDARY,
    )
    fig.supxlabel("WAPE", y=0.03, color=SECONDARY)
    fig.subplots_adjust(top=0.84, bottom=0.10, hspace=0.42, wspace=0.34)
    save_figure(fig, output_dir / "rolling_wape.png")


def plot_segment_wape(root: Path, output_dir: Path) -> None:
    data = pd.read_csv(root / "reports/rolling/segment_summary.csv")
    specifications = [
        ("peak", ["peak", "off_peak"], ["Weekday peak", "Other hours"]),
        ("zone_demand", ["top_20", "other"], ["Top 20 zones", "Other zones"]),
        ("day_type", ["weekday", "weekend"], ["Weekday", "Weekend"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True)
    for ax, (dimension, order, labels) in zip(axes, specifications):
        subset = data.loc[data["dimension"] == dimension].set_index("segment").reindex(order)
        values = subset["wape_mean"].to_numpy()[::-1]
        bars = ax.barh(labels[::-1], values, color=["#86b6ef", BLUE], height=0.54)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, 0.26)
        for bar, value in zip(bars, values):
            ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Time of day", loc="left")
    axes[1].set_title("Zone demand", loc="left")
    axes[2].set_title("Day type", loc="left")
    fig.suptitle("Relative error is lowest where demand is dense and regular", x=0.07, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.92, "Mean WAPE across four rolling windows · lower is better", color=SECONDARY)
    fig.subplots_adjust(top=0.78, wspace=0.48)
    save_figure(fig, output_dir / "segment_wape.png")


def plot_rebalancing_snapshot(root: Path, output_dir: Path) -> None:
    demo_dir = root / "reports" / "demo"
    moves = pd.read_csv(demo_dir / "rebalancing_moves.csv")
    summary = json.loads(
        (demo_dir / "rebalancing_summary.json").read_text(encoding="utf-8")
    )
    routes = moves.nlargest(10, "vehicles").sort_values("vehicles")
    labels = routes["origin_name"] + "  →  " + routes["destination_name"]

    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(
        2, 3, height_ratios=[0.7, 3.3], width_ratios=[1, 1, 1],
        hspace=0.36, wspace=0.38,
    )
    kpis = [
        ("Initial deficit", summary["initial_forecast_deficit"], "vehicles"),
        ("Vehicles moved", summary["vehicles_rebalanced"], "vehicles"),
        ("Remaining deficit", summary["remaining_forecast_deficit"], "vehicles"),
    ]
    for index, (label, value, unit) in enumerate(kpis):
        ax = fig.add_subplot(grid[0, index])
        ax.axis("off")
        ax.text(0, 0.70, f"{value:,.0f}", fontsize=25, fontweight="bold", color=BLUE)
        ax.text(0, 0.32, label, fontsize=11, fontweight="bold")
        ax.text(0, 0.04, unit, fontsize=9, color=SECONDARY)

    flow_ax = fig.add_subplot(grid[1, :2])
    bars = flow_ax.barh(labels, routes["vehicles"], color=BLUE, height=0.58)
    flow_ax.set_title("Largest suggested vehicle flows", loc="left", fontsize=12)
    flow_ax.set_xlabel("Vehicles moved")
    flow_ax.grid(axis="x")
    flow_ax.set_axisbelow(True)
    flow_ax.set_xlim(0, routes["vehicles"].max() * 1.20)
    for bar, value in zip(bars, routes["vehicles"]):
        flow_ax.text(
            value + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{int(value)}", va="center", fontsize=9, fontweight="bold",
        )

    cost_ax = fig.add_subplot(grid[1, 2])
    values = [
        summary["uniform_assignment_vehicle_minutes"],
        summary["travel_time_aware_vehicle_minutes"],
    ]
    cost_bars = cost_ax.bar(
        ["Unit-cost\nassignment", "Travel-time-aware\nassignment"],
        values,
        color=[LIGHT_GRAY, BLUE],
        width=0.58,
    )
    cost_ax.set_title("Estimated relocation cost", loc="left", fontsize=12)
    cost_ax.set_ylabel("Historical vehicle-minutes")
    cost_ax.grid(axis="y")
    cost_ax.set_axisbelow(True)
    cost_ax.set_ylim(0, max(values) * 1.22)
    for bar, value in zip(cost_bars, values):
        cost_ax.text(
            bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025,
            f"{value:,.0f}", ha="center", fontsize=9, fontweight="bold",
        )
    cost_ax.text(
        0.5, 0.91,
        f"{summary['vehicle_minutes_reduction_rate']:.1%} lower",
        transform=cost_ax.transAxes, ha="center", color=BLUE, fontweight="bold",
    )

    fig.suptitle(
        "Travel-time-aware rebalancing turns forecast deficits into vehicle moves",
        x=0.06, y=0.99, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.06, 0.945,
        "Offline snapshot · June 30, 2025 at 23:00 · median observed OD trip-time costs",
        color=SECONDARY,
    )
    fig.text(
        0.06, 0.01,
        f"{summary['observed_od_vehicle_share']:.1%} of moved vehicles use OD medians with ≥20 trips; "
        "the remainder use fallback estimates. Costs reflect past trips, not live traffic.",
        color=SECONDARY, fontsize=9,
    )
    save_figure(fig, output_dir / "rebalancing_snapshot.png")


def plot_feature_importance(root: Path, output_dir: Path) -> None:
    data = pd.read_csv(root / "reports/feature_importance.csv").nlargest(10, "importance")
    data = data.sort_values("importance")
    labels = [FEATURE_LABELS.get(feature, feature.replace("_", " ").title()) for feature in data["feature"]]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.barh(labels, data["importance"], color=BLUE, height=0.56)
    ax.set_title("Time-of-day and lag features drive the model", loc="left", pad=16)
    ax.text(0, 1.02, "XGBoost gain-based feature importance · top 10", transform=ax.transAxes, color=SECONDARY, va="bottom")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(0, data["importance"].max() * 1.17)
    for bar, value in zip(bars, data["importance"]):
        if value >= 0.01:
            ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9, fontweight="bold")
    save_figure(fig, output_dir / "feature_importance.png")


def generate_all(root: Path = PROJECT_ROOT) -> list[Path]:
    configure_style()
    output_dir = root / "reports" / "figures"
    plot_blind_comparison(root, output_dir)
    plot_top_zone_forecast(root, output_dir)
    plot_rolling_wape(root, output_dir)
    plot_segment_wape(root, output_dir)
    plot_feature_importance(root, output_dir)
    plot_rebalancing_snapshot(root, output_dir)
    return [
        output_dir / "blind_test_comparison.png",
        output_dir / "july_top_zone_forecast.png",
        output_dir / "rolling_wape.png",
        output_dir / "segment_wape.png",
        output_dir / "feature_importance.png",
        output_dir / "rebalancing_snapshot.png",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static project figures from saved reports.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    for path in generate_all(parse_args().root.resolve()):
        print(path)
