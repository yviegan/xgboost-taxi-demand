from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "reports" / "demo"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
LIGHT_BLUE = "#86b6ef"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    predictions = pd.read_parquet(DEMO_DIR / "blind_predictions.parquet")
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    zone_state = pd.read_csv(DEMO_DIR / "rebalancing_zone_state.csv")
    moves = pd.read_csv(DEMO_DIR / "rebalancing_moves.csv")
    summary = json.loads((DEMO_DIR / "rebalancing_summary.json").read_text(encoding="utf-8"))
    return predictions, zone_state, summary, moves


def forecast_figure(zone: pd.DataFrame, interval_label: str = "hour") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(zone["timestamp"], zone["pickups"], color=ORANGE, linewidth=2, label="Actual")
    ax.plot(
        zone["timestamp"], zone["prediction"], color=BLUE, linewidth=2,
        linestyle="--", label="XGBoost",
    )
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel(f"Pickups per {interval_label}")
    ax.grid(axis="y", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def flow_figure(moves: pd.DataFrame, limit: int = 12) -> plt.Figure:
    routes = moves.nlargest(limit, "vehicles").sort_values("vehicles")
    labels = routes["origin_name"] + "  →  " + routes["destination_name"]
    fig, ax = plt.subplots(figsize=(10, max(4.2, limit * 0.38)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(labels, routes["vehicles"], color=BLUE, height=0.58)
    ax.set_xlabel("Vehicles moved")
    ax.grid(axis="x", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, routes["vehicles"]):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center")
    ax.set_xlim(0, max(routes["vehicles"].max() * 1.18, 1))
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="NYC Taxi Demand & Rebalancing",
        page_icon="🚕",
        layout="wide",
    )
    st.title("NYC Taxi Demand Forecasting & Fleet Rebalancing")
    st.caption(
        "Explore the locked July forecast and an offline travel-time-aware dispatch snapshot."
    )

    required = [
        DEMO_DIR / "blind_predictions.parquet",
        DEMO_DIR / "rebalancing_zone_state.csv",
        DEMO_DIR / "rebalancing_moves.csv",
        DEMO_DIR / "rebalancing_summary.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        st.error(
            "Demo artifacts are missing: " + ", ".join(missing)
            + ". Run `python -m src.rebalancing_report` first."
        )
        st.stop()

    predictions, zone_state, summary, moves = load_demo_data()
    forecast_tab, rebalancing_tab, protocol_tab = st.tabs(
        ["July blind-test forecast", "June rebalancing snapshot", "Scope & protocol"]
    )

    with forecast_tab:
        st.subheader("Locked-parameter July forecast")
        st.write(
            "The model was trained through July 17, validated on July 18–24, and "
            "evaluated once on July 25–31."
        )
        zone_summary = (
            predictions.assign(absolute_error=lambda frame: (frame["pickups"] - frame["prediction"]).abs())
            .groupby(["zone_id", "zone_name", "borough"], as_index=False)
            .agg(
                actual_pickups=("pickups", "sum"),
                predicted_pickups=("prediction", "sum"),
                absolute_error=("absolute_error", "sum"),
            )
            .sort_values(["actual_pickups", "zone_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        zone_summary["wape"] = (
            zone_summary["absolute_error"]
            / zone_summary["actual_pickups"].replace(0, float("nan"))
        )
        zone_summary["demand_rank"] = zone_summary.index + 1
        zone_summary["label"] = zone_summary.apply(
            lambda row: (
                f"#{row['demand_rank']:.0f} {row['zone_name']} · {row['borough']} | "
                f"{row['actual_pickups']:,.0f} trips | WAPE {row['wape']:.1%}"
            ),
            axis=1,
        )

        controls = st.columns([1, 2])
        show_all_zones = controls[0].toggle(
            "Show all 226 zones",
            value=False,
            help="Off by default so the portfolio view focuses on the 20 highest-demand operational zones.",
        )
        granularity = controls[1].selectbox(
            "Chart granularity",
            ["3-hour totals", "Hourly"],
            index=0,
            help="Three-hour totals make the operational demand pattern easier to compare; hourly data remain available.",
        )
        visible_zones = zone_summary if show_all_zones else zone_summary.head(20)
        selected_label = st.selectbox("Taxi zone", visible_zones["label"].tolist(), index=0)
        selected = visible_zones.loc[visible_zones["label"] == selected_label].iloc[0]
        selected_zone = int(selected["zone_id"])
        zone = predictions[predictions["zone_id"] == selected_zone].sort_values("timestamp")

        if granularity == "3-hour totals":
            chart_data = (
                zone.set_index("timestamp")[["pickups", "prediction"]]
                .resample("3h")
                .sum()
                .reset_index()
            )
            interval_label = "3 hours"
        else:
            chart_data = zone
            interval_label = "hour"

        c1, c2, c3 = st.columns(3)
        c1.metric("Zone blind-test WAPE", f"{selected['wape']:.1%}")
        c2.metric("Actual pickups", f"{selected['actual_pickups']:,.0f}")
        c3.metric("Predicted pickups", f"{selected['predicted_pickups']:,.0f}")
        if selected["actual_pickups"] < len(zone):
            st.warning(
                "This is a sparse-demand zone averaging fewer than one pickup per hour. "
                "Occasional count spikes are intrinsically difficult to predict; use the Top-20 view "
                "for representative operational performance."
            )
        st.pyplot(forecast_figure(chart_data, interval_label), width="stretch")

    with rebalancing_tab:
        st.subheader("Travel-time-aware offline rebalancing snapshot")
        st.info(
            "This snapshot is 2025-06-30 23:00 and is separate from the July blind "
            "test. Route costs are median observed TLC trip times before the snapshot; "
            "sparse OD pairs use an explicit fallback."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial forecast deficit", f"{summary['initial_forecast_deficit']:,.0f}")
        c2.metric("Vehicles moved", f"{summary['vehicles_rebalanced']:,.0f}")
        c3.metric("Remaining deficit", f"{summary['remaining_forecast_deficit']:,.0f}")
        c4.metric(
            "Estimated relocation-time reduction",
            f"{summary['vehicle_minutes_reduction_rate']:.1%}",
            help="Compared with deterministic unit-cost greedy assignment using the same historical OD costs.",
        )
        st.caption(
            f"Direct OD medians backed by at least {summary['min_route_samples']} trips "
            f"cover {summary['observed_od_vehicle_share']:.1%} of moved vehicles; "
            "remaining routes use documented fallback estimates."
        )

        origin_boroughs = sorted(moves["origin_borough"].dropna().unique())
        destination_boroughs = sorted(moves["destination_borough"].dropna().unique())
        filters = st.columns(3)
        selected_origins = filters[0].multiselect(
            "Origin borough", origin_boroughs, default=origin_boroughs
        )
        selected_destinations = filters[1].multiselect(
            "Destination borough", destination_boroughs, default=destination_boroughs
        )
        minimum_vehicles = filters[2].slider(
            "Minimum vehicles per route",
            min_value=1,
            max_value=max(int(moves["vehicles"].max()), 1),
            value=1,
        )
        filtered = moves[
            moves["origin_borough"].isin(selected_origins)
            & moves["destination_borough"].isin(selected_destinations)
            & (moves["vehicles"] >= minimum_vehicles)
        ]
        if filtered.empty:
            st.warning("No routes match the selected filters.")
        else:
            st.pyplot(flow_figure(filtered), width="stretch")
            route_table = filtered[
                [
                    "origin_name", "origin_borough", "destination_name",
                    "destination_borough", "vehicles", "travel_minutes",
                    "vehicle_minutes", "cost_source", "route_samples",
                ]
            ].rename(
                columns={
                    "origin_name": "Origin",
                    "origin_borough": "Origin borough",
                    "destination_name": "Destination",
                    "destination_borough": "Destination borough",
                    "vehicles": "Vehicles",
                    "travel_minutes": "Median trip minutes",
                    "vehicle_minutes": "Estimated vehicle-minutes",
                    "cost_source": "Cost source",
                    "route_samples": "Historical trips",
                }
            )
            st.dataframe(route_table, hide_index=True, width="stretch")

        with st.expander("Inspect zone-level supply and deficit"):
            st.dataframe(
                zone_state[
                    [
                        "zone_name", "borough", "predicted_demand", "available_vehicles",
                        "initial_deficit", "vehicles_in", "vehicles_out", "remaining_deficit",
                    ]
                ].sort_values("initial_deficit", ascending=False),
                hide_index=True,
                width="stretch",
            )

    with protocol_tab:
        st.subheader("What this demo does—and does not claim")
        st.markdown(
            """
- **Forecast view:** saved July 25–31 blind-test predictions from parameters locked before July evaluation.
- **Rebalancing view:** a separate June 30 offline snapshot reconstructed from saved reports.
- **Supply proxy:** previous-hour drop-offs, not observed idle vehicles.
- **Route cost:** median observed June trip time for each OD pair before the snapshot; sparse pairs use an origin-, destination-, or global-median fallback.
- **Optimization:** cheapest-route-first greedy matching; it is not a global minimum-cost-flow solution.
- **No production claim:** historical trip time is an offline estimate, not live traffic, and the demo does not establish lower passenger wait time.
            """
        )


if __name__ == "__main__":
    main()
