# Zonal Taxi Demand Forecasting & Fleet Rebalancing with XGBoost

[中文](README.md) | [English](README_EN.md) · [GitHub](https://github.com/yviegan/xgboost-taxi-demand) · [Live Demo](https://xgboost-taxi-demand-dxfakmprsdudp2k9q3pkko.streamlit.app/)

> Forecasting next-hour pickups across 226 NYC taxi zones from approximately **27.76 million** Yellow Taxi trips, then converting predicted shortages into fleet-rebalancing suggestions.

**[Open the interactive demo](https://xgboost-taxi-demand-dxfakmprsdudp2k9q3pkko.streamlit.app/):** explore the locked July forecasts and a rebalancing snapshot costed with historical zone-to-zone trip times.

## Results at a glance

| Data | Final blind-test WAPE | Best baseline WAPE | Relative improvement |
|---:|---:|---:|---:|
| 27.76M valid trips | **15.83%** | 20.75% | **23.72%** |

- The final result comes from a **July 2025 blind test after parameter lock**; the blind-test outcome was not used for another tuning round.
- Mean WAPE across four rolling test windows was **16.48% ± 1.18%**, showing that performance was not isolated to one favorable week.
- The largest relative errors occurred in **lower-demand zones and weekends**.
- A vehicle-conserving rebalancing MVP closes the loop from demand prediction to shortage detection and dispatch suggestions.

![July blind-test WAPE comparison](reports/figures/blind_test_comparison.png)

## Business problem

The operational challenge is not simply citywide demand volume. It is the mismatch between **where vehicles are currently located and where riders will request trips during the next hour**. Anticipating demand by zone allows potential shortages to be identified before they occur.

The project therefore has two layers:

1. **Forecasting:** predict next-hour realized pickup demand for each taxi zone;
2. **Decision support:** combine the forecast with a supply proxy and suggest vehicle moves from surplus zones to deficit zones.

```mermaid
flowchart LR
    A[NYC TLC trips] --> B[Zone-hour panel]
    B --> C[Leakage-safe time features]
    C --> D[Poisson XGBoost]
    D --> E[Next-hour zone demand]
    E --> F[Demand minus supply proxy]
    F --> G[Fleet-rebalancing suggestions]
```

> TLC records contain completed trips only. The target is therefore **realized demand**, not the full latent demand that includes unserved requests.

## Data engineering

The project uses official [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) from January through July 2025.

| Dataset | Scale |
|---|---:|
| Valid trips, January–June | 23,896,401 |
| Valid trips, July | 3,866,132 |
| Taxi zones | 226 |
| Zone-hour rows, January–June | 981,744 |

The pipeline reads, cleans, and aggregates one month at a time rather than loading all trip-level records into memory. It:

- removes invalid timestamps and zones, reversed trips, and trips longer than six hours;
- aggregates hourly pickups and drop-offs by zone;
- completes the zone-hour grid to form a balanced panel;
- uses pickup timestamps—not trailing drop-offs—to define target-panel boundaries and prevent artificial future zero-demand rows;
- downloads through temporary files and validates Parquet and ZIP integrity before atomic replacement.

## Model and leakage controls

Taxi demand combines short-term momentum with daily and weekly seasonality. Features include:

- **demand lags:** 1, 2, 3, 24, and 168 hours;
- **historical statistics:** rolling means and standard deviations over 3, 6, 24, and 168 hours;
- **calendar signals:** hour, day of week, month, weekend indicator, and cyclic encodings;
- **supply proxy:** previous-hour drop-offs in each zone.

The core estimator is `XGBRegressor(objective="count:poisson")`. The Poisson objective matches the non-negative count target, while boosted trees capture nonlinear interactions among zones, time, and historical demand.

Four controls prevent optimistic leakage:

1. lag features are shifted within each zone;
2. every rolling statistic applies `shift(1)` before the rolling window;
3. train, validation, and test sets are split chronologically rather than randomly;
4. top-demand zones are identified from each fold's training period only.

These rules are covered by automated tests.

## Time-based validation

The data serve three distinct experimental roles:

| Stage | Role |
|---|---|
| January–April | Feature development and temporal tuning |
| Multiple April–June windows | Rolling robustness and segmented error analysis |
| July | Final evaluation after parameter lock |

Each experiment uses an expanding training window, the immediately following seven days for early stopping, and the next seven days for testing. XGBoost is compared with three transparent baselines: previous hour, same hour yesterday, and same hour one week ago.

### Rolling robustness

![Rolling validation WAPE](reports/figures/rolling_wape.png)

XGBoost beat all three baselines in all four windows. Test WAPE was **15.73%, 18.23%, 16.06%, and 15.88%**, averaging **16.48% ± 1.18%**. The mean test-minus-train WAPE gap was 2.54 percentage points: a normal generalization loss without a collapse in later windows.

### Temporal hyperparameter tuning

Sixteen reproducibly sampled parameter combinations and one manual baseline were evaluated on three early development folds. Selection used development-period mean WAPE only.

Locked parameters:

```json
{
  "learning_rate": 0.03,
  "max_depth": 10,
  "min_child_weight": 5,
  "subsample": 0.85,
  "colsample_bytree": 0.7,
  "reg_lambda": 20.0,
  "reg_alpha": 1.0
}
```

The selected parameters won in all three later comparison windows and reduced mean WAPE from 16.55% to **16.32%**, a **1.40% relative improvement**. The gain was small but stable; the value of tuning was a reproducible selection protocol rather than an exaggerated performance jump.

## Final July blind test

July data were downloaded and processed only after parameters had been locked. The model trained through July 17, used July 18–24 for early stopping, and was evaluated once on July 25–31.

| Model | MAE | RMSE | WAPE | Pooled R² |
|---|---:|---:|---:|---:|
| **XGBoost** | **3.89** | **9.84** | **15.83%** | **0.9741** |
| Previous week | 5.10 | 13.33 | 20.75% | 0.9525 |
| Previous hour | 6.67 | 17.49 | 27.16% | 0.9183 |
| Previous day | 7.73 | 23.45 | 31.45% | 0.8531 |

XGBoost reduced WAPE by **23.72%** relative to the strongest baseline. The following chart shows Midtown Center, the highest-demand zone during the blind-test week. The model tracks the dominant daily cycle but smooths or underestimates some sharp peaks.

![Actual and predicted hourly demand in Midtown Center](reports/figures/july_top_zone_forecast.png)

> The best iteration reached the configured 2,000-tree cap. Because that information became visible only after blind evaluation, the cap was not increased and July was not rerun. Any such change should be assessed on a new month.

## Where the model struggles

![Segment WAPE](reports/figures/segment_wape.png)

- **Weekday peak vs. other hours:** 14.39% vs. 17.28%;
- **Top-20 demand zones vs. other zones:** 12.39% vs. 21.94%;
- **Weekdays vs. weekends:** 15.87% vs. 17.96%.

High-demand zones have larger absolute errors but lower relative errors. Lower-demand zones and weekends are the main weaknesses because counts are sparse and irregular events represent a larger share of demand.

![Feature importance](reports/figures/feature_importance.png)

Built-in XGBoost importance indicates that cyclic time features and historical demand lags jointly drive predictions. It describes split contribution, **not causality**.

### Why pooled R² is so high

The 0.9741 value pools all zone-hours. Large differences in average demand across zones account for substantial variance, so it must not be described as “97.41% accuracy.” Median per-zone R² was 0.3279, and 96.46% of zones had positive R². WAPE, time stability, and segment-level results are therefore the primary evidence.

## From forecasts to travel-time-aware rebalancing

The decision layer uses previous-hour drop-offs as a proxy for currently available vehicles:

```text
forecast zone deficit = next-hour predicted demand − previous-hour drop-offs
```

The rebalancer accepts an origin-destination cost matrix, processes feasible routes from lowest to highest cost, and enforces origin capacity, destination need, and total vehicle conservation. Route cost is the **historical median observed trip time** for each origin-destination pair, estimated from TLC trips completed before the snapshot. OD pairs with fewer than 20 observations use an explicit origin-, destination-, or global-median fallback.

![Travel-time-aware rebalancing snapshot](reports/figures/rebalancing_snapshot.png)

The snapshot is from **June 30, 2025 at 23:00**, separate from the July blind test. All 858 units of forecast deficit were matched. Evaluated with the same historical OD times, estimated relocation cost was **13,121 vehicle-minutes** under travel-time-aware assignment versus 26,721 under unit-cost assignment—a **50.89% reduction**. Direct OD medians backed by at least 20 trips covered **75.99%** of moved vehicles; the remainder used explicit fallback estimates.

Historical completed-trip time is not live traffic, and the 50.89% figure is an offline objective result—not a measured reduction in rider waiting or real deadhead time. Greedy matching is also not guaranteed to solve the global minimum-cost transportation problem.

A production system would additionally require real-time vehicle state, road travel times, driver acceptance behavior, service fairness constraints, minimum-cost flow or rolling-horizon optimization, historical replay, and online experimentation.

## Interactive demo

The Streamlit app brings forecast traces, dispatch KPIs, major vehicle flows, and zone-level supply and deficits into one filterable view.

```bash
python -m src.rebalancing_report
python -m src.visualize
streamlit run app.py
```

The app provides:

- **July blind-test forecast:** select any zone and inspect actual versus predicted demand and zone WAPE;
- **June rebalancing snapshot:** inspect KPIs, filter flows by borough and route size, and view zone-level supply and deficits;
- **Scope & protocol:** see the explicit boundaries between the two snapshots, the supply proxy, geographic cost, and offline simulation.

The repository layout is compatible with Streamlit Community Cloud. Public deployment is an external publication step and is not performed automatically.

## Reproduce locally

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.download_data --year 2025 --start-month 1 --end-month 7
python -m src.pipeline --config configs/baseline.json
python -m src.tuning --config configs/tuning.json
python -m src.rolling_validation --config configs/rolling_validation.json
python -m src.blind_test --config configs/blind_test.json

# Saved-result reporting; no retraining
python -m src.rebalancing_report
python -m src.visualize
streamlit run app.py

python -m pytest
```

## Project structure

```text
configs/                  Experiment and locked-parameter configuration
src/data.py               Cleaning, aggregation, and panel extension
src/features.py           Leakage-safe temporal features
src/model.py              XGBoost, baselines, and metrics
src/evaluation.py         Temporal folds and segmented evaluation
src/tuning.py             Time-safe parameter search
src/blind_test.py         Final post-lock evaluation
src/rebalancing.py        Cost-aware fleet rebalancing
src/rebalancing_report.py Rebalancing snapshot and demo artifacts
src/visualize.py          Static result figures
app.py                    Streamlit report explorer
reports/                  Tuning, rolling, blind-test, and demo artifacts
tests/                    Leakage, splitting, selection, conservation, and chart tests
```

## Limitations and future work

The current version completes an offline loop from demand forecasting to dispatch suggestions, but its production boundaries are explicit: TLC contains completed trips rather than unmet demand; previous-hour drop-offs are only a supply proxy; historical OD trip times do not represent live traffic; and greedy matching is not globally optimal.

Future extensions could include:

1. **External signals:** add weather, holidays, major events, and live traffic to improve forecasts during unusual conditions and sudden demand shifts;
2. **Global optimization:** replace greedy matching with minimum-cost flow or linear programming under vehicle-conservation and capacity constraints;
3. **Real-time supply:** use live vehicle locations, occupancy, and in-transit state instead of previous-hour drop-offs;
4. **Live road costs:** replace historical median OD times with a road travel-time API that reflects current congestion;
5. **A new blind-test period:** reserve August and later data for subsequent model changes rather than repeatedly using feedback from July;
6. **Historical replay simulation:** simulate demand arrivals and vehicle movement over time to evaluate passenger wait time, fulfillment rate, and deadhead time rather than only a static cost function.

These are natural steps from an offline prototype toward a production system; they are not required for the current project to demonstrate XGBoost modeling, leakage-safe temporal validation, and a forecast-to-decision workflow.
