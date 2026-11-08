# lastmile-sim

A last-mile delivery simulator: generated orders and drivers, ETA prediction, and route assignment,
with three tracked metrics: **ETA error**, **on-time %**, and **route-cost improvement vs a simple baseline**.

```
pip install numpy pytest
python3 -m lastmile --seeds 10            # table + results/summary.json
python3 -m lastmile --baseline nn         # tougher baseline (nearest-neighbour sequencing)
python3 -m pytest -q
```

## How it works

- **World** (`world.py`): 24x24 km city, one depot. Orders arrive 08:00-18:00 with lunch/dinner peaks, clustered
  into neighbourhoods, each with a promise deadline (placed + 75-150 min), service time and 1-2 capacity units.
  Drivers have capacity 8/10/12 and a speed factor. **True traffic** (hidden from planners) depends on rush hour,
  a slow downtown, driver, per-leg lognormal noise and 4% incidents. Noise is keyed on the leg, so baseline and
  optimized runs face identical conditions.
- **Dispatch**: every 20 min, idle drivers get routes from the pending queue; trips run against true traffic.
- **ETA** (`eta.py`): `NaiveETA` = road km at 30 km/h + 4 min/stop. `LearnedETA` = ridge regression on leg time
  (distance x hour-of-day, downtown flag, first-leg flag), warm-started on historical legs and updated online, only
  from legs already completed at dispatch time (no peeking). ETAs are predicted at dispatch for every stop and
  scored against actual arrival.
- **Baseline router**: FIFO orders dealt round-robin to every idle driver, visited as dealt (`fifo`) or
  nearest-neighbour (`nn`).
- **Optimized router** (`routing.py`): holds tiny batches unless an order is near its deadline, takes the most urgent
  orders, sweep-clusters onto as few drivers as a route-length cap allows, then sequences each route with 2-opt +
  relocate minimising `km + 0.5 x predicted lateness`, using the learned ETA.
- **Cost** = $0.55/km + $20/driver-hour (dispatch to return).

## Results (10 simulated days, 180 orders/day, 12 drivers, mean ± sd)

| | baseline (fifo) | optimized |
|---|---|---|
| On-time % | 63.4 ± 5.6 | **88.7 ± 2.1** |
| Km / order | 16.5 | **11.3** |
| Cost / order | $23.47 | **$16.25** |

| ETA error | naive | learned |
|---|---|---|
| MAE (min) | 8.6 | **5.6** |
| P90 abs error (min) | 21.1 | **12.3** |
| Bias (min, + = actual later) | +4.8 | +1.7 |

**Route-cost improvement: 30.7% vs fifo, 23.7% vs nearest-neighbour baseline**
(on-time 80.7% -> 88.7% against that tougher baseline).

## Caveats

- Everything is synthetic; the gains show the mechanisms work, not what a real fleet would see.
- Results are sensitive to fleet size: at 8 drivers the system is overloaded (baseline on-time ~29%) and gains look
  inflated; at 16 the baseline is comfortable on service and the cost gap widens.
- The route-length cap (`MAX_ROUTE_UNITS`) matters: without it, batching cut cost but on-time fell *below* the
  baseline when drivers were plentiful. It is a tuned constant, not a derived one.
- Learned-ETA still carries ~+1.7 min of bias and can't see incidents or driver-specific speed.
