import pytest

from lastmile.__main__ import simulate
from lastmile.eta import LearnedETA, warm_start
from lastmile.sim import run
from lastmile.world import generate_drivers, generate_orders

SEED, N, D = 1, 120, 12


@pytest.fixture(scope="module")
def world():
    orders, drivers = generate_orders(N, SEED), generate_drivers(D, SEED)
    model = LearnedETA()
    warm_start(model, 3000, SEED)
    return orders, drivers, model


@pytest.mark.parametrize("strategy", ["baseline", "optimized"])
def test_every_order_delivered_exactly_once(world, strategy):
    orders, drivers, model = world
    res = run(strategy, orders, drivers, SEED, model)
    assert sorted(d.order_id for d in res.deliveries) == [o.id for o in orders]


@pytest.mark.parametrize("strategy", ["baseline", "optimized"])
def test_capacity_and_causality(world, strategy):
    orders, drivers, model = world
    by_id, cap = {o.id: o for o in orders}, {d.id: d.capacity for d in drivers}
    res = run(strategy, orders, drivers, SEED, model)
    trips = {}
    for d in res.deliveries:
        trips.setdefault((d.driver_id, d.dispatch), []).append(d)
        assert d.dispatch >= by_id[d.order_id].placed       # never dispatched before it exists
        assert d.arrival > d.dispatch
    for (drv, _), ds in trips.items():
        assert sum(by_id[d.order_id].units for d in ds) <= cap[drv]


def test_a_driver_never_runs_two_trips_at_once(world):
    orders, drivers, model = world
    res = run("optimized", orders, drivers, SEED, model)
    per = {}
    for tr in res.trips:
        per.setdefault(tr.driver_id, []).append((tr.dispatch, tr.dispatch + tr.hours * 60))
    for spans in per.values():
        spans.sort()
        assert all(a[1] <= b[0] + 1e-6 for a, b in zip(spans, spans[1:]))


def test_deterministic():
    assert simulate(3, 60, 8, "fifo") == simulate(3, 60, 8, "fifo")


def test_headline_claims_hold_across_seeds():
    rs = [simulate(s, 150, 12, "fifo") for s in range(3)]
    for r in rs:
        assert r["improvement"]["cost_per_order_pct"] > 10          # cheaper routes
        assert r["improvement"]["on_time_pts"] > 0                  # and not at the expense of service
        assert r["optimized"]["eta_learned"]["mae_min"] < r["optimized"]["eta_naive"]["mae_min"]
