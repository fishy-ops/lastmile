"""Route assignment strategies. Each plan() returns [(driver, [orders in visit order])]."""
from __future__ import annotations

import math

from .eta import NaiveETA
from .world import DEPOT, LAST_ORDER_MIN, Driver, Order, road_km

LATE_PENALTY_KM_PER_MIN = 0.5   # local-search objective: km + 0.5 * predicted late minutes
MIN_BATCH_UNITS = 6             # optimized planner holds small batches...
URGENT_SLACK_MIN = 30           # ...unless an order is about to run out of slack
EST_TRIP_MIN = 60
MAX_ROUTE_UNITS = 6             # cap route length: long routes push the last stops late


def tour_km(seq: list[Order]) -> float:
    if not seq:
        return 0.0
    km, prev = 0.0, DEPOT
    for o in seq:
        km += road_km(prev, o.pt)
        prev = o.pt
    return km + road_km(prev, DEPOT)


def nearest_neighbour(orders: list[Order]) -> list[Order]:
    left, out, cur = list(orders), [], DEPOT
    while left:
        nxt = min(left, key=lambda o: road_km(cur, o.pt))
        left.remove(nxt)
        out.append(nxt)
        cur = nxt.pt
    return out


def _objective(seq, depart, eta) -> float:
    arr = eta.predict_route(seq, depart)
    late = sum(max(0.0, a - o.deadline) for a, o in zip(arr, seq))
    return tour_km(seq) + LATE_PENALTY_KM_PER_MIN * late


def improve(seq: list[Order], depart: float, eta) -> list[Order]:
    """2-opt + relocate on (tour km + predicted lateness) until no move helps."""
    best, bo = list(seq), _objective(seq, depart, eta)
    n = len(best)
    improved = n > 2
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                o = _objective(cand, depart, eta)
                if o < bo - 1e-9:
                    best, bo, improved = cand, o, True
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = list(best)
                cand.insert(j, cand.pop(i))
                o = _objective(cand, depart, eta)
                if o < bo - 1e-9:
                    best, bo, improved = cand, o, True
    return best


def baseline_plan(t, pending, idle, ctx, mode: str = "fifo"):
    """Simple dispatcher: FIFO orders dealt round-robin to every idle driver, visited as dealt
    (mode='fifo') or nearest-neighbour from the depot (mode='nn')."""
    if not pending or not idle:
        return []
    drivers = sorted(idle, key=lambda d: d.id)
    loads = {d.id: [] for d in drivers}
    used = {d.id: 0 for d in drivers}
    k = 0
    for o in sorted(pending, key=lambda o: o.placed):
        for _ in range(len(drivers)):
            d = drivers[k % len(drivers)]
            k += 1
            if used[d.id] + o.units <= d.capacity:
                loads[d.id].append(o)
                used[d.id] += o.units
                break
    plan = []
    for d in drivers:
        if loads[d.id]:
            seq = loads[d.id] if mode == "fifo" else nearest_neighbour(loads[d.id])
            plan.append((d, seq))
    return plan


def _sweep_chunks(orders: list[Order], limits: list[int]) -> list[list[Order]]:
    """Order by angle around the depot (cut at the widest gap), then fill chunks by unit limits."""
    ang = sorted(orders, key=lambda o: math.atan2(o.y - DEPOT[1], o.x - DEPOT[0]))
    if len(ang) > 2:
        a = [math.atan2(o.y - DEPOT[1], o.x - DEPOT[0]) for o in ang]
        gaps = [(a[(i + 1) % len(a)] - a[i]) % (2 * math.pi) for i in range(len(a))]
        s = (gaps.index(max(gaps)) + 1) % len(a)
        ang = ang[s:] + ang[:s]
    chunks, cur, used, ci = [[] for _ in limits], 0, 0, 0
    for o in ang:
        while ci < len(limits) and used + o.units > limits[ci]:
            ci, used = ci + 1, 0
        if ci >= len(limits):
            break  # leftover stays pending for the next wave
        chunks[ci].append(o)
        used += o.units
    return [c for c in chunks if c]


def optimized_plan(t, pending, idle, ctx):
    """Hold small batches, take the most urgent orders, sweep-cluster onto as few drivers as
    capacity allows, then sequence each route with ETA-aware 2-opt/relocate."""
    if not pending or not idle:
        return []
    units_waiting = sum(o.units for o in pending)
    urgent = any(o.deadline - t - EST_TRIP_MIN < URGENT_SLACK_MIN for o in pending)
    flush = ctx["all_placed"] or t >= LAST_ORDER_MIN
    if units_waiting < MIN_BATCH_UNITS and not urgent and not flush:
        return []

    cap_total = sum(d.capacity for d in idle)
    chosen, units = [], 0
    for o in sorted(pending, key=lambda o: o.deadline):  # most urgent first
        if units + o.units <= cap_total:
            chosen.append(o)
            units += o.units
    if not chosen:
        return []

    drivers = sorted(idle, key=lambda d: -d.capacity)
    k = len(drivers)
    for kk in range(1, len(drivers) + 1):
        if math.ceil(units / kk) <= min(MAX_ROUTE_UNITS, min(d.capacity for d in drivers[:kk])):
            k = kk
            break
    drivers = drivers[:k]
    target = math.ceil(units / k)
    chunks = _sweep_chunks(chosen, [min(d.capacity, MAX_ROUTE_UNITS, target) for d in drivers])

    depart = t + ctx["load_min"]
    eta = ctx["eta"]
    return [(d, improve(nearest_neighbour(c), depart, eta)) for d, c in zip(drivers, chunks)]
