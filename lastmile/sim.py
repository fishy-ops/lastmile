"""Event loop: orders arrive, waves dispatch idle drivers, trips execute against true traffic."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .eta import LearnedETA, NaiveETA, leg_features
from .routing import baseline_plan, optimized_plan
from .world import DEPOT, LOAD_MIN, OPEN_MIN, Driver, Order, Traffic, road_km

FUEL_PER_KM = 0.55
WAGE_PER_HOUR = 20.0
WAVE_MIN = 20
HARD_STOP = 40 * 60


@dataclass
class Delivery:
    order_id: int
    driver_id: int
    dispatch: float
    stop_idx: int
    placed: float
    deadline: float
    arrival: float
    pred_naive: float
    pred_learned: float


@dataclass
class Trip:
    driver_id: int
    dispatch: float
    n_stops: int
    km: float
    hours: float

    @property
    def cost(self) -> float:
        return self.km * FUEL_PER_KM + self.hours * WAGE_PER_HOUR


@dataclass
class RunResult:
    strategy: str
    deliveries: list[Delivery] = field(default_factory=list)
    trips: list[Trip] = field(default_factory=list)


def run(strategy: str, orders: list[Order], drivers: list[Driver], seed: int,
        learned: LearnedETA, baseline_mode: str = "fifo") -> RunResult:
    """strategy: 'baseline' or 'optimized'. `learned` is copied so runs don't share state."""
    learned = copy.deepcopy(learned)
    naive = NaiveETA()
    traffic = Traffic(seed)
    res = RunResult(strategy)
    free_at = {d.id: float(OPEN_MIN) for d in drivers}
    obs: list[tuple[float, tuple]] = []  # (time observable, (p, q, depart, first, minutes))
    todo = sorted(orders, key=lambda o: o.placed)
    pending: list[Order] = []
    ip, t = 0, float(OPEN_MIN)

    while t < HARD_STOP:
        # the model only learns from legs already completed by now
        ready = [o for o in obs if o[0] <= t]
        if ready:
            obs = [o for o in obs if o[0] > t]
            for _, (p, q, dep, first, m) in ready:
                learned.add(p, q, dep, first, m)
            learned.refit()

        while ip < len(todo) and todo[ip].placed <= t:
            pending.append(todo[ip])
            ip += 1
        idle = [d for d in drivers if free_at[d.id] <= t]
        ctx = {"eta": learned, "all_placed": ip >= len(todo), "load_min": LOAD_MIN}
        if strategy == "baseline":
            plan = baseline_plan(t, pending, idle, ctx, baseline_mode)
        else:
            plan = optimized_plan(t, pending, idle, ctx)

        for drv, stops in plan:
            depart = t + LOAD_MIN
            pn = naive.predict_route(stops, depart)
            pl = learned.predict_route(stops, depart)
            clock, prev, km = depart, DEPOT, 0.0
            for i, o in enumerate(stops):
                svc_prev = 0.0 if i == 0 else stops[i - 1].service
                a_key = -1 if i == 0 else stops[i - 1].id
                leg = traffic.leg_minutes(a_key, o.id, prev, o.pt, clock, drv.speed)
                obs.append((clock + svc_prev + leg, (prev, o.pt, clock, i == 0, svc_prev + leg)))
                clock += svc_prev + leg
                km += road_km(prev, o.pt)
                res.deliveries.append(Delivery(o.id, drv.id, t, i, o.placed, o.deadline,
                                               clock, pn[i], pl[i]))
                prev = o.pt
            back = traffic.leg_minutes(stops[-1].id, -1, prev, DEPOT, clock + stops[-1].service, drv.speed)
            clock += stops[-1].service + back
            km += road_km(prev, DEPOT)
            free_at[drv.id] = clock
            res.trips.append(Trip(drv.id, t, len(stops), km, (clock - t) / 60))
            taken = {o.id for o in stops}
            pending = [o for o in pending if o.id not in taken]

        if ip >= len(todo) and not pending:
            break
        t += WAVE_MIN
    return res
