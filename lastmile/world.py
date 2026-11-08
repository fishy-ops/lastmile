"""World model: city geometry, generated orders/drivers, and the hidden 'true' traffic."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

CITY_KM = 24.0
DEPOT = (12.0, 12.0)
CENTER = (12.0, 12.0)
ROAD_FACTOR = 1.3          # road km = euclidean km * factor
BASE_SPEED_KMH = 36.0
OPEN_MIN = 8 * 60          # orders start arriving / drivers start
LAST_ORDER_MIN = 18 * 60   # last order placed
LOAD_MIN = 6.0             # loading time at depot before departing


@dataclass(frozen=True)
class Order:
    id: int
    x: float
    y: float
    placed: float      # minutes since midnight
    deadline: float    # promised-by time, minutes since midnight
    service: float     # minutes spent at the door
    units: int         # capacity units used

    @property
    def pt(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Driver:
    id: int
    capacity: int      # units
    speed: float       # multiplier on traffic speed (driver skill / vehicle)


def euclid(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def road_km(p, q) -> float:
    return ROAD_FACTOR * euclid(p, q)


def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_orders(n: int, seed: int) -> list[Order]:
    """n orders for one day: lunch/dinner-peaked arrivals, clustered neighbourhoods."""
    rng = random.Random(f"orders:{seed}")
    centers = [(rng.uniform(3, CITY_KM - 3), rng.uniform(3, CITY_KM - 3)) for _ in range(5)]
    weights = [rng.uniform(0.5, 1.5) for _ in centers]

    def intensity(t: float) -> float:
        h = t / 60
        return 0.5 + 1.2 * _gauss(h, 12.0, 1.0) + 0.9 * _gauss(h, 16.5, 1.5)

    peak = 2.7
    times: list[float] = []
    while len(times) < n:  # rejection sampling of a non-homogeneous arrival profile
        t = rng.uniform(OPEN_MIN, LAST_ORDER_MIN)
        if rng.random() * peak <= intensity(t):
            times.append(t)
    times.sort()

    orders = []
    for i, t in enumerate(times):
        if rng.random() < 0.2:
            x, y = rng.uniform(0, CITY_KM), rng.uniform(0, CITY_KM)
        else:
            cx, cy = rng.choices(centers, weights)[0]
            x = min(CITY_KM, max(0.0, rng.gauss(cx, 3.0)))
            y = min(CITY_KM, max(0.0, rng.gauss(cy, 3.0)))
        orders.append(Order(
            id=i, x=x, y=y, placed=t,
            deadline=t + rng.uniform(75, 150),
            service=rng.uniform(2.0, 6.0),
            units=1 if rng.random() < 0.75 else 2,
        ))
    return orders


def generate_drivers(n: int, seed: int) -> list[Driver]:
    rng = random.Random(f"drivers:{seed}")
    return [Driver(id=i, capacity=rng.choice([8, 10, 12]), speed=rng.uniform(0.9, 1.1))
            for i in range(n)]


class Traffic:
    """Ground truth travel times. Planners never see this; they only observe outcomes.

    Speed depends on time of day (AM/PM rush), zone (slow downtown) and driver, with
    deterministic per-leg lognormal noise and occasional incidents. Deterministic noise
    (keyed on leg endpoints + 30-min bucket) means baseline and optimized runs face the
    same conditions on the same leg.
    """

    def __init__(self, seed: int):
        self.seed = seed

    @staticmethod
    def rush(t: float) -> float:
        h = t / 60
        return 1 - 0.35 * _gauss(h, 8.5, 1.0) - 0.40 * _gauss(h, 17.5, 1.2)

    @staticmethod
    def zone(p, q) -> float:
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        return 0.65 + 0.35 * min(1.0, euclid(mid, CENTER) / 6.0)

    def leg_minutes(self, a_key: int, b_key: int, p, q, depart: float, driver_speed: float) -> float:
        rng = random.Random(f"leg:{self.seed}:{a_key}:{b_key}:{int(depart // 30)}")
        v = BASE_SPEED_KMH * self.rush(depart) * self.zone(p, q) * driver_speed
        minutes = road_km(p, q) / v * 60
        minutes *= math.exp(0.15 * rng.gauss(0, 1))
        if rng.random() < 0.04:
            minutes *= 1.8  # incident
        return minutes
