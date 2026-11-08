"""ETA predictors: a naive constant-speed baseline and an online-learned ridge model."""
from __future__ import annotations

import math
import random

import numpy as np

from .world import CENTER, CITY_KM, DEPOT, ROAD_FACTOR, Order, Traffic, euclid, road_km

N_FEATURES = 4 + 24


def _central(p, q) -> int:
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    return 1 if euclid(mid, CENTER) < 6.0 else 0


def leg_features(p, q, depart: float, first: bool) -> np.ndarray:
    """[1, first_leg, d, d*central, d*onehot(hour)] where d = road km."""
    d = road_km(p, q)
    f = np.zeros(N_FEATURES)
    f[0] = 1.0
    f[1] = 1.0 if first else 0.0
    f[2] = d
    f[3] = d * _central(p, q)
    f[4 + int(depart // 60) % 24] = d
    return f


class NaiveETA:
    """Road distance at a flat 30 km/h plus a flat 4 min per stop. What a first version ships."""
    name = "naive"

    def leg(self, p, q, depart: float, first: bool) -> float:
        return road_km(p, q) / 30.0 * 60 + (0.0 if first else 4.0)

    def predict_route(self, stops: list[Order], depart: float) -> list[float]:
        t, prev, out = depart, DEPOT, []
        for i, o in enumerate(stops):
            t += self.leg(prev, o.pt, t, i == 0)
            out.append(t)
            prev = o.pt
        return out


class LearnedETA(NaiveETA):
    """Ridge regression on leg time (travel + previous stop's service), updated online.

    Sufficient statistics (X'X, X'y) are accumulated so refits are O(features^3), not O(history).
    """
    name = "learned"

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.xtx = np.zeros((N_FEATURES, N_FEATURES))
        self.xty = np.zeros(N_FEATURES)
        self.n = 0
        self.w = np.zeros(N_FEATURES)
        self._fallback = NaiveETA()

    def add(self, p, q, depart: float, first: bool, minutes: float) -> None:
        f = leg_features(p, q, depart, first)
        self.xtx += np.outer(f, f)
        self.xty += f * minutes
        self.n += 1

    def refit(self) -> None:
        if self.n >= 30:
            self.w = np.linalg.solve(self.xtx + self.lam * np.eye(N_FEATURES), self.xty)

    def leg(self, p, q, depart: float, first: bool) -> float:
        if self.n < 30:
            return self._fallback.leg(p, q, depart, first)
        d = road_km(p, q)
        w = self.w
        m = w[0] + w[1] * first + d * (w[2] + w[3] * _central(p, q) + w[4 + int(depart // 60) % 24])
        return max(0.5, m)


def warm_start(model: LearnedETA, n: int, seed: int) -> None:
    """Pre-train on historical legs sampled across the day (what a fleet logs before launch)."""
    rng = random.Random(f"history:{seed}")
    traffic = Traffic(seed=10_000 + seed)
    for k in range(n):
        p = (rng.uniform(0, CITY_KM), rng.uniform(0, CITY_KM))
        q = (rng.uniform(0, CITY_KM), rng.uniform(0, CITY_KM))
        depart = rng.uniform(8 * 60, 22 * 60)
        first = rng.random() < 0.15
        minutes = traffic.leg_minutes(100_000 + k, 200_000 + k, p, q, depart, rng.uniform(0.9, 1.1))
        if not first:
            minutes += rng.uniform(2.0, 6.0)
        model.add(p, q, depart, first, minutes)
    model.refit()
