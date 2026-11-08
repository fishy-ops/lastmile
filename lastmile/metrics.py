"""Metrics: ETA error, on-time %, and route-cost improvement vs the baseline."""
from __future__ import annotations

import numpy as np

from .sim import RunResult


def eta_metrics(res: RunResult, which: str) -> dict:
    pred = np.array([getattr(d, f"pred_{which}") for d in res.deliveries])
    act = np.array([d.arrival for d in res.deliveries])
    disp = np.array([d.dispatch for d in res.deliveries])
    err = act - pred
    return {
        "mae_min": float(np.abs(err).mean()),
        "bias_min": float(err.mean()),          # >0: actual later than predicted
        "p90_abs_min": float(np.percentile(np.abs(err), 90)),
        "mape_pct": float((np.abs(err) / np.maximum(act - disp, 1.0)).mean() * 100),
    }


def delivery_metrics(res: RunResult) -> dict:
    arr = np.array([d.arrival for d in res.deliveries])
    dl = np.array([d.deadline for d in res.deliveries])
    placed = np.array([d.placed for d in res.deliveries])
    late = np.maximum(0.0, arr - dl)
    n = len(arr)
    return {
        "orders": n,
        "on_time_pct": float((arr <= dl).mean() * 100),
        "mean_late_min": float(late[late > 0].mean()) if (late > 0).any() else 0.0,
        "mean_lead_min": float((arr - placed).mean()),
        "trips": len(res.trips),
        "stops_per_trip": n / max(1, len(res.trips)),
        "km": float(sum(t.km for t in res.trips)),
        "km_per_order": float(sum(t.km for t in res.trips) / n),
        "cost": float(sum(t.cost for t in res.trips)),
        "cost_per_order": float(sum(t.cost for t in res.trips) / n),
    }


def summarize(res: RunResult) -> dict:
    return {**delivery_metrics(res),
            "eta_naive": eta_metrics(res, "naive"),
            "eta_learned": eta_metrics(res, "learned")}


def improvement(base: dict, opt: dict) -> dict:
    pct = lambda b, o: (b - o) / b * 100 if b else 0.0
    return {
        "cost_per_order_pct": pct(base["cost_per_order"], opt["cost_per_order"]),
        "km_per_order_pct": pct(base["km_per_order"], opt["km_per_order"]),
        "on_time_pts": opt["on_time_pct"] - base["on_time_pct"],
        "eta_mae_pct": pct(opt["eta_naive"]["mae_min"], opt["eta_learned"]["mae_min"]),
    }
