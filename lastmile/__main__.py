"""CLI: python -m lastmile [--seeds 5] [--orders 180] [--drivers 8] [--baseline fifo|nn]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from .eta import LearnedETA, warm_start
from .metrics import improvement, summarize
from .sim import run
from .world import generate_drivers, generate_orders


def simulate(seed: int, n_orders: int, n_drivers: int, baseline: str) -> dict:
    orders = generate_orders(n_orders, seed)
    drivers = generate_drivers(n_drivers, seed)
    model = LearnedETA()
    warm_start(model, 3000, seed)
    base = summarize(run("baseline", orders, drivers, seed, model, baseline))
    opt = summarize(run("optimized", orders, drivers, seed, model))
    return {"seed": seed, "baseline": base, "optimized": opt, "improvement": improvement(base, opt)}


def ms(xs):
    return f"{st.mean(xs):7.2f} ± {st.pstdev(xs):.2f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lastmile")
    ap.add_argument("--seeds", type=int, default=5, help="number of simulated days")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--orders", type=int, default=180)
    ap.add_argument("--drivers", type=int, default=12)
    ap.add_argument("--baseline", choices=["fifo", "nn"], default="fifo")
    ap.add_argument("--out", default="results/summary.json")
    a = ap.parse_args(argv)

    runs = [simulate(s, a.orders, a.drivers, a.baseline) for s in range(a.seed0, a.seed0 + a.seeds)]

    def get(d, keys):
        for k in keys:
            d = d[k]
        return d

    col = lambda side, *keys: [get(r[side], keys) for r in runs]

    print(f"\nLast-mile sim: {a.seeds} days x {a.orders} orders, {a.drivers} drivers, baseline={a.baseline}\n")
    print(f"{'metric':<28}{'baseline':>18}{'optimized':>18}")
    rows = [("On-time %", ("on_time_pct",)), ("Mean lateness (late only, min)", ("mean_late_min",)),
            ("Mean order->door (min)", ("mean_lead_min",)), ("Stops per trip", ("stops_per_trip",)),
            ("Km per order", ("km_per_order",)), ("Cost per order ($)", ("cost_per_order",))]
    for label, keys in rows:
        print(f"{label:<28}{ms(col('baseline', *keys)):>18}{ms(col('optimized', *keys)):>18}")
    print(f"\n{'ETA error (optimized fleet)':<28}{'naive':>18}{'learned':>18}")
    for label, key in [("MAE (min)", "mae_min"), ("Bias (min, +=late)", "bias_min"),
                       ("P90 abs error (min)", "p90_abs_min"), ("MAPE (%)", "mape_pct")]:
        print(f"{label:<28}{ms(col('optimized', 'eta_naive', key)):>18}{ms(col('optimized', 'eta_learned', key)):>18}")
    print("\nImprovement vs baseline (mean ± sd over days)")
    for label, key in [("Route cost / order (%)", "cost_per_order_pct"), ("Km / order (%)", "km_per_order_pct"),
                       ("On-time (pct points)", "on_time_pts"), ("ETA MAE, learned vs naive (%)", "eta_mae_pct")]:
        print(f"  {label:<30}{ms([r['improvement'][key] for r in runs])}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "runs": runs}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
