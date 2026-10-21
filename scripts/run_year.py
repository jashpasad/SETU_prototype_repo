"""Full-year simulation of the reference feeder.

Runs two experiments:
  1. SETU with the fairness constraint enforced (the product)
  2. SETU with the fairness constraint disabled (experimental control)

Both are compared against the baseline - binary feeder control - which is
computed inside the same run from the same load and scarcity realisation,
so the comparison is like for like.

Usage:
    python scripts/run_year.py [--days 365] [--no-control]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setu.config import SimConfig, STEPS_PER_DAY          # noqa: E402
from setu.simulate import run, summarise                  # noqa: E402

WARMUP_DAYS = 30   # ledger window; Gini statistics exclude this period


def gini_stats(daily_gini: np.ndarray, warmup: int = WARMUP_DAYS) -> dict:
    g = daily_gini[warmup:] if daily_gini.size > warmup else daily_gini
    return {
        "gini_mean_steady": round(float(g.mean()), 3),
        "gini_p95_steady": round(float(np.percentile(g, 95)), 3),
        "gini_max_steady": round(float(g.max()), 3),
        "gini_within_limit_pct": round(float((g <= 0.25).mean() * 100), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(days=args.days)

    print(f"SETU simulation - {args.days} days, "
          f"{cfg.feeder.n_households} households + "
          f"{cfg.feeder.n_enterprises} micro-enterprises")

    print("\n[1/2] fairness constraint ENFORCED")
    t0 = time.time()
    res = run(cfg, enforce_fairness=True)
    print(f"      done in {time.time() - t0:.0f}s")
    summary = summarise(res)
    summary.update(gini_stats(res["daily_gini"]))
    summary.update({f"ledger_{k}": v for k, v in res["ledger"].summary().items()})

    np.savez_compressed(out / "traces_fair.npz", **res["traces"],
                        daily_gini=res["daily_gini"],
                        burden_share=res["ledger"].burden_share())

    if not args.no_control:
        print("\n[2/2] fairness constraint DISABLED (control)")
        t0 = time.time()
        res_c = run(cfg, enforce_fairness=False)
        print(f"      done in {time.time() - t0:.0f}s")
        sc = summarise(res_c)
        gc = gini_stats(res_c["daily_gini"])
        summary["control_gini_mean_steady"] = gc["gini_mean_steady"]
        summary["control_gini_p95_steady"] = gc["gini_p95_steady"]
        summary["control_burden_share_max_pct"] = round(
            res_c["ledger"].summary()["burden_share_max_pct"], 1)
        summary["control_full_outage_h_per_day"] = sc["setu_full_outage_h_per_day"]
        summary["fairness_cost_outage_h_per_day"] = round(
            summary["setu_full_outage_h_per_day"]
            - sc["setu_full_outage_h_per_day"], 4)
        summary["control_unserved_critical_mwh_yr"] = sc["setu_unserved_critical_mwh_yr"]
        np.savez_compressed(out / "traces_control.npz", **res_c["traces"],
                            daily_gini=res_c["daily_gini"],
                            burden_share=res_c["ledger"].burden_share())

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    for k, v in summary.items():
        print(f"  {k:<42} {v}")
    print(f"\nSaved to {out}/summary.json and {out}/traces_*.npz")


if __name__ == "__main__":
    main()
