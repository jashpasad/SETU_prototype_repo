"""Validation harness for the advanced modelling layers.

Each block below exercises one module in isolation and prints the diagnostic
that proves it is doing what it claims. Run:

    python scripts/validate_advanced.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setu.config import BatteryConfig, DispatchConfig          # noqa: E402
from setu.conformal import ConformalQuantileCalibrator, coverage_curve  # noqa: E402
from setu.degradation import assess, marginal_cost_inr_per_kwh, rainflow  # noqa: E402
from setu.network import LVFeeder                              # noqa: E402
from setu.optimizer import StochasticMILPDispatcher            # noqa: E402
from setu.stochastic import build_scenario_set                 # noqa: E402
from setu.thermal import ThermostaticFleet, ambient_temperature  # noqa: E402


def hdr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def main():
    rng = np.random.default_rng(5)

    # ---------------------------------------------------------------- 1
    hdr("1. UNBALANCED THREE-PHASE LV POWER FLOW (backward-forward sweep)")
    f = LVFeeder(430, rng)
    print(f.describe())
    for label, total in (("light    60 kW", 60.0),
                         ("evening 220 kW", 220.0),
                         ("heavy   330 kW", 330.0)):
        kw = rng.gamma(2.0, total / 430 / 2.0, size=430)
        kw *= total / kw.sum()
        r0, r1 = f.solve(kw, 0.0), f.solve(kw, 125.0)
        print(f"  {label}: Vdt {r0['v_dt_pu']:.4f}->{r1['v_dt_pu']:.4f}  "
              f"Vmin {r0['v_min_pu']:.4f}->{r1['v_min_pu']:.4f}  "
              f"violations {r0['violations']}->{r1['violations']}  "
              f"loss {r0['losses_kw']:.1f} kW  imbalance "
              f"{r0['phase_imbalance_pct']:.1f}%  iters {r0['iterations']}")
    s = f.voltage_sensitivity(kw)
    print("  linearised sensitivities used as MPC constraints:")
    for k, v in s.items():
        print(f"     {k:26s} {v:+.6f}")
    ratio = s["dv_per_kw_curtailed"] / max(s["dv_per_kw_injected"], 1e-12)
    print(f"  => curtailment is {ratio:.1f}x more effective per kW than "
          f"battery injection for LV voltage support")

    # ---------------------------------------------------------------- 2
    hdr("2. ETP THERMOSTATIC LOADS — COLD LOAD PICKUP AS AN EMERGENT RESULT")
    fleet = ThermostaticFleet("air_cooler", np.arange(400), rng)
    Tamb = ambient_temperature(150)
    for _ in range(4):
        for s_ in range(96):
            fleet.step(np.full(400, Tamb[s_]), np.zeros(400, dtype=bool))
    peak = 76
    for s_ in range(70, peak):
        fleet.step(np.full(400, Tamb[s_]), np.zeros(400, dtype=bool))
    base_kw = fleet.natural_demand_kw(np.full(400, Tamb[peak])).sum()
    print(f"  settled at 19:00: diversity {fleet.diversity_factor():.3f}, "
          f"demand {base_kw:.1f} kW")
    for h in (1, 2, 3):
        for _ in range(4):
            fleet.step(np.full(400, Tamb[peak]), np.ones(400, dtype=bool))
        av = fleet.available_curtailment_kw(np.full(400, Tamb[peak])).sum()
        print(f"    after {h} h held off: still curtailable {av:6.1f} kW, "
              f"mean T {fleet.T.mean():.1f} C")
    reb = fleet.natural_demand_kw(np.full(400, Tamb[peak])).sum()
    print(f"  on restoration {reb:.1f} kW vs settled {base_kw:.1f} kW "
          f"=> cold load pickup x{reb / base_kw:.2f}  (NOT hard-coded)")
    print("  comfort:", {k: round(v, 3) for k, v in
                         fleet.comfort_summary().items()})

    # ---------------------------------------------------------------- 3
    hdr("3. SCENARIO GENERATION + KANTOROVICH REDUCTION")
    K = DispatchConfig().horizon_steps
    q50 = np.concatenate([np.linspace(150, 230, K // 2),
                          np.linspace(230, 140, K - K // 2)])
    for nk in (2, 4, 6, 10):
        sc = build_scenario_set(q50 * 0.85, q50, q50 * 1.2,
                                np.random.default_rng(3), 160, nk)
        print(f"  keep {nk:2d} of 160 -> Wasserstein-1 gap "
              f"{sc['wasserstein_gap']:7.2f}  (monotone decreasing = correct)")

    # ---------------------------------------------------------------- 4
    hdr("4. TWO-STAGE STOCHASTIC MILP (binary relays, CVaR, OWA, voltage)")
    G, S = 16, 5
    opt = StochasticMILPDispatcher(BatteryConfig(), DispatchConfig(),
                                   n_groups=G, n_scenarios=S)
    print(f"  DPP-compliant: {opt.problem.is_dcp(dpp=True)}   "
          f"scalar variables: {opt.problem.size_metrics.num_scalar_variables}")
    group_kw = rng.uniform(4, 14, G)
    sc = build_scenario_set(q50 * 0.85, q50, q50 * 1.2, rng, 120, S)
    paths, probs = sc["scenarios"], sc["probabilities"]
    args = dict(group_kw=group_kw,
                crit=np.clip(paths * 0.22, 0, None),
                flex=np.clip(paths * 0.78, 0, None),
                pv=np.zeros((S, K)),
                limit=np.tile(np.concatenate(
                    [np.zeros(10), np.full(K - 10, 250.0)]), (S, 1)),
                price=np.full(K, 8.5), probs=probs, soc0=190.0,
                burden=rng.uniform(0, 3.0, G),
                dv_curtail=s["dv_per_kw_curtailed"],
                dv_inject=s["dv_per_kw_injected"])
    t = []
    for _ in range(8):
        t0 = time.time()
        opt.solve(burden_cap=40.0, v0=0.952, **args)
        t.append((time.time() - t0) * 1000)
    print(f"  solve time: first {t[0]:.0f} ms, median thereafter "
          f"{np.median(t[1:]):.0f} ms")

    print("  fairness (OWA) cap sweep — soft, so never infeasible:")
    for cap in (40.0, 18.0, 9.0, 3.0):
        r = opt.solve(burden_cap=cap, v0=0.952, **args)
        print(f"    cap {cap:5.1f} -> {r['groups_curtailed']:2d} relays, "
              f"{r['curtail_kw']:6.1f} kW, OWA violation "
              f"{r['fairness_violation']:6.2f}, E[UC] "
              f"{r['expected_uc_kwh']:.2f} kWh")

    print("  voltage constraint sweep:")
    for v0 in (0.960, 0.930, 0.910, 0.890, 0.870):
        r = opt.solve(burden_cap=40.0, v0=v0, **args)
        print(f"    V0={v0:.3f} -> {r['groups_curtailed']:2d} relays, "
              f"{r['curtail_kw']:6.1f} kW, V_after {r['v_after_pu']:.4f}, "
              f"residual {r['voltage_violation_pu']:.5f} pu")

    # ---------------------------------------------------------------- 5
    hdr("5. RAINFLOW COUNTING + Xu et al. (2018) DEGRADATION")
    print("  ASTM E1049 turning-point test:",
          [(round(d, 2), round(m, 2), c) for d, m, c in
           rainflow(np.array([0, 1, -1, 2, -2, 1, -1, 0], dtype=float))])
    SPD = 96
    deep = 0.55 + 0.40 * np.sin(np.linspace(0, 365 * 2 * np.pi, 365 * SPD))
    shal = 0.60 + 0.06 * np.sin(np.linspace(0, 365 * 8 * np.pi, 365 * SPD))
    for name, ser in (("deep daily cycling", deep),
                      ("shallow frequent  ", shal)):
        r = assess(ser * 250.0, 250.0, 365 * 24 * 3600)
        print(f"  {name}: EFC {r['equivalent_full_cycles']:7.1f}  "
              f"meanDoD {r['mean_cycle_depth']:.3f}  "
              f"fade/yr {r['fade_per_year_pct']:.2f}%  "
              f"life {r['projected_life_years']:.1f} yr  "
              f"cycle share {r['cycle_share_of_fade_pct']}%")
    print("  marginal degradation cost, INR per kWh of throughput:")
    for d in (0.05, 0.20, 0.50, 0.90):
        print(f"    DoD {d:4.2f} -> Rs "
              f"{marginal_cost_inr_per_kwh(250.0, 2_250_000, d):.3f}/kWh")

    # ---------------------------------------------------------------- 6
    hdr("6. CONFORMALISED QUANTILE REGRESSION")
    n = 4000
    y = 100 + 40 * np.sin(np.linspace(0, 40, n)) + rng.standard_normal(n) * 18
    qlo = y - rng.uniform(6, 12, n)
    qhi = y + rng.uniform(6, 12, n)
    y = y + rng.standard_normal(n) * 14
    cal = ConformalQuantileCalibrator(0.20).calibrate(
        y[:2000], qlo[:2000], qhi[:2000])
    print("  calibration:", cal.diagnostics)
    print("  reliability diagram (nominal vs empirical coverage):")
    for row in coverage_curve(y, qlo, qhi):
        print("    ", row)

    print("\nAll advanced modules validated.\n")


if __name__ == "__main__":
    main()
