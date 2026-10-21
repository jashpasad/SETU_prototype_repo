"""Battery degradation from rainflow cycle counting.

Counting "equivalent full cycles" as throughput divided by capacity is a
convenient fiction. Degradation is strongly non-linear in depth of discharge:
a hundred 10% cycles do far less damage than ten 100% cycles with the same
throughput. For a controller whose whole job is to decide when to cycle, that
distinction is the difference between a seven-year asset and a four-year one.

This module implements:

  1. ASTM E1049 rainflow counting on the state-of-charge series, extracting
     half and full cycles with their depth and mean state of charge;
  2. the Xu et al. (2018) semi-empirical life model, which decomposes fade
     into cycle stress (depth, mean SoC, temperature) and calendar stress
     (time, mean SoC, temperature);
  3. a marginal degradation cost in rupees per kWh of throughput, which is
     what the dispatch optimiser actually needs as a price signal.

Reference: B. Xu, A. Oudalov, A. Ulbig, G. Andersson, D. Kirschen,
"Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment",
IEEE Transactions on Smart Grid, 2018.
"""
from __future__ import annotations

import numpy as np

# LFP stress-model coefficients
K_D1, K_D2, K_D3 = 1.4e5, -5.01e-1, -1.23e5
K_SIGMA, SIGMA_REF = 1.04, 0.5
K_T, T_REF_K = 6.93e-2, 298.15
K_CAL = 4.14e-10          # per second, calendar
NON_LINEAR_A = 5.75e-2


def _extrema(x: np.ndarray) -> np.ndarray:
    """Reduce a series to its turning points."""
    if x.size < 3:
        return x
    d = np.diff(x)
    sign = np.sign(d)
    sign[sign == 0] = 1
    turn = np.where(np.diff(sign) != 0)[0] + 1
    return np.concatenate(([x[0]], x[turn], [x[-1]]))


def rainflow(series: np.ndarray) -> list[tuple]:
    """ASTM E1049 rainflow counting.

    Returns a list of (depth, mean, count) where count is 0.5 for a half
    cycle and 1.0 for a full cycle.
    """
    pts = _extrema(np.asarray(series, dtype=float))
    stack: list[float] = []
    cycles: list[tuple] = []

    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            r1 = abs(stack[-2] - stack[-3])
            r2 = abs(stack[-1] - stack[-2])
            if r2 < r1:
                break
            # the middle range is closed as a full cycle
            depth = r1
            mean = (stack[-2] + stack[-3]) / 2.0
            cycles.append((depth, mean, 1.0))
            del stack[-3:-1]

    # residue counted as half cycles
    for i in range(len(stack) - 1):
        depth = abs(stack[i + 1] - stack[i])
        mean = (stack[i + 1] + stack[i]) / 2.0
        cycles.append((depth, mean, 0.5))

    return [c for c in cycles if c[0] > 1e-9]


# ----------------------------------------------------------------------
def _stress_depth(d: np.ndarray) -> np.ndarray:
    return 1.0 / (K_D1 * np.power(np.clip(d, 1e-4, 1.0), K_D2) + K_D3)


def _stress_soc(sigma: np.ndarray) -> np.ndarray:
    return np.exp(K_SIGMA * (sigma - SIGMA_REF))


def _stress_temp(T_K: float) -> float:
    return float(np.exp(K_T * (T_K - T_REF_K) * T_REF_K / T_K))


def assess(soc_series_kwh: np.ndarray, capacity_kwh: float,
           seconds: float, cell_temp_c: float = 32.0) -> dict:
    """Full degradation assessment over a state-of-charge trajectory."""
    soc = np.clip(np.asarray(soc_series_kwh) / capacity_kwh, 0.0, 1.0)
    cycles = rainflow(soc)
    T_K = cell_temp_c + 273.15
    sT = _stress_temp(T_K)

    if cycles:
        d = np.array([c[0] for c in cycles])
        m = np.array([c[1] for c in cycles])
        n = np.array([c[2] for c in cycles])
        f_cyc = float((n * _stress_depth(d) * _stress_soc(m) * sT).sum())
        efc = float((n * d).sum())
    else:
        d = m = n = np.array([])
        f_cyc, efc = 0.0, 0.0

    f_cal = float(K_CAL * seconds * _stress_soc(np.array([soc.mean()]))[0] * sT)
    f_total = f_cyc + f_cal
    # non-linear capacity fade; saturating form from the reference model
    fade = 1.0 - NON_LINEAR_A * np.exp(-f_total) - (1.0 - NON_LINEAR_A) * np.exp(-f_total * 0.9)

    years = seconds / (365.25 * 24 * 3600)
    fade_per_year = fade / max(years, 1e-9)
    life_years = 0.20 / max(fade_per_year, 1e-12)   # end of life at 80% SoH

    return {
        "n_cycles_counted": len(cycles),
        "equivalent_full_cycles": round(efc, 1),
        "mean_cycle_depth": round(float(d.mean()) if d.size else 0.0, 4),
        "p95_cycle_depth": round(float(np.percentile(d, 95)) if d.size else 0.0, 4),
        "cycle_stress": round(f_cyc, 6),
        "calendar_stress": round(f_cal, 6),
        "cycle_share_of_fade_pct": round(
            100 * f_cyc / max(f_total, 1e-12), 1),
        "capacity_fade_pct": round(100 * fade, 3),
        "fade_per_year_pct": round(100 * fade_per_year, 3),
        "projected_life_years": round(min(life_years, 25.0), 1),
        "naive_efc_throughput": round(
            float(np.abs(np.diff(soc)).sum() / 2.0), 1),
    }


def marginal_cost_inr_per_kwh(capacity_kwh: float, cost_inr: float,
                              depth: float, mean_soc: float = 0.6,
                              cell_temp_c: float = 32.0) -> float:
    """Degradation cost of one kWh of throughput at a given cycle depth.

    This is the number the optimiser should be using as its cycling weight,
    and it varies by roughly an order of magnitude across the depth range -
    which is precisely why a single constant weight mis-prices shallow
    cycling.
    """
    sT = _stress_temp(cell_temp_c + 273.15)
    f = float(_stress_depth(np.array([depth]))[0]
              * _stress_soc(np.array([mean_soc]))[0] * sT)
    throughput_kwh = 2.0 * depth * capacity_kwh
    fade_fraction = f / 0.20          # share of usable life consumed
    return float(cost_inr * fade_fraction / max(throughput_kwh, 1e-9))
