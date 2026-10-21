"""Fairness-constrained allocation of curtailment across connections.

The optimiser decides HOW MUCH flexible load to curtail. This module decides
WHOSE. Without a fairness rule, any sensible allocator repeatedly targets the
same high-flexibility connections, because they are the cheapest place to
find kilowatts - and the scheme loses consent within weeks.

SETU allocates to the connections with the least accumulated curtailment
first, and reports the Gini coefficient of that accumulation as a published
accountability metric.
"""
from __future__ import annotations

import numpy as np


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector. 0 = perfectly equal."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    if x.sum() <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1.0) / n)


def allocate_curtailment(target_kw: float,
                         flex_available_kw: np.ndarray,
                         cumulative_curtailed_kwh: np.ndarray,
                         enforce_fairness: bool = True,
                         tol: float = 1e-9) -> np.ndarray:
    """Split an aggregate curtailment target across connections.

    enforce_fairness=True
        Connections are drawn from the least-burdened quartile outward, and
        within each pool the target is water-filled equally. This uses
        history (who has already paid) and spreads the instantaneous burden
        at the same time.

    enforce_fairness=False
        Largest flexible loads first - the naive allocation, kept as an
        experimental control so the fairness gain can be measured rather
        than asserted.
    """
    n = flex_available_kw.size
    alloc = np.zeros(n)
    if target_kw <= tol:
        return alloc
    remaining = float(target_kw)

    if not enforce_fairness:
        for i in np.argsort(-flex_available_kw, kind="stable"):
            if remaining <= tol:
                break
            take = min(flex_available_kw[i], remaining)
            alloc[i] = take
            remaining -= take
        return alloc

    cum = cumulative_curtailed_kwh
    for pct in (10, 20, 30, 40, 50, 65, 80, 100):
        if remaining <= tol:
            break
        thresh = np.percentile(cum, pct)
        pool = np.where((cum <= thresh) & ((flex_available_kw - alloc) > tol))[0]
        for _ in range(40):
            if remaining <= tol or pool.size == 0:
                break
            headroom = flex_available_kw[pool] - alloc[pool]
            live = pool[headroom > tol]
            if live.size == 0:
                break
            quantum = remaining / live.size
            take = np.minimum(flex_available_kw[live] - alloc[live], quantum)
            alloc[live] += take
            remaining -= float(take.sum())
    return alloc


class FairnessLedger:
    """Rolling record of who has been curtailed, and how heavily.

    Fairness is measured as BURDEN SHARE - the fraction of a connection's own
    deferrable energy that SETU has taken - not as absolute kWh. Absolute kWh
    across connections with very different deferrable capacity is unequal by
    construction and says nothing about whether the scheme is fair. Burden
    share asks the question a participant actually cares about: how much of
    what I offered did you use, compared with everyone else?
    """

    MIN_AVAILABILITY_KWH = 5.0   # ignore connections with nothing to defer

    def __init__(self, n_connections: int, window_days: int = 30,
                 steps_per_day: int = 96):
        self.n = n_connections
        self.window = window_days * steps_per_day
        self.cumulative_kwh = np.zeros(n_connections)
        self.available_kwh = np.zeros(n_connections)
        self._buf_curt: list[np.ndarray] = []
        self._buf_avail: list[np.ndarray] = []
        self.gini_history: list[float] = []

    def record(self, curtailed_kwh: np.ndarray, available_kwh: np.ndarray):
        self._buf_curt.append(curtailed_kwh.copy())
        self._buf_avail.append(available_kwh.copy())
        self.cumulative_kwh += curtailed_kwh
        self.available_kwh += available_kwh
        if len(self._buf_curt) > self.window:
            self.cumulative_kwh -= self._buf_curt.pop(0)
            self.available_kwh -= self._buf_avail.pop(0)
        self.gini_history.append(self.burden_gini())

    def burden_share(self) -> np.ndarray:
        """Per-connection curtailed / available, for eligible participants."""
        mask = self.available_kwh > self.MIN_AVAILABILITY_KWH
        if not mask.any():
            return np.zeros(0)
        return self.cumulative_kwh[mask] / self.available_kwh[mask]

    def burden_gini(self) -> float:
        b = self.burden_share()
        return gini(b) if b.size else 0.0

    @property
    def current_gini(self) -> float:
        return self.gini_history[-1] if self.gini_history else 0.0

    def summary(self) -> dict:
        g = np.array(self.gini_history) if self.gini_history else np.array([0.0])
        b = self.burden_share()
        return {
            "gini_mean": float(g.mean()),
            "gini_p95": float(np.percentile(g, 95)),
            "gini_max": float(g.max()),
            "burden_share_mean_pct": float(b.mean() * 100) if b.size else 0.0,
            "burden_share_max_pct": float(b.max() * 100) if b.size else 0.0,
            "eligible_participants": int(b.size),
            "curtailed_kwh_mean": float(self.cumulative_kwh.mean()),
            "curtailed_kwh_max": float(self.cumulative_kwh.max()),
        }
