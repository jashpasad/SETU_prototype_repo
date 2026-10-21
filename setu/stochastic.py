"""Scenario generation and reduction for the stochastic dispatch problem.

Planning against a single forecast path - even a pessimistic one - is a
heuristic dressed up as optimisation. The honest formulation is a two-stage
stochastic program: commit relay states now, let continuous recourse adapt
once the realisation is known, and optimise the expectation plus a tail risk
measure over a set of scenarios.

Generating many scenarios is cheap; solving over them is not. We therefore
generate a large sample and reduce it to a small representative set using
fast-forward selection under the Kantorovich (Wasserstein-1) distance, the
standard construction from Dupacova, Growe-Kuska and Roemisch (2003). The
reduced set preserves the probability mass of the discarded scenarios by
assigning each discarded path to its nearest retained neighbour.
"""
from __future__ import annotations

import numpy as np


def generate_scenarios(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray,
                       n: int, rng: np.random.Generator,
                       ar_rho: float = 0.82) -> np.ndarray:
    """Sample net-load paths consistent with a quantile forecast.

    Forecast error is not white: a cloudy half hour is followed by another.
    We therefore drive the paths with an AR(1) latent process mapped through
    the forecast quantiles, so the scenarios have realistic temporal
    correlation rather than independent jitter at each step.

    Returns an array of shape (n, horizon).
    """
    K = q50.size
    z = np.zeros((n, K))
    eps = rng.standard_normal((n, K))
    z[:, 0] = eps[:, 0]
    sd = np.sqrt(1.0 - ar_rho ** 2)
    for k in range(1, K):
        z[:, k] = ar_rho * z[:, k - 1] + sd * eps[:, k]

    # map the standard-normal latent onto the forecast interval; 1.2816 is
    # the standard normal 90th percentile, so z = +1.2816 lands on q90
    upper = (q90 - q50) / 1.2816
    lower = (q50 - q10) / 1.2816
    spread = np.where(z >= 0, upper[None, :], lower[None, :])
    return q50[None, :] + z * spread


def kantorovich_reduce(paths: np.ndarray, n_keep: int,
                       probs: np.ndarray | None = None) -> tuple:
    """Fast-forward selection under the Wasserstein-1 distance.

    Returns (indices_kept, probabilities_kept).
    """
    n = paths.shape[0]
    if probs is None:
        probs = np.full(n, 1.0 / n)
    if n_keep >= n:
        return np.arange(n), probs

    # pairwise L1 distance between scenario paths
    D = np.abs(paths[:, None, :] - paths[None, :, :]).sum(axis=2)

    kept: list[int] = []
    # first pick: the scenario minimising the probability-weighted distance
    # to every other scenario
    first = int(np.argmin((probs[None, :] * D).sum(axis=1)))
    kept.append(first)
    nearest = D[first].copy()

    while len(kept) < n_keep:
        best, best_cost = -1, np.inf
        for c in range(n):
            if c in kept:
                continue
            cost = float((probs * np.minimum(nearest, D[c])).sum())
            if cost < best_cost:
                best, best_cost = c, cost
        kept.append(best)
        nearest = np.minimum(nearest, D[best])

    kept_arr = np.array(sorted(kept))
    # redistribute discarded mass onto the nearest retained scenario
    assign = np.argmin(D[np.ix_(np.arange(n), kept_arr)], axis=1)
    new_probs = np.zeros(kept_arr.size)
    np.add.at(new_probs, assign, probs)
    new_probs /= new_probs.sum()
    return kept_arr, new_probs


def wasserstein_gap(full: np.ndarray, kept: np.ndarray,
                    probs_kept: np.ndarray) -> float:
    """Wasserstein-1 distance between the full sample and the reduced set.

    Reported as a diagnostic: it is the quantity the reduction minimises, and
    it bounds how much the reduced problem's optimal value can differ from
    the full one for Lipschitz cost functions.
    """
    D = np.abs(full[:, None, :] - kept[None, :, :]).sum(axis=2)
    return float((np.min(D, axis=1)).mean())


def build_scenario_set(q10, q50, q90, rng, n_sample: int = 160,
                       n_keep: int = 6) -> dict:
    """Generate, reduce, and report the reduction quality."""
    full = generate_scenarios(q10, q50, q90, n_sample, rng)
    idx, probs = kantorovich_reduce(full, n_keep)
    kept = full[idx]
    return {
        "scenarios": kept,
        "probabilities": probs,
        "n_sample": n_sample,
        "wasserstein_gap": wasserstein_gap(full, kept, probs),
        "sample_spread": float(full.std(axis=0).mean()),
    }
