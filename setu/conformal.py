"""Conformalised quantile regression for the net-load forecast.

A gradient-boosted quantile model trained with the pinball loss is not
calibrated: its nominal 80% interval routinely covers 60-70% of outcomes out
of sample, and a dispatch optimiser told to plan against a "P90" that is
really a P72 will under-reserve energy and lose critical load.

Conformalised quantile regression (Romano, Patterson and Candes, NeurIPS
2019) repairs this with a finite-sample distribution-free guarantee: given
exchangeable data, the conformalised interval covers at least 1 - alpha of
outcomes, whatever the underlying model is doing.

Construction
------------
On a held-out calibration split, score each point by how far outside the
predicted interval it fell:

    E_i = max( q_lo(x_i) - y_i ,  y_i - q_hi(x_i) )

Take Q as the ceil((n+1)(1-alpha))/n empirical quantile of E, then widen:

    C(x) = [ q_lo(x) - Q , q_hi(x) + Q ]

The adjustment is signed, so a model whose intervals are already too wide is
tightened rather than inflated.
"""
from __future__ import annotations

import numpy as np


class ConformalQuantileCalibrator:
    """Split-conformal wrapper around a pair of predicted quantiles."""

    def __init__(self, alpha: float = 0.20):
        self.alpha = alpha
        self.q_adjust: float | None = None
        self.diagnostics: dict = {}

    # ------------------------------------------------------------------
    def calibrate(self, y: np.ndarray, q_lo: np.ndarray,
                  q_hi: np.ndarray) -> "ConformalQuantileCalibrator":
        y, q_lo, q_hi = map(np.asarray, (y, q_lo, q_hi))
        scores = np.maximum(q_lo - y, y - q_hi)
        n = scores.size
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_adjust = float(np.quantile(scores, level, method="higher"))

        raw_cov = float(((y >= q_lo) & (y <= q_hi)).mean())
        lo, hi = self.apply(q_lo, q_hi)
        cal_cov = float(((y >= lo) & (y <= hi)).mean())
        self.diagnostics = {
            "nominal_coverage": 1 - self.alpha,
            "raw_coverage": round(raw_cov, 4),
            "calibrated_coverage": round(cal_cov, 4),
            "adjustment_kw": round(self.q_adjust, 3),
            "raw_mean_width_kw": round(float((q_hi - q_lo).mean()), 2),
            "calibrated_mean_width_kw": round(float((hi - lo).mean()), 2),
            "n_calibration": int(n),
        }
        return self

    # ------------------------------------------------------------------
    def apply(self, q_lo: np.ndarray, q_hi: np.ndarray) -> tuple:
        if self.q_adjust is None:
            return q_lo, q_hi
        return q_lo - self.q_adjust, q_hi + self.q_adjust

    # ------------------------------------------------------------------
    def upper(self, q_hi: np.ndarray) -> np.ndarray:
        """Calibrated upper bound - what dispatch should reserve against."""
        return q_hi + (self.q_adjust or 0.0)


def coverage_curve(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                   alphas=(0.05, 0.10, 0.20, 0.30, 0.40)) -> list[dict]:
    """Reliability diagram: nominal versus empirical coverage.

    A well-calibrated forecaster lies on the diagonal. Reporting this is the
    difference between claiming the forecast is probabilistic and showing it.
    """
    out = []
    n = y.size
    half = n // 2
    for a in alphas:
        cal = ConformalQuantileCalibrator(alpha=a).calibrate(
            y[:half], q_lo[:half], q_hi[:half])
        lo, hi = cal.apply(q_lo[half:], q_hi[half:])
        out.append({
            "alpha": a,
            "nominal": round(1 - a, 3),
            "raw": round(float(((y[half:] >= q_lo[half:])
                                & (y[half:] <= q_hi[half:])).mean()), 3),
            "conformal": round(float(((y[half:] >= lo)
                                      & (y[half:] <= hi)).mean()), 3),
        })
    return out
