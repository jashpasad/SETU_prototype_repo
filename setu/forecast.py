"""Scarcity-event generation and probabilistic net-load forecasting.

The dispatch optimiser is given distributions, not point estimates, because
the decision it makes is a risk decision: spend stored energy now, or hold it
against a deficit that may or may not arrive.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .config import STEPS_PER_DAY, STEPS_PER_HOUR, ScarcityConfig

QUANTILES = (0.10, 0.50, 0.90)


def generate_scarcity(doy: int, cfg: ScarcityConfig,
                      rng: np.random.Generator) -> np.ndarray:
    """Import limit in kW for each step of the day.

    Normal operation sits at the full import limit. A scarcity event either
    drives it to zero ('shed' - the upstream breaker opens) or reduces it to
    a fraction of the limit ('cap').
    """
    limit = np.full(STEPS_PER_DAY, cfg.normal_import_limit_kw)
    month = min(11, int((doy - 1) / 30.5))
    p = cfg.daily_event_probability * cfg.seasonal_multiplier[month]
    if rng.random() > min(p, 0.99):
        return limit

    dur = float(np.clip(rng.normal(cfg.mean_duration_h, cfg.duration_sd_h),
                        cfg.min_duration_h, cfg.max_duration_h))
    start = rng.uniform(cfg.earliest_start_h, cfg.latest_start_h)
    i0 = int(start * STEPS_PER_HOUR)
    i1 = min(STEPS_PER_DAY, i0 + int(round(dur * STEPS_PER_HOUR)))

    if rng.random() < cfg.shed_probability:
        limit[i0:i1] = 0.0
    else:
        frac = rng.uniform(*cfg.cap_fraction_range)
        limit[i0:i1] = cfg.normal_import_limit_kw * frac
    return limit


# ----------------------------------------------------------------------
def _features(net_load: np.ndarray, pv: np.ndarray, doy_arr: np.ndarray) -> np.ndarray:
    """Build the feature matrix for the quantile forecaster."""
    n = net_load.size
    idx = np.arange(n)
    step = idx % STEPS_PER_DAY
    hour = step / STEPS_PER_HOUR
    lag1 = np.roll(net_load, 1)
    lag4 = np.roll(net_load, 4)
    lag_day = np.roll(net_load, STEPS_PER_DAY)
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy_arr / 365), np.cos(2 * np.pi * doy_arr / 365),
        lag1, lag4, lag_day, pv,
    ])


class QuantileNetLoadForecaster:
    """Gradient-boosted quantile regression on feeder net load.

    Trained on a warm-up history, then used to produce P10/P50/P90 net-load
    forecasts over the optimiser's horizon.
    """

    def __init__(self, quantiles=QUANTILES, n_estimators: int = 120):
        self.quantiles = quantiles
        self.models = {
            q: GradientBoostingRegressor(loss="quantile", alpha=q,
                                         n_estimators=n_estimators,
                                         max_depth=3, learning_rate=0.1,
                                         random_state=7)
            for q in quantiles
        }
        self.fitted = False

    def fit(self, net_load: np.ndarray, pv: np.ndarray, doy_arr: np.ndarray):
        X = _features(net_load, pv, doy_arr)
        valid = slice(STEPS_PER_DAY, None)
        for q, m in self.models.items():
            m.fit(X[valid], net_load[valid])
        self.fitted = True
        return self

    def predict(self, net_load: np.ndarray, pv: np.ndarray,
                doy_arr: np.ndarray) -> dict:
        X = _features(net_load, pv, doy_arr)
        return {q: m.predict(X) for q, m in self.models.items()}


def horizon_quantiles(forecaster: QuantileNetLoadForecaster,
                      history: np.ndarray, pv_fc: np.ndarray,
                      doy: int, k: int) -> dict:
    """P10/P50/P90 net load for the next k steps, given recent history."""
    if not forecaster.fitted:
        return {q: pv_fc * 0.0 for q in QUANTILES}
    pad = np.concatenate([history[-STEPS_PER_DAY:], np.repeat(history[-1], k)])
    pv_pad = np.concatenate([np.repeat(pv_fc[0], STEPS_PER_DAY), pv_fc])
    doy_pad = np.full(pad.size, doy)
    out = forecaster.predict(pad, pv_pad, doy_pad)
    return {q: v[-k:] for q, v in out.items()}
