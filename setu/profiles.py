"""Appliance-level load synthesis and rooftop PV output for the feeder.

Load is built bottom-up from appliances rather than top-down from a feeder
shape, because SETU's whole mechanism depends on knowing which watts are
critical and which are deferrable. A top-down profile cannot answer that.
"""
from __future__ import annotations

import numpy as np

from .config import STEPS_PER_DAY, STEPS_PER_HOUR, DT_HOURS, FeederConfig

# name, kW, tier, hours/day, window start h, window end h, ownership share
HOUSEHOLD_APPLIANCES = [
    ("lights",        0.036, "critical", 6.0, 18.0, 24.0, 1.00),
    ("fan",           0.055, "critical", 9.0, 11.0, 24.0, 0.95),
    ("phone_charge",  0.010, "critical", 4.0, 19.0, 24.0, 1.00),
    ("tv",            0.060, "critical", 3.5, 19.0, 23.0, 0.78),
    ("refrigerator",  0.150, "flexible", 8.0,  0.0, 24.0, 0.62),
    ("water_pump",    0.750, "flexible", 0.8,  5.0, 22.0, 0.55),
    ("water_heater",  2.000, "flexible", 0.5,  5.5, 21.0, 0.34),
    ("air_cooler",    1.400, "flexible", 5.0, 12.0, 24.0, 0.40),
    ("ev_charger",    1.300, "flexible", 3.0, 18.0, 24.0, 0.12),
]

ENTERPRISE_APPLIANCES = [
    ("lighting_fans", 0.400, "critical", 11.0,  8.0, 21.0, 1.00),
    ("pos_cold",      0.600, "flexible", 10.0,  8.0, 21.0, 0.55),
    ("machinery",     2.600, "flexible",  5.5,  9.0, 19.0, 0.70),
]


def _window_weights(start_h: float, end_h: float, peakiness: float = 1.6) -> np.ndarray:
    """A smooth within-window usage shape, peaking near the window centre."""
    t = np.arange(STEPS_PER_DAY) / STEPS_PER_HOUR
    w = np.zeros(STEPS_PER_DAY)
    inside = (t >= start_h) & (t < end_h)
    if not inside.any():
        return w
    span = end_h - start_h
    x = (t[inside] - start_h) / span
    w[inside] = np.sin(np.pi * x) ** peakiness + 0.18
    return w / w.sum()


def _seasonal_factor(doy: int, appliance: str) -> float:
    """Cooling and heating loads are strongly seasonal in western India."""
    summer = np.exp(-((doy - 140) ** 2) / (2 * 55.0 ** 2))      # peaks mid-May
    winter = np.exp(-((((doy - 15) % 365) - 0) ** 2) / (2 * 45.0 ** 2))
    if appliance == "air_cooler":
        return 0.12 + 1.55 * summer
    if appliance == "water_heater":
        return 0.25 + 1.40 * winter
    if appliance == "fan":
        return 0.55 + 0.75 * summer
    if appliance == "refrigerator":
        return 0.80 + 0.45 * summer
    return 1.0


class FeederPopulation:
    """Owns the per-connection appliance set and generates daily profiles."""

    def __init__(self, cfg: FeederConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.n_households + cfg.n_enterprises
        self.is_enterprise = np.zeros(self.n, dtype=bool)
        self.is_enterprise[cfg.n_households:] = True

        self.owned: list[list[tuple]] = []
        for i in range(self.n):
            spec = ENTERPRISE_APPLIANCES if self.is_enterprise[i] else HOUSEHOLD_APPLIANCES
            mine = []
            for name, kw, tier, hrs, s, e, share in spec:
                if rng.random() <= share:
                    # per-connection variation in rating and usage habit
                    kw_i = kw * rng.normal(1.0, 0.08)
                    hrs_i = max(0.1, hrs * rng.normal(1.0, 0.16))
                    shift = rng.normal(0.0, 0.7)
                    mine.append((name, kw_i, tier, hrs_i,
                                 max(0.0, s + shift), min(24.0, e + shift)))
            self.owned.append(mine)

        # Rooftop PV allocation
        self.pv_kwp = np.zeros(self.n)
        hh = np.arange(cfg.n_households)
        chosen = rng.choice(hh, size=int(cfg.pv_household_fraction * cfg.n_households),
                            replace=False)
        self.pv_kwp[chosen] = cfg.pv_kwp_per_household * rng.normal(1.0, 0.12, chosen.size)
        self.pv_kwp[cfg.n_households:] = cfg.pv_kwp_enterprise
        self.pv_kwp = np.clip(self.pv_kwp, 0, None)
        self.total_pv_kwp = float(self.pv_kwp.sum())

    # ------------------------------------------------------------------
    def day_profiles(self, doy: int) -> dict:
        """Return per-connection critical and flexible demand for one day.

        Shapes: (n_connections, STEPS_PER_DAY) in kW.
        """
        crit = np.zeros((self.n, STEPS_PER_DAY))
        flex = np.zeros((self.n, STEPS_PER_DAY))
        for i, mine in enumerate(self.owned):
            for name, kw, tier, hrs, s, e in mine:
                f = _seasonal_factor(doy, name)
                energy = kw * hrs * f                      # kWh for the day
                shape = _window_weights(s, e)
                if shape.sum() == 0:
                    continue
                series = energy * shape / DT_HOURS         # kWh share -> kW
                series = series * self.rng.normal(1.0, 0.07, STEPS_PER_DAY).clip(0.6, 1.4)
                if tier == "critical":
                    crit[i] += series
                else:
                    flex[i] += series
        return {"critical": crit, "flexible": flex}

    # ------------------------------------------------------------------
    def pv_output(self, doy: int, cloud: float) -> np.ndarray:
        """Aggregate feeder PV output in kW for one day."""
        t = np.arange(STEPS_PER_DAY) / STEPS_PER_HOUR
        decl = 23.45 * np.sin(np.deg2rad(360.0 * (284 + doy) / 365.0))
        lat = np.deg2rad(self.cfg.latitude_deg)
        d = np.deg2rad(decl)
        ha = np.deg2rad(15.0 * (t - 12.0))
        cos_z = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(ha)
        clear = np.clip(cos_z, 0, None) ** 1.18
        if clear.max() > 0:
            clear = clear / clear.max()
        # intra-day cloud variability on top of the daily cloud index
        noise = 1.0 - cloud * (0.55 + 0.45 * self.rng.random(STEPS_PER_DAY))
        out = self.total_pv_kwp * 0.86 * clear * np.clip(noise, 0.05, 1.0)
        return np.clip(out, 0, None)


def cloud_index(doy: int, rng: np.random.Generator) -> float:
    """Daily cloud index in [0, 1]; monsoon months are much cloudier."""
    monsoon = np.exp(-((doy - 205) ** 2) / (2 * 42.0 ** 2))
    base = 0.08 + 0.62 * monsoon
    return float(np.clip(rng.beta(2.0, 2.0) * 0.9 * (0.35 + base * 1.6), 0.0, 0.95))
