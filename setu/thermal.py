"""Equivalent-thermal-parameter (ETP) models for thermostatically controlled loads.

Why this matters: in the first version of this prototype, flexible load was a
number of kilowatts that could be subtracted at will. That is wrong in a way
that flatters the result. Air coolers, refrigerators and water heaters are
thermal integrators with deadband control. Curtailing them does not remove
energy, it defers it into a temperature excursion, and when supply returns
every unit that drifted outside its deadband switches on at once.

Modelling the state explicitly gives us three things the lumped model could
never produce:

  1. a real comfort cost for curtailment (degree-hours outside the deadband),
  2. cold load pickup as an EMERGENT phenomenon rather than an assumption,
  3. an upper bound on how long each appliance can honestly be held off.

Each appliance is a first-order lumped-capacitance system

    C dT/dt = (T_amb - T)/R + Q

integrated exactly over a step of length dt:

    T(k+1) = T_inf + (T(k) - T_inf) * exp(-dt / (R C)),   T_inf = T_amb + Q R

which is unconditionally stable for any dt, unlike a forward-Euler update.
"""
from __future__ import annotations

import numpy as np

from .config import DT_HOURS

# kind: (R [degC/kW], C [kWh/degC], P_rated [kW], COP, setpoint, deadband, cooling)
ETP_CLASSES = {
    "air_cooler":   dict(R=(3.2, 6.5), C=(0.9, 2.4), P=(1.20, 1.60),
                         cop=2.6, setpoint=(26.0, 29.0), deadband=1.6, cooling=True),
    "refrigerator": dict(R=(42.0, 78.0), C=(0.10, 0.22), P=(0.13, 0.19),
                         cop=1.9, setpoint=(3.0, 6.0), deadband=2.2, cooling=True),
    "water_heater": dict(R=(95.0, 160.0), C=(0.14, 0.26), P=(1.80, 2.20),
                         cop=1.0, setpoint=(52.0, 60.0), deadband=5.0, cooling=False),
}

# How far outside the deadband each class may be pushed before SETU must
# restore it, regardless of what the optimiser would prefer. These are the
# hard comfort/safety limits, in degrees C beyond the deadband edge.
EXCURSION_LIMIT = {"air_cooler": 3.5, "refrigerator": 2.5, "water_heater": 9.0}


class ThermostaticFleet:
    """A vectorised population of ETP appliances of one class.

    State is carried across the whole simulation, so a unit curtailed at
    19:00 is genuinely warmer at 19:15 and draws for longer once released.
    """

    def __init__(self, kind: str, owner_idx: np.ndarray, rng: np.random.Generator):
        spec = ETP_CLASSES[kind]
        self.kind = kind
        self.owner = owner_idx                    # connection index per unit
        n = owner_idx.size
        self.n = n

        self.R = rng.uniform(*spec["R"], size=n)
        self.C = rng.uniform(*spec["C"], size=n)
        self.P = rng.uniform(*spec["P"], size=n)
        self.cop = spec["cop"]
        self.setpoint = rng.uniform(*spec["setpoint"], size=n)
        self.deadband = np.full(n, spec["deadband"])
        self.cooling = spec["cooling"]
        self.limit = EXCURSION_LIMIT[kind]

        # thermal power delivered to the mass when the unit is energised
        self.Q_on = (-self.P * self.cop) if self.cooling else (self.P * self.cop)

        self.T = self.setpoint.copy()
        self.on = rng.random(n) < 0.3
        self.tau = self.R * self.C                      # time constant, hours
        self.excursion_degh = np.zeros(n)               # comfort accounting
        self.forced_off_steps = np.zeros(n, dtype=int)

    # ------------------------------------------------------------------
    @property
    def upper(self) -> np.ndarray:
        return self.setpoint + self.deadband / 2.0

    @property
    def lower(self) -> np.ndarray:
        return self.setpoint - self.deadband / 2.0

    def natural_demand_kw(self, T_amb: np.ndarray) -> np.ndarray:
        """Power the fleet would draw this step with no SETU intervention."""
        want = self._thermostat(T_amb)
        return self.P * want

    def _thermostat(self, T_amb: np.ndarray) -> np.ndarray:
        """Deadband hysteresis: returns the desired on/off state."""
        want = self.on.copy()
        if self.cooling:
            want[self.T > self.upper] = True
            want[self.T < self.lower] = False
        else:
            want[self.T < self.lower] = True
            want[self.T > self.upper] = False
        return want

    def available_curtailment_kw(self, T_amb: np.ndarray) -> np.ndarray:
        """Power that may be deferred WITHOUT breaching the excursion limit.

        A unit already beyond its limit is not curtailable at any price; this
        is what stops the optimiser from treating thermal loads as a free
        battery and is the main reason the lumped model over-promised.
        """
        want = self._thermostat(T_amb)
        if self.cooling:
            headroom = (self.upper + self.limit) - self.T
        else:
            headroom = self.T - (self.lower - self.limit)
        ok = want & (headroom > 0.0)
        return self.P * ok

    # ------------------------------------------------------------------
    def step(self, T_amb: np.ndarray, curtail_mask: np.ndarray) -> np.ndarray:
        """Advance one step. Returns actual power drawn per unit, kW."""
        want = self._thermostat(T_amb)
        actual = want & ~curtail_mask

        # units held off against their own thermostat
        self.forced_off_steps += (want & curtail_mask).astype(int)

        Q = np.where(actual, self.Q_on, 0.0)
        T_inf = T_amb + Q * self.R
        decay = np.exp(-DT_HOURS / self.tau)
        self.T = T_inf + (self.T - T_inf) * decay
        self.on = actual

        # comfort accounting in degree-hours outside the deadband
        if self.cooling:
            over = np.clip(self.T - self.upper, 0, None)
        else:
            over = np.clip(self.lower - self.T, 0, None)
        self.excursion_degh += over * DT_HOURS

        return self.P * actual

    # ------------------------------------------------------------------
    def diversity_factor(self) -> float:
        """Fraction of the fleet currently energised.

        A settled fleet sits near its duty cycle. After a long interruption
        it approaches 1.0 - that collapse of diversity IS cold load pickup.
        """
        return float(self.on.mean())

    def comfort_summary(self) -> dict:
        return {
            "mean_excursion_degh": float(self.excursion_degh.mean()),
            "p95_excursion_degh": float(np.percentile(self.excursion_degh, 95)),
            "share_ever_beyond_limit": float(
                (self.excursion_degh > self.limit * DT_HOURS).mean()),
        }


def ambient_temperature(doy: int, steps_per_day: int = 96) -> np.ndarray:
    """Dry-bulb ambient profile for western India, degC.

    Seasonal mean with a diurnal swing lagging solar noon by about 2.5 hours.
    """
    t = np.arange(steps_per_day) / (steps_per_day / 24.0)
    seasonal = 26.5 + 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.0)
    monsoon = -4.0 * np.exp(-((doy - 205) ** 2) / (2 * 38.0 ** 2))
    swing = 6.5 + 2.0 * np.sin(2 * np.pi * (doy - 110) / 365.0)
    return seasonal + monsoon + swing * np.sin(2 * np.pi * (t - 9.0) / 24.0)
