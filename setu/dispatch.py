"""Rolling-horizon dispatch optimiser.

The problem is built once with CVXPY Parameters and re-solved every 15-minute
step. That matters: a 365-day run is ~35,000 solves, and rebuilding the
problem each time would make the simulation unusable.

Decision at each step: how much of the forecast deficit to cover from the
community battery, how much flexible load to curtail, and how much to import
- minimising expected unserved CRITICAL energy above all else.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .config import DT_HOURS, BatteryConfig, DispatchConfig


class RollingHorizonDispatcher:
    def __init__(self, bat: BatteryConfig, dsp: DispatchConfig):
        self.bat = bat
        self.dsp = dsp
        K = dsp.horizon_steps
        self.K = K

        # ---- parameters updated every step -------------------------------
        self.p_crit = cp.Parameter(K, nonneg=True)
        self.p_flex = cp.Parameter(K, nonneg=True)
        self.p_pv = cp.Parameter(K, nonneg=True)
        self.p_limit = cp.Parameter(K, nonneg=True)
        self.p_price = cp.Parameter(K, nonneg=True)
        self.p_soc0 = cp.Parameter(nonneg=True)
        self.p_soc_target = cp.Parameter(nonneg=True)

        # ---- decision variables ------------------------------------------
        dis = cp.Variable(K, nonneg=True)     # battery discharge, kW
        ch = cp.Variable(K, nonneg=True)      # battery charge, kW
        curt = cp.Variable(K, nonneg=True)    # flexible load curtailed, kW
        uc = cp.Variable(K, nonneg=True)      # unserved CRITICAL load, kW
        imp = cp.Variable(K, nonneg=True)     # import from the DISCOM, kW
        spill = cp.Variable(K, nonneg=True)   # surplus PV not absorbed, kW
        soc = cp.Variable(K + 1, nonneg=True) # state of charge, kWh

        cons = [
            dis <= bat.power_kw,
            ch <= bat.power_kw,
            curt <= self.p_flex,
            uc <= self.p_crit,
            imp <= self.p_limit,
            soc[0] == self.p_soc0,
            soc >= bat.capacity_kwh * bat.soc_min,
            soc <= bat.capacity_kwh * bat.soc_max,
        ]

        # power balance at the distribution transformer bus
        served = self.p_crit - uc + self.p_flex - curt
        cons.append(served + ch + spill == imp + self.p_pv + dis)

        # state of charge dynamics
        for k in range(K):
            cons.append(
                soc[k + 1] == soc[k]
                + (ch[k] * bat.eta_charge - dis[k] / bat.eta_discharge) * DT_HOURS
            )

        obj = (
            dsp.w_unserved_critical * cp.sum(uc) * DT_HOURS
            + dsp.w_curtail_flexible * cp.sum(curt) * DT_HOURS
            + dsp.w_import_energy * cp.sum(cp.multiply(imp, self.p_price)) * DT_HOURS
            + dsp.w_cycling * cp.sum(dis + ch) * DT_HOURS
            + dsp.w_soc_target * cp.pos(self.p_soc_target - soc[K])
            - dsp.w_terminal_soc * soc[K]
        )

        self.vars = dict(dis=dis, ch=ch, curt=curt, uc=uc, imp=imp,
                         spill=spill, soc=soc)
        self.problem = cp.Problem(cp.Minimize(obj), cons)

    # ------------------------------------------------------------------
    def step(self, crit, flex, pv, limit, price, soc0,
             normal_limit: float | None = None,
             soc_target: float = 0.0,
             force_mpc: bool = False) -> dict:
        """Return the first-step actions for this horizon.

        The optimiser is only invoked when a deficit is actually visible
        within the horizon. The rest of the time the substation controller
        runs its deterministic rule, which is what it would do in the field:
        an edge device should not solve an LP every fifteen minutes to
        decide to do nothing. This also means the solver is never on the
        critical path during normal operation.
        """
        if normal_limit is not None and not force_mpc:
            deficit_ahead = (
                np.any(limit < normal_limit - 1e-6)
                or np.any(crit + flex - pv > limit + 1e-6)
            )
            if not deficit_ahead:
                out = self._fallback(crit[0], flex[0], pv[0], limit[0], soc0)
                out["mode"] = "rule"
                return out

        self.p_crit.value = np.maximum(crit, 0.0)
        self.p_flex.value = np.maximum(flex, 0.0)
        self.p_pv.value = np.maximum(pv, 0.0)
        self.p_limit.value = np.maximum(limit, 0.0)
        self.p_price.value = np.maximum(price, 0.0)
        self.p_soc0.value = float(soc0)
        self.p_soc_target.value = float(max(soc_target, 0.0))

        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            self.problem.solve(solver=cp.SCS, warm_start=True)

        if self.vars["dis"].value is None:
            out = self._fallback(crit[0], flex[0], pv[0], limit[0], soc0)
            out["mode"] = "fallback"
            return out

        return {
            "discharge_kw": float(self.vars["dis"].value[0]),
            "charge_kw": float(self.vars["ch"].value[0]),
            "curtail_kw": float(self.vars["curt"].value[0]),
            "unserved_critical_kw": float(self.vars["uc"].value[0]),
            "import_kw": float(self.vars["imp"].value[0]),
            "fallback": False,
            "mode": "mpc",
        }

    # ------------------------------------------------------------------
    def _fallback(self, crit, flex, pv, limit, soc0) -> dict:
        """Deterministic safety envelope.

        This is what the substation controller runs when the optimiser, the
        forecast or the network link is unavailable. It is intentionally
        simple: protect critical load, shed flexible load, never violate the
        battery limits. The island stays up, just less efficiently.
        """
        bat = self.bat
        avail_kwh = max(0.0, soc0 - bat.capacity_kwh * bat.soc_min)
        max_dis = min(bat.power_kw, avail_kwh / DT_HOURS * bat.eta_discharge)

        supply = limit + pv
        curt = 0.0
        if crit + flex > supply:
            curt = min(flex, crit + flex - supply)
        deficit = max(0.0, crit + flex - curt - supply)
        dis = min(max_dis, deficit)
        uc = max(0.0, deficit - dis)

        surplus = max(0.0, supply - (crit - uc + flex - curt))
        ch = min(bat.power_kw, surplus) if dis == 0 else 0.0
        return {
            "discharge_kw": dis, "charge_kw": ch, "curtail_kw": curt,
            "unserved_critical_kw": uc,
            "import_kw": min(limit, max(0.0, crit - uc + flex - curt + ch - pv - dis)),
            "fallback": True,
            "mode": "rule",
        }
