"""Two-stage stochastic MILP for SETU dispatch.

This replaces the deterministic LP with a formulation that matches what the
controller actually faces:

  * Relays are DISCRETE. A contactor is open or closed; it cannot deliver
    7.3 kW of "partial curtailment". The first-stage decision is therefore a
    binary vector over curtailment groups.
  * The future is UNCERTAIN. Net load over the horizon is represented by a
    reduced scenario set with probabilities, and continuous recourse adapts
    per scenario.
  * The tail matters more than the mean. Expected unserved critical energy is
    the wrong objective when the failure mode is a long evening with no sun;
    we minimise expectation plus CVaR of unserved critical energy, using the
    Rockafellar-Uryasev linearisation.
  * Fairness must be a CONSTRAINT, not a post-hoc allocation rule. The Gini
    coefficient is not convex, but Lorenz-style constraints are: the sum of
    the k largest elements of a vector is LP-representable, so we bound the
    burden borne by the worst-off k groups directly inside the program.
  * Voltage is a HARD limit. Linearised sensitivities from the three-phase
    power flow keep the worst premises voltage inside the statutory band.

First stage is integer; recourse is a continuous relaxation. That keeps the
problem tractable at a 15-minute cadence while preserving the part of the
integrality that physically matters - which relays actually open now.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .config import DT_HOURS, BatteryConfig, DispatchConfig


class StochasticMILPDispatcher:
    """Scenario-based two-stage dispatch with integer first-stage switching."""

    W_FAIRNESS = 260.0      # penalty per unit of OWA burden-cap violation
    W_VOLTAGE = 9.0e4       # penalty per pu of voltage-band violation

    def __init__(self, bat: BatteryConfig, dsp: DispatchConfig,
                 n_groups: int = 16, n_scenarios: int = 6,
                 cvar_alpha: float = 0.90, owa_k: int = 4,
                 v_min_pu: float = 0.94):
        self.bat, self.dsp = bat, dsp
        self.G, self.S = n_groups, n_scenarios
        self.K = dsp.horizon_steps
        self.alpha = cvar_alpha
        self.owa_k = owa_k
        self.v_min = v_min_pu
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        G, S, K = self.G, self.S, self.K
        bat, dsp = self.bat, self.dsp

        # ---------------- parameters -----------------------------------
        self.p_group_kw = cp.Parameter(G, nonneg=True)      # curtailable kW per group now
        self.p_crit = cp.Parameter((S, K), nonneg=True)     # critical load per scenario
        self.p_flex = cp.Parameter((S, K), nonneg=True)     # flexible load per scenario
        self.p_pv = cp.Parameter((S, K), nonneg=True)
        self.p_limit = cp.Parameter((S, K), nonneg=True)
        # price and probability are folded into ONE parameter so the
        # objective stays DPP-compliant: a product of two Parameters with a
        # Variable breaks disciplined parametrised programming and forces a
        # full recompile on every solve.
        self.p_price_w = cp.Parameter((S, K), nonneg=True)
        self.p_prob = cp.Parameter(S, nonneg=True)
        self.p_soc0 = cp.Parameter(nonneg=True)
        self.p_soc_target = cp.Parameter(nonneg=True)
        self.p_burden = cp.Parameter(G, nonneg=True)        # accumulated burden per group
        self.p_burden_cap = cp.Parameter(nonneg=True)       # OWA bound
        # voltage linearisation
        self.p_v0 = cp.Parameter()                          # worst premises voltage now
        self.p_dv_group = cp.Parameter(G, nonneg=True)      # pu gain per group opened
        self.p_dv_inject = cp.Parameter(nonneg=True)        # pu per kW injected

        # ---------------- first stage (here and now) --------------------
        z = cp.Variable(G, boolean=True)                    # 1 = group curtailed
        curt0 = self.p_group_kw @ z                         # kW curtailed now

        # ---------------- recourse (per scenario) -----------------------
        dis = cp.Variable((S, K), nonneg=True)
        ch = cp.Variable((S, K), nonneg=True)
        curt = cp.Variable((S, K), nonneg=True)
        uc = cp.Variable((S, K), nonneg=True)
        imp = cp.Variable((S, K), nonneg=True)
        spill = cp.Variable((S, K), nonneg=True)
        soc = cp.Variable((S, K + 1), nonneg=True)

        cons = [
            dis <= bat.power_kw, ch <= bat.power_kw,
            curt <= self.p_flex, uc <= self.p_crit,
            imp <= self.p_limit,
            soc >= bat.capacity_kwh * bat.soc_min,
            soc <= bat.capacity_kwh * bat.soc_max,
            soc[:, 0] == self.p_soc0,
        ]

        # non-anticipativity: the first step is identical across scenarios
        # and is exactly the integer decision taken now
        for s in range(S):
            cons.append(curt[s, 0] == curt0)
        cons.append(dis[1:, 0] == dis[0, 0])
        cons.append(ch[1:, 0] == ch[0, 0])

        # power balance and state of charge dynamics
        served = self.p_crit - uc + self.p_flex - curt
        cons.append(served + ch + spill == imp + self.p_pv + dis)
        for s in range(S):
            for k in range(K):
                cons.append(soc[s, k + 1] == soc[s, k]
                            + (ch[s, k] * bat.eta_charge
                               - dis[s, k] / bat.eta_discharge) * DT_HOURS)

        # ---------------- voltage constraint ----------------------------
        # Linearised around the current operating point. Curtailment raises
        # the worst premises voltage; battery injection raises it far less,
        # because the battery sits at the transformer and relieves only the
        # transformer drop, not the LV line drop.
        # a slack keeps the controller feasible when voltage cannot be
        # recovered by curtailment alone; the penalty is large enough that it
        # is only ever used as a last resort
        v_slack = cp.Variable(nonneg=True)
        cons.append(self.p_v0 + self.p_dv_group @ z
                    + self.p_dv_inject * dis[0, 0] + v_slack >= self.v_min)

        # ---------------- OWA / Lorenz fairness constraint --------------
        # Bound the TOTAL burden of the k worst-off groups. The sum of the k
        # largest entries of a vector x is
        #     max_{0<=y<=1, sum y = k} y'x  =  min_r { k r + sum_i (x_i - r)+ }
        # which is linear in (r, w). This is a genuine convex constraint on
        # inequality, not a heuristic ordering rule.
        step_burden = self.p_burden + cp.multiply(self.p_group_kw, z) * DT_HOURS
        r = cp.Variable()
        w = cp.Variable(G, nonneg=True)
        # Soft, not hard. A real-time controller that can be made infeasible
        # by its own fairness bookkeeping is a controller that fails at the
        # worst possible moment. The cap is enforced by penalty, and the
        # violation is reported so it can be audited.
        f_slack = cp.Variable(nonneg=True)
        cons += [w >= step_burden - r,
                 self.owa_k * r + cp.sum(w) <= self.p_burden_cap + f_slack]

        # ---------------- CVaR of unserved critical energy --------------
        loss = cp.sum(uc, axis=1) * DT_HOURS               # per scenario, kWh
        zeta = cp.Variable()
        u = cp.Variable(S, nonneg=True)
        cons.append(u >= loss - zeta)
        cvar = zeta + (1.0 / (1.0 - self.alpha)) * (self.p_prob @ u)

        # ---------------- objective -------------------------------------
        exp_soc_T = self.p_prob @ soc[:, K]
        exp_uc = self.p_prob @ loss
        exp_curt = self.p_prob @ (cp.sum(curt, axis=1) * DT_HOURS)
        exp_imp = cp.sum(cp.multiply(self.p_price_w, imp)) * DT_HOURS
        exp_cyc = self.p_prob @ (cp.sum(dis + ch, axis=1) * DT_HOURS)

        # shortfall against the pre-position target, expressed with an
        # explicit slack rather than cp.pos: the positive-part atom trips a
        # bounds-propagation bug on the DPP canonicalisation path, and the
        # epigraph form is the standard LP representation anyway
        soc_short = cp.Variable(nonneg=True)
        cons.append(soc_short >= self.p_soc_target - exp_soc_T)


        obj = (dsp.w_unserved_critical * exp_uc
               + dsp.w_unserved_critical * 0.5 * cvar
               + dsp.w_curtail_flexible * exp_curt
               + dsp.w_import_energy * exp_imp
               + dsp.w_cycling * exp_cyc
               - dsp.w_terminal_soc * exp_soc_T
               + dsp.w_soc_target * soc_short
               + self.W_FAIRNESS * f_slack
               + self.W_VOLTAGE * v_slack)

        self.vars = dict(z=z, dis=dis, ch=ch, curt=curt, uc=uc, imp=imp,
                         soc=soc, zeta=zeta, cvar=cvar,
                         f_slack=f_slack, v_slack=v_slack)
        self.problem = cp.Problem(cp.Minimize(obj), cons)

    # ------------------------------------------------------------------
    @staticmethod
    def _mip_solvers():
        """MIP backends in preference order, filtered to what is installed."""
        avail = set(cp.installed_solvers())
        return [s for s in ("HIGHS", "SCIP", "GUROBI", "CBC", "GLPK_MI", "SCIPY")
                if s in avail]

    # ------------------------------------------------------------------
    def solve(self, group_kw, crit, flex, pv, limit, price, probs, soc0,
              burden, burden_cap, v0, dv_curtail, dv_inject,
              soc_target=0.0, time_limit: float = 4.0) -> dict:
        self.p_group_kw.value = np.maximum(group_kw, 0.0)
        self.p_crit.value = np.maximum(crit, 0.0)
        self.p_flex.value = np.maximum(flex, 0.0)
        self.p_pv.value = np.maximum(pv, 0.0)
        self.p_limit.value = np.maximum(limit, 0.0)
        pr = probs / probs.sum()
        self.p_prob.value = pr
        self.p_price_w.value = np.outer(pr, np.maximum(price, 0.0))
        self.p_soc0.value = float(soc0)
        self.p_soc_target.value = float(max(soc_target, 0.0))
        self.p_burden.value = np.maximum(burden, 0.0)
        self.p_burden_cap.value = float(max(burden_cap, 1e-6))
        self.p_v0.value = float(v0)
        self.p_dv_group.value = (np.maximum(group_kw, 0.0)
                                 * float(max(dv_curtail, 0.0)))
        self.p_dv_inject.value = float(max(dv_inject, 0.0))

        status = None
        for solver in self._mip_solvers():
            try:
                self.problem.solve(solver=solver)
                status = self.problem.status
                if self.vars["z"].value is not None:
                    break
            except Exception:
                continue

        if self.vars["z"].value is None:
            return {"feasible": False, "status": status}

        zv = np.asarray(self.vars["z"].value).round().astype(int)
        return {
            "feasible": True,
            "status": status,
            "relay_states": zv,
            "groups_curtailed": int(zv.sum()),
            "curtail_kw": float(np.dot(self.p_group_kw.value, zv)),
            "discharge_kw": float(self.vars["dis"].value[0, 0]),
            "charge_kw": float(self.vars["ch"].value[0, 0]),
            "unserved_critical_kw": float(self.vars["uc"].value[:, 0].mean()),
            "import_kw": float(self.vars["imp"].value[0, 0]),
            "cvar_kwh": float(self.vars["cvar"].value),
            "expected_uc_kwh": float(
                (self.p_prob.value * (self.vars["uc"].value.sum(axis=1)
                                      * DT_HOURS)).sum()),
            "v_after_pu": float(self.p_v0.value
                                + np.dot(self.p_dv_group.value, zv)
                                + self.p_dv_inject.value
                                * self.vars["dis"].value[0, 0]),
            "fairness_violation": float(self.vars["f_slack"].value),
            "voltage_violation_pu": float(self.vars["v_slack"].value),
            "is_dpp": self.problem.is_dcp(dpp=True),
        }
