"""Year-long simulation of one feeder, with and without SETU.

Baseline represents what a DISCOM can actually do today: control is binary,
so when demand exceeds the import limit the feeder breaker opens and every
connection on it loses supply - critical load included.

SETU keeps the feeder inside its import limit by curtailing flexible load and
discharging the community battery, so the critical tier stays energised.
"""
from __future__ import annotations

import numpy as np

from .config import (STEPS_PER_DAY, STEPS_PER_HOUR, DT_HOURS, SimConfig)
from .profiles import FeederPopulation, cloud_index
from .forecast import (QuantileNetLoadForecaster, generate_scarcity,
                       horizon_quantiles)
from .dispatch import RollingHorizonDispatcher
from .fairness import FairnessLedger, allocate_curtailment, gini


def _price_vector(cfg) -> np.ndarray:
    """Time-of-day energy price, used by the optimiser to value imports."""
    t = np.arange(STEPS_PER_DAY) / STEPS_PER_HOUR
    peak = ((t >= 7) & (t < 11)) | ((t >= 18) & (t < 23))
    p = np.full(STEPS_PER_DAY, cfg.economics.tariff_offpeak_inr_per_kwh)
    p[peak] = cfg.economics.tariff_peak_inr_per_kwh
    return p


def run(cfg: SimConfig, enforce_fairness: bool | None = None,
        progress: bool = True) -> dict:
    """Run the simulation and return per-step traces and summary metrics."""
    rng = np.random.default_rng(cfg.seed)
    pop = FeederPopulation(cfg.feeder, rng)
    bat, dsp = cfg.battery, cfg.dispatch
    if enforce_fairness is None:
        enforce_fairness = dsp.enforce_fairness

    dispatcher = RollingHorizonDispatcher(bat, dsp)
    ledger = FairnessLedger(pop.n, dsp.fairness_window_days, STEPS_PER_DAY)
    forecaster = QuantileNetLoadForecaster()
    price = _price_vector(cfg)

    N = cfg.days * STEPS_PER_DAY
    tr = {k: np.zeros(N) for k in (
        "critical_kw", "flexible_kw", "pv_kw", "import_limit_kw",
        "discharge_kw", "charge_kw", "curtail_kw", "unserved_critical_kw",
        "import_kw", "soc_kwh", "base_unserved_critical_kw",
        "base_unserved_flexible_kw", "gini")}
    tr["mpc_used"] = np.zeros(N, dtype=bool)
    tr["fallback"] = np.zeros(N, dtype=bool)
    daily_gini = []

    soc = bat.capacity_kwh * bat.soc_init
    history = np.full(STEPS_PER_DAY, 60.0)
    warmup_net, warmup_pv, warmup_doy = [], [], []
    K = dsp.horizon_steps

    for d in range(cfg.days):
        doy = ((cfg.start_month - 1) * 30 + d) % 365 + 1
        prof = pop.day_profiles(doy)
        crit_i, flex_i = prof["critical"], prof["flexible"]
        crit = crit_i.sum(axis=0)
        flex = flex_i.sum(axis=0)
        pv = pop.pv_output(doy, cloud_index(doy, rng))
        limit = generate_scarcity(doy, cfg.scarcity, rng)
        net = crit + flex - pv

        # --- warm the forecaster on the first two weeks of history --------
        if d < 14:
            warmup_net.append(net); warmup_pv.append(pv)
            warmup_doy.append(np.full(STEPS_PER_DAY, doy))
        elif d == 14:
            forecaster.fit(np.concatenate(warmup_net),
                           np.concatenate(warmup_pv),
                           np.concatenate(warmup_doy))

        # extend the day so the horizon never runs off the end
        crit_x = np.concatenate([crit, crit[:K]])
        flex_x = np.concatenate([flex, flex[:K]])
        pv_x = np.concatenate([pv, pv[:K]])
        lim_x = np.concatenate([limit, limit[:K]])
        pr_x = np.concatenate([price, price[:K]])

        for s in range(STEPS_PER_DAY):
            g = d * STEPS_PER_DAY + s

            # ---------- BASELINE: binary feeder control -------------------
            demand = crit[s] + flex[s]
            net_demand = max(0.0, demand - pv[s])
            if net_demand > limit[s] + 1e-6:
                tr["base_unserved_critical_kw"][g] = crit[s]
                tr["base_unserved_flexible_kw"][g] = flex[s]

            # ---------- SETU ---------------------------------------------
            fc = horizon_quantiles(forecaster, history, pv_x[s:s + K], doy, K)
            # plan against the pessimistic (P90) net-load quantile
            if forecaster.fitted:
                scale = np.clip(fc[0.90] / np.maximum(fc[0.50], 1e-3), 0.95, 1.30)
            else:
                scale = np.ones(K)
            hour = s / STEPS_PER_HOUR
            prepos = dsp.preposition_start_h <= hour < dsp.preposition_end_h
            soc_target = (bat.capacity_kwh * dsp.preposition_soc_target
                          if prepos else 0.0)
            act = dispatcher.step(
                crit=crit_x[s:s + K] * scale,
                flex=flex_x[s:s + K],
                pv=pv_x[s:s + K],
                limit=lim_x[s:s + K],
                price=pr_x[s:s + K],
                soc0=soc,
                normal_limit=cfg.scarcity.normal_import_limit_kw,
                soc_target=soc_target,
                force_mpc=prepos,
            )

            curt_kw = min(act["curtail_kw"], flex[s])
            # allocate against burden share, so the allocator chases the
            # same quantity the fairness metric reports
            burden = ledger.cumulative_kwh / np.maximum(ledger.available_kwh, 1.0)
            alloc = allocate_curtailment(
                curt_kw,
                flex_i[:, s],
                burden,
                enforce_fairness=enforce_fairness,
            )
            ledger.record(alloc * DT_HOURS, flex_i[:, s] * DT_HOURS)

            soc = float(np.clip(
                soc + (act["charge_kw"] * bat.eta_charge
                       - act["discharge_kw"] / bat.eta_discharge) * DT_HOURS,
                bat.capacity_kwh * bat.soc_min, bat.capacity_kwh * bat.soc_max))

            tr["critical_kw"][g] = crit[s]
            tr["flexible_kw"][g] = flex[s]
            tr["pv_kw"][g] = pv[s]
            tr["import_limit_kw"][g] = limit[s]
            tr["discharge_kw"][g] = act["discharge_kw"]
            tr["charge_kw"][g] = act["charge_kw"]
            tr["curtail_kw"][g] = curt_kw
            tr["unserved_critical_kw"][g] = act["unserved_critical_kw"]
            tr["import_kw"][g] = act["import_kw"]
            tr["soc_kwh"][g] = soc
            tr["gini"][g] = ledger.current_gini
            tr["mpc_used"][g] = (act["mode"] == "mpc")
            tr["fallback"][g] = (act["mode"] == "fallback")

            history = np.roll(history, -1)
            history[-1] = net[s]

        daily_gini.append(ledger.current_gini)
        if progress and (d + 1) % 30 == 0:
            print(f"  simulated day {d + 1}/{cfg.days}", flush=True)

    return {"traces": tr, "population": pop, "ledger": ledger,
            "daily_gini": np.array(daily_gini), "config": cfg}


# ----------------------------------------------------------------------
def summarise(res: dict) -> dict:
    """Headline metrics, computed from the simulation traces."""
    tr, cfg = res["traces"], res["config"]
    days = cfg.days
    ec = cfg.economics

    # A baseline outage is total: the breaker opens and 100% of critical
    # load is lost. Under SETU the optimiser rations instead, so a step with
    # a small shortfall is not the same event. We therefore report both a
    # strict full-outage measure and total unserved critical energy.
    FULL_OUTAGE_FRACTION = 0.50
    crit_kw = np.maximum(tr["critical_kw"], 1e-9)
    base_out_steps = (tr["base_unserved_critical_kw"] > 1e-6).sum()
    setu_any_steps = (tr["unserved_critical_kw"] > 1e-6).sum()
    setu_out_steps = ((tr["unserved_critical_kw"] / crit_kw)
                      > FULL_OUTAGE_FRACTION).sum()

    base_out_h_day = base_out_steps * DT_HOURS / days
    setu_out_h_day = setu_out_steps * DT_HOURS / days
    setu_partial_h_day = setu_any_steps * DT_HOURS / days

    crit_energy = tr["critical_kw"].sum() * DT_HOURS
    base_uce = tr["base_unserved_critical_kw"].sum() * DT_HOURS
    setu_uce = tr["unserved_critical_kw"].sum() * DT_HOURS
    base_unserved_flex = tr["base_unserved_flexible_kw"].sum() * DT_HOURS
    deferred_flex = tr["curtail_kw"].sum() * DT_HOURS

    scale = 365.0 / days

    scarce = tr["import_limit_kw"] < cfg.scarcity.normal_import_limit_kw - 1e-6
    net_demand = tr["critical_kw"] + tr["flexible_kw"] - tr["pv_kw"]
    peak_base = float(net_demand[scarce].max()) if scarce.any() else float(net_demand.max())
    peak_setu = float(tr["import_kw"][scarce].max()) if scarce.any() else float(tr["import_kw"].max())

    throughput = tr["discharge_kw"].sum() * DT_HOURS
    cycles = throughput / cfg.battery.capacity_kwh

    capex = (ec.battery_cost_inr + ec.pcs_cost_inr + ec.dt_sensing_controller_inr
             + ec.relay_cost_inr * (cfg.feeder.n_households + cfg.feeder.n_enterprises)
             + ec.install_civil_inr)

    voll_avoided = (
        (base_uce - setu_uce) * ec.voll_household_inr_per_kwh
    ) * scale  # flexible energy is deferred, not lost, so it is not valued at VoLL

    return {
        "days_simulated": days,
        "connections": cfg.feeder.n_households + cfg.feeder.n_enterprises,
        "total_pv_kwp": round(res["population"].total_pv_kwp, 1),

        "baseline_outage_h_per_day": round(base_out_h_day, 3),
        "setu_full_outage_h_per_day": round(setu_out_h_day, 3),
        "setu_any_shortfall_h_per_day": round(setu_partial_h_day, 3),
        "mean_shortfall_when_rationing_pct": round(float(
            100 * (tr["unserved_critical_kw"][tr["unserved_critical_kw"] > 1e-6]
                   / crit_kw[tr["unserved_critical_kw"] > 1e-6]).mean())
            if setu_any_steps else 0.0, 1),
        "full_outage_reduction_pct": round(100 * (1 - setu_out_h_day / max(base_out_h_day, 1e-9)), 1),
        "critical_energy_availability_pct": round(
            100 * (1 - setu_uce / max(crit_energy, 1e-9)), 2),
        "baseline_critical_energy_availability_pct": round(
            100 * (1 - base_uce / max(crit_energy, 1e-9)), 2),
        "unserved_critical_energy_reduction_pct": round(
            100 * (1 - setu_uce / max(base_uce, 1e-9)), 1),

        "baseline_time_availability_pct": round(100 * (1 - base_out_steps / (days * 96)), 2),
        "setu_time_availability_pct": round(100 * (1 - setu_out_steps / (days * 96)), 2),

        "baseline_unserved_critical_mwh_yr": round(base_uce * scale / 1000, 2),
        "setu_unserved_critical_mwh_yr": round(setu_uce * scale / 1000, 2),
        "critical_energy_served_mwh_yr": round(crit_energy * scale / 1000, 2),

        "flexible_energy_deferred_mwh_yr": round(deferred_flex * scale / 1000, 2),
        "deferred_share_of_flexible_pct": round(
            100 * deferred_flex / max(tr["flexible_kw"].sum() * DT_HOURS, 1e-9), 2),

        "scarcity_hours_per_day": round(float(scarce.sum()) * DT_HOURS / days, 2),
        "peak_net_demand_in_scarcity_kw": round(peak_base, 1),
        "peak_import_in_scarcity_kw": round(peak_setu, 1),
        "peak_shaved_kw": round(peak_base - peak_setu, 1),

        "battery_equivalent_full_cycles_yr": round(cycles * scale, 1),
        "optimiser_invoked_share_pct": round(100 * tr["mpc_used"].mean(), 2),
        "solver_failure_share_pct": round(100 * tr["fallback"].mean(), 3),

        "gini_mean": round(float(res["daily_gini"].mean()), 3),
        "gini_p95": round(float(np.percentile(res["daily_gini"], 95)), 3),
        "gini_max": round(float(res["daily_gini"].max()), 3),
        "share_never_curtailed_pct": round(
            100 * float((res["ledger"].cumulative_kwh <= 1e-6).mean()), 1),

        "capex_inr": int(capex),
        "capex_inr_per_connection": int(capex / (cfg.feeder.n_households
                                                 + cfg.feeder.n_enterprises)),
        "voll_avoided_inr_yr": int(voll_avoided),
    }
