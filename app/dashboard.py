"""SETU operator console.

Reads the traces produced by scripts/run_year.py and presents them the way a
DISCOM control-room operator and a cooperative administrator would use them.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

INK, AMBER, GREEN, TEAL, CORAL, MAUVE = (
    "#141B2E", "#F2A626", "#3B9E67", "#2196A8", "#D9584A", "#C77DBA")

st.set_page_config(page_title="SETU Operator Console", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
 .stApp {{ background:{INK}; }}
 h1,h2,h3,h4,p,span,label,div {{ color:#E8ECF3 !important; }}
 .kpi {{ background:#1F2A47; border-radius:10px; padding:14px 16px; }}
 .kpi .v {{ font-size:28px; font-weight:700; line-height:1.1; }}
 .kpi .l {{ font-size:12px; color:#9BA6BC !important; }}
 .tag {{ display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:12px; font-weight:700; }}
 [data-testid="stMetricValue"] {{ color:#F2A626 !important; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    s = json.loads((RESULTS / "summary.json").read_text())
    t = dict(np.load(RESULTS / "traces_fair.npz"))
    c = {}
    if (RESULTS / "traces_control.npz").exists():
        c = dict(np.load(RESULTS / "traces_control.npz"))
    return s, t, c


try:
    S, T, C = load()
except FileNotFoundError:
    st.error("No results found. Run `python scripts/run_year.py` first.")
    st.stop()

SPD = 96


def kpi(col, value, label, colour=AMBER):
    col.markdown(
        f"<div class='kpi'><div class='v' style='color:{colour}'>{value}</div>"
        f"<div class='l'>{label}</div></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------- header
left, right = st.columns([3, 1])
left.markdown("## SETU · Operator Console")
left.caption("Feeder DT-4471 · 400 households + 30 micro-enterprises · "
             "315 kVA distribution transformer · 250 kWh / 125 kW community battery")
right.markdown(
    f"<div style='text-align:right;margin-top:18px'>"
    f"<span class='tag' style='background:{GREEN};color:#0B1220'>SIMULATION · 365 DAYS</span>"
    f"</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Reliability", "Evening replay", "Fairness ledger", "Method & economics"])

# ------------------------------------------------------------- reliability
with tab1:
    c = st.columns(4)
    kpi(c[0], f"{S['critical_energy_availability_pct']}%",
        f"Critical-supply availability (baseline {S['baseline_critical_energy_availability_pct']}%)", GREEN)
    kpi(c[1], f"−{S['unserved_critical_energy_reduction_pct']}%",
        "Unserved critical energy vs baseline", AMBER)
    kpi(c[2], f"{S['baseline_outage_h_per_day']} → {S['setu_full_outage_h_per_day']}",
        "Full outage hours per day", TEAL)
    kpi(c[3], f"{S['peak_shaved_kw']} kW",
        "Peak import shaved during scarcity", MAUVE)

    st.write("")
    a, b = st.columns([3, 2])

    uc = T["unserved_critical_kw"]
    buc = T["base_unserved_critical_kw"]
    days = uc.size // SPD
    month_idx = ((np.arange(uc.size) // SPD) // 30.5).astype(int).clip(0, 11)
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bm = [buc[month_idx == m].sum() * 0.25 / 1000 for m in range(12)]
    sm = [uc[month_idx == m].sum() * 0.25 / 1000 for m in range(12)]

    fig = go.Figure()
    fig.add_bar(x=names, y=bm, name="Baseline (binary feeder control)",
                marker_color="#8E97A8")
    fig.add_bar(x=names, y=sm, name="With SETU", marker_color=AMBER)
    fig.update_layout(
        barmode="group", height=330, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=36, b=10),
        title="Unserved critical energy by month (MWh)",
        legend=dict(orientation="h", y=-0.18))
    a.plotly_chart(fig, use_container_width=True)

    b.markdown("#### How the residual is served")
    b.markdown(
        f"""SETU almost never loses critical supply outright
({S['setu_full_outage_h_per_day']} h/day). Instead, for
**{S['setu_any_shortfall_h_per_day']} h/day** the optimiser **rations** —
holding an average shortfall of **{S['mean_shortfall_when_rationing_pct']}%**
across the critical tier rather than disconnecting it.

That is graceful degradation, not an outage: lights and fans dim in duty
cycle instead of a neighbourhood going dark. The baseline has no equivalent
state — the breaker is open or closed.""")
    b.metric("Flexible energy deferred (recovered off-peak later)",
             f"{S['flexible_energy_deferred_mwh_yr']} MWh/yr",
             f"{S['deferred_share_of_flexible_pct']}% of flexible load")

# ----------------------------------------------------------- evening replay
with tab2:
    scarce_days = sorted({int(i // SPD) for i in np.where(
        T["import_limit_kw"] < 249.9)[0]})
    d = st.select_slider("Day of simulation", options=scarce_days,
                         value=scarce_days[len(scarce_days) // 2])
    sl = slice(d * SPD, (d + 1) * SPD)
    t = np.arange(SPD) / 4

    crit, flex = T["critical_kw"][sl], T["flexible_kw"][sl]
    curt, pv = T["curtail_kw"][sl], T["pv_kw"][sl]
    lim, imp = T["import_limit_kw"][sl], T["import_kw"][sl]
    dis, soc = T["discharge_kw"][sl], T["soc_kwh"][sl]

    f = go.Figure()
    f.add_scatter(x=t, y=crit, name="Tier 0 critical", stackgroup="s",
                  line=dict(width=0), fillcolor=GREEN)
    f.add_scatter(x=t, y=flex - curt, name="Flexible served", stackgroup="s",
                  line=dict(width=0), fillcolor="#3C4F74")
    f.add_scatter(x=t, y=curt, name="Flexible curtailed", stackgroup="s",
                  line=dict(width=0), fillcolor=CORAL)
    f.add_scatter(x=t, y=lim, name="DISCOM import limit",
                  line=dict(color=AMBER, width=3, dash="dash"))
    f.add_scatter(x=t, y=dis, name="Battery discharge",
                  line=dict(color=MAUVE, width=2))
    f.add_scatter(x=t, y=pv, name="Rooftop PV",
                  line=dict(color="#C9D44B", width=2))
    f.update_layout(height=400, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=36, b=10),
                    title=f"Day {d} · feeder power (kW) by hour",
                    xaxis_title="hour", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(f, use_container_width=True)

    g = go.Figure()
    g.add_scatter(x=t, y=soc, name="State of charge",
                  line=dict(color=TEAL, width=3), fill="tozeroy",
                  fillcolor="rgba(33,150,168,0.18)")
    g.update_layout(height=220, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=34, b=10),
                    title="Community battery state of charge (kWh)",
                    xaxis_title="hour", showlegend=False)
    st.plotly_chart(g, use_container_width=True)

    shed = lim < 1e-6
    c = st.columns(4)
    kpi(c[0], f"{shed.sum() * 0.25:.2f} h", "Upstream shed duration today", CORAL)
    kpi(c[1], f"{curt.max():.0f} kW", "Peak flexible load curtailed", AMBER)
    kpi(c[2], f"{dis.max():.0f} kW", "Peak battery discharge", MAUVE)
    kpi(c[3], f"{soc.min():.0f} kWh", "Minimum state of charge", TEAL)

# ---------------------------------------------------------- fairness ledger
with tab3:
    c = st.columns(4)
    kpi(c[0], f"{S['gini_mean_steady']}", "Curtailment Gini, SETU (target ≤ 0.25)", GREEN)
    kpi(c[1], f"{S.get('control_gini_mean_steady', '—')}",
        "Gini with fairness constraint OFF (control)", CORAL)
    kpi(c[2], f"{S['ledger_burden_share_mean_pct']:.1f}%",
        "Mean burden share — fraction of own flexible energy taken", AMBER)
    kpi(c[3], f"{S.get('fairness_cost_outage_h_per_day', 0.0)}",
        "Reliability cost of fairness (outage h/day)", TEAL)

    st.write("")
    a, b = st.columns(2)

    fig = go.Figure()
    fig.add_scatter(y=T["daily_gini"], name="Fairness constraint ON",
                    line=dict(color=GREEN, width=2))
    if "daily_gini" in C:
        fig.add_scatter(y=C["daily_gini"], name="Control — constraint OFF",
                        line=dict(color=CORAL, width=2))
    fig.add_hline(y=0.25, line=dict(color=AMBER, dash="dash"),
                  annotation_text="target 0.25")
    fig.update_layout(height=330, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=36, b=10),
                      title="Rolling 30-day curtailment Gini",
                      xaxis_title="day", legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, use_container_width=True)

    h = go.Figure()
    h.add_histogram(x=T["burden_share"] * 100, nbinsx=34, name="Fairness ON",
                    marker_color=GREEN, opacity=0.78)
    if "burden_share" in C:
        h.add_histogram(x=C["burden_share"] * 100, nbinsx=34, name="Control",
                        marker_color=CORAL, opacity=0.6)
    h.update_layout(barmode="overlay", height=330, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=36, b=10),
                    title="Distribution of burden share across participants (%)",
                    xaxis_title="% of own flexible energy curtailed",
                    legend=dict(orientation="h", y=-0.2))
    b.plotly_chart(h, use_container_width=True)

    st.info(
        "Fairness is measured as **burden share** — the fraction of a "
        "connection's own deferrable energy that SETU has taken — not absolute "
        "kWh. Absolute kWh across connections with very different deferrable "
        "capacity is unequal by construction. The constraint is published "
        "monthly to members as the cooperative's accountability metric.")

# ------------------------------------------------------- method & economics
with tab4:
    a, b = st.columns(2)
    a.markdown("#### Method")
    a.markdown(f"""
- **{S['days_simulated']} days** at 15-minute resolution, {S['connections']} connections
- Load synthesised **bottom-up from appliances**, each tagged critical or flexible
- {S['total_pv_kwp']} kWp of rooftop PV with a daily cloud index and clear-sky geometry
- Scarcity events generated stochastically; the **same realisation** drives both
  baseline and SETU, so the comparison is like-for-like
- Forecast: gradient-boosted **quantile regression** (P10/P50/P90) on feeder net load
- Dispatch: **rolling-horizon MPC** (6 h, CVXPY + CLARABEL), re-solved every 15 min
- Optimiser invoked on **{S['optimiser_invoked_share_pct']}%** of steps — only when a
  deficit is within the horizon; a deterministic rule runs the rest of the time
- Solver failures: **{S['solver_failure_share_pct']}%**
""")
    b.markdown("#### Prototype economics (per feeder)")
    b.markdown(f"""
| Item | Value |
|---|---|
| Capex | ₹{S['capex_inr']:,} |
| Capex per connection | ₹{S['capex_inr_per_connection']:,} |
| Value of lost load avoided | ₹{S['voll_avoided_inr_yr']:,}/yr |
| Battery equivalent full cycles | {S['battery_equivalent_full_cycles_yr']}/yr |
| Peak import shaved in scarcity | {S['peak_shaved_kw']} kW |
""")
    b.caption("Value of lost load valued at ₹95/kWh residential — the "
              "conservative end of published Indian VoLL estimates. Deferred "
              "flexible energy is recovered off-peak and is NOT valued as loss.")

    with st.expander("Full results JSON"):
        st.json(S)
