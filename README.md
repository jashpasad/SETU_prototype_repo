# SETU — Neighbourhood Reliability Layer for Renewable-Heavy Feeders

**Shed appliances, not neighbourhoods.**

A distribution-transformer-level control layer that replaces binary feeder load
shedding with graduated, appliance-level curtailment — backed by probabilistic
forecasting, a shared community battery, and a fairness-constrained dispatch
engine.

Yuva Yodha Energy Tech Hackathon 2026 · Schneider Electric
Challenge 03 — Grid Reliability & Renewable Intermittency · **Prototype Round**

---

## The problem this prototype attacks

A DISCOM's control over an LV feeder is binary. Facing a 30 kW deficit on a
220 kW distribution transformer, it cannot ask for 30 kW — it opens the breaker
and takes all 220, so every home, shop and clinic on that feeder goes dark
together. Yet most of the load is deferrable: pumps, geysers, coolers,
e-rickshaw chargers and refrigeration can shift by minutes to hours unnoticed.
Only a small critical slice — a few lights, a fan, phone charging, a clinic
fridge — must never be interrupted.

SETU adds the missing resolution: it segments every connection into a **Tier 0
critical** circuit and a **flexible** circuit, then keeps the feeder inside its
import limit by curtailing flexible load and discharging a shared battery, so
the critical tier stays energised.

## What this repository contains

This is a **working simulation**, not a mockup. It synthesises a year of
appliance-level demand for 430 connections, generates stochastic scarcity
events, forecasts net load probabilistically, and solves a rolling-horizon
dispatch problem every 15 minutes — about 35,000 optimisation decisions per run.

```
setu/
  config.py      all physical, economic and solver assumptions in one place
  profiles.py    bottom-up appliance load synthesis + rooftop PV geometry
  forecast.py    scarcity-event generator; gradient-boosted quantile forecaster
  dispatch.py    rolling-horizon MPC (CVXPY) + deterministic safety envelope
  fairness.py    burden-share allocation and Gini ledger
  simulate.py    year-long engine; baseline and SETU on the same realisation
scripts/
  run_year.py    main experiment (fairness ON) + ablation
  run_control.py fairness-OFF control run
app/
  dashboard.py   Streamlit operator console
results/         summary.json + compressed traces
docs/screenshots four console views
```

## Results

Single feeder, 365 simulated days at 15-minute resolution. 400 households and
30 micro-enterprises on one 315 kVA distribution transformer, 338 kWp of rooftop
PV, a 250 kWh / 125 kW community battery.

**Baseline and SETU are driven by the same load and the same scarcity
realisation**, so the comparison is like for like.

| Metric | Baseline (binary feeder control) | SETU |
|---|---|---|
| Critical-supply availability (energy basis) | 65.36% | **98.59%** |
| Unserved critical energy | 58.16 MWh/yr | **2.36 MWh/yr**  (−95.9%) |
| Full outage hours per day | 3.282 | **0.002** |
| Rationing instead of outage | no such state — control is binary | 0.856 h/day at 17.7% mean shortfall |
| Peak import during scarcity | 373.2 kW | **193.2 kW**  (−180.0 kW) |
| Flexible energy deferred (recovered off-peak) | — | 110.45 MWh/yr (15.11% of flexible load) |
| Curtailment Gini, rolling 30 days | 0.524  *(constraint OFF)* | **0.298**  *(constraint ON)* |
| Reliability cost of the fairness constraint | — | 0.0 outage h/day, 0.0 MWh/yr |
| Battery equivalent full cycles | — | 253.4/yr |
| Optimiser invoked | — | 45.71% of steps; solver failures 0.0% |

### How to read the outage numbers

The baseline outage is total: the breaker opens and 100% of critical load is
lost. SETU almost never reaches that state. Instead, for 0.86 h/day it
**rations** — holding an average 17.7% shortfall across the critical tier rather
than disconnecting it. Lights and fans dim in duty cycle instead of a
neighbourhood going dark. Because these are different states, we report both a
strict full-outage measure and total unserved critical energy, and we treat
**unserved critical energy (−95.9%)** as the headline.

### The fairness result

With the fairness constraint disabled, the rolling 30-day curtailment Gini
settles at **0.524** and never falls below the 0.25 target. With it enabled it
runs at **0.298** — and costs **nothing**: identical full-outage hours and
identical unserved critical energy in both runs. Equity was free.

Fairness is measured as **burden share** — the fraction of a connection's own
deferrable energy that SETU has taken — not absolute kWh. Absolute kWh across
connections with very different deferrable capacity is unequal by construction
and tells a participant nothing useful.

## Quick start

```bash
pip install -r requirements.txt

python scripts/run_year.py --days 365      # main run + ablation (~5 min)
python scripts/run_year.py --days 30       # fast demo run
python scripts/run_control.py              # fairness-OFF control

streamlit run app/dashboard.py             # operator console
```

## Design decisions worth knowing

**The optimiser is not a safety component.** A deterministic rule engine in
`dispatch.py` enforces the battery and power limits and can run the island
alone. The MPC is invoked on only **45.7%** of steps — when a deficit is
actually within the horizon — and the rule handles the rest. An edge controller
should not solve an LP every fifteen minutes to decide to do nothing.

**Dispatch plans against the P90 net-load quantile, not the median,** because
the decision is a risk decision: spend stored energy now, or hold it against a
deficit that may not arrive.

**The battery is pre-positioned.** Between 11:00 and 17:30 the optimiser carries
a state-of-charge target so it enters the evening scarcity window full. Removing
this materially degraded performance in testing.

## Two bugs this simulation caught in our own design

1. With our first objective weights, curtailing flexible load was *cheaper than
   importing energy*, so the controller curtailed continuously even with no
   scarcity — 48% of all flexible energy deferred. Corrected to 15%.
2. The first fair allocator sorted by absolute cumulative curtailment, which
   concentrated burden on large flexible loads and left Gini at 0.41. Switching
   the metric and the allocator to burden share fixed it.

Both are documented rather than quietly patched, because the point of building
a prototype is to find things like this.

## Advanced modelling layers

Six modules beyond the baseline engine. Each is validated in isolation by
`scripts/validate_advanced.py`, whose full output is saved to
`docs/validation_output.txt`.

| Module | What it does | Why the simpler version was wrong |
|---|---|---|
| `network.py` | Unbalanced three-phase LV feeder, 33 nodes, backward-forward sweep power flow, transformer impedance, per-phase voltages | A single-bus model hides the constraint that actually binds. The feeder reaches 0.873 pu and 321 voltage violations under heavy load |
| `thermal.py` | ETP (lumped-capacitance) models of coolers, fridges and water heaters with deadband hysteresis, integrated exactly | Treating flexible load as kW you can subtract at will is wrong in a way that flatters the result. Cold load pickup and comfort limits now emerge from physics |
| `stochastic.py` | AR(1)-correlated scenario paths; fast-forward reduction under the Wasserstein-1 metric | Planning against one pessimistic path is a heuristic dressed up as optimisation |
| `optimizer.py` | Two-stage stochastic MILP: binary relay states, scenario recourse, non-anticipativity, CVaR, OWA fairness, voltage constraints | Relays are discrete; an LP that curtails 7.3 kW describes nothing physical |
| `degradation.py` | ASTM E1049 rainflow counting + Xu et al. (2018) stress model | Throughput divided by capacity mis-prices depth of discharge by a wide margin |
| `conformal.py` | Conformalised quantile regression with coverage diagnostics | The raw GBM's nominal 80% interval covered 48%. Dispatch was reserving against a "P90" that was really a P72 |

### Validation highlights

**Voltage.** Curtailment is about 7x more effective per kilowatt than battery
injection for LV voltage support. The battery sits at the transformer, so it
relieves the transformer drop but not the LV line drop. A DT-sited battery
cannot fix a voltage problem at the tail of a feeder; curtailment can. This
was invisible in the single-bus model.

**Cold load pickup is derived, not assumed.** Hold the cooler fleet off for
three hours and settled demand of 382 kW returns as 563 kW - a 1.47x rebound
produced by 400 independent thermostats drifting into alignment. After three
hours, 93% of units are past their comfort limit, which puts a hard ceiling
on how long load can honestly be deferred.

**Scenario reduction converges.** Wasserstein-1 gap falls monotonically
486 -> 443 -> 413 -> 377 as the retained set grows 2 -> 4 -> 6 -> 10.

**Both constraints bind.** Tightening the OWA fairness cap from 18 to 9
moves the solution from 5 relays to 7 - spreading the same curtailment
wider - at zero cost in unserved energy. Below 0.910 pu the voltage
constraint forces extra relays open; at 0.870 pu the optimiser reports a
residual violation rather than pretending it can fix the feeder.

**Degradation separates depth from throughput.** Deep daily cycling gives
3.60%/yr fade and a 5.6-year life; shallow cycling at lower throughput gives
2.90%/yr and 6.9 years.

**Conformal calibration works.** Coverage moves from 0.48 to 0.82 against a
nominal 0.80, and the reliability diagram sits on the diagonal at every
alpha tested.

**It is fast enough to be real.** The MILP is DPP-compliant: 312 ms to
compile, then ~18 ms per solve.

### Not yet integrated

These six modules are validated in isolation but the year-long run in
`scripts/run_year.py` still uses the original LP, single-bus, lumped-load
path. Integrating them will move the headline numbers DOWN, correctly:
thermal limits cap deferral, voltage constraints block curtailment that is
currently free, and conformal calibration widens the reserve that must be
held. That re-baselining is the next piece of work, and the numbers in the
results table above should be read as the pre-integration figures.

## Limitations

- Load profiles are **synthesised from appliance assumptions**, not metered.
  The appliance set, ownership shares and seasonality in `profiles.py` are the
  principal modelling risk and the first thing a field pilot should falsify.
- Scarcity events are stochastic with parameters chosen to reproduce roughly
  3.2 outage hours/day; they are not drawn from a specific DISCOM's logs.
- The LV network is modelled as a single bus. Voltage and phase-imbalance
  constraints are **not** enforced; a feeder-level power-flow model is the next
  step before any islanding claim is made to a utility.
- No hardware was built. Relay response times and LoRaWAN behaviour are assumed.

## Team

Jash Bharat Pasad · Sanidhya Bhandari · Shreyas Purkar

## Licence

MIT
