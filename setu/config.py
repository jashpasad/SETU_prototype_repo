"""Feeder, asset and simulation configuration for SETU.

All parameters are explicit and in SI / INR units so that any number
reported by the simulation can be traced back to an assumption here.
"""
from dataclasses import dataclass, field

STEPS_PER_HOUR = 4          # 15-minute resolution
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
DT_HOURS = 1.0 / STEPS_PER_HOUR


@dataclass
class FeederConfig:
    """The physical neighbourhood SETU is deployed on."""
    n_households: int = 400
    n_enterprises: int = 30
    dt_rating_kva: float = 315.0          # distribution transformer rating

    # Rooftop PV on and around the feeder
    pv_household_fraction: float = 0.35   # share of homes with rooftop PV
    pv_kwp_per_household: float = 2.0
    pv_kwp_enterprise: float = 2.0

    latitude_deg: float = 23.2            # Ahmedabad / Gandhinagar


@dataclass
class BatteryConfig:
    """Shared community battery at the distribution transformer."""
    capacity_kwh: float = 250.0
    power_kw: float = 125.0
    eta_charge: float = 0.94
    eta_discharge: float = 0.94
    soc_min: float = 0.10
    soc_max: float = 1.00
    soc_init: float = 0.80
    # Energy held back so a scarcity event always has something to draw on
    reserve_fraction: float = 0.20


@dataclass
class ScarcityConfig:
    """How often the upstream network fails to deliver, and how badly.

    Two event types are modelled:
      'shed' - upstream breaker opens, import limit falls to zero
      'cap'  - import is curtailed to a fraction of demand
    """
    daily_event_probability: float = 0.85
    mean_duration_h: float = 3.70
    duration_sd_h: float = 1.10
    min_duration_h: float = 0.75
    max_duration_h: float = 6.50
    earliest_start_h: float = 17.5
    latest_start_h: float = 21.0
    shed_probability: float = 0.60        # remainder are 'cap' events
    cap_fraction_range: tuple = (0.55, 0.78)
    # Seasonal multiplier on event probability, by month (Jan..Dec)
    seasonal_multiplier: tuple = (
        0.80, 0.85, 1.00, 1.25, 1.40, 1.30,
        1.15, 1.05, 0.95, 0.85, 0.80, 0.80,
    )
    normal_import_limit_kw: float = 250.0


@dataclass
class DispatchConfig:
    """Objective weights and horizon for the rolling-horizon optimiser."""
    horizon_steps: int = 24               # 6 hours at 15-minute resolution
    w_unserved_critical: float = 1000.0   # alpha - dominates everything
    w_curtail_flexible: float = 14.0      # beta  - discomfort, not loss
    w_import_energy: float = 1.00         # gamma - cost of imported energy
    w_terminal_soc: float = 3.5           # value of energy left in the battery
    w_cycling: float = 0.60               # battery throughput / degradation cost
    w_soc_target: float = 45.0            # penalty for missing the pre-position target
    preposition_start_h: float = 11.0     # charge the battery ahead of the evening
    preposition_end_h: float = 17.5
    preposition_soc_target: float = 0.95
    enforce_fairness: bool = True
    gini_limit: float = 0.25
    fairness_window_days: int = 30


@dataclass
class EconomicsConfig:
    """Capital and operating assumptions, INR."""
    battery_cost_inr: float = 2_250_000
    pcs_cost_inr: float = 600_000
    dt_sensing_controller_inr: float = 150_000
    relay_cost_inr: float = 900
    install_civil_inr: float = 340_000

    tariff_peak_inr_per_kwh: float = 8.50
    tariff_offpeak_inr_per_kwh: float = 4.20
    voll_household_inr_per_kwh: float = 95.0     # value of lost load
    voll_enterprise_inr_per_kwh: float = 320.0

    subscription_household_inr_per_month: float = 90.0
    subscription_enterprise_inr_per_month: float = 450.0
    dr_capacity_inr_per_kw_year: float = 1800.0


@dataclass
class SimConfig:
    days: int = 365
    seed: int = 20261207
    start_month: int = 1
    feeder: FeederConfig = field(default_factory=FeederConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    scarcity: ScarcityConfig = field(default_factory=ScarcityConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
