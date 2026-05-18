"""
ROI & financial analysis engine for PowerRECO.
Uses TNB tariff data from the BOLT SAM Calculator.
"""
from __future__ import annotations
import math

# ── TNB tariff constants (Peninsular Malaysia, from SAM Calculator) ───────────
MD_CHARGE_NEW = 97.06       # RM/kW  — updated MD tariff
MD_CHARGE_OLD = 30.30       # RM/kW  — legacy MD tariff
ENERGY_RATE_C1 = 0.365      # RM/kWh — Tariff C1 (medium voltage general)
NEM_BUYBACK = 0.31          # RM/kWh — TNB NEM 3.0 solar export credit (≤ 72 kWp)

# Levy & tax (non-domestic)
KWTBB = 0.016               # 1.6% renewable energy fund
SERVICE_TAX = 0.06          # 6% service tax on energy + demand charges
TAX_MULTIPLIER = (1 + KWTBB) * (1 + SERVICE_TAX)

# ── Default installed cost benchmarks (Malaysia 2024–2025) ───────────────────
DEFAULT_SOLAR_COST_PER_KWP = 3_500.0    # RM/kWp — grid-tied, inverter included
DEFAULT_BATT_COST_PER_KWH  = 2_500.0    # RM/kWh — LFP, installed
DEFAULT_SELF_CONSUMPTION   = 0.65       # 65% of solar used on-site

# ── System lifetime assumptions ───────────────────────────────────────────────
SOLAR_LIFE_YEARS  = 25      # panel warranty life
BATTERY_LIFE_YEARS = 10     # LFP cycle life ≈ 10 yrs at moderate DOD
BATT_REPLACE_DISCOUNT = 0.20   # 20% cost reduction at first replacement
SOLAR_DEGRADATION  = 0.005  # 0.5%/year panel output loss
BATT_DEGRADATION   = 0.02   # 2%/year capacity fade
DISCOUNT_RATE      = 0.06   # 6% cost of capital (WACC proxy)

# Malaysia grid emission factor (tCO₂/kWh, Suruhanjaya Tenaga 2022)
GRID_EMISSION_FACTOR = 0.000585


def calculate_roi(
    solar_kwp: float,
    battery_kwh: float,
    monthly_generation_kwh: float,
    md_reduction_kw: float,
    self_consumption_pct: float = DEFAULT_SELF_CONSUMPTION,
    use_new_tariff: bool = True,
    solar_cost_per_kwp: float = DEFAULT_SOLAR_COST_PER_KWP,
    battery_cost_per_kwh: float = DEFAULT_BATT_COST_PER_KWH,
) -> dict:
    """
    Full 25-year financial model for a solar + battery system.

    Returns a dict with CAPEX, annual savings, payback, NPV, IRR, and the
    year-by-year cumulative NPV series for charting.
    """
    md_rate = MD_CHARGE_NEW if use_new_tariff else MD_CHARGE_OLD

    # ── Capital costs ──────────────────────────────────────────────────────────
    solar_capex   = solar_kwp  * solar_cost_per_kwp
    battery_capex = battery_kwh * battery_cost_per_kwh
    total_capex   = solar_capex + battery_capex

    # ── Year-1 annual savings ──────────────────────────────────────────────────
    annual_gen_kwh     = monthly_generation_kwh * 12
    self_consumed_kwh  = annual_gen_kwh * self_consumption_pct
    exported_kwh       = annual_gen_kwh * (1.0 - self_consumption_pct)

    # Energy offset: self-consumed solar avoids buying from grid
    annual_energy_save = self_consumed_kwh * ENERGY_RATE_C1 * TAX_MULTIPLIER
    # NEM export credit (no tax on credit)
    annual_nem_credit  = exported_kwh * NEM_BUYBACK
    # MD peak-shaving via battery
    annual_md_save     = md_reduction_kw * md_rate * 12 * TAX_MULTIPLIER

    total_annual_benefit_yr1 = annual_energy_save + annual_nem_credit + annual_md_save

    # ── Simple payback ─────────────────────────────────────────────────────────
    simple_payback = (
        total_capex / total_annual_benefit_yr1
        if total_annual_benefit_yr1 > 0 else float("inf")
    )

    # ── 25-year DCF ───────────────────────────────────────────────────────────
    npv = -total_capex
    cashflows = []
    cumulative_npv = []

    for yr in range(1, SOLAR_LIFE_YEARS + 1):
        solar_f = (1.0 - SOLAR_DEGRADATION) ** yr
        batt_f  = (1.0 - BATT_DEGRADATION)  ** min(yr, BATTERY_LIFE_YEARS)

        yr_benefit = (
            self_consumed_kwh  * solar_f * ENERGY_RATE_C1 * TAX_MULTIPLIER
            + exported_kwh     * solar_f * NEM_BUYBACK
            + annual_md_save   * batt_f
        )

        # Battery replacement capex at year 10
        yr_cost = 0.0
        if yr == BATTERY_LIFE_YEARS and battery_kwh > 0:
            yr_cost = battery_capex * (1.0 - BATT_REPLACE_DISCOUNT)

        net_cf = yr_benefit - yr_cost
        cashflows.append(net_cf)

        pv   = net_cf / ((1.0 + DISCOUNT_RATE) ** yr)
        npv += pv
        cumulative_npv.append(round(npv))

    irr = _irr(total_capex, cashflows)

    # ── CO₂ offset ────────────────────────────────────────────────────────────
    co2_offset = annual_gen_kwh * GRID_EMISSION_FACTOR

    return {
        # Costs
        "solar_capex_rm":          round(solar_capex),
        "battery_capex_rm":        round(battery_capex),
        "total_capex_rm":          round(total_capex),
        # Annual benefits (Year 1)
        "annual_energy_savings_rm": round(annual_energy_save),
        "annual_nem_credit_rm":     round(annual_nem_credit),
        "annual_md_savings_rm":     round(annual_md_save),
        "total_annual_benefit_rm":  round(total_annual_benefit_yr1),
        "monthly_net_benefit_rm":   round(total_annual_benefit_yr1 / 12),
        # Financial metrics
        "simple_payback_years":    round(simple_payback, 1),
        "npv_25yr_rm":             round(npv),
        "irr_pct":                 round(irr * 100, 1) if irr is not None else None,
        "cumulative_npv":          cumulative_npv,
        # Generation details
        "annual_gen_kwh":          round(annual_gen_kwh),
        "self_consumed_kwh_annual": round(self_consumed_kwh),
        "exported_kwh_annual":      round(exported_kwh),
        # Environment
        "co2_offset_tonnes_yr":    round(co2_offset, 1),
        # Rates used (for transparency)
        "md_rate_used":            md_rate,
        "energy_rate_used":        ENERGY_RATE_C1,
        "nem_rate_used":           NEM_BUYBACK,
    }


def _irr(capex: float, cashflows: list[float]) -> float | None:
    """Newton-Raphson IRR. Returns None if no solution in (0%, 200%)."""
    r = 0.10
    for _ in range(200):
        npv  = -capex + sum(cf / (1 + r) ** (i + 1) for i, cf in enumerate(cashflows))
        dnpv = sum(-(i + 1) * cf / (1 + r) ** (i + 2) for i, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-10:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < 1e-8:
            r = r_new
            break
        r = max(-0.99, min(r_new, 9.99))   # clamp to avoid divergence
    return r if 0.0 < r < 2.0 else None
