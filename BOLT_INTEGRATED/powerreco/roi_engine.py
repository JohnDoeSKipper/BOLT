"""
ROI & financial analysis engine for PowerRECO.
25-year DCF model with IRR (Newton-Raphson), NPV at 6% WACC,
battery replacement at year 10.
"""
from __future__ import annotations
import math

MD_CHARGE_NEW = 97.06
MD_CHARGE_OLD = 30.30
ENERGY_RATE_C1 = 0.365
NEM_BUYBACK = 0.31

KWTBB = 0.016
SERVICE_TAX = 0.06
TAX_MULTIPLIER = (1 + KWTBB) * (1 + SERVICE_TAX)

DEFAULT_SOLAR_COST_PER_KWP = 3_500.0
DEFAULT_BATT_COST_PER_KWH  = 2_500.0
DEFAULT_SELF_CONSUMPTION   = 0.65

SOLAR_LIFE_YEARS   = 25
BATTERY_LIFE_YEARS = 10
BATT_REPLACE_DISCOUNT = 0.20
SOLAR_DEGRADATION  = 0.005
BATT_DEGRADATION   = 0.02
DISCOUNT_RATE      = 0.06

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

    Returns CAPEX, annual savings, payback, NPV, IRR,
    and year-by-year cumulative NPV series for charting.
    """
    md_rate = MD_CHARGE_NEW if use_new_tariff else MD_CHARGE_OLD

    solar_capex   = solar_kwp   * solar_cost_per_kwp
    battery_capex = battery_kwh * battery_cost_per_kwh
    total_capex   = solar_capex + battery_capex

    annual_gen_kwh    = monthly_generation_kwh * 12
    self_consumed_kwh = annual_gen_kwh * self_consumption_pct
    exported_kwh      = annual_gen_kwh * (1.0 - self_consumption_pct)

    annual_energy_save = self_consumed_kwh * ENERGY_RATE_C1 * TAX_MULTIPLIER
    annual_nem_credit  = exported_kwh * NEM_BUYBACK
    annual_md_save     = md_reduction_kw * md_rate * 12 * TAX_MULTIPLIER

    total_annual_benefit_yr1 = annual_energy_save + annual_nem_credit + annual_md_save

    simple_payback = (
        total_capex / total_annual_benefit_yr1
        if total_annual_benefit_yr1 > 0 else float("inf")
    )

    npv = -total_capex
    cashflows = []
    cumulative_npv = []

    for yr in range(1, SOLAR_LIFE_YEARS + 1):
        solar_f = (1.0 - SOLAR_DEGRADATION) ** yr
        batt_f  = (1.0 - BATT_DEGRADATION) ** min(yr, BATTERY_LIFE_YEARS)

        yr_benefit = (
            self_consumed_kwh * solar_f * ENERGY_RATE_C1 * TAX_MULTIPLIER
            + exported_kwh    * solar_f * NEM_BUYBACK
            + annual_md_save  * batt_f
        )

        yr_cost = 0.0
        if yr == BATTERY_LIFE_YEARS and battery_kwh > 0:
            yr_cost = battery_capex * (1.0 - BATT_REPLACE_DISCOUNT)

        net_cf = yr_benefit - yr_cost
        cashflows.append(net_cf)

        pv  = net_cf / ((1.0 + DISCOUNT_RATE) ** yr)
        npv += pv
        cumulative_npv.append(round(npv))

    irr = _irr(total_capex, cashflows)
    co2_offset = annual_gen_kwh * GRID_EMISSION_FACTOR

    return {
        "solar_capex_rm":           round(solar_capex),
        "battery_capex_rm":         round(battery_capex),
        "total_capex_rm":           round(total_capex),
        "annual_energy_savings_rm": round(annual_energy_save),
        "annual_nem_credit_rm":     round(annual_nem_credit),
        "annual_md_savings_rm":     round(annual_md_save),
        "total_annual_benefit_rm":  round(total_annual_benefit_yr1),
        "monthly_net_benefit_rm":   round(total_annual_benefit_yr1 / 12),
        "simple_payback_years":     round(simple_payback, 1),
        "npv_25yr_rm":              round(npv),
        "irr_pct":                  round(irr * 100, 1) if irr is not None else None,
        "cumulative_npv":           cumulative_npv,
        "annual_gen_kwh":           round(annual_gen_kwh),
        "self_consumed_kwh_annual": round(self_consumed_kwh),
        "exported_kwh_annual":      round(exported_kwh),
        "co2_offset_tonnes_yr":     round(co2_offset, 1),
        "md_rate_used":             md_rate,
        "energy_rate_used":         ENERGY_RATE_C1,
        "nem_rate_used":            NEM_BUYBACK,
    }


def _irr(capex: float, cashflows: list[float]) -> float | None:
    """Newton-Raphson IRR. Returns None if no solution found in (0%, 200%)."""
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
        r = max(-0.99, min(r_new, 9.99))
    return r if 0.0 < r < 2.0 else None
