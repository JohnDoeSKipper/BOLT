"""
Solar sizing engine for PowerRECO.
Calculates optimal solar panel array size based on available roof area.
"""
from __future__ import annotations
import math

# Physical constants for modern monocrystalline PERC/TOPCon panels
_PANEL_W_M = 1.134          # panel width in metres
_PANEL_H_M = 1.762          # panel height in metres
_PANEL_AREA_M2 = _PANEL_W_M * _PANEL_H_M          # ~2.00 m²
_SPACING_FACTOR = 1.05      # 5% extra for mounting rail clearance

USABLE_ROOF_FRACTION = 0.85                               # 85% of total roof
EFFECTIVE_PANEL_AREA_M2 = _PANEL_AREA_M2 * _SPACING_FACTOR  # ~2.10 m²
SYSTEM_DERATE = 0.80        # 80%: inverter, wiring, soiling, temperature losses

# Malaysia seasonal irradiance multipliers (Peninsular, Jan–Dec)
SEASONAL_FACTORS = [
    0.92, 0.97, 1.05, 1.08, 1.05, 0.98,
    0.97, 0.99, 0.96, 0.94, 0.88, 0.87,
]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def calculate_solar_sizing(
    roof_area_m2: float,
    panel_wattage_w: int = 415,
    psh: float = 4.5,
) -> dict:
    """
    Return optimal solar sizing for a given roof area.

    Args:
        roof_area_m2:    Total roof area in square metres.
        panel_wattage_w: Rated output per panel in watts (default 415 W).
        psh:             Peak sun hours per day for the site (Malaysia avg 4.5).

    Returns:
        dict with all sizing parameters and generation estimates.
    """
    usable_m2 = roof_area_m2 * USABLE_ROOF_FRACTION
    n_panels = math.floor(usable_m2 / EFFECTIVE_PANEL_AREA_M2)
    system_kwp = round(n_panels * panel_wattage_w / 1_000, 2)

    annual_kwh = system_kwp * psh * 365 * SYSTEM_DERATE
    monthly_kwh_avg = annual_kwh / 12
    daily_kwh_avg = annual_kwh / 365

    monthly_breakdown = [
        round(monthly_kwh_avg * f) for f in SEASONAL_FACTORS
    ]

    return {
        "roof_area_m2": roof_area_m2,
        "usable_area_m2": round(usable_m2, 2),
        "coverage_pct": USABLE_ROOF_FRACTION * 100,
        "n_panels": n_panels,
        "panel_wattage_w": panel_wattage_w,
        "effective_panel_area_m2": round(EFFECTIVE_PANEL_AREA_M2, 2),
        "system_kwp": system_kwp,
        "psh": psh,
        "system_derate_pct": SYSTEM_DERATE * 100,
        "annual_generation_kwh": round(annual_kwh),
        "monthly_generation_kwh_avg": round(monthly_kwh_avg),
        "daily_generation_kwh_avg": round(daily_kwh_avg, 1),
        "monthly_breakdown_kwh": monthly_breakdown,
        "month_labels": MONTH_LABELS,
    }
