"""
TNB Tariff Engine — Peninsular Malaysia
Tariff rates: official TNB schedule (revised 2014, updated 2022).
Covers: Tariff A (domestic tiered), B, C1, C2, D, E1, E2.
Includes: ICPT, KWTBB (1.6%), Service Tax (6%), NEM solar credit.

Voltage tiers (used for auto-detection):
  Low Voltage  (LV)  240 V / 415 V   → Tariff A, B         peak demand < 25 kW
  Medium Voltage(MV) 6.6 kV / 11 kV / 22 kV → Tariff C1, C2  25–999 kW
  High Voltage (HV)  33 kV / 132 kV  → Tariff D, E1, E2   ≥ 1,000 kW
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal

TariffCode = Literal["A", "B", "C1", "C2", "D", "E1", "E2"]

# ─── Rate tables ──────────────────────────────────────────────────────────────

# Tariff A — Domestic (progressive tiers)
# (block_limit_kwh | None=unlimited,  rate_rm_per_kwh,  display_label)
TARIFF_A_BLOCKS: list[tuple] = [
    (200,  0.218, "First 200 kWh"),
    (100,  0.334, "Next 100 kWh  (201 – 300)"),
    (300,  0.516, "Next 300 kWh  (301 – 600)"),
    (300,  0.546, "Next 300 kWh  (601 – 900)"),
    (None, 0.571, "Remaining kWh (> 900)"),
]
TARIFF_A_MIN = 3.00

# Tariff B — Low Voltage Commercial
TARIFF_B_ENERGY = 0.435
TARIFF_B_MIN    = 7.20

# Tariff C1 — Medium Voltage General
TARIFF_C1_ENERGY = 0.365
TARIFF_C1_MD     = 30.30
TARIFF_C1_MIN    = 600.00

# Tariff C2 — Medium Voltage Time-of-Use (08:00–22:00 Mon–Sat = peak)
TARIFF_C2_PEAK    = 0.365
TARIFF_C2_OFFPEAK = 0.219
TARIFF_C2_MD      = 30.30
TARIFF_C2_MIN     = 600.00

# Tariff D — High Voltage General
TARIFF_D_ENERGY = 0.337
TARIFF_D_MD     = 29.60
TARIFF_D_MIN    = 600.00

# Tariff E1 — High Voltage Peak / Off-Peak
TARIFF_E1_PEAK    = 0.337
TARIFF_E1_OFFPEAK = 0.202
TARIFF_E1_MD      = 29.60
TARIFF_E1_MIN     = 600.00

# Tariff E2 — High Voltage TOU (MD ≥ 1,500 kW, same rates as E1 in 2024)
TARIFF_E2_PEAK    = 0.337
TARIFF_E2_OFFPEAK = 0.202
TARIFF_E2_MD      = 29.60
TARIFF_E2_MIN     = 600.00

SERVICE_TAX_RATE = 0.06    # 6 % on (energy + MD + KWTBB) for non-domestic
KWTBB_RATE       = 0.016   # 1.6 % Kumpulan Wang Tenaga Boleh Baharu (non-domestic)

# NEM buyback default — TNB NEM 3.0 (≤ 72 kWp, residential/commercial)
NEM_DEFAULT_RATE = 0.31    # RM/kWh exported to grid

TARIFF_META: dict[str, dict] = {
    "A":  {
        "name": "Tariff A — Domestic",
        "voltage": "Low Voltage (240 V / 415 V)",
        "category": "Residential",
        "tou": False, "has_md": False,
        "typical_demand": "< 12 kW",
        "description": "Single- or three-phase residential supply.",
    },
    "B":  {
        "name": "Tariff B — Low Voltage Commercial",
        "voltage": "Low Voltage (240 V / 415 V)",
        "category": "Commercial (small)",
        "tou": False, "has_md": False,
        "typical_demand": "12 – 25 kW",
        "description": "Shops, small offices, clinics on LV supply.",
    },
    "C1": {
        "name": "Tariff C1 — Medium Voltage General",
        "voltage": "Medium Voltage (6.6 kV / 11 kV / 22 kV)",
        "category": "Industrial / Commercial (medium)",
        "tou": False, "has_md": True,
        "typical_demand": "25 – 999 kW",
        "description": "Factories, malls, hotels with MV metered supply.",
    },
    "C2": {
        "name": "Tariff C2 — Medium Voltage Time-of-Use",
        "voltage": "Medium Voltage (6.6 kV / 11 kV / 22 kV)",
        "category": "Industrial / Commercial (medium, TOU)",
        "tou": True, "has_md": True,
        "typical_demand": "25 – 999 kW",
        "description": "Same as C1 but with peak/off-peak pricing — suits heavy night-shift loads.",
    },
    "D":  {
        "name": "Tariff D — High Voltage General",
        "voltage": "High Voltage (33 kV / 132 kV)",
        "category": "Industrial (large)",
        "tou": False, "has_md": True,
        "typical_demand": "1,000 – 4,999 kW",
        "description": "Large factories and industrial parks on HV supply.",
    },
    "E1": {
        "name": "Tariff E1 — High Voltage Peak / Off-Peak",
        "voltage": "High Voltage (33 kV / 132 kV)",
        "category": "Industrial (large, TOU)",
        "tou": True, "has_md": True,
        "typical_demand": "≥ 1,000 kW",
        "description": "Large HV industrial supply with TOU pricing.",
    },
    "E2": {
        "name": "Tariff E2 — High Voltage TOU (MD ≥ 1,500 kW)",
        "voltage": "High Voltage (33 kV / 132 kV)",
        "category": "Industrial (extra-large, TOU)",
        "tou": True, "has_md": True,
        "typical_demand": "≥ 1,500 kW",
        "description": "Extra-large HV industrial supply, mandatory TOU.",
    },
}

_MD_RATES: dict[str, float] = {
    "C1": TARIFF_C1_MD, "C2": TARIFF_C2_MD,
    "D": TARIFF_D_MD, "E1": TARIFF_E1_MD, "E2": TARIFF_E2_MD,
}
_MIN_CHARGES: dict[str, float] = {
    "A": TARIFF_A_MIN, "B": TARIFF_B_MIN,
    "C1": TARIFF_C1_MIN, "C2": TARIFF_C2_MIN,
    "D": TARIFF_D_MIN, "E1": TARIFF_E1_MIN, "E2": TARIFF_E2_MIN,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_peak_hour(ts: pd.Timestamp) -> bool:
    """Peak window = 08:00–22:00, Monday–Saturday. Sunday & PH = off-peak."""
    if ts.dayofweek == 6:   # Sunday
        return False
    return 8 <= ts.hour < 22


def get_md_rate(tariff: TariffCode) -> float:
    return _MD_RATES.get(tariff, 0.0)


def get_min_charge(tariff: TariffCode) -> float:
    return _MIN_CHARGES.get(tariff, 0.0)


# ─── Bill calculator ──────────────────────────────────────────────────────────

def calculate_bill(
    tariff: TariffCode,
    monthly_kwh: float,
    peak_kwh: float = 0.0,
    offpeak_kwh: float = 0.0,
    max_demand_kw: float = 0.0,
    icpt_sen_per_kwh: float = 0.0,
    apply_service_tax: bool = True,
    apply_kwtbb: bool = True,
) -> dict:
    """
    Calculate a full TNB bill for one billing month.

    For flat tariffs (A, B, C1, D): pass monthly_kwh.
    For TOU tariffs (C2, E1, E2): pass peak_kwh + offpeak_kwh.
    ICPT in sen/kWh: positive = surcharge, negative = rebate.

    Returns a dict with every component and the total.
    NOTE: this calculates the GROSS bill (before any NEM solar credit).
          Deduct NEM credit separately using compute_nem_credit().
    """
    is_domestic = tariff == "A"
    result: dict = {
        "tariff": tariff,
        "tariff_name": TARIFF_META[tariff]["name"],
        "total_kwh": monthly_kwh,
        "peak_kwh": peak_kwh,
        "offpeak_kwh": offpeak_kwh,
        "max_demand_kw": max_demand_kw,
        "energy_charge": 0.0,
        "md_charge": 0.0,
        "icpt_charge": 0.0,
        "kwtbb_charge": 0.0,
        "service_tax": 0.0,
        "subtotal_pretax": 0.0,
        "minimum_charge_applied": False,
        "total_bill": 0.0,
        "blocks": [],
        "notes": [],
    }

    # ── Energy charges ────────────────────────────────────────────────────────
    if tariff == "A":
        remaining, ec = monthly_kwh, 0.0
        for limit, rate, label in TARIFF_A_BLOCKS:
            if remaining <= 0:
                break
            used = remaining if limit is None else min(remaining, limit)
            cost = used * rate
            result["blocks"].append({
                "Block": label,
                "kWh": round(used, 3),
                "Rate (RM/kWh)": rate,
                "Charge (RM)": round(cost, 2),
            })
            ec += cost
            remaining -= used
        result["energy_charge"] = round(ec, 2)  # minimum enforced at total-bill stage below

    elif tariff == "B":
        ec = round(monthly_kwh * TARIFF_B_ENERGY, 2)
        result["energy_charge"] = ec
        result["blocks"] = [{"Block": "All kWh (flat rate)",
                              "kWh": monthly_kwh, "Rate (RM/kWh)": TARIFF_B_ENERGY, "Charge (RM)": ec}]

    elif tariff == "C1":
        ec = round(monthly_kwh * TARIFF_C1_ENERGY, 2)
        result["energy_charge"] = ec
        result["md_charge"]     = round(max_demand_kw * TARIFF_C1_MD, 2)
        result["blocks"] = [{"Block": "All kWh (flat rate)",
                              "kWh": monthly_kwh, "Rate (RM/kWh)": TARIFF_C1_ENERGY, "Charge (RM)": ec}]

    elif tariff == "C2":
        ep = round(peak_kwh    * TARIFF_C2_PEAK,    2)
        eo = round(offpeak_kwh * TARIFF_C2_OFFPEAK, 2)
        result["energy_charge"] = ep + eo
        result["md_charge"]     = round(max_demand_kw * TARIFF_C2_MD, 2)
        result["blocks"] = [
            {"Block": "Peak kWh  (08:00–22:00, Mon–Sat)",
             "kWh": peak_kwh,    "Rate (RM/kWh)": TARIFF_C2_PEAK,    "Charge (RM)": ep},
            {"Block": "Off-Peak kWh  (all other times)",
             "kWh": offpeak_kwh, "Rate (RM/kWh)": TARIFF_C2_OFFPEAK, "Charge (RM)": eo},
        ]

    elif tariff == "D":
        ec = round(monthly_kwh * TARIFF_D_ENERGY, 2)
        result["energy_charge"] = ec
        result["md_charge"]     = round(max_demand_kw * TARIFF_D_MD, 2)
        result["blocks"] = [{"Block": "All kWh (flat rate)",
                              "kWh": monthly_kwh, "Rate (RM/kWh)": TARIFF_D_ENERGY, "Charge (RM)": ec}]

    elif tariff in ("E1", "E2"):
        ep = round(peak_kwh    * TARIFF_E1_PEAK,    2)
        eo = round(offpeak_kwh * TARIFF_E1_OFFPEAK, 2)
        result["energy_charge"] = ep + eo
        result["md_charge"]     = round(max_demand_kw * TARIFF_E1_MD, 2)
        result["blocks"] = [
            {"Block": "Peak kWh  (08:00–22:00, Mon–Sat)",
             "kWh": peak_kwh,    "Rate (RM/kWh)": TARIFF_E1_PEAK,    "Charge (RM)": ep},
            {"Block": "Off-Peak kWh  (all other times)",
             "kWh": offpeak_kwh, "Rate (RM/kWh)": TARIFF_E1_OFFPEAK, "Charge (RM)": eo},
        ]

    # ── ICPT ──────────────────────────────────────────────────────────────────
    kwh_for_icpt = monthly_kwh if not TARIFF_META[tariff]["tou"] else (peak_kwh + offpeak_kwh)
    result["icpt_charge"] = round(kwh_for_icpt * icpt_sen_per_kwh / 100, 2)

    # ── KWTBB (non-domestic only) ──────────────────────────────────────────────
    if apply_kwtbb and not is_domestic:
        result["kwtbb_charge"] = round(result["energy_charge"] * KWTBB_RATE, 2)

    # ── Service Tax (non-domestic only) ───────────────────────────────────────
    if apply_service_tax and not is_domestic:
        taxable = result["energy_charge"] + result["md_charge"] + result["kwtbb_charge"]
        result["service_tax"] = round(taxable * SERVICE_TAX_RATE, 2)

    # ── Subtotal & minimum charge ──────────────────────────────────────────────
    raw = (result["energy_charge"] + result["md_charge"] + result["icpt_charge"] +
           result["kwtbb_charge"]  + result["service_tax"])
    result["subtotal_pretax"] = round(
        result["energy_charge"] + result["md_charge"] + result["icpt_charge"], 2)

    min_c = _MIN_CHARGES[tariff]
    if raw < min_c:
        result["total_bill"] = min_c
        result["minimum_charge_applied"] = True
        result["notes"].append(f"Minimum monthly charge of RM {min_c:.2f} applied.")
    else:
        result["total_bill"] = round(raw, 2)

    if is_domestic:
        result["notes"].append("Tariff A: Service Tax and KWTBB are not applicable.")
        if icpt_sen_per_kwh == 0:
            result["notes"].append("ICPT: set to 0 — domestic customers are generally exempt.")

    return result


# ─── NEM solar credit ─────────────────────────────────────────────────────────

def compute_nem_credit(export_kwh: float, nem_rate_rm: float = NEM_DEFAULT_RATE) -> dict:
    """
    NEM (Net Energy Metering) credit for solar export to grid.
    Applied AFTER the gross bill (post all taxes/levies).

    nem_rate_rm: TNB buyback rate in RM/kWh.
      • NEM 3.0 (≤ 72 kWp residential/commercial): RM 0.31/kWh
      • Larger / commercial systems: check current TNB NEM schedule.
    """
    credit = round(export_kwh * nem_rate_rm, 2)
    return {
        "export_kwh": round(export_kwh, 3),
        "nem_rate_rm": nem_rate_rm,
        "nem_credit_rm": credit,
    }


# ─── Auto-detect tariff ───────────────────────────────────────────────────────

def auto_detect_tariff(df: pd.DataFrame, interval_minutes: int = 30) -> tuple[TariffCode, str, dict]:
    """
    Infer the most likely TNB tariff from a load profile.
    Uses peak demand (voltage proxy), load factor, and daily usage pattern.

    Returns (tariff_code, reason_string, stats_dict).
    """
    kfac = interval_minutes / 60.0

    df = df.copy()
    df["month"]   = df["timestamp"].dt.to_period("M")
    df["hour"]    = df["timestamp"].dt.hour
    df["dow"]     = df["timestamp"].dt.dayofweek  # 0=Mon … 6=Sun

    # Monthly stats
    monthly = (df.groupby("month")["kw_import"]
                 .agg(total_kwh=lambda x: (x * kfac).sum(), peak_kw="max")
                 .reset_index())

    avg_monthly_kwh = float(monthly["total_kwh"].mean())
    abs_peak_kw     = float(df["kw_import"].max())
    avg_kw          = float(df["kw_import"].mean())
    load_factor     = avg_kw / abs_peak_kw if abs_peak_kw > 0 else 0.0

    # Usage pattern — business-hours vs off-hours
    biz_mask  = (df["hour"] >= 8) & (df["hour"] < 18) & (df["dow"] < 5)
    biz_avg   = float(df.loc[biz_mask,  "kw_import"].mean()) if biz_mask.any() else 0.0
    off_avg   = float(df.loc[~biz_mask, "kw_import"].mean()) if (~biz_mask).any() else 0.0
    biz_ratio = biz_avg / off_avg if off_avg > 0 else 1.0

    # Weekend vs weekday
    wkday_avg = float(df.loc[df["dow"] < 5,  "kw_import"].mean())
    wkend_avg = float(df.loc[df["dow"] >= 5, "kw_import"].mean())
    wkend_ratio = wkend_avg / wkday_avg if wkday_avg > 0 else 1.0

    # Peak-hour concentration (residential = strong evening peak)
    eve_mask  = (df["hour"] >= 18) & (df["hour"] < 23)
    noon_mask = (df["hour"] >= 9)  & (df["hour"] < 17)
    eve_avg   = float(df.loc[eve_mask,  "kw_import"].mean()) if eve_mask.any() else 0.0
    noon_avg  = float(df.loc[noon_mask, "kw_import"].mean()) if noon_mask.any() else 0.0
    res_shape = eve_avg > noon_avg  # true for typical residential profile

    stats = {
        "avg_monthly_kwh": round(avg_monthly_kwh, 1),
        "abs_peak_kw":     round(abs_peak_kw, 1),
        "avg_kw":          round(avg_kw, 1),
        "load_factor":     round(load_factor, 3),
        "biz_ratio":       round(biz_ratio, 2),
        "wkend_ratio":     round(wkend_ratio, 2),
    }

    # ── Decision tree (voltage-based, demand as proxy) ─────────────────────────
    if abs_peak_kw >= 5000:
        code = "E2" if abs_peak_kw >= 1500 and load_factor > 0.6 else "E1"
        why  = (f"Peak demand {abs_peak_kw:,.0f} kW ≥ 5,000 kW → Extra High Voltage supply. "
                f"Load factor {load_factor:.0%}.")

    elif abs_peak_kw >= 1000:
        code = "D"
        why  = (f"Peak demand {abs_peak_kw:,.0f} kW falls in 1,000–4,999 kW range → "
                f"High Voltage (33 kV / 132 kV) supply, Tariff D.")

    elif abs_peak_kw >= 25:
        # Distinguish C1 vs C2: C2 saves money when off-peak is heavy (load_factor high + night usage)
        night_mask = (df["hour"] < 8) | (df["hour"] >= 22)
        night_avg  = float(df.loc[night_mask, "kw_import"].mean()) if night_mask.any() else 0.0
        night_frac = night_avg / avg_kw if avg_kw > 0 else 0.0
        if night_frac >= 0.35 or load_factor >= 0.70:
            code = "C2"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW → Medium Voltage (MV) supply. "
                    f"High off-peak usage ({night_frac:.0%} of avg load overnight) — "
                    f"Tariff C2 (TOU) is likely more economical than C1.")
        else:
            code = "C1"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW → Medium Voltage (MV) supply, Tariff C1. "
                    f"Load factor {load_factor:.0%}, daytime-heavy profile.")

    elif avg_monthly_kwh > 2000 or (biz_ratio > 2.5 and not res_shape):
        code = "B"
        why  = (f"Peak demand {abs_peak_kw:.1f} kW < 25 kW (Low Voltage supply), "
                f"avg monthly {avg_monthly_kwh:,.0f} kWh with a commercial daytime pattern "
                f"(business-hours ratio {biz_ratio:.1f}×) → Tariff B.")

    else:
        code = "A"
        why  = (f"Peak demand {abs_peak_kw:.1f} kW, avg monthly {avg_monthly_kwh:,.0f} kWh "
                f"with a {'residential evening-peak' if res_shape else 'low-demand'} profile "
                f"→ Domestic Tariff A (Low Voltage supply).")

    return code, why, stats


# ─── Monthly statistics ───────────────────────────────────────────────────────

def compute_monthly_stats(df: pd.DataFrame, interval_minutes: int = 30) -> pd.DataFrame:
    """
    Aggregate a load profile into per-month billing stats.
    Returns columns: month, total_kwh, peak_kwh, offpeak_kwh,
                     max_demand_kw, export_kwh
    """
    kfac = interval_minutes / 60.0
    df = df.copy()
    df["month"]      = df["timestamp"].dt.to_period("M")
    df["is_peak"]    = df["timestamp"].apply(is_peak_hour)
    df["kwh"]        = df["kw_import"] * kfac
    df["export_kwh_pt"] = df["kw_export"] * kfac

    stats = df.groupby("month", group_keys=False).apply(
        lambda g: pd.Series({
            "total_kwh":    round(float(g["kwh"].sum()), 3),
            "peak_kwh":     round(float(g.loc[g["is_peak"],  "kwh"].sum()), 3),
            "offpeak_kwh":  round(float(g.loc[~g["is_peak"], "kwh"].sum()), 3),
            "max_demand_kw": round(float(g["kw_import"].max()), 3),
            "export_kwh":   round(float(g["export_kwh_pt"].sum()), 3),
        })
    ).reset_index()

    return stats
