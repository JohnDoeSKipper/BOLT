"""
AI Energy Manager  v5  |  Flask backend
────────────────────────────────────────────────────────────────────────────────
New in v5:

  ① TNB-billing-aware dispatch  (run_ai_manager_tnb)
     • Tracks BILLING-MONTH MD — the single highest 30-min kVA in the month
       (not a rolling daily peak).  For TOU tariffs (C2/E1/E2) MD is measured
       only during peak hours 08:00–22:00 Mon–Sat (excl. public holidays).
       For flat tariffs (C1, D) MD is measured 24 × 7.
     • Strategic off-peak charging: TOU tariffs fill the battery during
       off-peak hours when energy is cheap and there is zero MD risk.
     • Adaptive MD target: if an unavoidable new peak is recorded, the target
       adjusts upward and the remainder of the month protects the new level.

  ② Live data feed  (POST /api/live/push)
     • The Predictor posts one 30-min reading every half-hour.
     • Per-session state (SOC, billing-month peak, dispatch log) is kept in
       memory across calls.
     • Optionally POST a 48-step forecast to /api/live/forecast so the
       dispatcher can see the day ahead and hold battery in reserve.

  ③ What-if simulator  (POST /api/simulate)
     • User selects solar kWp + battery kWh to install.
     • Generates solar generation profile (Gaussian bell, parameterised by
       peak-sun-hours for the site).
     • Applies solar to offset load, then runs TNB-aware battery dispatch.
     • Returns per-interval synthetic profile + before/after bill comparison.

Backward compatibility: all v4 endpoints (/api/upload, /api/optimize) still work.
────────────────────────────────────────────────────────────────────────────────
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import math, os, traceback, re
from io import StringIO
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

app = Flask(__name__, static_folder='static')
CORS(app)


# ══════════════════════════════════════════════════════════════════════════════
#  TNB TARIFF CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

TNB_PEAK_START_H = 8    # 08:00 inclusive
TNB_PEAK_END_H   = 22   # 22:00 exclusive  →  peak = [08:00, 22:00)

# Tariffs where energy uses peak/off-peak rates AND where MD is only measured
# during peak hours (Mon–Sat 08:00–22:00, excluding public holidays).
TOU_TARIFFS = frozenset({'C2', 'E1', 'E2'})

# MD rate (RM per kVA or kW per month) — note: TNB bills MD in kVA for MV
MD_RATES = {
    'C1': 30.30,
    'C2': 30.30,
    'D':  29.60,
    'E1': 29.60,
    'E2': 29.60,
}

# Energy rates (RM/kWh)
ENERGY_RATES = {
    'C1': {'flat': 0.365},
    'C2': {'peak': 0.365, 'offpeak': 0.219},
    'D':  {'flat': 0.337},
    'E1': {'peak': 0.337, 'offpeak': 0.202},
    'E2': {'peak': 0.337, 'offpeak': 0.202},
}

# Service tax + KWTBB multiplier for non-domestic tariffs (6% + 1.6% applied correctly)
# service_tax is on (energy + MD + kwtbb), kwtbb is on energy only
# Combined effective uplift ≈ 1.016 * 1.06 = 1.07696 on energy, 1.06 on MD
SERVICE_TAX = 0.08   # 8% (raised Mar 2024)
KWTBB_RATE  = 0.016  # 1.6% on energy

# NEM buyback rate (RM/kWh exported to grid) — TNB NEM 3.0 default
NEM_RATE_DEFAULT = 0.31

# Malaysian public holidays (Federal + common state — extend as needed)
_PH = {
    # 2024
    date(2024,  1,  1), date(2024,  1, 25), date(2024,  2, 10), date(2024,  2, 11),
    date(2024,  2, 12), date(2024,  4, 10), date(2024,  4, 11), date(2024,  5,  1),
    date(2024,  5, 22), date(2024,  6,  3), date(2024,  6, 17), date(2024,  7,  7),
    date(2024,  8, 31), date(2024,  9, 16), date(2024, 10, 31), date(2024, 12, 11),
    date(2024, 12, 25),
    # 2025
    date(2025,  1,  1), date(2025,  1, 29), date(2025,  1, 30), date(2025,  3, 11),
    date(2025,  3, 31), date(2025,  4,  1), date(2025,  5,  1), date(2025,  5, 12),
    date(2025,  6,  2), date(2025,  6,  7), date(2025,  6,  8), date(2025,  8, 31),
    date(2025,  9, 16), date(2025, 10, 20), date(2025, 10, 29), date(2025, 12, 25),
}
MALAYSIA_PUBLIC_HOLIDAYS: frozenset = frozenset(_PH)


# ══════════════════════════════════════════════════════════════════════════════
#  TNB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calc_kva(kw_net: float, kvar_net: float) -> float:
    return math.sqrt(kw_net ** 2 + kvar_net ** 2)


def is_tnb_peak(ts: pd.Timestamp, tariff: str) -> bool:
    """
    True if this 30-min interval falls within TNB peak hours for the tariff.

    For TOU tariffs (C2, E1, E2): peak = 08:00–22:00 Mon–Sat, excl. public holidays.
    For flat tariffs (C1, D):     all intervals are "peak" for MD purposes (24/7).

    Note: the MD charge is based on this window — discharging outside it gives
    no MD benefit for TOU tariffs.
    """
    if tariff not in TOU_TARIFFS:
        return True  # C1, D: MD measured all day
    if ts.dayofweek == 6:  # Sunday
        return False
    if ts.date() in MALAYSIA_PUBLIC_HOLIDAYS:
        return False
    return TNB_PEAK_START_H <= ts.hour < TNB_PEAK_END_H


def preferred_charge_window(ts: pd.Timestamp, tariff: str) -> bool:
    """
    True if this interval is ideal for charging the battery.

    TOU tariffs: off-peak hours (cheap energy, zero MD risk).
    Flat tariffs: night-time (22:00–07:00) to avoid raising daytime kVA.
    """
    if tariff in TOU_TARIFFS:
        return not is_tnb_peak(ts, tariff)
    # Non-TOU: charge between 22:00 and 07:00
    return ts.hour >= 22 or ts.hour < 7


def tnb_bill_quick(
    total_kwh: float,
    peak_kwh: float,
    offpeak_kwh: float,
    md_kva: float,
    tariff: str,
    nem_export_kwh: float = 0.0,
    nem_rate: float = NEM_RATE_DEFAULT,
) -> dict:
    """
    Simplified TNB bill calculator for before/after comparisons.
    Covers energy + MD + KWTBB + Service Tax (8%).
    Does not enforce minimum monthly charges (add separately if needed).
    """
    rates = ENERGY_RATES.get(tariff)
    if rates is None:
        return {'energy_rm': 0, 'md_rm': 0, 'kwtbb_rm': 0, 'st_rm': 0,
                'nem_credit_rm': 0, 'gross_rm': 0, 'net_rm': 0}

    if 'flat' in rates:
        energy_rm = total_kwh * rates['flat']
    else:
        energy_rm = peak_kwh * rates['peak'] + offpeak_kwh * rates['offpeak']

    md_rm    = md_kva * MD_RATES.get(tariff, 0.0)
    kwtbb_rm = energy_rm * KWTBB_RATE
    st_rm    = (energy_rm + md_rm + kwtbb_rm) * SERVICE_TAX
    gross_rm = energy_rm + md_rm + kwtbb_rm + st_rm
    nem_rm   = round(nem_export_kwh * nem_rate, 2)
    net_rm   = max(0.0, round(gross_rm - nem_rm, 2))

    return {
        'energy_rm':    round(energy_rm, 2),
        'md_rm':        round(md_rm, 2),
        'kwtbb_rm':     round(kwtbb_rm, 2),
        'st_rm':        round(st_rm, 2),
        'nem_credit_rm': nem_rm,
        'gross_rm':     round(gross_rm, 2),
        'net_rm':       net_rm,
        'total_kwh':    round(total_kwh, 1),
        'peak_kwh':     round(peak_kwh, 1),
        'offpeak_kwh':  round(offpeak_kwh, 1),
        'md_kva':       round(md_kva, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SOLAR GENERATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

# Build the normalised half-hourly bell curve once at import time.
# Bell centred at 13:00 (hh index 26), sigma ≈ 3 h (6 half-hour steps).
_HH       = np.arange(48)
_RAW_BELL = np.exp(-0.5 * ((_HH - 26) / 6.0) ** 2)
_RAW_BELL[_RAW_BELL < 0.01] = 0.0   # no generation before ~06:00 / after ~20:00
# Normalise: raw[hh] * kWp = kW output when PSH = 1.
# At PSH = psh, scale by psh / (sum(raw) * 0.5h).
# Precompute for psh=1 so runtime only needs: curve * psh * kwp
_BELL_NORM = _RAW_BELL / (_RAW_BELL.sum() * 0.5)  # shape factor (kW per kWp per psh)


def generate_solar_kw(timestamps: pd.Series, solar_kwp: float, psh: float = 4.5) -> pd.Series:
    """
    Instantaneous solar kW output for each half-hourly timestamp.

    solar_kwp : installed DC nameplate capacity
    psh       : peak sun hours (daily kWh/kWp).  Malaysia defaults:
                West coast 4.0–5.0, East coast 3.8–4.5; use 4.5 if unsure.

    The bell curve peaks at 13:00 and integrates to psh kWh/day/kWp.
    Multiply by kWp to get kW; divide by 2 to convert to 30-min kWh.
    """
    hh = (timestamps.dt.hour * 2 + timestamps.dt.minute // 30).astype(int) % 48
    kw_per_kwp = _BELL_NORM[hh.values] * psh
    return pd.Series(kw_per_kwp * solar_kwp, index=timestamps.index, dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LiveSession:
    """Per-session state for the live 30-min data feed."""

    tariff:           str   = 'C1'
    battery_kwh:      float = 200.0
    battery_kw:       float = 100.0    # power rating (= battery_kwh × c_rate)
    c_rate:           float = 0.5
    md_target_kva:    float = 0.0      # 0 = auto-set from first observed peak × target_pct
    peak_target_pct:  float = 0.85     # aim to stay below this fraction of initial peak
    soc_init_pct:     float = 0.50

    # Running state — updated per push
    soc_kwh:              float = 0.0
    billing_month:        str   = ''       # 'YYYY-MM' of current billing month
    month_peak_kva:       float = 0.0     # highest MD-measuring kVA this billing month
    initialised:          bool  = False

    # Forecast from Predictor (48 steps ahead), refreshed each push
    forecast_kw:  list = field(default_factory=list)   # median kW predictions
    forecast_ts:  list = field(default_factory=list)   # ISO timestamps

    # Rolling history (last 48 intervals = 24 h)
    history:      list = field(default_factory=list)

    # Full dispatch log for this session
    dispatch_log: list = field(default_factory=list)


# One live session per session_id key
_LIVE_SESSIONS: dict[str, LiveSession] = {}


def _get_session(session_id: str) -> LiveSession:
    if session_id not in _LIVE_SESSIONS:
        _LIVE_SESSIONS[session_id] = LiveSession()
    return _LIVE_SESSIONS[session_id]


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-INTERVAL TNB-AWARE DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

_INTERVAL_H        = 0.5   # 30-minute intervals
_BAT_DISCHARGE_MIN = 0.15  # never discharge below 15% SOC
_BAT_CHARGE_FULL   = 0.90  # stop normal charging above 90%
_BAT_EMERGENCY_SOC = 0.15  # emergency charging kicks in below 15%
_CHARGE_GUARD_PCT  = 0.95  # charging must not push kVA above 95% of MD target


def dispatch_interval(
    ts: pd.Timestamp,
    kw_net: float,
    kvar_net: float,
    soc_kwh: float,
    bat_kwh: float,
    bat_kw: float,
    tariff: str,
    md_target_kva: float,
    forecast_kw: list | None = None,
    lookahead_steps: int = 4,
) -> dict:
    """
    Run one 30-min dispatch interval with TNB billing rules.

    Returns a dict with kw_managed, kvar_managed, kva_managed, soc_kwh,
    battery_charge_kw, battery_discharge_kw, is_peak, md_counting,
    actions, and notes.

    Strategy:
    - If in MD-measuring window AND kva >= md_target → discharge to protect MD.
    - Look-ahead: if any of the next `lookahead_steps` forecast intervals will
      exceed md_target, pre-discharge now even if current kVA is below target.
    - If in preferred charge window AND SOC < 90% AND charging won't breach
      md_target → charge at full rate (fill up during off-peak / cheap energy).
    - If NOT in preferred charge window but SOC is critical (<15%) → emergency charge.
    """
    bat_min_kwh       = bat_kwh * _BAT_DISCHARGE_MIN
    bat_full_kwh      = bat_kwh * _BAT_CHARGE_FULL
    bat_emergency_kwh = bat_kwh * _BAT_EMERGENCY_SOC

    is_peak    = is_tnb_peak(ts, tariff)
    is_md_tick = is_peak  # for TOU tariffs MD counts only during peak; for flat C1/D always True
    prefer_chg = preferred_charge_window(ts, tariff)

    kw   = kw_net
    kvar = kvar_net
    kva  = calc_kva(kw, kvar)

    bat_chg_kw = 0.0
    bat_dis_kw = 0.0
    actions    = []

    # ── Look-ahead peak flag ──────────────────────────────────────────────────
    # If the next N forecast steps are expected to exceed md_target, start
    # discharging now to build down kVA before the spike arrives.
    upcoming_spike = False
    if forecast_kw and md_target_kva > 0:
        upcoming_spike = any(
            fkw >= md_target_kva
            for fkw in forecast_kw[:lookahead_steps]
        )

    # ── STEP 1: Battery discharge ─────────────────────────────────────────────
    # Discharge when:
    #   (a) This is an MD-measuring interval (protects monthly peak), AND
    #   (b) Current kVA is at or above the MD target (or a spike is coming), AND
    #   (c) SOC is above the minimum floor.
    should_discharge = (
        is_md_tick
        and (kva >= md_target_kva or upcoming_spike)
        and soc_kwh > bat_min_kwh
        and md_target_kva > 0
    )
    if should_discharge:
        # How much kW reduction do we need?  Decompose along the power-factor axis.
        needed_kva    = max(0.0, kva - md_target_kva)
        pf            = (kw / kva) if kva > 1e-3 else 1.0
        needed_kw_red = needed_kva * pf
        avail_kwh     = soc_kwh - bat_min_kwh

        dis_kwh = min(
            needed_kw_red * _INTERVAL_H,   # energy to close the kVA gap
            bat_kw * _INTERVAL_H,           # C-rate cap
            avail_kwh,                      # what's available
        )
        dis_kw    = dis_kwh / _INTERVAL_H
        soc_kwh  -= dis_kwh
        kw       -= dis_kw
        kva       = calc_kva(kw, kvar)
        bat_dis_kw = dis_kw

        trigger = 'look-ahead' if (upcoming_spike and calc_kva(kw_net, kvar_net) < md_target_kva) else 'kVA ≥ target'
        actions.append({
            'type': 'battery_discharge',
            'discharge_kw': round(dis_kw, 2),
            'kva_before': round(calc_kva(kw_net, kvar_net), 2),
            'kva_after': round(kva, 2),
            'soc_pct': round(soc_kwh / bat_kwh * 100, 1),
            'trigger': trigger,
            'note': f'Discharged {dis_kw:.1f} kW ({trigger}) → kVA {calc_kva(kw_net,kvar_net):.1f}→{kva:.1f}',
        })

    # ── STEP 2: Battery charge ────────────────────────────────────────────────
    # Charge when NOT discharging AND:
    #   (A) preferred_charge_window (off-peak for TOU, night for flat): fill up.
    #   (B) Emergency: SOC < 15%, charge regardless of time (with higher kVA ceiling).
    if bat_dis_kw == 0:
        emergency = soc_kwh < bat_emergency_kwh
        is_offpeak_charge = prefer_chg and soc_kwh < bat_full_kwh
        is_emergency_charge = emergency

        if is_offpeak_charge or is_emergency_charge:
            # Ceiling: charging must not push kVA above the charge guard threshold.
            # For emergency, allow up to 100% of md_target; normal = 95%.
            guard_pct   = 1.0 if is_emergency_charge else _CHARGE_GUARD_PCT
            kva_ceiling = (md_target_kva * guard_pct) if md_target_kva > 0 else float('inf')
            kva_headroom = kva_ceiling - kva  # headroom available before we'd breach ceiling

            if kva_headroom > 0.5:   # at least 0.5 kVA headroom before we attempt charging
                room_in_bat = bat_kwh - soc_kwh
                chg_kwh = min(
                    bat_kw * _INTERVAL_H,           # C-rate cap
                    kva_headroom * _INTERVAL_H,     # kVA headroom cap
                    room_in_bat,                    # space in battery
                )
                if chg_kwh > 0.01:
                    chg_kw     = chg_kwh / _INTERVAL_H
                    soc_kwh   += chg_kwh
                    kw        += chg_kw              # charger draws real kW
                    kva        = calc_kva(kw, kvar)
                    bat_chg_kw = chg_kw

                    reason = 'emergency (SOC<15%)' if is_emergency_charge else (
                        'off-peak preferred window' if tariff in TOU_TARIFFS else 'night-time window'
                    )
                    actions.append({
                        'type': 'battery_charge',
                        'charge_kw': round(chg_kw, 2),
                        'kva_after': round(kva, 2),
                        'soc_pct': round(soc_kwh / bat_kwh * 100, 1),
                        'reason': reason,
                        'note': f'Charged {chg_kw:.1f} kW ({reason}) → SOC {soc_kwh/bat_kwh*100:.0f}%',
                    })

    soc_kwh = max(bat_kwh * 0.05, min(bat_kwh, soc_kwh))

    return {
        'kw_original':        round(kw_net, 3),
        'kvar_original':      round(kvar_net, 3),
        'kva_original':       round(calc_kva(kw_net, kvar_net), 3),
        'kw_managed':         round(kw, 3),
        'kvar_managed':       round(kvar, 3),
        'kva_managed':        round(kva, 3),
        'battery_charge_kw':  round(bat_chg_kw, 3),
        'battery_discharge_kw': round(bat_dis_kw, 3),
        'battery_action_kw':  round(bat_dis_kw - bat_chg_kw, 3),
        'battery_soc_kwh':    round(soc_kwh, 2),
        'battery_soc_pct':    round(soc_kwh / bat_kwh * 100, 1),
        'is_peak_hour':       bool(is_peak),
        'md_counting':        bool(is_md_tick),
        'prefer_charge':      bool(prefer_chg),
        'md_target_kva':      round(md_target_kva, 2),
        'actions':            actions,
    }, soc_kwh


# ══════════════════════════════════════════════════════════════════════════════
#  TNB-AWARE BATCH DISPATCH  (replaces run_ai_manager for historical datasets)
# ══════════════════════════════════════════════════════════════════════════════

def run_ai_manager_tnb(
    df: pd.DataFrame,
    tariff: str = 'C1',
    battery_kwh: float = 200.0,
    c_rate: float = 0.5,
    md_target_pct: float = 0.85,
    initial_soc_pct: float = 0.50,
    peak_reference_kva: float | None = None,
    lookahead_intervals: int = 4,
    proportions: dict | None = None,
    priority_order: list | None = None,
    max_cut_pct: dict | None = None,
) -> list[dict]:
    """
    TNB-billing-aware batch dispatch over a historical DataFrame.

    Key differences from run_ai_manager (v4):
    - MD is tracked across the FULL BILLING MONTH (not reset daily).
    - MD target is set once per billing month; adapts upward if breached.
    - For TOU tariffs: battery charges at FULL rate during off-peak windows
      (cheap energy, no MD risk).  No charging during peak hours unless emergency.
    - For flat tariffs (C1/D): charges at night; avoids raising daytime kVA.
    - MD measurement: TOU = peak hours only (08:00–22:00 Mon–Sat excl. PH).
                      Flat = 24 × 7.
    """
    if proportions is None:
        proportions = {'ev': 0.30, 'hvac': 0.40, 'misc': 0.30}
    if priority_order is None:
        priority_order = ['misc', 'hvac', 'ev']
    if max_cut_pct is None:
        max_cut_pct = {'ev': 0.10, 'hvac': 0.15, 'misc': 0.20}

    bat_kw = battery_kwh * c_rate
    soc    = battery_kwh * initial_soc_pct

    # Normalise proportions
    tot = sum(proportions.values()) or 1.0
    ev_p   = proportions.get('ev',   0.30) / tot
    hvac_p = proportions.get('hvac', 0.40) / tot
    misc_p = proportions.get('misc', 0.30) / tot

    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['_month'] = df['timestamp'].dt.to_period('M').astype(str)

    # MD target: per billing month.  Seed it from the first month's data or
    # from a user-supplied absolute ceiling.
    billing_month_targets: dict[str, float] = {}
    billing_month_peaks: dict[str, float]   = {}

    # Pre-compute each month's reference peak.
    # For the FIRST month, use the actual month's MD × 1/peak_target_pct
    # (i.e., what MD the site currently runs, and we'll try to beat it).
    # For subsequent months, carry over the managed peak from the prior month.
    months_ordered = sorted(df['_month'].unique())
    if peak_reference_kva is not None:
        for m in months_ordered:
            billing_month_targets[m] = float(peak_reference_kva) * md_target_pct
    else:
        # Bootstrap: use each month's own peak × target_pct as the target.
        # In live use the prior month's managed peak is a better seed.
        for m in months_ordered:
            m_kva = df.loc[df['_month'] == m, 'kva'].max()
            billing_month_targets[m] = m_kva * md_target_pct

    results = []

    for idx, row in df.iterrows():
        ts    = row['timestamp']
        month = row['_month']

        md_target = billing_month_targets[month]

        # Build a short look-ahead from the next N rows of the SAME dataset
        ahead = df.loc[
            (df.index > idx) & (df.index <= idx + lookahead_intervals),
            'kva'
        ].tolist()

        result, soc = dispatch_interval(
            ts             = ts,
            kw_net         = float(row['kw_net']),
            kvar_net       = float(row['kvar_net']),
            soc_kwh        = soc,
            bat_kwh        = battery_kwh,
            bat_kw         = bat_kw,
            tariff         = tariff,
            md_target_kva  = md_target,
            forecast_kw    = ahead,
            lookahead_steps= lookahead_intervals,
        )

        # ── Load reduction (if battery alone didn't close the gap) ────────────
        remaining_gap = max(0.0, result['kva_managed'] - md_target)
        kw   = result['kw_managed']
        kvar = result['kvar_managed']
        kva  = result['kva_managed']

        ev_kva   = kva * ev_p
        hvac_kva = kva * hvac_p
        misc_kva = kva * misc_p
        ev_f = hvac_f = misc_f = 1.0

        if remaining_gap > 1e-2 and result['is_peak_hour']:
            for load in priority_order:
                if remaining_gap <= 1e-2:
                    break
                if load == 'misc' and misc_kva > 0:
                    cap = misc_kva * max_cut_pct.get('misc', 0.20)
                    cut = min(remaining_gap, cap)
                    misc_f = 1.0 - cut / misc_kva
                    pf = kw / kva if kva > 1e-3 else 1.0
                    qf = kvar / kva if kva > 1e-3 else 0.0
                    kw -= cut * pf; kvar -= cut * qf
                    kva = calc_kva(kw, kvar); remaining_gap = max(0.0, kva - md_target)
                    result['actions'].append({'type': 'load_reduction', 'load': 'Miscellaneous',
                        'cut_kva': round(cut, 2), 'note': f'Misc cut {cut:.1f} kVA → {kva:.1f}'})
                elif load == 'hvac' and hvac_kva > 0:
                    cap = hvac_kva * max_cut_pct.get('hvac', 0.15)
                    cut = min(remaining_gap, cap)
                    hvac_f = 1.0 - cut / hvac_kva
                    pf = kw / kva if kva > 1e-3 else 1.0
                    qf = kvar / kva if kva > 1e-3 else 0.0
                    kw -= cut * pf; kvar -= cut * qf
                    kva = calc_kva(kw, kvar); remaining_gap = max(0.0, kva - md_target)
                    result['actions'].append({'type': 'load_reduction', 'load': 'HVAC',
                        'cut_kva': round(cut, 2), 'note': f'HVAC cut {cut:.1f} kVA → {kva:.1f}'})
                elif load == 'ev' and ev_kva > 0:
                    cap = ev_kva * max_cut_pct.get('ev', 0.10)
                    cut = min(remaining_gap, cap)
                    ev_f = 1.0 - cut / ev_kva
                    pf = kw / kva if kva > 1e-3 else 1.0
                    qf = kvar / kva if kva > 1e-3 else 0.0
                    kw -= cut * pf; kvar -= cut * qf
                    kva = calc_kva(kw, kvar); remaining_gap = max(0.0, kva - md_target)
                    result['actions'].append({'type': 'load_reduction', 'load': 'EV Charger',
                        'cut_kva': round(cut, 2), 'note': f'EV cut {cut:.1f} kVA → {kva:.1f}'})

            result['kw_managed']   = round(kw, 3)
            result['kvar_managed'] = round(kvar, 3)
            result['kva_managed']  = round(kva, 3)

        # ── Update billing-month running MD ───────────────────────────────────
        if result['md_counting']:
            prev = billing_month_peaks.get(month, 0.0)
            if kva > prev:
                billing_month_peaks[month] = kva
                # If we breached the target (battery couldn't prevent it),
                # raise the target for this month to the new actual peak.
                # Battery will now protect THIS new level for the rest of the month.
                if kva > md_target:
                    billing_month_targets[month] = kva

        result.update({
            'timestamp':          ts.isoformat(),
            'billing_month':      month,
            'month_peak_kva_so_far': round(billing_month_peaks.get(month, 0.0), 2),
            'month_md_target':    round(billing_month_targets[month], 2),
            'ev_kva':             round(ev_kva, 2),
            'hvac_kva':           round(hvac_kva, 2),
            'misc_kva':           round(misc_kva, 2),
            'ev_factor':          round(ev_f, 3),
            'hvac_factor':        round(hvac_f, 3),
            'misc_factor':        round(misc_f, 3),
        })
        results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  WHAT-IF SIMULATOR  (solar + battery sizing)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_solar_battery(
    df: pd.DataFrame,
    solar_kwp: float,
    battery_kwh: float,
    battery_kw: float | None = None,
    tariff: str = 'C1',
    psh: float = 4.5,
    md_target_pct: float = 0.85,
    initial_soc_pct: float = 0.50,
    c_rate: float = 0.5,
    nem_rate: float = NEM_RATE_DEFAULT,
) -> dict:
    """
    Simulate the grid-side load profile after adding solar PV and/or BESS.

    Energy flow per 30-min interval (in order):
      1. Solar generation reduces the gross load first.
      2. Excess solar (generation > load) charges the battery if space available.
      3. The residual net load (after solar but before battery dispatch) is
         passed through the TNB-aware dispatcher.
      4. Battery discharges to protect MD target; also charges from grid during
         preferred windows if not already full from solar.
      5. Any solar surplus after load + battery charging → NEM grid export.

    Returns:
      'profile'   : per-interval DataFrame
      'baseline'  : bill without any changes (original load)
      'simulated' : bill after solar + battery
      'savings'   : key financial metrics
      'monthly'   : per-month before/after summary
    """
    if battery_kw is None:
        battery_kw = battery_kwh * c_rate

    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['_month'] = df['timestamp'].dt.to_period('M').astype(str)

    # ── Solar generation ──────────────────────────────────────────────────────
    df['solar_kw'] = generate_solar_kw(df['timestamp'], solar_kwp, psh).values

    # ── Per-interval simulation ───────────────────────────────────────────────
    soc = battery_kwh * initial_soc_pct
    bat_min = battery_kwh * _BAT_DISCHARGE_MIN

    # Pre-compute MD targets per month (based on BASELINE load)
    months = sorted(df['_month'].unique())
    md_targets: dict[str, float] = {}
    for m in months:
        m_kva = df.loc[df['_month'] == m, 'kva'].max()
        md_targets[m] = m_kva * md_target_pct

    rows = []

    for _, row in df.iterrows():
        ts       = row['timestamp']
        month    = row['_month']
        kw_orig  = float(row['kw_net'])
        kvar     = float(row['kvar_net'])
        solar_kw = float(row['solar_kw'])

        # 1. Solar offsets gross load
        net_after_solar = kw_orig - solar_kw
        solar_excess    = max(0.0, -net_after_solar)   # solar > load
        net_load        = max(0.0, net_after_solar)     # remaining demand on grid+battery

        # 2. Excess solar → charge battery (if space available)
        bat_chg_solar = 0.0
        if solar_excess > 0 and soc < battery_kwh:
            room         = battery_kwh - soc
            chg_kwh      = min(solar_excess * _INTERVAL_H, battery_kw * _INTERVAL_H, room)
            bat_chg_solar = chg_kwh / _INTERVAL_H
            soc          += chg_kwh
            solar_excess -= bat_chg_solar   # remaining excess goes to export

        # 3. TNB-aware battery dispatch on the net load
        md_target = md_targets[month]
        res, soc = dispatch_interval(
            ts             = ts,
            kw_net         = net_load + bat_chg_solar,  # kW seen at meter after solar
            kvar_net       = kvar,
            soc_kwh        = soc,
            bat_kwh        = battery_kwh,
            bat_kw         = battery_kw,
            tariff         = tariff,
            md_target_kva  = md_target,
            lookahead_steps= 0,    # no explicit forecast in batch mode
        )

        # Reduce solar charging from result to avoid double-counting
        net_grid_import = max(0.0, res['kw_managed'])
        nem_export_kw   = max(0.0, solar_excess)   # solar surplus not absorbed by battery

        rows.append({
            'timestamp':             ts.isoformat(),
            'billing_month':         month,
            'kw_original':           round(kw_orig, 3),
            'kvar_original':         round(kvar, 3),
            'kva_original':          round(calc_kva(kw_orig, kvar), 3),
            'solar_gen_kw':          round(solar_kw, 3),
            'solar_to_load_kw':      round(min(solar_kw, kw_orig), 3),
            'solar_to_battery_kw':   round(bat_chg_solar, 3),
            'solar_to_export_kw':    round(nem_export_kw, 3),
            'net_load_after_solar_kw': round(net_load, 3),
            'battery_charge_kw':     round(res['battery_charge_kw'] - bat_chg_solar, 3),  # grid charge only
            'battery_charge_solar_kw': round(bat_chg_solar, 3),
            'battery_discharge_kw':  round(res['battery_discharge_kw'], 3),
            'battery_soc_kwh':       round(soc, 2),
            'battery_soc_pct':       round(soc / battery_kwh * 100, 1),
            'kw_managed':            round(net_grid_import, 3),
            'kvar_managed':          round(res['kvar_managed'], 3),
            'kva_managed':           round(calc_kva(net_grid_import, res['kvar_managed']), 3),
            'nem_export_kw':         round(nem_export_kw, 3),
            'is_peak_hour':          res['is_peak_hour'],
            'md_counting':           res['md_counting'],
        })

    profile_df = pd.DataFrame(rows)

    # ── Monthly bill comparison ───────────────────────────────────────────────
    monthly_summary = []
    total_baseline = {'energy_rm': 0, 'md_rm': 0, 'gross_rm': 0, 'net_rm': 0}
    total_simulated = {'energy_rm': 0, 'md_rm': 0, 'gross_rm': 0, 'net_rm': 0}

    for m in months:
        mdf_base = df[df['_month'] == m]
        mdf_sim  = profile_df[profile_df['billing_month'] == m]

        # Baseline
        base_kwh  = float((mdf_base['kw_net'] * _INTERVAL_H).sum())
        base_peak = float(mdf_base['peak_kwh'] if 'peak_kwh' in mdf_base.columns else 0)
        is_tou    = tariff in TOU_TARIFFS

        if is_tou:
            base_peak_kwh   = float(mdf_base.loc[mdf_base['timestamp'].apply(lambda t: is_tnb_peak(t, tariff)), 'kw_net'].sum() * _INTERVAL_H)
            base_offpk_kwh  = base_kwh - base_peak_kwh
        else:
            base_peak_kwh  = base_kwh
            base_offpk_kwh = 0.0

        if is_tou:
            _peak_mask = mdf_base['timestamp'].apply(lambda t: is_tnb_peak(t, tariff))
            _peak_kva_series = mdf_base.loc[_peak_mask, 'kva']
            base_md_kva = float(_peak_kva_series.max()) if len(_peak_kva_series) else 0.0
        else:
            base_md_kva = float(mdf_base['kva'].max())
        base_md_kva = 0.0 if (base_md_kva != base_md_kva) else base_md_kva  # NaN guard
        base_bill = tnb_bill_quick(base_kwh, base_peak_kwh, base_offpk_kwh, base_md_kva, tariff)

        # Simulated
        sim_kwh = float((mdf_sim['kw_managed'] * _INTERVAL_H).sum())
        if is_tou:
            sim_peak_kwh  = float(mdf_sim.loc[mdf_sim['is_peak_hour'], 'kw_managed'].sum() * _INTERVAL_H)
            sim_offpk_kwh = sim_kwh - sim_peak_kwh
        else:
            sim_peak_kwh  = sim_kwh
            sim_offpk_kwh = 0.0

        if is_tou:
            _sim_peak = mdf_sim.loc[mdf_sim['is_peak_hour'], 'kva_managed']
            sim_md_kva = float(_sim_peak.max()) if len(_sim_peak) else 0.0
        else:
            sim_md_kva = float(mdf_sim['kva_managed'].max())
        sim_md_kva = 0.0 if (sim_md_kva != sim_md_kva) else sim_md_kva  # NaN guard
        sim_export_kwh = float((mdf_sim['nem_export_kw'] * _INTERVAL_H).sum())
        sim_bill = tnb_bill_quick(sim_kwh, sim_peak_kwh, sim_offpk_kwh, sim_md_kva,
                                  tariff, sim_export_kwh, nem_rate)

        saving_rm = base_bill['net_rm'] - sim_bill['net_rm']
        monthly_summary.append({
            'month':                   m,
            'baseline_kwh':            round(base_kwh, 1),
            'baseline_md_kva':         round(base_md_kva, 1),
            'baseline_bill_rm':        round(base_bill['net_rm'], 2),
            'sim_kwh':                 round(sim_kwh, 1),
            'sim_md_kva':              round(sim_md_kva, 1),
            'sim_solar_gen_kwh':       round(float((mdf_sim['solar_gen_kw'] * _INTERVAL_H).sum()), 1),
            'sim_nem_export_kwh':      round(sim_export_kwh, 1),
            'sim_nem_credit_rm':       round(sim_bill['nem_credit_rm'], 2),
            'sim_bill_rm':             round(sim_bill['net_rm'], 2),
            'saving_rm':               round(saving_rm, 2),
            'md_reduction_kva':        round(base_md_kva - sim_md_kva, 1),
            'md_reduction_pct':        round((base_md_kva - sim_md_kva) / base_md_kva * 100, 1) if base_md_kva else 0,
            'energy_reduction_kwh':    round(base_kwh - sim_kwh, 1),
        })

        for k in total_baseline:
            total_baseline[k] += base_bill.get(k, 0)
        for k in total_simulated:
            total_simulated[k] += sim_bill.get(k, 0)

    total_saving = total_baseline['net_rm'] - total_simulated['net_rm']
    self_consumption = 0.0
    total_solar_gen  = float((profile_df['solar_gen_kw'] * _INTERVAL_H).sum())
    total_solar_load = float((profile_df['solar_to_load_kw'] * _INTERVAL_H).sum())
    if total_solar_gen > 0:
        self_consumption = total_solar_load / total_solar_gen * 100.0

    return {
        'profile':    profile_df.to_dict(orient='records'),
        'monthly':    monthly_summary,
        'baseline':   total_baseline,
        'simulated':  total_simulated,
        'savings': {
            'annual_saving_rm':        round(total_saving * 12 / max(len(months), 1), 2),
            'total_saving_rm':         round(total_saving, 2),
            'avg_monthly_saving_rm':   round(total_saving / max(len(months), 1), 2),
            'solar_kwp':               solar_kwp,
            'battery_kwh':             battery_kwh,
            'psh':                     psh,
            'total_solar_gen_kwh':     round(total_solar_gen, 1),
            'self_consumption_pct':    round(self_consumption, 1),
            'total_nem_export_kwh':    round(float((profile_df['nem_export_kw'] * _INTERVAL_H).sum()), 1),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY HELPERS  (v4 — unchanged, kept for /api/upload & /api/optimize)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_cols(df):
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace('﻿', '', regex=False)
                  .str.replace(r'\s+', ' ', regex=True))
    return df


def _is_valid_header_row(cols):
    col_str = ' '.join(cols)
    has_time  = any(t in col_str for t in ('date', 'time', 'start', 'end', 'timestamp', 'datetime'))
    has_power = any(t in col_str for t in ('kw', 'kvar', 'kwh', 'power', 'watt', 'energy', 'var'))
    return has_time and has_power


def _find_datetime_col(df):
    cols = list(df.columns)
    for col in cols:
        if ('date' in col and 'time' in col) or 'datetime' in col or 'timestamp' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    for col in cols:
        if col in ('end_time', 'end time'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    for col in cols:
        if col in ('start_time', 'start time'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    date_col = next((c for c in cols if c == 'date' or c.startswith('date')), None)
    time_col = next((c for c in cols if c in ('time', 'end time', 'end_time', 'start_time') and c != date_col), None)
    if date_col and time_col:
        combined = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
        df['_dt'] = combined
        return '_dt', df
    for col in cols[:10]:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() / max(len(parsed), 1) >= 0.8:
                df[col] = parsed
                return col, df
        except Exception:
            continue
    raise ValueError(f"No date/time column found. Columns: {cols}")


def _map_power_cols(df):
    col_map = {}
    for col in df.columns:
        is_kvar = 'kvar' in col or ('var' in col and 'kw' not in col)
        is_kw   = ('kw' in col or 'kwh' in col or 'watt' in col or 'active' in col
                   or 'power' in col or 'energy' in col) and not is_kvar
        is_import = 'import' in col
        is_export = 'export' in col
        if is_kvar and is_import and 'kvar_import' not in col_map:   col_map['kvar_import'] = col
        elif is_kvar and is_export and 'kvar_export' not in col_map: col_map['kvar_export'] = col
        elif is_kw   and is_import and 'kw_import'   not in col_map: col_map['kw_import']   = col
        elif is_kw   and is_export and 'kw_export'   not in col_map: col_map['kw_export']   = col
    for key in ('kw_import', 'kw_export', 'kvar_import', 'kvar_export'):
        col_map.setdefault(key, None)
    return col_map


def parse_uploaded_data(file_content, filename):
    try:
        df = None
        if filename.lower().endswith('.csv'):
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            for hrow in range(12):
                for enc in encodings:
                    try:
                        tmp = pd.read_csv(StringIO(file_content.decode(enc)), header=hrow)
                        tmp = _normalize_cols(tmp)
                        if len(tmp.columns) >= 4 and _is_valid_header_row(list(tmp.columns)):
                            df = tmp
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
            if df is None:
                raise ValueError("Could not locate a valid header row in CSV.")
        elif filename.lower().endswith(('.xlsx', '.xls')):
            import io
            engine = 'xlrd' if filename.lower().endswith('.xls') else 'openpyxl'
            fb = io.BytesIO(file_content)
            for hrow in range(12):
                try:
                    fb.seek(0)
                    tmp = pd.read_excel(fb, header=hrow, engine=engine)
                    tmp = _normalize_cols(tmp)
                    if len(tmp.columns) >= 4 and _is_valid_header_row(list(tmp.columns)):
                        df = tmp
                        break
                except Exception:
                    continue
            if df is None:
                raise ValueError("Could not locate a valid header row in Excel.")
        else:
            raise ValueError(f"Unsupported format '{filename}'. Use CSV or Excel.")

        df = df.dropna(how='all').reset_index(drop=True)
        df = df.loc[:, df.columns.notna()]
        df = df.loc[:, df.columns.astype(str) != '']

        start_col, df = _find_datetime_col(df)
        df = df.dropna(subset=[start_col]).reset_index(drop=True)
        df = df.sort_values(start_col).reset_index(drop=True)

        col_map = _map_power_cols(df)

        def _s(key):
            col = col_map.get(key)
            if col and col in df.columns:
                return pd.to_numeric(df[col], errors='coerce').fillna(0)
            return pd.Series([0.0] * len(df), index=df.index)

        result = pd.DataFrame()
        result['timestamp']   = df[start_col].values
        result['kw_import']   = _s('kw_import').values
        result['kw_export']   = _s('kw_export').values
        result['kvar_import'] = _s('kvar_import').values
        result['kvar_export'] = _s('kvar_export').values
        result['kw_net']   = result['kw_import'] - result['kw_export']
        result['kvar_net'] = result['kvar_import'] - result['kvar_export']
        result['kva']      = result.apply(lambda r: calc_kva(r['kw_net'], r['kvar_net']), axis=1)

        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result = result.set_index('timestamp').resample('30min').mean().dropna(how='all').reset_index()
        result['kva'] = result.apply(lambda r: calc_kva(r['kw_net'], r['kvar_net']), axis=1)
        if result.empty:
            raise ValueError("Empty after resampling.")
        return result
    except Exception as e:
        raise ValueError(f"Data parsing error: {str(e)}")


def run_ai_manager(df, proportions, battery_capacity_kwh, priority_order,
                   peak_target_pct, max_cut_pct, bat_charge_upper_pct,
                   bat_charge_lower_pct, c_rate=0.5, initial_soc_pct=0.50,
                   peak_reference_kva=None, lookahead_intervals=3):
    """Legacy v4 rule-based dispatch (day-rolling reference peak). Kept for /api/optimize."""
    results    = []
    INTERVAL_H = 0.5

    BAT_EMERGENCY_PCT        = 0.15
    BAT_CHARGE_FULL_PCT      = 0.90
    BAT_DISCHARGE_MIN_PCT    = 0.15
    DISCHARGE_PROXIMITY_PCT  = 0.80
    CHARGE_PEAK_GUARD_PCT    = 0.92
    EMERGENCY_PEAK_GUARD_PCT = 1.00

    ev_p   = proportions.get('ev',   0.3)
    hvac_p = proportions.get('hvac', 0.4)
    misc_p = proportions.get('misc', 0.3)
    tot = ev_p + hvac_p + misc_p or 1
    ev_p /= tot; hvac_p /= tot; misc_p /= tot

    df = df.copy()
    df['date'] = df['timestamp'].dt.date

    bat_max           = battery_capacity_kwh
    bat_min_abs       = battery_capacity_kwh * 0.05
    bat_emergency_abs = battery_capacity_kwh * BAT_EMERGENCY_PCT
    bat_full          = battery_capacity_kwh * BAT_CHARGE_FULL_PCT
    bat_dis_min       = battery_capacity_kwh * BAT_DISCHARGE_MIN_PCT
    chg_rate_kw       = battery_capacity_kwh * c_rate
    dis_rate_kw       = battery_capacity_kwh * c_rate
    bat_soc           = battery_capacity_kwh * initial_soc_pct

    df_sorted = df.sort_values('timestamp')
    if peak_reference_kva is not None:
        df['_ref_peak'] = float(peak_reference_kva)
    else:
        daily_max = df_sorted.groupby('date')['kva'].max().sort_index()
        rolling_ref = daily_max.shift(1).rolling(30, min_periods=1).max()
        first_day_fallback = daily_max.iloc[0] * 1.10
        rolling_ref = rolling_ref.fillna(first_day_fallback)
        ref_map = rolling_ref.to_dict()
        df['_ref_peak'] = df['date'].map(ref_map)

    for date_val, day_df in df.groupby('date'):
        day_df = day_df.sort_values('timestamp').reset_index(drop=True)
        ref_peak             = day_df['_ref_peak'].iloc[0]
        day_peak             = day_df['kva'].max()
        discharge_trigger    = ref_peak * peak_target_pct
        discharge_proximity  = ref_peak * DISCHARGE_PROXIMITY_PCT
        charge_upper         = ref_peak * bat_charge_upper_pct
        charge_lower         = ref_peak * bat_charge_lower_pct
        charge_kva_ceiling   = discharge_trigger * CHARGE_PEAK_GUARD_PCT
        emergency_kva_ceiling= discharge_trigger * EMERGENCY_PEAK_GUARD_PCT
        lookahead_high = set(day_df.index[day_df['kva'] >= ref_peak * 0.90].tolist())

        for idx, row in day_df.iterrows():
            kva_orig = row['kva']
            kw       = row['kw_net']
            kvar     = row['kvar_net']

            ev_kva   = kva_orig * ev_p
            hvac_kva = kva_orig * hvac_p
            misc_kva = kva_orig * misc_p

            ev_f = hvac_f = misc_f = 1.0
            bat_chg_kw = bat_dis_kw = 0.0
            actions = []

            mgd_kw   = kw
            mgd_kvar = kvar
            mgd_kva  = calc_kva(mgd_kw, mgd_kvar)

            upcoming_high = any((idx + k) in lookahead_high for k in range(1, lookahead_intervals + 1))
            in_peak_zone  = kva_orig >= discharge_proximity or upcoming_high
            soc_ok_dis    = bat_soc > bat_dis_min

            if in_peak_zone and mgd_kva >= discharge_trigger and soc_ok_dis:
                needed_kva         = mgd_kva - discharge_trigger
                avail_kwh          = bat_soc - bat_dis_min
                pf                 = (mgd_kw / mgd_kva) if mgd_kva > 0 else 1.0
                needed_kw_reduction= needed_kva * pf
                dis_kwh = min(needed_kw_reduction * INTERVAL_H, dis_rate_kw * INTERVAL_H, avail_kwh)
                dis_kw  = dis_kwh / INTERVAL_H
                soc_before = bat_soc
                bat_soc   -= dis_kwh
                mgd_kw    -= dis_kw
                mgd_kva    = calc_kva(mgd_kw, mgd_kvar)
                mgd_kva    = max(mgd_kva, 0.0)
                bat_dis_kw = dis_kw
                actions.append({
                    'type': 'battery_discharge', 'load': 'Battery',
                    'discharge_kw': round(dis_kw, 2),
                    'soc_before_kwh': round(soc_before, 1), 'soc_after_kwh': round(bat_soc, 1),
                    'kva_before': round(kva_orig, 2), 'kva_after': round(mgd_kva, 2),
                    'lookahead_triggered': bool(upcoming_high and kva_orig < discharge_proximity),
                    'note': (f'Battery discharged {round(dis_kw,1)} kW → kVA {round(kva_orig,1)}→{round(mgd_kva,1)}'),
                })

            remaining_cut = max(0.0, mgd_kva - discharge_trigger)
            if remaining_cut > 1e-3:
                for load in priority_order:
                    if remaining_cut <= 1e-3:
                        break
                    if load == 'misc' and misc_kva > 0:
                        max_possible = misc_kva * max_cut_pct.get('misc', 0.20)
                        cut_kva = min(remaining_cut, max_possible)
                        misc_f  = 1.0 - cut_kva / misc_kva
                        pf_misc = mgd_kw / mgd_kva if mgd_kva > 0 else 1.0
                        qf_misc = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                        mgd_kw -= cut_kva * pf_misc; mgd_kvar -= cut_kva * qf_misc
                        mgd_kva = calc_kva(mgd_kw, mgd_kvar)
                        remaining_cut = max(0.0, mgd_kva - discharge_trigger)
                        if cut_kva > 0.05:
                            actions.append({'type': 'load_reduction', 'load': 'Miscellaneous',
                                'cut_kva': round(cut_kva, 2), 'factor_pct': round(misc_f * 100, 1),
                                'note': f'Misc cut {round(cut_kva,1)} kVA'})
                    elif load == 'hvac' and hvac_kva > 0:
                        max_possible = hvac_kva * max_cut_pct.get('hvac', 0.15)
                        cut_kva = min(remaining_cut, max_possible)
                        hvac_f  = 1.0 - cut_kva / hvac_kva
                        pf_hvac = mgd_kw / mgd_kva if mgd_kva > 0 else 1.0
                        qf_hvac = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                        mgd_kw -= cut_kva * pf_hvac; mgd_kvar -= cut_kva * qf_hvac
                        mgd_kva = calc_kva(mgd_kw, mgd_kvar)
                        remaining_cut = max(0.0, mgd_kva - discharge_trigger)
                        if cut_kva > 0.05:
                            actions.append({'type': 'load_reduction', 'load': 'HVAC',
                                'cut_kva': round(cut_kva, 2), 'factor_pct': round(hvac_f * 100, 1),
                                'note': f'HVAC cut {round(cut_kva,1)} kVA'})
                    elif load == 'ev' and ev_kva > 0:
                        max_possible = ev_kva * max_cut_pct.get('ev', 0.10)
                        cut_kva = min(remaining_cut, max_possible)
                        ev_f    = 1.0 - cut_kva / ev_kva
                        pf_ev   = mgd_kw / mgd_kva if mgd_kva > 0 else 1.0
                        qf_ev   = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                        mgd_kw -= cut_kva * pf_ev; mgd_kvar -= cut_kva * qf_ev
                        mgd_kva = calc_kva(mgd_kw, mgd_kvar)
                        remaining_cut = max(0.0, mgd_kva - discharge_trigger)
                        if cut_kva > 0.05:
                            actions.append({'type': 'load_reduction', 'load': 'EV Charger',
                                'cut_kva': round(cut_kva, 2), 'factor_pct': round(ev_f * 100, 1),
                                'note': f'EV cut {round(cut_kva,1)} kVA'})

            ev_m   = ev_kva   * ev_f
            hvac_m = hvac_kva * hvac_f
            misc_m = misc_kva * misc_f

            if bat_dis_kw == 0 and bat_soc < bat_max:
                emergency_charge = bat_soc < bat_emergency_abs
                active_ceiling   = emergency_kva_ceiling if emergency_charge else charge_kva_ceiling
                kva_headroom     = active_ceiling - mgd_kva
                if kva_headroom > 0:
                    normal_charge = (kva_orig < charge_upper) and (bat_soc < bat_full)
                    if emergency_charge or normal_charge:
                        room_in_bat = bat_max - bat_soc
                        chg_kwh = min(chg_rate_kw * INTERVAL_H, kva_headroom * INTERVAL_H, room_in_bat)
                        if chg_kwh > 0.01:
                            soc_before  = bat_soc
                            bat_soc    += chg_kwh
                            bat_chg_kw  = chg_kwh / INTERVAL_H
                            mgd_kw    += bat_chg_kw
                            mgd_kva    = calc_kva(mgd_kw, mgd_kvar)
                            reason = ('Emergency charge' if emergency_charge else
                                      f'Normal charge (kVA {round(kva_orig,0):.0f} < {round(charge_upper,0):.0f})')
                            actions.append({'type': 'battery_charge', 'load': 'Battery',
                                'charge_kw': round(bat_chg_kw, 2),
                                'soc_before_kwh': round(soc_before, 1), 'soc_after_kwh': round(bat_soc, 1),
                                'kva_ceiling': round(active_ceiling, 2), 'kva_after_charge': round(mgd_kva, 2),
                                'charge_trigger': 'emergency' if emergency_charge else 'normal',
                                'note': f'{reason}: +{round(bat_chg_kw,1)} kW'})

            bat_soc    = max(bat_min_abs, min(bat_max, bat_soc))
            bat_action = bat_dis_kw - bat_chg_kw
            ev_kvah_cut   = (ev_kva   - ev_m)   * INTERVAL_H
            hvac_kvah_cut = (hvac_kva - hvac_m) * INTERVAL_H
            misc_kvah_cut = (misc_kva - misc_m) * INTERVAL_H

            results.append({
                'timestamp':              row['timestamp'].isoformat(),
                'date':                   str(date_val),
                'kva_original':           round(kva_orig, 2),
                'kw_original':            round(kw, 2),
                'kvar_original':          round(kvar, 2),
                'kw_managed':             round(mgd_kw, 2),
                'kvar_managed':           round(mgd_kvar, 2),
                'ev_kva':                 round(ev_kva, 2),
                'hvac_kva':              round(hvac_kva, 2),
                'misc_kva':              round(misc_kva, 2),
                'ev_managed':             round(ev_m, 2),
                'hvac_managed':           round(hvac_m, 2),
                'misc_managed':           round(misc_m, 2),
                'ev_factor':              round(ev_f, 3),
                'hvac_factor':            round(hvac_f, 3),
                'misc_factor':            round(misc_f, 3),
                'ev_kvah_cut':            round(ev_kvah_cut, 3),
                'hvac_kvah_cut':          round(hvac_kvah_cut, 3),
                'misc_kvah_cut':          round(misc_kvah_cut, 3),
                'battery_action_kw':      round(bat_action, 2),
                'battery_charge_kw':      round(bat_chg_kw, 2),
                'battery_discharge_kw':   round(bat_dis_kw, 2),
                'battery_soc_kwh':        round(bat_soc, 2),
                'battery_soc_pct':        round(bat_soc / bat_max * 100, 1) if bat_max else 0,
                'kva_managed':            round(mgd_kva, 2),
                'target_peak':            round(discharge_trigger, 2),
                'ref_peak':               round(ref_peak, 2),
                'discharge_proximity':    round(discharge_proximity, 2),
                'charge_kva_ceiling':     round(charge_kva_ceiling, 2),
                'emergency_kva_ceiling':  round(emergency_kva_ceiling, 2),
                'charge_threshold_upper': round(charge_upper, 2),
                'charge_threshold_lower': round(charge_lower, 2),
                'day_peak':               round(day_peak, 2),
                'bat_capacity':           round(bat_max, 2),
                'actions':                actions,
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES — EXISTING (backward compatible)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'manager_index.html')


@app.route('/api/upload', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        f       = request.files['file']
        content = f.read()
        df      = parse_uploaded_data(content, f.filename)
        records = [{'timestamp': row['timestamp'].isoformat(),
                    'kva':       round(row['kva'], 2),
                    'kw_net':    round(row['kw_net'], 2),
                    'kvar_net':  round(row['kvar_net'], 2),
                    'kw_import': round(row['kw_import'], 2),
                    'kw_export': round(row['kw_export'], 2)}
                   for _, row in df.iterrows()]
        dates = sorted(df['timestamp'].dt.date.astype(str).unique().tolist())
        return jsonify({'success': True, 'records': records, 'dates': dates,
                        'total_intervals': len(records),
                        'peak_kva': round(df['kva'].max(), 2),
                        'avg_kva':  round(df['kva'].mean(), 2)})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Legacy v4 route — uses day-rolling reference peak (run_ai_manager)."""
    try:
        data             = request.json
        records          = data.get('records', [])
        proportions      = data.get('proportions', {'ev': 0.3, 'hvac': 0.4, 'misc': 0.3})
        battery_capacity = float(data.get('battery_capacity_kwh', 200))
        priority_order   = data.get('priority_order', ['misc', 'hvac', 'ev'])
        peak_target_pct  = float(data.get('peak_target_pct', 0.85))
        max_cut_pct      = data.get('max_cut_pct', {'ev': 0.10, 'hvac': 0.15, 'misc': 0.20})
        bat_charge_upper = float(data.get('bat_charge_upper_pct', 0.70))
        bat_charge_lower = float(data.get('bat_charge_lower_pct', 0.60))
        c_rate           = float(data.get('c_rate', 0.5))
        initial_soc_pct  = float(data.get('initial_soc_pct', 0.50))
        peak_ref_kva     = data.get('peak_reference_kva', None)
        if peak_ref_kva is not None:
            peak_ref_kva = float(peak_ref_kva)
        lookahead        = int(data.get('lookahead_intervals', 3))
        max_cut_pct = {k: float(v) for k, v in max_cut_pct.items()}

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        results = run_ai_manager(df, proportions, battery_capacity, priority_order,
                                 peak_target_pct, max_cut_pct,
                                 bat_charge_upper, bat_charge_lower,
                                 c_rate=c_rate, initial_soc_pct=initial_soc_pct,
                                 peak_reference_kva=peak_ref_kva,
                                 lookahead_intervals=lookahead)

        day_summaries = {}
        for r in results:
            d = r['date']
            if d not in day_summaries:
                day_summaries[d] = {
                    'orig_peak': 0, 'managed_peak': 0,
                    'total_discharge_kwh': 0, 'total_charge_kwh': 0,
                    'intervals_battery_discharge': 0, 'intervals_battery_charge': 0,
                    'intervals_load_reduced': 0,
                    'ev_total_kvah_cut': 0, 'hvac_total_kvah_cut': 0, 'misc_total_kvah_cut': 0,
                }
            s = day_summaries[d]
            s['orig_peak']    = max(s['orig_peak'],    r['kva_original'])
            s['managed_peak'] = max(s['managed_peak'], r['kva_managed'])
            s['total_discharge_kwh'] += r['battery_discharge_kw'] * 0.5
            s['total_charge_kwh']    += r['battery_charge_kw']    * 0.5
            s['ev_total_kvah_cut']   += r.get('ev_kvah_cut',   0)
            s['hvac_total_kvah_cut'] += r.get('hvac_kvah_cut', 0)
            s['misc_total_kvah_cut'] += r.get('misc_kvah_cut', 0)
            for act in r['actions']:
                if act['type'] == 'battery_discharge': s['intervals_battery_discharge'] += 1
                elif act['type'] == 'battery_charge':  s['intervals_battery_charge']    += 1
                elif act['type'] == 'load_reduction':  s['intervals_load_reduced']      += 1

        for s in day_summaries.values():
            s['ev_total_kvah_cut']   = round(s['ev_total_kvah_cut'],   2)
            s['hvac_total_kvah_cut'] = round(s['hvac_total_kvah_cut'], 2)
            s['misc_total_kvah_cut'] = round(s['misc_total_kvah_cut'], 2)

        overall_orig    = max((s['orig_peak']    for s in day_summaries.values()), default=0)
        overall_managed = max((s['managed_peak'] for s in day_summaries.values()), default=0)
        peak_red = ((overall_orig - overall_managed) / overall_orig * 100) if overall_orig else 0

        return jsonify({
            'success': True, 'results': results, 'day_summaries': day_summaries,
            'summary': {
                'original_peak_kva':           round(overall_orig, 2),
                'managed_peak_kva':            round(overall_managed, 2),
                'peak_reduction_pct':          round(peak_red, 1),
                'total_battery_discharge_kwh': round(sum(s['total_discharge_kwh'] for s in day_summaries.values()), 1),
                'total_battery_charge_kwh':    round(sum(s['total_charge_kwh']    for s in day_summaries.values()), 1),
                'avg_original_kva': round(sum(r['kva_original'] for r in results) / len(results), 2) if results else 0,
                'avg_managed_kva':  round(sum(r['kva_managed']  for r in results) / len(results), 2) if results else 0,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES — NEW v5
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/optimize/tnb', methods=['POST'])
def optimize_tnb():
    """
    TNB-billing-aware batch optimisation.
    Uses run_ai_manager_tnb — tracks billing-month MD, respects peak windows.

    Body (JSON):
      records            : [{timestamp, kw_net, kvar_net, kva, ...}]
      tariff             : 'C1' | 'C2' | 'D' | 'E1' | 'E2'  (default 'C1')
      battery_kwh        : float  (default 200)
      c_rate             : float  (default 0.5)
      md_target_pct      : float  (default 0.85 = try to stay below 85% of baseline MD)
      initial_soc_pct    : float  (default 0.50)
      peak_reference_kva : float | null  (override all months; null = auto)
      lookahead_intervals: int    (default 4)
      proportions        : {ev, hvac, misc}
      priority_order     : ['misc','hvac','ev']
      max_cut_pct        : {ev, hvac, misc}
    """
    try:
        data = request.json

        records      = data.get('records', [])
        tariff       = data.get('tariff', 'C1').upper()
        battery_kwh  = float(data.get('battery_kwh', 200))
        c_rate       = float(data.get('c_rate', 0.5))
        md_tgt_pct   = float(data.get('md_target_pct', 0.85))
        soc_init     = float(data.get('initial_soc_pct', 0.50))
        peak_ref     = data.get('peak_reference_kva', None)
        if peak_ref is not None:
            peak_ref = float(peak_ref)
        lookahead    = int(data.get('lookahead_intervals', 4))
        proportions  = data.get('proportions', {'ev': 0.3, 'hvac': 0.4, 'misc': 0.3})
        priority     = data.get('priority_order', ['misc', 'hvac', 'ev'])
        max_cut      = {k: float(v) for k, v in
                        data.get('max_cut_pct', {'ev': 0.10, 'hvac': 0.15, 'misc': 0.20}).items()}

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'kva' not in df.columns:
            df['kva'] = df.apply(lambda r: calc_kva(float(r.get('kw_net', 0)),
                                                     float(r.get('kvar_net', 0))), axis=1)
        if 'kw_net' not in df.columns:
            df['kw_net']   = df.get('kw_import', 0) - df.get('kw_export', 0)
            df['kvar_net'] = df.get('kvar_import', 0) - df.get('kvar_export', 0)

        results = run_ai_manager_tnb(
            df, tariff=tariff, battery_kwh=battery_kwh, c_rate=c_rate,
            md_target_pct=md_tgt_pct, initial_soc_pct=soc_init,
            peak_reference_kva=peak_ref, lookahead_intervals=lookahead,
            proportions=proportions, priority_order=priority, max_cut_pct=max_cut,
        )

        # Per-month summary
        month_summary = {}
        for r in results:
            m = r['billing_month']
            if m not in month_summary:
                month_summary[m] = {
                    'orig_md_kva': 0, 'managed_md_kva': 0,
                    'md_target_kva': r['month_md_target'],
                    'total_discharge_kwh': 0, 'total_charge_kwh': 0,
                    'total_grid_kwh': 0, 'intervals': 0,
                }
            s = month_summary[m]
            if r['md_counting']:
                s['orig_md_kva']    = max(s['orig_md_kva'],    r['kva_original'])
                s['managed_md_kva'] = max(s['managed_md_kva'], r['kva_managed'])
            s['total_discharge_kwh'] += r['battery_discharge_kw'] * 0.5
            s['total_charge_kwh']    += r['battery_charge_kw'] * 0.5
            s['total_grid_kwh']      += r['kw_managed'] * 0.5
            s['intervals']           += 1

        for m, s in month_summary.items():
            orig = s['orig_md_kva']; mgd = s['managed_md_kva']
            md_rate = MD_RATES.get(tariff, 0)
            s['md_saving_rm']        = round((orig - mgd) * md_rate, 2)
            s['peak_reduction_pct']  = round((orig - mgd) / orig * 100, 1) if orig else 0
            s['total_discharge_kwh'] = round(s['total_discharge_kwh'], 1)
            s['total_charge_kwh']    = round(s['total_charge_kwh'], 1)
            s['total_grid_kwh']      = round(s['total_grid_kwh'], 1)

        overall_orig    = max((s['orig_md_kva']    for s in month_summary.values()), default=0)
        overall_managed = max((s['managed_md_kva'] for s in month_summary.values()), default=0)
        md_rate         = MD_RATES.get(tariff, 0)
        total_md_saving = sum(s['md_saving_rm'] for s in month_summary.values())

        return jsonify({
            'success': True,
            'tariff': tariff,
            'results': results,
            'month_summary': month_summary,
            'summary': {
                'original_peak_kva':         round(overall_orig, 2),
                'managed_peak_kva':          round(overall_managed, 2),
                'peak_reduction_pct':        round((overall_orig - overall_managed) / overall_orig * 100, 1) if overall_orig else 0,
                'md_rate_rm_per_kva':        md_rate,
                'total_md_saving_rm':        round(total_md_saving, 2),
                'total_battery_discharge_kwh': round(sum(s['total_discharge_kwh'] for s in month_summary.values()), 1),
                'total_battery_charge_kwh':    round(sum(s['total_charge_kwh']    for s in month_summary.values()), 1),
                'tnb_peak_window':           '08:00–22:00 Mon–Sat (excl. PH)' if tariff in TOU_TARIFFS else '24/7 (all hours)',
                'md_measurement_note':       (
                    f'For {tariff} (TOU): MD counted only during peak hours 08:00–22:00 Mon–Sat.'
                    if tariff in TOU_TARIFFS else
                    f'For {tariff}: MD measured 24 hours a day, 7 days a week.'
                ),
            },
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


# ── Live data feed ────────────────────────────────────────────────────────────

@app.route('/api/live/push', methods=['POST'])
def live_push():
    """
    Accept one 30-min reading from the Predictor.  Runs TNB-aware dispatch
    for that interval and returns the decision.  Session state (SOC, month MD)
    persists across calls.

    Body (JSON):
      session_id   : str   (optional, default 'default')
      timestamp    : ISO-8601 string  e.g. '2024-06-15T14:30:00'
      kw_import    : float
      kvar_import  : float  (optional, default 0)
      kw_export    : float  (optional, default 0)
      kvar_export  : float  (optional, default 0)

      # Session config — only needed on first push or to override:
      tariff       : 'C1' | 'C2' | 'D' | 'E1' | 'E2'
      battery_kwh  : float
      c_rate       : float  (default 0.5)
      md_target_pct: float  (default 0.85)
      soc_init_pct : float  (default 0.50)
    """
    try:
        data       = request.json
        sid        = data.get('session_id', 'default')
        session    = _get_session(sid)

        # Update session config if provided
        if 'tariff' in data:
            session.tariff = data['tariff'].upper()
        if 'battery_kwh' in data:
            session.battery_kwh = float(data['battery_kwh'])
            session.battery_kw  = session.battery_kwh * session.c_rate
        if 'c_rate' in data:
            session.c_rate      = float(data['c_rate'])
            session.battery_kw  = session.battery_kwh * session.c_rate
        if 'md_target_pct' in data:
            session.peak_target_pct = float(data['md_target_pct'])

        if not session.initialised:
            session.soc_kwh     = session.battery_kwh * session.soc_init_pct
            session.initialised = True

        # Parse the new reading
        ts          = pd.Timestamp(data['timestamp'])
        kw_import   = float(data.get('kw_import',   0.0))
        kvar_import = float(data.get('kvar_import', 0.0))
        kw_export   = float(data.get('kw_export',   0.0))
        kvar_export = float(data.get('kvar_export', 0.0))
        kw_net      = kw_import  - kw_export
        kvar_net    = kvar_import - kvar_export
        kva         = calc_kva(kw_net, kvar_net)

        # Billing month management
        month_str = ts.to_period('M').strftime('%Y-%m')
        if month_str != session.billing_month:
            # New billing month — reset month MD, keep SOC
            session.billing_month  = month_str
            session.month_peak_kva = 0.0
            # Seed md_target from the 30-day forecast peak if available, else
            # use the last observed month_peak × target_pct
            if session.dispatch_log:
                last_peak = max(
                    (r['kva_original'] for r in session.dispatch_log[-1440:]
                     if r.get('md_counting')),
                    default=kva,
                )
                session.md_target_kva = last_peak * session.peak_target_pct
            elif session.md_target_kva == 0.0:
                # Very first interval: set a conservative target from this reading
                session.md_target_kva = kva * session.peak_target_pct

        # Run dispatch
        result, new_soc = dispatch_interval(
            ts             = ts,
            kw_net         = kw_net,
            kvar_net       = kvar_net,
            soc_kwh        = session.soc_kwh,
            bat_kwh        = session.battery_kwh,
            bat_kw         = session.battery_kw,
            tariff         = session.tariff,
            md_target_kva  = session.md_target_kva,
            forecast_kw    = session.forecast_kw[:8],  # next 4 hours of forecast
            lookahead_steps= 4,
        )
        session.soc_kwh = new_soc

        # Update running month MD
        if result['md_counting']:
            managed_kva = result['kva_managed']
            if managed_kva > session.month_peak_kva:
                session.month_peak_kva = managed_kva
            # Adaptive: if we exceeded target, adjust upward
            if managed_kva > session.md_target_kva:
                session.md_target_kva = managed_kva

        # Append to rolling 24h history (keep last 48 intervals)
        record = {**result, 'timestamp': ts.isoformat(), 'kva_original': round(kva, 3),
                  'kw_import': round(kw_import, 3), 'kvar_import': round(kvar_import, 3),
                  'billing_month': month_str, 'month_peak_kva': round(session.month_peak_kva, 2),
                  'md_target_kva': round(session.md_target_kva, 2)}
        session.history.append(record)
        if len(session.history) > 48:
            session.history.pop(0)
        session.dispatch_log.append(record)

        return jsonify({
            'success':         True,
            'session_id':      sid,
            'timestamp':       ts.isoformat(),
            'billing_month':   month_str,
            'dispatch': {
                'kw_original':         result['kva_original'],
                'kva_original':        round(kva, 2),
                'kva_managed':         result['kva_managed'],
                'kw_managed':          result['kw_managed'],
                'battery_charge_kw':   result['battery_charge_kw'],
                'battery_discharge_kw':result['battery_discharge_kw'],
                'battery_soc_kwh':     result['battery_soc_kwh'],
                'battery_soc_pct':     result['battery_soc_pct'],
                'is_peak_hour':        result['is_peak_hour'],
                'md_counting':         result['md_counting'],
                'prefer_charge':       result['prefer_charge'],
                'actions':             result['actions'],
            },
            'month_status': {
                'billing_month':   month_str,
                'month_peak_kva':  round(session.month_peak_kva, 2),
                'md_target_kva':   round(session.md_target_kva, 2),
                'md_rate_rm_kva':  MD_RATES.get(session.tariff, 0),
                'projected_md_charge_rm': round(session.month_peak_kva * MD_RATES.get(session.tariff, 0), 2),
                'target_md_charge_rm':    round(session.md_target_kva * MD_RATES.get(session.tariff, 0), 2),
            },
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/api/live/forecast', methods=['POST'])
def live_forecast():
    """
    Push the Predictor's 48-step forecast to the session so the dispatcher
    can use look-ahead for pre-discharge decisions.

    Body (JSON):
      session_id : str
      forecast   : [{timestamp: ISO, median_kw: float, p10_kw: float, p90_kw: float}, ...]
                   Must be sorted ascending and cover the next 24 h (48 × 30-min steps).
    """
    try:
        data    = request.json
        sid     = data.get('session_id', 'default')
        session = _get_session(sid)
        fc      = data.get('forecast', [])

        session.forecast_kw = [float(f.get('median_kw', 0)) for f in fc]
        session.forecast_ts = [f.get('timestamp', '') for f in fc]

        return jsonify({
            'success':      True,
            'session_id':   sid,
            'forecast_steps': len(session.forecast_kw),
            'forecast_peak_kw': round(max(session.forecast_kw), 2) if session.forecast_kw else 0,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/api/live/state', methods=['GET'])
def live_state():
    """Return current session state (SOC, month MD, recent history)."""
    try:
        sid     = request.args.get('session_id', 'default')
        session = _get_session(sid)
        return jsonify({
            'session_id':      sid,
            'tariff':          session.tariff,
            'battery_kwh':     session.battery_kwh,
            'battery_kw':      session.battery_kw,
            'c_rate':          session.c_rate,
            'soc_kwh':         round(session.soc_kwh, 2),
            'soc_pct':         round(session.soc_kwh / session.battery_kwh * 100, 1) if session.battery_kwh else 0,
            'billing_month':   session.billing_month,
            'month_peak_kva':  round(session.month_peak_kva, 2),
            'md_target_kva':   round(session.md_target_kva, 2),
            'md_rate_rm_kva':  MD_RATES.get(session.tariff, 0),
            'projected_md_charge_rm': round(session.month_peak_kva * MD_RATES.get(session.tariff, 0), 2),
            'target_md_charge_rm':    round(session.md_target_kva * MD_RATES.get(session.tariff, 0), 2),
            'history_count':   len(session.dispatch_log),
            'recent_history':  session.history[-12:],  # last 6 hours
            'has_forecast':    len(session.forecast_kw) > 0,
            'forecast_peak_kw': round(max(session.forecast_kw), 2) if session.forecast_kw else 0,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/api/live/reset', methods=['DELETE'])
def live_reset():
    """Clear a session (reset SOC, month MD, history)."""
    try:
        sid = request.args.get('session_id', 'default')
        if sid in _LIVE_SESSIONS:
            del _LIVE_SESSIONS[sid]
        return jsonify({'success': True, 'session_id': sid, 'message': 'Session cleared.'})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/api/live/sessions', methods=['GET'])
def live_sessions():
    """List all active sessions."""
    return jsonify({
        'sessions': [
            {'session_id': sid, 'tariff': s.tariff, 'battery_kwh': s.battery_kwh,
             'billing_month': s.billing_month, 'soc_pct': round(s.soc_kwh / s.battery_kwh * 100, 1),
             'history_count': len(s.dispatch_log)}
            for sid, s in _LIVE_SESSIONS.items()
        ]
    })


# ── What-if simulator ─────────────────────────────────────────────────────────

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Solar + battery what-if simulator.

    Body (JSON):
      records        : [{timestamp, kw_net, kvar_net, kva, kw_import, kw_export, ...}]
                       (output of /api/upload)
      solar_kwp      : float   installed solar PV capacity (kWp)
      battery_kwh    : float   battery storage capacity
      battery_kw     : float   battery power rating  (null = kwh × c_rate)
      tariff         : 'C1' | 'C2' | 'D' | 'E1' | 'E2'
      psh            : float   peak sun hours (default 4.5 for West Malaysia)
      md_target_pct  : float   target MD reduction as fraction (default 0.85)
      c_rate         : float   default 0.5
      nem_rate       : float   NEM buyback rate RM/kWh (default 0.31)
      initial_soc_pct: float   default 0.50
    """
    try:
        data = request.json

        records     = data.get('records', [])
        solar_kwp   = float(data.get('solar_kwp',   0.0))
        battery_kwh = float(data.get('battery_kwh', 0.0))
        battery_kw  = data.get('battery_kw', None)
        if battery_kw is not None:
            battery_kw = float(battery_kw)
        tariff      = data.get('tariff', 'C1').upper()
        psh         = float(data.get('psh', 4.5))
        md_tgt_pct  = float(data.get('md_target_pct', 0.85))
        c_rate      = float(data.get('c_rate', 0.5))
        nem_rate    = float(data.get('nem_rate', NEM_RATE_DEFAULT))
        soc_init    = float(data.get('initial_soc_pct', 0.50))

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Ensure required columns exist
        if 'kw_net' not in df.columns:
            df['kw_net']   = df.get('kw_import', df.get('kw_original', 0)) - df.get('kw_export', 0)
        if 'kvar_net' not in df.columns:
            df['kvar_net'] = df.get('kvar_import', df.get('kvar_original', 0)) - df.get('kvar_export', 0)
        if 'kva' not in df.columns:
            df['kva'] = df.apply(lambda r: calc_kva(float(r['kw_net']), float(r['kvar_net'])), axis=1)

        sim = simulate_solar_battery(
            df           = df,
            solar_kwp    = solar_kwp,
            battery_kwh  = battery_kwh,
            battery_kw   = battery_kw,
            tariff       = tariff,
            psh          = psh,
            md_target_pct= md_tgt_pct,
            c_rate       = c_rate,
            nem_rate     = nem_rate,
            initial_soc_pct = soc_init,
        )

        return jsonify({'success': True, 'tariff': tariff, **sim})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("=" * 60)
    print("  AI Energy Manager  v5  |  http://localhost:5050")
    print("  New endpoints:")
    print("    POST /api/optimize/tnb    — TNB-billing-aware batch dispatch")
    print("    POST /api/live/push       — stream 30-min readings from Predictor")
    print("    POST /api/live/forecast   — push 24h forecast for look-ahead")
    print("    GET  /api/live/state      — current session status")
    print("    DELETE /api/live/reset    — clear session")
    print("    POST /api/simulate        — solar+battery what-if simulator")
    print("  Legacy: POST /api/upload, POST /api/optimize")
    print("=" * 60)
    app.run(debug=True, port=5050)
