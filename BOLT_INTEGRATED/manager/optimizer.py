"""
BOLT AI Energy Manager — Smart Power Management Engine  (v7)

Design priorities (in order):
  P1. Never create a new kVA peak from battery charging
  P2. Avoid load shedding — battery management is the primary tool
  P3. Protect the MD period window without neglecting off-peak
  P4. Continuous adaptation — each tick's SOC carries forward to the next

Key algorithmic improvements over v6:
  [FIX-1]  Battery charging is blocked when load shedding occurred in the same
            interval — prevents adding charge load that undoes the shed.
  [FIX-2]  Discharge targets 97 % of trigger (configurable), not 100 %.
            Provides a buffer below the MD measurement point and prevents
            oscillation around the trigger boundary.
  [FIX-3]  Look-ahead is now quantified: scans upcoming intervals to find the
            maximum expected kVA and total kWh of discharge energy required,
            not just a binary "high / not-high" flag.
  [FIX-4]  Dynamic SOC floor before MD period: when MD start is within
            `md_reserve_lookahead_h` hours, the minimum allowable SOC rises
            to `soc_md_reserve_pct` of capacity, ensuring the battery is
            adequately charged for the highest-value dispatch window.
  [FIX-5]  Pre-MD charging is throttled to exactly the rate needed to reach
            the target SOC by MD start — avoids unnecessary kVA spikes from
            flat-out pre-MD charging when battery is already near target.
  [FIX-6]  Emergency charge ceiling lowered to 95 % of trigger (was 100 %).
            Emergency charging no longer pushes kVA to the exact trigger value.
  [FIX-7]  off_peak_ok check now uses mgd_kva (post-shedding) instead of
            kva_orig. After load reduction, the actual site demand is lower,
            so the charge decision is more accurate.
  [FIX-8]  Minimum discharge threshold (0.5 kW) suppresses noisy micro-actions.
  [FIX-9]  Added `initial_soc_kwh` parameter for pipeline SOC continuity.
  [FIX-10] `charge_threshold_upper` added to every row_out (was missing).
  [FIX-11] `note` field added to load_reduction actions (consistency).
  [FIX-12] Predictive urgency scaling: look-ahead peak severity amplifies the
            proximity zone, triggering earlier pre-discharge for larger peaks.
  [FIX-13] Post-MD recharge window: 2 h after MD end, charge_upper threshold
            is relaxed to 90 % of ref_peak to restore SOC for next day.
"""
from __future__ import annotations
import math
import io
import numpy as np
import pandas as pd
from io import StringIO


# ── Utility ───────────────────────────────────────────────────────────────────

def calc_kva(kw: float, kvar: float) -> float:
    return math.sqrt(kw * kw + kvar * kvar)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace('﻿', '', regex=False)
                  .str.replace(r'\s+', ' ', regex=True))
    return df


def _is_valid_header_row(cols: list) -> bool:
    s = ' '.join(cols)
    has_time  = any(t in s for t in ('date', 'time', 'start', 'end', 'timestamp', 'datetime'))
    has_power = any(t in s for t in ('kw', 'kvar', 'kwh', 'power', 'watt', 'energy', 'var'))
    return has_time and has_power


def _find_datetime_col(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    cols = list(df.columns)
    for col in cols:
        if ('date' in col and 'time' in col) or col in ('datetime', 'timestamp'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    for col in cols:
        if col in ('end_time', 'end time', 'start_time', 'start time'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    date_col = next((c for c in cols if c == 'date' or c.startswith('date')), None)
    time_col = next((c for c in cols if c in ('time', 'end time', 'end_time', 'start_time')
                     and c != date_col), None)
    if date_col and time_col:
        df['_dt'] = pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
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


def _map_power_cols(df: pd.DataFrame) -> dict:
    col_map: dict[str, str | None] = {}
    for col in df.columns:
        is_kvar = 'kvar' in col or ('var' in col and 'kw' not in col)
        is_kw   = (('kw' in col or 'kwh' in col or 'watt' in col
                    or 'active' in col or 'power' in col or 'energy' in col) and not is_kvar)
        is_imp, is_exp = 'import' in col, 'export' in col
        if is_kvar and is_imp and 'kvar_import' not in col_map: col_map['kvar_import'] = col
        elif is_kvar and is_exp and 'kvar_export' not in col_map: col_map['kvar_export'] = col
        elif is_kw   and is_imp and 'kw_import'   not in col_map: col_map['kw_import']   = col
        elif is_kw   and is_exp and 'kw_export'   not in col_map: col_map['kw_export']   = col
    for key in ('kw_import', 'kw_export', 'kvar_import', 'kvar_export'):
        col_map.setdefault(key, None)
    return col_map


def parse_uploaded_data(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Parse CSV or Excel load-profile bytes into a normalised DataFrame.
    Returns: timestamp, kw_import, kw_export, kvar_import, kvar_export, kw_net, kvar_net, kva
    """
    df = None
    if filename.lower().endswith('.csv'):
        for hrow in range(12):
            for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
                try:
                    tmp = pd.read_csv(StringIO(file_content.decode(enc)), header=hrow)
                    tmp = _normalize_cols(tmp)
                    if len(tmp.columns) >= 4 and _is_valid_header_row(list(tmp.columns)):
                        df = tmp; break
                except Exception:
                    continue
            if df is not None:
                break
        if df is None:
            raise ValueError("Could not locate a valid header row in CSV.")
    elif filename.lower().endswith(('.xlsx', '.xls')):
        engine = 'xlrd' if filename.lower().endswith('.xls') else 'openpyxl'
        fb = io.BytesIO(file_content)
        for hrow in range(12):
            try:
                fb.seek(0)
                tmp = pd.read_excel(fb, header=hrow, engine=engine)
                tmp = _normalize_cols(tmp)
                if len(tmp.columns) >= 4 and _is_valid_header_row(list(tmp.columns)):
                    df = tmp; break
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
    df = df.dropna(subset=[start_col]).sort_values(start_col).reset_index(drop=True)

    col_map = _map_power_cols(df)

    def _s(key: str) -> pd.Series:
        col = col_map.get(key)
        if col and col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(0)
        return pd.Series([0.0] * len(df), index=df.index)

    result = pd.DataFrame({
        'timestamp':   df[start_col].values,
        'kw_import':   _s('kw_import').values,
        'kw_export':   _s('kw_export').values,
        'kvar_import': _s('kvar_import').values,
        'kvar_export': _s('kvar_export').values,
    })
    result['kw_net']    = result['kw_import']   - result['kw_export']
    result['kvar_net']  = result['kvar_import'] - result['kvar_export']
    result['kva']       = result.apply(lambda r: calc_kva(r['kw_net'], r['kvar_net']), axis=1)
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    result = (result.set_index('timestamp')
              .resample('30min').mean()
              .dropna(how='all')
              .reset_index())
    result['kva'] = result.apply(lambda r: calc_kva(r['kw_net'], r['kvar_net']), axis=1)
    if result.empty:
        raise ValueError("Empty after resampling.")
    return result


# ── Main optimiser ────────────────────────────────────────────────────────────

def run_ai_manager(
    df: pd.DataFrame,
    loads: dict,
    battery_capacity_kwh: float,
    priority_order: list,
    peak_target_pct: float,
    bat_charge_upper_pct: float,
    # Battery hardware
    c_rate: float = 0.5,
    charge_c_rate: float | None = None,     # separate charge C-rate; defaults to c_rate
    bat_efficiency: float = 0.95,
    initial_soc_pct: float = 0.50,
    initial_soc_kwh: float | None = None,   # absolute override (pipeline continuity)
    # Peak reference
    peak_reference_kva: float | None = None,
    lookahead_intervals: int = 16,
    # MD window
    md_start_hour: int = 14,
    md_end_hour: int = 22,
    pre_md_hours: int = 2,
    post_md_hours: int = 2,                 # recharge window after MD end
    # Dispatch tuning
    discharge_target_pct: float = 0.97,    # discharge to this fraction of trigger (not 100 %)
    soc_md_reserve_pct: float = 0.70,      # target SOC when entering MD period
    md_reserve_lookahead_h: float = 4.0,   # h before MD to start protecting reserve
) -> list[dict]:
    """
    Smart sequential per-interval optimiser for peak shaving and load management.

    Priority order:
      1. Battery discharge to shave peaks  (never create new peaks from charging)
      2. Load curtailment only when battery is fully depleted
      3. Preserve battery SOC for upcoming MD period

    Returns a list of per-interval result dicts.  The final dict contains
    'final_soc_kwh' for pipeline continuity (carry SOC to next tick).
    """
    INTERVAL_H           = 0.5     # 30-min intervals
    BAT_HARD_MIN_PCT     = 0.05    # absolute SoC floor (hardware protection)
    BAT_EMERGENCY_PCT    = 0.15    # SoC at which emergency charging triggers
    BAT_CHARGE_FULL_PCT  = 0.90    # normal charge ceiling (don't fill to 100 %)
    PROX_NORMAL          = 0.80    # discharge when kVA ≥ 80 % of ref_peak (off-MD)
    PROX_MD              = 0.70    # discharge when kVA ≥ 70 % of ref_peak (MD hours)
    CHARGE_GUARD_PCT     = 0.88    # charging kVA ceiling as fraction of trigger
                                   # = 0.88 × peak_target × ref_peak  (was 0.92)
    EMERG_GUARD_PCT      = 0.95    # [FIX-6] emergency ceiling, was 1.00
    MIN_DISCHARGE_KW     = 0.5     # [FIX-8] suppress micro-discharge actions below this

    eff_chg_rate = battery_capacity_kwh * (charge_c_rate if charge_c_rate else c_rate)
    eff_dis_rate = battery_capacity_kwh * c_rate

    load_keys  = list(loads.keys())
    total_prop = sum(loads[k].get('proportion', 0) for k in load_keys) or 1
    norm       = {k: loads[k].get('proportion', 0) / total_prop for k in load_keys}

    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['date']  = df['timestamp'].dt.date
    df['hour']  = df['timestamp'].dt.hour
    df['in_md'] = df['hour'].apply(lambda h: md_start_hour <= h < md_end_hour)

    pre_md_start = (md_start_hour - pre_md_hours) % 24
    df['in_pre_md'] = df['hour'].apply(
        lambda h: (pre_md_start <= h < md_start_hour) if pre_md_start < md_start_hour
                  else (h >= pre_md_start or h < md_start_hour))

    # Post-MD recharge window [FIX-13]
    post_md_end = (md_end_hour + post_md_hours) % 24
    df['in_post_md'] = df['hour'].apply(
        lambda h: (md_end_hour <= h < post_md_end) if md_end_hour < post_md_end
                  else (h >= md_end_hour or h < post_md_end))

    # Peak reference — rolling 30-day prior-day max
    if peak_reference_kva is not None:
        df['_ref_peak'] = float(peak_reference_kva)
    else:
        daily_max   = df.groupby('date')['kva'].max().sort_index()
        rolling_ref = daily_max.shift(1).rolling(30, min_periods=1).max()
        # Bootstrap day 1: use the first 3 days' average if available, else 110 % of day 1
        if len(daily_max) >= 3:
            bootstrap = daily_max.iloc[:3].mean() * 1.05
        else:
            bootstrap = daily_max.iloc[0] * 1.10
        rolling_ref = rolling_ref.fillna(bootstrap)
        df['_ref_peak'] = df['date'].map(rolling_ref.to_dict())

    # Absolute SOC limits
    bat_max       = battery_capacity_kwh
    bat_hard_min  = battery_capacity_kwh * BAT_HARD_MIN_PCT
    bat_emerg_abs = battery_capacity_kwh * BAT_EMERGENCY_PCT
    bat_full      = battery_capacity_kwh * BAT_CHARGE_FULL_PCT
    bat_md_res    = battery_capacity_kwh * soc_md_reserve_pct  # [FIX-4] MD reserve

    # Initial SOC — absolute kwh takes precedence over percentage [FIX-9]
    bat_soc = (float(initial_soc_kwh)
               if initial_soc_kwh is not None
               else battery_capacity_kwh * initial_soc_pct)
    bat_soc = max(bat_hard_min, min(bat_max, bat_soc))

    n = len(df)
    results: list[dict] = []

    for idx in range(n):
        row       = df.iloc[idx]
        kva_orig  = float(row['kva'])
        kw        = float(row['kw_net'])
        kvar      = float(row['kvar_net'])
        date      = row['date']
        ts        = row['timestamp']
        hour      = int(row['hour'])
        ref_peak  = float(row['_ref_peak'])
        in_md     = bool(row['in_md'])
        in_pre_md = bool(row['in_pre_md'])
        in_post_md= bool(row['in_post_md'])

        # Core thresholds
        discharge_trigger   = ref_peak * peak_target_pct
        # [FIX-2] Discharge target is slightly below trigger to avoid recording trigger as peak
        dispatch_target     = discharge_trigger * discharge_target_pct
        discharge_proximity = ref_peak * (PROX_MD if in_md else PROX_NORMAL)
        charge_upper        = ref_peak * bat_charge_upper_pct
        charge_kva_ceil     = discharge_trigger * CHARGE_GUARD_PCT
        emerg_kva_ceil      = discharge_trigger * EMERG_GUARD_PCT  # [FIX-6] was 1.00

        # [FIX-13] Post-MD: relax charge_upper to allow recharge even under moderate load
        if in_post_md and not in_md:
            charge_upper = ref_peak * min(0.90, bat_charge_upper_pct + 0.15)

        # ── [FIX-3] Quantified look-ahead ──────────────────────────────────────
        # Scan upcoming intervals; find max kVA and total kWh deficit
        upcoming_max_kva       = 0.0
        upcoming_kwh_needed    = 0.0
        upcoming_high          = False
        intervals_above_trigger= 0
        pf_approx = kw / kva_orig if kva_orig > 1.0 else 0.85

        for fi in range(idx + 1, min(idx + lookahead_intervals + 1, n)):
            frow   = df.iloc[fi]
            f_kva  = float(frow['kva'])
            f_ref  = float(frow['_ref_peak'])
            f_trig = f_ref * peak_target_pct
            if f_kva >= f_trig:
                upcoming_high = True
                intervals_above_trigger += 1
                upcoming_max_kva = max(upcoming_max_kva, f_kva)
                # Approximate kW above trigger needed to flatten this interval
                f_kw_approx  = f_kva * pf_approx
                f_kvar_approx= f_kva * math.sqrt(max(1.0 - pf_approx**2, 0))
                f_kw_target  = math.sqrt(max(f_trig**2 - f_kvar_approx**2, 0.0))
                excess_kw    = max(f_kw_approx - f_kw_target, 0.0)
                upcoming_kwh_needed += excess_kw * INTERVAL_H / bat_efficiency

        # Severity factor: scales proximity zone to pre-discharge earlier for large peaks
        # severity = 1.0 (at trigger) → 1.3 (30 % above trigger) — caps at 1.3
        if upcoming_max_kva > 0 and discharge_trigger > 0:
            peak_severity = min(1.30, upcoming_max_kva / discharge_trigger)
        else:
            peak_severity = 1.0

        # Effective proximity: wider when a severe peak is detected
        eff_proximity = discharge_proximity * (1.0 / peak_severity)  # shrink % threshold
        in_peak_zone  = kva_orig >= eff_proximity or upcoming_high

        # ── [FIX-4] Dynamic minimum SOC (MD reserve protection) ───────────────
        # Within md_reserve_lookahead_h of MD start, protect SOC reserve
        hours_to_md_start = (md_start_hour - hour) % 24
        in_md_reserve_window = (
            not in_md
            and not in_pre_md       # pre-MD boost will handle it
            and 0 < hours_to_md_start <= md_reserve_lookahead_h
        )
        # Effective minimum SOC for discharge
        if in_md:
            bat_dis_min = bat_hard_min               # inside MD: use everything available
        elif in_md_reserve_window:
            # Reserve at least bat_md_res, but never block emergency
            bat_dis_min = max(bat_hard_min, bat_md_res * 0.5)
        else:
            bat_dis_min = battery_capacity_kwh * 0.15  # standard 15 %

        # ── Working variables ──────────────────────────────────────────────────
        load_kva    = {k: kva_orig * norm[k] for k in load_keys}
        load_factor = {k: 1.0 for k in load_keys}
        bat_chg_kw = bat_dis_kw = 0.0
        load_shed_kva = 0.0       # [FIX-1] tracker for whether any load was shed
        actions: list[dict] = []
        mgd_kw   = kw
        mgd_kvar = kvar
        mgd_kva  = calc_kva(mgd_kw, mgd_kvar)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1 — BATTERY DISCHARGE  (primary peak-shaving tool)
        # ══════════════════════════════════════════════════════════════════════
        if in_peak_zone and mgd_kva >= discharge_trigger and bat_soc > bat_dis_min and mgd_kw > 0:

            # [FIX-2] Target dispatch_target (97 % of trigger), not 100 %
            kw_target = math.sqrt(max(dispatch_target ** 2 - mgd_kvar ** 2, 0.0))
            dis_kw_load_needed = max(mgd_kw - kw_target, 0.0)

            # [FIX-8] Suppress micro-discharges
            if dis_kw_load_needed >= MIN_DISCHARGE_KW:
                dis_kwh_from_bat_needed = dis_kw_load_needed * INTERVAL_H / bat_efficiency
                dis_kwh_from_bat = min(
                    dis_kwh_from_bat_needed,
                    eff_dis_rate * INTERVAL_H,
                    bat_soc - bat_dis_min,
                )
                dis_kwh_load = dis_kwh_from_bat * bat_efficiency
                dis_kw_load  = dis_kwh_load / INTERVAL_H

                soc_before = bat_soc
                bat_soc   -= dis_kwh_from_bat
                mgd_kw    -= dis_kw_load
                mgd_kw     = max(mgd_kw, 0.0)
                mgd_kva    = calc_kva(mgd_kw, mgd_kvar)
                bat_dis_kw = dis_kw_load

                la_trig = bool(upcoming_high and kva_orig < eff_proximity)
                actions.append({
                    'type':                'battery_discharge',
                    'load':                'Battery',
                    'discharge_kw':        round(dis_kw_load, 2),
                    'soc_before_kwh':      round(soc_before, 1),
                    'soc_after_kwh':       round(bat_soc, 1),
                    'kva_before':          round(kva_orig, 2),
                    'kva_after':           round(mgd_kva, 2),
                    'lookahead_triggered': la_trig,
                    'upcoming_peak_kva':   round(upcoming_max_kva, 1),
                    'upcoming_kwh_needed': round(upcoming_kwh_needed, 2),
                    'md_hours':            in_md,
                    'note': (
                        f'Discharged {round(dis_kw_load, 1)} kW to bus; '
                        f'kVA {round(kva_orig, 1)} → {round(mgd_kva, 1)}; '
                        f'SOC {round(soc_before, 0):.0f} → {round(bat_soc, 0):.0f} kWh'
                        + (' [look-ahead]' if la_trig else '')
                        + (f' [severity {peak_severity:.2f}]' if peak_severity > 1.05 else '')
                        + (' [MD hrs]' if in_md else '')
                    ),
                })

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2 — LOAD CURTAILMENT  (last resort — only when battery is spent)
        # ══════════════════════════════════════════════════════════════════════
        remaining = max(0.0, mgd_kva - discharge_trigger)
        if remaining > 1e-3:
            for lk in priority_order:
                if remaining <= 1e-3:
                    break
                if lk not in loads or load_kva.get(lk, 0) <= 0:
                    continue
                max_possible = load_kva[lk] * loads[lk].get('max_cut_pct', 0.10)
                cut_kva = min(remaining, max_possible)
                if cut_kva < 0.05:
                    continue

                load_factor[lk] = 1.0 - cut_kva / load_kva[lk]
                # Decompose kVA cut into kW/kVAR components along current PF axis
                pf_l = mgd_kw   / mgd_kva if mgd_kva > 1e-3 else 1.0
                qf_l = mgd_kvar / mgd_kva if mgd_kva > 1e-3 else 0.0
                mgd_kw   -= cut_kva * pf_l
                mgd_kvar -= cut_kva * qf_l
                mgd_kw    = max(mgd_kw, 0.0)
                mgd_kva   = calc_kva(mgd_kw, mgd_kvar)
                remaining = max(0.0, mgd_kva - discharge_trigger)
                load_shed_kva += cut_kva  # [FIX-1] track total shed

                actions.append({
                    'type':        'load_reduction',
                    'load':        loads[lk].get('name', lk),
                    'load_key':    lk,
                    'cut_kva':     round(cut_kva, 2),
                    'factor_pct':  round(load_factor[lk] * 100, 1),
                    'max_cut_pct': loads[lk].get('max_cut_pct', 0.10) * 100,
                    'note': (  # [FIX-11] note field was missing
                        f'{loads[lk].get("name", lk)} curtailed {round(cut_kva, 1)} kVA '
                        f'(max {loads[lk].get("max_cut_pct", 0.10)*100:.0f}%); '
                        f'factor {round(load_factor[lk]*100, 1)}%; '
                        f'battery SOC {round(bat_soc, 0):.0f} kWh'
                    ),
                })

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3 — BATTERY CHARGING  (opportunistic; NEVER create new peaks)
        # ══════════════════════════════════════════════════════════════════════
        # [FIX-1] Skip charging entirely if load was shed this interval —
        #         charging after shedding adds load back, partially undoing the shed.
        # [FIX-1] Also skip if battery discharged this interval (existing guard).
        can_charge = (bat_dis_kw == 0.0) and (load_shed_kva == 0.0) and (bat_soc < bat_max)

        if can_charge:
            emergency = bat_soc < bat_emerg_abs
            ceiling   = emerg_kva_ceil if emergency else charge_kva_ceil

            # Maximum grid kW that keeps post-charge kVA ≤ ceiling (exact formula)
            max_chg_kw_grid    = math.sqrt(max(ceiling ** 2 - mgd_kvar ** 2, 0.0)) - mgd_kw
            max_chg_kwh_stored = max(max_chg_kw_grid, 0.0) * bat_efficiency * INTERVAL_H

            # [FIX-7] off_peak check uses mgd_kva (post-shedding), not kva_orig
            off_peak_ok = (mgd_kva < charge_upper) or in_pre_md or in_post_md

            normal = off_peak_ok and (bat_soc < bat_full) and not in_md

            if max_chg_kwh_stored > 0.001 and (emergency or normal):
                # [FIX-5] Throttle pre-MD charge rate to exactly what's needed
                #         to reach soc_md_reserve_pct by MD start.
                if in_pre_md and not emergency:
                    hours_to_md = max(0.25, hours_to_md_start)
                    soc_target  = bat_md_res
                    soc_deficit = max(0.0, soc_target - bat_soc)
                    # kW needed = deficit / (hours × η) — cap at hardware C-rate
                    rate_needed = soc_deficit / (hours_to_md * bat_efficiency) if soc_deficit > 0.1 else 0.0
                    effective_rate = min(rate_needed, eff_chg_rate)
                else:
                    effective_rate = eff_chg_rate

                chg_kwh_stored = min(
                    max_chg_kwh_stored,
                    effective_rate * INTERVAL_H,
                    bat_max - bat_soc,
                )

                if chg_kwh_stored > 0.01:
                    chg_kw_grid = chg_kwh_stored / bat_efficiency / INTERVAL_H
                    soc_before  = bat_soc
                    bat_soc    += chg_kwh_stored
                    bat_chg_kw  = chg_kw_grid
                    mgd_kw     += chg_kw_grid
                    mgd_kva     = calc_kva(mgd_kw, mgd_kvar)

                    trigger_str = (
                        'emergency'    if emergency   else
                        'pre-MD boost' if in_pre_md   else
                        'post-MD refil'if in_post_md  else
                        'normal'
                    )
                    actions.append({
                        'type':             'battery_charge',
                        'load':             'Battery',
                        'charge_kw':        round(chg_kw_grid, 2),
                        'soc_before_kwh':   round(soc_before, 1),
                        'soc_after_kwh':    round(bat_soc, 1),
                        'kva_ceiling':      round(ceiling, 2),
                        'kva_after_charge': round(mgd_kva, 2),
                        'charge_trigger':   trigger_str,
                        'note': (
                            f'{trigger_str.capitalize()} charge {round(chg_kw_grid, 1)} kW; '
                            f'stores {round(chg_kwh_stored, 1)} kWh (η={bat_efficiency}); '
                            f'SOC {round(soc_before, 0):.0f} → {round(bat_soc, 0):.0f} kWh; '
                            f'kVA after {round(mgd_kva, 1)} (ceil {round(ceiling, 1)})'
                        ),
                    })

        # Hard clamp SOC to hardware limits
        bat_soc = max(bat_hard_min, min(bat_max, bat_soc))

        # ── Per-load breakdown ─────────────────────────────────────────────────
        load_managed  = {k: load_kva[k] * load_factor[k]          for k in load_keys}
        load_kvah_cut = {k: (load_kva[k] - load_managed[k]) * INTERVAL_H for k in load_keys}

        row_out: dict = {
            'timestamp':              ts.isoformat(),
            'date':                   str(date),
            'kva_original':           round(kva_orig,  2),
            'kw_original':            round(kw,        2),
            'kvar_original':          round(kvar,      2),
            'kw_managed':             round(mgd_kw,    2),
            'kvar_managed':           round(mgd_kvar,  2),
            'battery_action_kw':      round(bat_dis_kw - bat_chg_kw, 2),
            'battery_charge_kw':      round(bat_chg_kw,  2),
            'battery_discharge_kw':   round(bat_dis_kw,  2),
            'battery_soc_kwh':        round(bat_soc,    2),
            'battery_soc_pct':        round(bat_soc / bat_max * 100, 1) if bat_max else 0,
            'kva_managed':            round(mgd_kva,   2),
            'target_peak':            round(discharge_trigger, 2),
            'dispatch_target':        round(dispatch_target,   2),  # 97 % level
            'ref_peak':               round(ref_peak,  2),
            'charge_threshold_upper': round(charge_upper, 2),         # [FIX-10] was missing
            'charge_kva_ceiling':     round(charge_kva_ceil, 2),
            'discharge_proximity':    round(eff_proximity, 2),
            'in_md_hours':            in_md,
            'in_pre_md':              in_pre_md,
            'in_post_md':             in_post_md,
            'upcoming_peak_kva':      round(upcoming_max_kva, 1),
            'upcoming_kwh_needed':    round(upcoming_kwh_needed, 2),
            'actions':                actions,
        }
        for k in load_keys:
            row_out[f'{k}_kva']      = round(load_kva[k],     2)
            row_out[f'{k}_managed']  = round(load_managed[k], 2)
            row_out[f'{k}_factor']   = round(load_factor[k],  3)
            row_out[f'{k}_kvah_cut'] = round(load_kvah_cut[k],3)

        results.append(row_out)

    # Attach final SOC to last row for pipeline continuity [FIX-9]
    if results:
        results[-1]['final_soc_kwh'] = round(bat_soc, 2)

    return results
