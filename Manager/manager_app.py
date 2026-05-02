"""
AI Energy Manager - Backend Server (v4)
Battery rules:
  DISCHARGE when ALL hold:
    - kva_original >= 80% of day_peak  (in the peak zone)
    - kva_after_loads >= target_peak   (85% of day_peak)
    - SOC > 15% of capacity            (discharge floor)
    - Discharge capped to exact gap needed, 0.5C rate, and available SOC
  CHARGE when:
    - Not discharging this interval
    - SOC < bat_max
    - Charge would not push kva above 92% of discharge_trigger (peak-creation guard)
    - A) Emergency: SOC < 15%  → charge regardless of kva level
    - B) Normal   : kva < charge_upper_pct (70%) of day_peak AND SOC < 90%
    - Charge capped to 0.5C rate and kva ceiling headroom
  Priority: battery discharge FIRST, then load reduction
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import math, os, traceback, re
from io import StringIO

app = Flask(__name__, static_folder='static')
CORS(app)

# ── helpers ──────────────────────────────────────────────────────────────────
def calc_kva(kw_net, kvar_net):
    return math.sqrt(kw_net**2 + kvar_net**2)

def _normalize_cols(df):
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace('\ufeff','',regex=False)
                  .str.replace(r'\s+',' ',regex=True))
    return df

def _is_valid_header_row(cols):
    col_str = ' '.join(cols)
    has_time  = any(t in col_str for t in ('date','time','start','end','timestamp','datetime'))
    has_power = any(t in col_str for t in ('kw','kvar','kwh','power','watt','energy','var'))
    return has_time and has_power

def _find_datetime_col(df):
    cols = list(df.columns)
    for col in cols:
        if ('date' in col and 'time' in col) or 'datetime' in col or 'timestamp' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    for col in cols:
        if col in ('end_time','end time'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    for col in cols:
        if col in ('start_time','start time'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col, df
    date_col = next((c for c in cols if c == 'date' or c.startswith('date')), None)
    time_col = next((c for c in cols if c in ('time','end time','end_time','start_time') and c != date_col), None)
    if date_col and time_col:
        combined = pd.to_datetime(df[date_col].astype(str)+' '+df[time_col].astype(str), errors='coerce')
        df['_dt'] = combined
        return '_dt', df
    for col in cols[:10]:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() / max(len(parsed),1) >= 0.8:
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
    for key in ('kw_import','kw_export','kvar_import','kvar_export'):
        col_map.setdefault(key, None)
    return col_map

# ── data parsing (unchanged from v2) ─────────────────────────────────────────
def parse_uploaded_data(file_content, filename):
    try:
        df = None
        if filename.lower().endswith('.csv'):
            encodings = ['utf-8-sig','utf-8','latin-1','cp1252']
            for hrow in range(12):
                for enc in encodings:
                    try:
                        tmp = pd.read_csv(StringIO(file_content.decode(enc)), header=hrow)
                        tmp = _normalize_cols(tmp)
                        if len(tmp.columns) >= 4 and _is_valid_header_row(list(tmp.columns)):
                            df = tmp; break
                    except Exception: continue
                if df is not None: break
            if df is None: raise ValueError("Could not locate a valid header row in CSV.")
        elif filename.lower().endswith(('.xlsx','.xls')):
            import io
            engine = 'xlrd' if filename.lower().endswith('.xls') else 'openpyxl'
            fb = io.BytesIO(file_content)
            for hrow in range(12):
                try:
                    fb.seek(0)
                    tmp = pd.read_excel(fb, header=hrow, engine=engine)
                    tmp = _normalize_cols(tmp)
                    if len(tmp.columns) >= 4 and _is_valid_header_row(list(tmp.columns)):
                        df = tmp; break
                except Exception: continue
            if df is None: raise ValueError("Could not locate a valid header row in Excel.")
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
            return pd.Series([0.0]*len(df), index=df.index)

        result = pd.DataFrame()
        result['timestamp']   = df[start_col].values
        result['kw_import']   = _s('kw_import').values
        result['kw_export']   = _s('kw_export').values
        result['kvar_import'] = _s('kvar_import').values
        result['kvar_export'] = _s('kvar_export').values
        result['kw_net']   = result['kw_import']   - result['kw_export']
        result['kvar_net'] = result['kvar_import'] - result['kvar_export']
        result['kva']      = result.apply(lambda r: calc_kva(r['kw_net'],r['kvar_net']), axis=1)

        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result = result.set_index('timestamp').resample('30min').mean().dropna(how='all').reset_index()
        result['kva'] = result.apply(lambda r: calc_kva(r['kw_net'],r['kvar_net']), axis=1)
        if result.empty: raise ValueError("Empty after resampling.")
        return result
    except Exception as e:
        raise ValueError(f"Data parsing error: {str(e)}")


# ── AI MANAGER v4 ─────────────────────────────────────────────────────────────
def run_ai_manager(df, proportions, battery_capacity_kwh, priority_order,
                   peak_target_pct, max_cut_pct, bat_charge_upper_pct,
                   bat_charge_lower_pct, c_rate=0.5, initial_soc_pct=0.50,
                   peak_reference_kva=None, lookahead_intervals=3):
    """
    Per-day 24h optimisation  (v4 — corrected charge/discharge guards).

    FIX 1  — kW/kVAR correctly tracked: battery dispatch and load cuts both
             decompose into kW and kVAR components; kVA is always recomputed
             as √(kW²+kVAR²) rather than linearly decremented.
    FIX 2  — SOC carries over between days; only initialised once at the
             start of the full dataset using initial_soc_pct.
    FIX 3  — Peak reference is a 30-day rolling maximum of prior days (or a
             user-supplied absolute ceiling) — not the current day's own peak.
    FIX 4  — Look-ahead: if any of the next `lookahead_intervals` intervals
             are flagged high-load, the battery starts discharging early.
    FIX 5  — C-rate is a configurable parameter (default 0.5C).
    FIX 6  — Per-interval kVAh cut computed for each load type and aggregated.
    FIX 7  — Emergency charging uses a separate 100%-of-target ceiling so a
             critically low battery can always recover.

    ── DISCHARGE rules (ALL must hold) ────────────────────────────────────────
      1. kva_orig ≥ DISCHARGE_PROXIMITY_PCT (80%) of reference peak,
         OR any of the next `lookahead_intervals` intervals are high-load.
      2. managed kVA ≥ discharge_trigger (peak_target_pct of reference peak).
      3. SOC > BAT_DISCHARGE_MIN_PCT (15%) floor.
      Discharge kW is decomposed along the active-power axis (same power
      factor), then kVA is recomputed.

    ── CHARGE rules ────────────────────────────────────────────────────────────
      A) Emergency (SOC < 15%): ceiling = 100% of discharge_trigger.
      B) Normal (kva < charge_upper AND SOC < 90%): ceiling = 92% of trigger.
      Charge power adds to kW, then kVA is recomputed.
      Never charge and discharge in the same interval.
    """
    results    = []
    INTERVAL_H = 0.5   # 30-minute intervals

    # ── tuneable constants ───────────────────────────────────────────────────
    BAT_EMERGENCY_PCT        = 0.15   # SOC below this → emergency charge
    BAT_CHARGE_FULL_PCT      = 0.90   # SOC ceiling for normal charging
    BAT_DISCHARGE_MIN_PCT    = 0.15   # SOC floor — never discharge below this
    DISCHARGE_PROXIMITY_PCT  = 0.80   # kva must be ≥ 80% of reference peak to discharge
    CHARGE_PEAK_GUARD_PCT    = 0.92   # charging must not push kva above 92% of target
    EMERGENCY_PEAK_GUARD_PCT = 1.00   # emergency charging allowed up to 100% of target

    # normalise proportions
    ev_p   = proportions.get('ev',   0.3)
    hvac_p = proportions.get('hvac', 0.4)
    misc_p = proportions.get('misc', 0.3)
    tot = ev_p + hvac_p + misc_p or 1
    ev_p /= tot; hvac_p /= tot; misc_p /= tot

    df = df.copy()
    df['date'] = df['timestamp'].dt.date

    # FIX 2: SOC carries across days — initialise once here, not per-day.
    bat_max     = battery_capacity_kwh
    bat_min_abs = battery_capacity_kwh * 0.05
    bat_emergency_abs = battery_capacity_kwh * BAT_EMERGENCY_PCT
    bat_full    = battery_capacity_kwh * BAT_CHARGE_FULL_PCT
    bat_dis_min = battery_capacity_kwh * BAT_DISCHARGE_MIN_PCT
    # FIX 5: C-rate is now a user-configurable parameter (default 0.5C).
    chg_rate_kw = battery_capacity_kwh * c_rate
    dis_rate_kw = battery_capacity_kwh * c_rate
    bat_soc     = battery_capacity_kwh * initial_soc_pct   # only set once at the very start

    # FIX 3: Build a rolling 30-day peak reference per calendar day so the
    # optimizer does not "know" the day's own maximum before it acts.
    # If the caller supplies an absolute ceiling, use that instead.
    df_sorted = df.sort_values('timestamp')
    if peak_reference_kva is not None:
        # Absolute ceiling provided by user — use it for every day.
        df['_ref_peak'] = float(peak_reference_kva)
    else:
        # Rolling 30-day maximum of kva (using data prior to each day).
        # We compute a per-day maximum then roll over calendar days.
        daily_max = df_sorted.groupby('date')['kva'].max().sort_index()
        rolling_ref = daily_max.shift(1).rolling(30, min_periods=1).max()
        # For the very first day there is no prior history — fall back to
        # 110% of the first day's actual peak as a conservative estimate.
        first_day_fallback = daily_max.iloc[0] * 1.10
        rolling_ref = rolling_ref.fillna(first_day_fallback)
        ref_map = rolling_ref.to_dict()
        df['_ref_peak'] = df['date'].map(ref_map)

    for date, day_df in df.groupby('date'):
        day_df = day_df.sort_values('timestamp').reset_index(drop=True)

        # FIX 3: Use the rolling reference peak, not the day's own maximum.
        ref_peak             = day_df['_ref_peak'].iloc[0]
        day_peak             = day_df['kva'].max()           # kept for reporting only
        discharge_trigger    = ref_peak  * peak_target_pct
        discharge_proximity  = ref_peak  * DISCHARGE_PROXIMITY_PCT
        charge_upper         = ref_peak  * bat_charge_upper_pct
        charge_lower         = ref_peak  * bat_charge_lower_pct
        # FIX 7: Normal charging ceiling (92%), emergency ceiling (100%).
        charge_kva_ceiling   = discharge_trigger * CHARGE_PEAK_GUARD_PCT
        emergency_kva_ceiling= discharge_trigger * EMERGENCY_PEAK_GUARD_PCT

        # FIX 4: Build a simple look-ahead profile from today's data.
        # "High-load" intervals are those whose kva ≥ 90% of reference peak.
        lookahead_high = set(
            day_df.index[day_df['kva'] >= ref_peak * 0.90].tolist()
        )

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

            # FIX 1: Track kW and kVAR separately; recompute kVA each time.
            mgd_kw   = kw
            mgd_kvar = kvar
            mgd_kva  = calc_kva(mgd_kw, mgd_kvar)

            # FIX 4: Look-ahead — check if any of the next N intervals are
            # flagged as high-load.  If so, treat *this* interval as also in
            # the peak zone so the battery starts discharging earlier.
            upcoming_high = any(
                (idx + k) in lookahead_high
                for k in range(1, lookahead_intervals + 1)
            )

            # ═══════════════════════════════════════════════════════════════
            # STEP 1 — BATTERY DISCHARGE
            #   Guard 1: kva_orig in peak zone (≥ 80% of reference peak)
            #            OR look-ahead shows a peak coming soon.
            #   Guard 2: current managed kVA ≥ discharge_trigger
            #   Guard 3: SOC > bat_dis_min  (15% floor)
            #   Discharge amount = exact kW needed to reach trigger via a
            #     proper kW/kVAR decomposition, capped by C-rate and SOC.
            # ═══════════════════════════════════════════════════════════════
            in_peak_zone = kva_orig >= discharge_proximity or upcoming_high
            soc_ok_dis   = bat_soc > bat_dis_min

            if in_peak_zone and mgd_kva >= discharge_trigger and soc_ok_dis:
                needed_kva = mgd_kva - discharge_trigger          # kVA gap to close
                avail_kwh  = bat_soc - bat_dis_min

                # FIX 1: Decompose the needed kVA reduction into kW by
                # scaling along the active-power axis (same power factor).
                # dis_kw reduces kw_net; kvar is unchanged.
                pf = (mgd_kw / mgd_kva) if mgd_kva > 0 else 1.0
                needed_kw_reduction = needed_kva * pf   # kW needed to close gap

                dis_kwh    = min(
                    needed_kw_reduction * INTERVAL_H,   # energy to cover gap
                    dis_rate_kw * INTERVAL_H,            # C-rate cap
                    avail_kwh                             # what's in the battery
                )
                dis_kw     = dis_kwh / INTERVAL_H
                soc_before = bat_soc
                bat_soc   -= dis_kwh
                mgd_kw    -= dis_kw
                mgd_kva    = calc_kva(mgd_kw, mgd_kvar)   # recompute properly
                mgd_kva    = max(mgd_kva, 0.0)
                bat_dis_kw = dis_kw
                actions.append({
                    'type':           'battery_discharge',
                    'load':           'Battery',
                    'discharge_kw':   round(dis_kw, 2),
                    'soc_before_kwh': round(soc_before, 1),
                    'soc_after_kwh':  round(bat_soc, 1),
                    'kva_before':     round(kva_orig, 2),
                    'kva_after':      round(mgd_kva, 2),
                    'lookahead_triggered': bool(upcoming_high and kva_orig < discharge_proximity),
                    'note': (f'Battery discharged {round(dis_kw,1)} kW '
                             f'(SOC {round(soc_before,0):.0f}→{round(bat_soc,0):.0f} kWh / '
                             f'{round(bat_soc/bat_max*100,0):.0f}%) '
                             f'→ kVA {round(kva_orig,1)}→{round(mgd_kva,1)}'
                             + (' [look-ahead]' if upcoming_high and kva_orig < discharge_proximity else ''))
                })

            # ═══════════════════════════════════════════════════════════════
            # STEP 2 — LOAD REDUCTION  (if still above target after discharge)
            # FIX 1: Each load cut reduces kW and kVAR proportionally
            # (same power factor as the full load), then kVA is recomputed
            # via √(kW²+kVAR²) rather than being decremented directly.
            # ═══════════════════════════════════════════════════════════════
            remaining_cut = max(0.0, mgd_kva - discharge_trigger)
            if remaining_cut > 1e-3:
                for load in priority_order:
                    if remaining_cut <= 1e-3:
                        break

                    if load == 'misc' and misc_kva > 0:
                        max_possible = misc_kva * max_cut_pct.get('misc', 0.20)
                        cut_kva = min(remaining_cut, max_possible)
                        misc_f  = 1.0 - cut_kva / misc_kva
                        # Decompose cut into kW and kVAR using load's power factor
                        pf_misc   = mgd_kw   / mgd_kva if mgd_kva > 0 else 1.0
                        qf_misc   = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                        mgd_kw   -= cut_kva * pf_misc
                        mgd_kvar -= cut_kva * qf_misc
                        mgd_kva   = calc_kva(mgd_kw, mgd_kvar)
                        remaining_cut = max(0.0, mgd_kva - discharge_trigger)
                        if cut_kva > 0.05:
                            actions.append({
                                'type': 'load_reduction', 'load': 'Miscellaneous',
                                'cut_kva': round(cut_kva,2), 'factor_pct': round(misc_f*100,1),
                                'max_cut_pct': max_cut_pct.get('misc',0.20)*100,
                                'note': (f'Misc cut {round(cut_kva,1)} kVA '
                                         f'(max {round(max_cut_pct.get("misc",0.20)*100,0):.0f}%) '
                                         f'→ {round(misc_f*100,1)}%')
                            })

                    elif load == 'hvac' and hvac_kva > 0:
                        max_possible = hvac_kva * max_cut_pct.get('hvac', 0.15)
                        cut_kva = min(remaining_cut, max_possible)
                        hvac_f  = 1.0 - cut_kva / hvac_kva
                        pf_hvac   = mgd_kw   / mgd_kva if mgd_kva > 0 else 1.0
                        qf_hvac   = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                        mgd_kw   -= cut_kva * pf_hvac
                        mgd_kvar -= cut_kva * qf_hvac
                        mgd_kva   = calc_kva(mgd_kw, mgd_kvar)
                        remaining_cut = max(0.0, mgd_kva - discharge_trigger)
                        if cut_kva > 0.05:
                            actions.append({
                                'type': 'load_reduction', 'load': 'HVAC',
                                'cut_kva': round(cut_kva,2), 'factor_pct': round(hvac_f*100,1),
                                'max_cut_pct': max_cut_pct.get('hvac',0.15)*100,
                                'note': (f'HVAC cut {round(cut_kva,1)} kVA '
                                         f'(max {round(max_cut_pct.get("hvac",0.15)*100,0):.0f}%) '
                                         f'→ {round(hvac_f*100,1)}%')
                            })

                    elif load == 'ev' and ev_kva > 0:
                        max_possible = ev_kva * max_cut_pct.get('ev', 0.10)
                        cut_kva = min(remaining_cut, max_possible)
                        ev_f    = 1.0 - cut_kva / ev_kva
                        pf_ev     = mgd_kw   / mgd_kva if mgd_kva > 0 else 1.0
                        qf_ev     = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                        mgd_kw   -= cut_kva * pf_ev
                        mgd_kvar -= cut_kva * qf_ev
                        mgd_kva   = calc_kva(mgd_kw, mgd_kvar)
                        remaining_cut = max(0.0, mgd_kva - discharge_trigger)
                        if cut_kva > 0.05:
                            actions.append({
                                'type': 'load_reduction', 'load': 'EV Charger',
                                'cut_kva': round(cut_kva,2), 'factor_pct': round(ev_f*100,1),
                                'max_cut_pct': max_cut_pct.get('ev',0.10)*100,
                                'note': (f'EV cut {round(cut_kva,1)} kVA '
                                         f'(max {round(max_cut_pct.get("ev",0.10)*100,0):.0f}%) '
                                         f'→ {round(ev_f*100,1)}%')
                            })

            ev_m   = ev_kva   * ev_f
            hvac_m = hvac_kva * hvac_f
            misc_m = misc_kva * misc_f

            # ═══════════════════════════════════════════════════════════════
            # STEP 3 — BATTERY CHARGE
            #
            # Hard pre-conditions (skip entirely if any fail):
            #   • Not discharging this interval
            #   • Battery is not already full (SOC < bat_max)
            #   • Remaining headroom under the charge ceiling > 0
            #     (ceiling = 92% of discharge_trigger, prevents charger
            #      from itself creating a peak in the 90-95% zone)
            #
            # Trigger:
            #   A) Emergency: SOC < 15% of capacity  (charge regardless of kva)
            #   B) Normal   : kva_original < charge_upper AND SOC < 90%
            #
            # Charge amount = min(0.5C × 0.5h,  headroom_to_ceiling,  room_in_battery)
            # ═══════════════════════════════════════════════════════════════
            if bat_dis_kw == 0 and bat_soc < bat_max:
                emergency_charge = bat_soc < bat_emergency_abs
                # FIX 7: Emergency uses a higher ceiling (100% of target, not 92%).
                active_ceiling   = emergency_kva_ceiling if emergency_charge else charge_kva_ceiling
                # FIX 1: headroom computed against current mgd_kva (already correct kVA)
                kva_headroom = active_ceiling - mgd_kva
                if kva_headroom > 0:
                    normal_charge    = (kva_orig < charge_upper) and (bat_soc < bat_full)

                    if emergency_charge or normal_charge:
                        room_in_bat = bat_max - bat_soc
                        chg_kwh = min(
                            chg_rate_kw * INTERVAL_H,
                            kva_headroom * INTERVAL_H,
                            room_in_bat
                        )
                        if chg_kwh > 0.01:
                            soc_before  = bat_soc
                            bat_soc    += chg_kwh
                            bat_chg_kw  = chg_kwh / INTERVAL_H
                            # FIX 1: charger draws real kW — add to kw_net, recompute kVA.
                            mgd_kw    += bat_chg_kw
                            mgd_kva    = calc_kva(mgd_kw, mgd_kvar)
                            reason = ('Emergency charge (SOC < 15%)'
                                      if emergency_charge else
                                      f'Normal charge (kVA {round(kva_orig,0):.0f} '
                                      f'< {round(charge_upper,0):.0f} upper threshold)')
                            actions.append({
                                'type':             'battery_charge',
                                'load':             'Battery',
                                'charge_kw':        round(bat_chg_kw, 2),
                                'soc_before_kwh':   round(soc_before, 1),
                                'soc_after_kwh':    round(bat_soc, 1),
                                'kva_ceiling':      round(active_ceiling, 2),
                                'kva_after_charge': round(mgd_kva, 2),
                                'charge_trigger':   'emergency' if emergency_charge else 'normal',
                                'note': (f'{reason}: +{round(bat_chg_kw,1)} kW '
                                         f'(SOC {round(soc_before,0):.0f}→{round(bat_soc,0):.0f} kWh / '
                                         f'{round(bat_soc/bat_max*100,0):.0f}%) '
                                         f'kVA now {round(mgd_kva,1)} '
                                         f'(ceiling {round(active_ceiling,1)})')
                            })

            bat_soc    = max(bat_min_abs, min(bat_max, bat_soc))
            bat_action = bat_dis_kw - bat_chg_kw

            # FIX 6: Compute per-load kVAh cut for this interval (kVA × 0.5 h).
            ev_kvah_cut   = (ev_kva   - ev_m)   * INTERVAL_H
            hvac_kvah_cut = (hvac_kva - hvac_m) * INTERVAL_H
            misc_kvah_cut = (misc_kva - misc_m) * INTERVAL_H

            results.append({
                'timestamp':              row['timestamp'].isoformat(),
                'date':                   str(date),
                'kva_original':           round(kva_orig, 2),
                'kw_original':            round(kw, 2),
                'kvar_original':          round(kvar, 2),
                'kw_managed':             round(mgd_kw, 2),
                'kvar_managed':           round(mgd_kvar, 2),
                'ev_kva':                 round(ev_kva, 2),
                'hvac_kva':               round(hvac_kva, 2),
                'misc_kva':              round(misc_kva, 2),
                'ev_managed':             round(ev_m, 2),
                'hvac_managed':           round(hvac_m, 2),
                'misc_managed':           round(misc_m, 2),
                'ev_factor':              round(ev_f, 3),
                'hvac_factor':            round(hvac_f, 3),
                'misc_factor':            round(misc_f, 3),
                # FIX 6: per-load curtailment energy this interval
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


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        f = request.files['file']
        content = f.read()
        df = parse_uploaded_data(content, f.filename)
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
    try:
        data             = request.json
        records          = data.get('records', [])
        proportions      = data.get('proportions', {'ev':0.3,'hvac':0.4,'misc':0.3})
        battery_capacity = float(data.get('battery_capacity_kwh', 200))
        priority_order   = data.get('priority_order', ['misc','hvac','ev'])
        peak_target_pct  = float(data.get('peak_target_pct', 0.85))
        max_cut_pct      = data.get('max_cut_pct', {'ev':0.10,'hvac':0.15,'misc':0.20})
        bat_charge_upper = float(data.get('bat_charge_upper_pct', 0.70))
        bat_charge_lower = float(data.get('bat_charge_lower_pct', 0.60))
        # FIX 5: C-rate is now user-configurable (default 0.5C).
        c_rate           = float(data.get('c_rate', 0.5))
        # FIX 2: Initial SOC for the very first interval (default 50%).
        initial_soc_pct  = float(data.get('initial_soc_pct', 0.50))
        # FIX 3: Optional absolute kVA ceiling; None = use 30-day rolling max.
        peak_ref_kva     = data.get('peak_reference_kva', None)
        if peak_ref_kva is not None:
            peak_ref_kva = float(peak_ref_kva)
        # FIX 4: Number of look-ahead intervals (default 3 = 1.5 hours).
        lookahead        = int(data.get('lookahead_intervals', 3))

        # ensure float values
        max_cut_pct = {k: float(v) for k,v in max_cut_pct.items()}

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
                    'orig_peak':0,'managed_peak':0,
                    'total_discharge_kwh':0,'total_charge_kwh':0,
                    'intervals_battery_discharge':0,'intervals_battery_charge':0,
                    'intervals_load_reduced':0,
                    # FIX 6: aggregate per-load curtailment energy
                    'ev_total_kvah_cut':0,'hvac_total_kvah_cut':0,'misc_total_kvah_cut':0,
                }
            s = day_summaries[d]
            s['orig_peak']    = max(s['orig_peak'],    r['kva_original'])
            s['managed_peak'] = max(s['managed_peak'], r['kva_managed'])
            s['total_discharge_kwh'] += r['battery_discharge_kw'] * 0.5
            s['total_charge_kwh']    += r['battery_charge_kw']    * 0.5
            # FIX 6: accumulate per-load kVAh cuts
            s['ev_total_kvah_cut']   += r.get('ev_kvah_cut',   0)
            s['hvac_total_kvah_cut'] += r.get('hvac_kvah_cut', 0)
            s['misc_total_kvah_cut'] += r.get('misc_kvah_cut', 0)
            for act in r['actions']:
                if act['type']=='battery_discharge': s['intervals_battery_discharge']+=1
                elif act['type']=='battery_charge':  s['intervals_battery_charge']+=1
                elif act['type']=='load_reduction':  s['intervals_load_reduced']+=1

        # Round kVAh summaries
        for s in day_summaries.values():
            s['ev_total_kvah_cut']   = round(s['ev_total_kvah_cut'],   2)
            s['hvac_total_kvah_cut'] = round(s['hvac_total_kvah_cut'], 2)
            s['misc_total_kvah_cut'] = round(s['misc_total_kvah_cut'], 2)

        overall_orig    = max((s['orig_peak']    for s in day_summaries.values()), default=0)
        overall_managed = max((s['managed_peak'] for s in day_summaries.values()), default=0)
        peak_red = ((overall_orig-overall_managed)/overall_orig*100) if overall_orig else 0

        return jsonify({
            'success': True, 'results': results, 'day_summaries': day_summaries,
            'summary': {
                'original_peak_kva':           round(overall_orig,2),
                'managed_peak_kva':            round(overall_managed,2),
                'peak_reduction_pct':          round(peak_red,1),
                'total_battery_discharge_kwh': round(sum(s['total_discharge_kwh'] for s in day_summaries.values()),1),
                'total_battery_charge_kwh':    round(sum(s['total_charge_kwh']    for s in day_summaries.values()),1),
                'avg_original_kva': round(sum(r['kva_original'] for r in results)/len(results),2) if results else 0,
                'avg_managed_kva':  round(sum(r['kva_managed']  for r in results)/len(results),2) if results else 0,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("="*55)
    print("  AI Energy Manager  v4  |  http://localhost:5050")
    print("="*55)
    app.run(debug=True, port=5050)
