"""
AI Energy Manager — pure Python optimization engine (v6).
Extracted from the original Flask backend for direct Streamlit integration.

Battery model notes:
  Discharge: dis_kw = kW - sqrt(max(0, trigger² - kVAR²))  [exact formula, F1]
  Charge kW ceiling converted from kVA to kW [F2]
  Round-trip efficiency modelled at default 0.95 one-way [F3]
  Pre-MD charge boost window (2h before MD start) [F5]
"""
from __future__ import annotations
import math
import io
import numpy as np
import pandas as pd
from io import StringIO


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
        is_kw = (('kw' in col or 'kwh' in col or 'watt' in col or 'active' in col
                  or 'power' in col or 'energy' in col) and not is_kvar)
        is_imp, is_exp = 'import' in col, 'export' in col
        if is_kvar and is_imp and 'kvar_import' not in col_map:
            col_map['kvar_import'] = col
        elif is_kvar and is_exp and 'kvar_export' not in col_map:
            col_map['kvar_export'] = col
        elif is_kw and is_imp and 'kw_import' not in col_map:
            col_map['kw_import'] = col
        elif is_kw and is_exp and 'kw_export' not in col_map:
            col_map['kw_export'] = col
    for key in ('kw_import', 'kw_export', 'kvar_import', 'kvar_export'):
        col_map.setdefault(key, None)
    return col_map


def parse_uploaded_data(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Parse CSV or Excel load profile bytes into a normalized DataFrame.
    Returns columns: timestamp, kw_import, kw_export, kvar_import, kvar_export,
                     kw_net, kvar_net, kva
    """
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
    result['kw_net']   = result['kw_import']   - result['kw_export']
    result['kvar_net'] = result['kvar_import'] - result['kvar_export']
    result['kva']      = result.apply(lambda r: calc_kva(r['kw_net'], r['kvar_net']), axis=1)
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    result = (result.set_index('timestamp')
              .resample('30min').mean()
              .dropna(how='all')
              .reset_index())
    result['kva'] = result.apply(lambda r: calc_kva(r['kw_net'], r['kvar_net']), axis=1)
    if result.empty:
        raise ValueError("Empty after resampling.")
    return result


def run_ai_manager(
    df: pd.DataFrame,
    loads: dict,
    battery_capacity_kwh: float,
    priority_order: list,
    peak_target_pct: float,
    bat_charge_upper_pct: float,
    c_rate: float = 0.5,
    initial_soc_pct: float = 0.50,
    bat_efficiency: float = 0.95,
    peak_reference_kva: float | None = None,
    lookahead_intervals: int = 16,
    md_start_hour: int = 14,
    md_end_hour: int = 22,
    pre_md_hours: int = 2,
) -> list[dict]:
    """
    Sequential per-interval optimization across the full dataset.

    Discharge (F1 — exact formula):
      dis_kw_needed = mgd_kw - sqrt(max(0, trigger² - mgd_kvar²))
    Charge (F2 — exact kW ceiling):
      max_chg_kw_grid = sqrt(max(0, ceiling² - mgd_kvar²)) - mgd_kw
    Efficiency (F3):
      One-way efficiency default 0.95; round-trip ~0.90.
    Pre-MD boost (F5):
      `pre_md_hours` before MD window: charge regardless of charge_upper threshold.
    """
    INTERVAL_H             = 0.5
    BAT_EMERGENCY_PCT      = 0.15
    BAT_CHARGE_FULL_PCT    = 0.90
    BAT_DISCHARGE_MIN_PCT  = 0.15
    PROX_NORMAL            = 0.80
    PROX_MD                = 0.70
    CHARGE_GUARD_PCT       = 0.92
    EMERG_GUARD_PCT        = 1.00

    load_keys = list(loads.keys())
    total_prop = sum(loads[k].get('proportion', 0) for k in load_keys) or 1
    norm = {k: loads[k].get('proportion', 0) / total_prop for k in load_keys}

    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['date']  = df['timestamp'].dt.date
    df['hour']  = df['timestamp'].dt.hour
    df['in_md'] = df['hour'].apply(lambda h: md_start_hour <= h < md_end_hour)

    pre_md_start = (md_start_hour - pre_md_hours) % 24
    df['in_pre_md'] = df['hour'].apply(
        lambda h: (pre_md_start <= h < md_start_hour) if pre_md_start < md_start_hour
                  else (h >= pre_md_start or h < md_start_hour))

    if peak_reference_kva is not None:
        df['_ref_peak'] = float(peak_reference_kva)
    else:
        daily_max   = df.groupby('date')['kva'].max().sort_index()
        rolling_ref = daily_max.shift(1).rolling(30, min_periods=1).max()
        rolling_ref = rolling_ref.fillna(daily_max.iloc[0] * 1.10)
        df['_ref_peak'] = df['date'].map(rolling_ref.to_dict())

    day_actual_peak = df.groupby('date')['kva'].max().to_dict()

    bat_max           = battery_capacity_kwh
    bat_min_abs       = battery_capacity_kwh * 0.05
    bat_emergency_abs = battery_capacity_kwh * BAT_EMERGENCY_PCT
    bat_full          = battery_capacity_kwh * BAT_CHARGE_FULL_PCT
    bat_dis_min       = battery_capacity_kwh * BAT_DISCHARGE_MIN_PCT
    chg_rate_kw       = battery_capacity_kwh * c_rate
    dis_rate_kw       = battery_capacity_kwh * c_rate
    bat_soc           = battery_capacity_kwh * initial_soc_pct

    results = []
    n = len(df)

    for idx in range(n):
        row      = df.iloc[idx]
        kva_orig = float(row['kva'])
        kw       = float(row['kw_net'])
        kvar     = float(row['kvar_net'])
        date     = row['date']
        ts       = row['timestamp']
        ref_peak = float(row['_ref_peak'])
        in_md    = bool(row['in_md'])
        in_pre_md = bool(row['in_pre_md'])

        discharge_trigger   = ref_peak * peak_target_pct
        discharge_proximity = ref_peak * (PROX_MD if in_md else PROX_NORMAL)
        charge_upper        = ref_peak * bat_charge_upper_pct
        charge_kva_ceil     = discharge_trigger * CHARGE_GUARD_PCT
        emerg_kva_ceil      = discharge_trigger * EMERG_GUARD_PCT

        load_kva    = {k: kva_orig * norm[k] for k in load_keys}
        load_factor = {k: 1.0 for k in load_keys}

        bat_chg_kw = bat_dis_kw = 0.0
        actions = []
        mgd_kw   = kw
        mgd_kvar = kvar
        mgd_kva  = calc_kva(mgd_kw, mgd_kvar)

        upcoming_high = False
        for fi in range(idx + 1, min(idx + lookahead_intervals + 1, n)):
            frow = df.iloc[fi]
            if float(frow['kva']) >= float(frow['_ref_peak']) * peak_target_pct:
                upcoming_high = True
                break

        # STEP 1: DISCHARGE
        in_peak_zone = kva_orig >= discharge_proximity or upcoming_high
        if (in_peak_zone and mgd_kva >= discharge_trigger
                and bat_soc > bat_dis_min and mgd_kw > 0):

            kw_target = math.sqrt(max(discharge_trigger ** 2 - mgd_kvar ** 2, 0.0))
            dis_kw_load_needed = max(mgd_kw - kw_target, 0.0)

            dis_kwh_from_bat_needed = dis_kw_load_needed * INTERVAL_H / bat_efficiency
            dis_kwh_from_bat = min(
                dis_kwh_from_bat_needed,
                dis_rate_kw * INTERVAL_H,
                bat_soc - bat_dis_min
            )
            dis_kwh_load = dis_kwh_from_bat * bat_efficiency
            dis_kw_load  = dis_kwh_load / INTERVAL_H

            soc_before = bat_soc
            bat_soc   -= dis_kwh_from_bat
            mgd_kw    -= dis_kw_load
            mgd_kw     = max(mgd_kw, 0.0)
            mgd_kva    = calc_kva(mgd_kw, mgd_kvar)
            bat_dis_kw = dis_kw_load

            la_trig = bool(upcoming_high and kva_orig < discharge_proximity)
            actions.append({
                'type':                'battery_discharge',
                'load':                'Battery',
                'discharge_kw':        round(dis_kw_load, 2),
                'soc_before_kwh':      round(soc_before, 1),
                'soc_after_kwh':       round(bat_soc, 1),
                'kva_before':          round(kva_orig, 2),
                'kva_after':           round(mgd_kva, 2),
                'lookahead_triggered': la_trig,
                'md_hours':            in_md,
            })

        # STEP 2: LOAD REDUCTION
        remaining = max(0.0, mgd_kva - discharge_trigger)
        if remaining > 1e-3:
            for lk in priority_order:
                if remaining <= 1e-3:
                    break
                if lk not in loads or load_kva.get(lk, 0) <= 0:
                    continue
                max_possible = load_kva[lk] * loads[lk].get('max_cut_pct', 0.10)
                cut_kva = min(remaining, max_possible)
                load_factor[lk] = 1.0 - cut_kva / load_kva[lk]
                pf_l = mgd_kw   / mgd_kva if mgd_kva > 0 else 1.0
                qf_l = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                mgd_kw   -= cut_kva * pf_l
                mgd_kvar -= cut_kva * qf_l
                mgd_kw    = max(mgd_kw, 0.0)
                mgd_kva   = calc_kva(mgd_kw, mgd_kvar)
                remaining = max(0.0, mgd_kva - discharge_trigger)
                if cut_kva > 0.05:
                    actions.append({
                        'type':        'load_reduction',
                        'load':        loads[lk].get('name', lk),
                        'load_key':    lk,
                        'cut_kva':     round(cut_kva, 2),
                        'factor_pct':  round(load_factor[lk] * 100, 1),
                        'max_cut_pct': loads[lk].get('max_cut_pct', 0.10) * 100,
                    })

        # STEP 3: CHARGE
        if bat_dis_kw == 0 and bat_soc < bat_max:
            emergency = bat_soc < bat_emergency_abs
            ceiling   = emerg_kva_ceil if emergency else charge_kva_ceil

            max_chg_kw_grid = (math.sqrt(max(ceiling ** 2 - mgd_kvar ** 2, 0.0)) - mgd_kw)
            max_chg_kwh_stored = max(max_chg_kw_grid, 0.0) * bat_efficiency * INTERVAL_H

            off_peak_ok = (kva_orig < charge_upper) or in_pre_md
            normal = off_peak_ok and (bat_soc < bat_full) and not in_md

            if max_chg_kwh_stored > 0.001 and (emergency or normal):
                chg_kwh_stored = min(
                    max_chg_kwh_stored,
                    chg_rate_kw * INTERVAL_H,
                    bat_max - bat_soc
                )
                if chg_kwh_stored > 0.01:
                    chg_kw_grid = chg_kwh_stored / bat_efficiency / INTERVAL_H
                    soc_before  = bat_soc
                    bat_soc    += chg_kwh_stored
                    bat_chg_kw  = chg_kw_grid
                    mgd_kw     += chg_kw_grid
                    mgd_kva     = calc_kva(mgd_kw, mgd_kvar)
                    trigger_str = ('emergency' if emergency else
                                   'pre-MD boost' if in_pre_md else 'normal')
                    actions.append({
                        'type':             'battery_charge',
                        'load':             'Battery',
                        'charge_kw':        round(chg_kw_grid, 2),
                        'soc_before_kwh':   round(soc_before, 1),
                        'soc_after_kwh':    round(bat_soc, 1),
                        'kva_ceiling':      round(ceiling, 2),
                        'kva_after_charge': round(mgd_kva, 2),
                        'charge_trigger':   trigger_str,
                    })

        bat_soc = max(bat_min_abs, min(bat_max, bat_soc))

        load_managed  = {k: load_kva[k] * load_factor[k] for k in load_keys}
        load_kwah_cut = {k: (load_kva[k] - load_managed[k]) * INTERVAL_H for k in load_keys}

        row_out = {
            'timestamp':              ts.isoformat(),
            'date':                   str(date),
            'kva_original':           round(kva_orig, 2),
            'kw_original':            round(kw, 2),
            'kvar_original':          round(kvar, 2),
            'kw_managed':             round(mgd_kw, 2),
            'kvar_managed':           round(mgd_kvar, 2),
            'battery_action_kw':      round(bat_dis_kw - bat_chg_kw, 2),
            'battery_charge_kw':      round(bat_chg_kw, 2),
            'battery_discharge_kw':   round(bat_dis_kw, 2),
            'battery_soc_kwh':        round(bat_soc, 2),
            'battery_soc_pct':        round(bat_soc / bat_max * 100, 1) if bat_max else 0,
            'kva_managed':            round(mgd_kva, 2),
            'target_peak':            round(discharge_trigger, 2),
            'ref_peak':               round(ref_peak, 2),
            'in_md_hours':            in_md,
            'in_pre_md':              in_pre_md,
            'actions':                actions,
        }
        for k in load_keys:
            row_out[f'{k}_kva']      = round(load_kva[k], 2)
            row_out[f'{k}_managed']  = round(load_managed[k], 2)
            row_out[f'{k}_factor']   = round(load_factor[k], 3)
            row_out[f'{k}_kwah_cut'] = round(load_kwah_cut[k], 3)
        results.append(row_out)

    return results
