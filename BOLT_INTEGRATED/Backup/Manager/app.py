"""
AI Energy Manager – Backend Server (v6)  [ORIGINAL BACKUP – DO NOT MODIFY]

FIXES applied (see flaw report):
  F1 CRITICAL – Discharge kW now uses exact formula: dis_kw = kW − sqrt(T²−kVAR²)
  F2 HIGH     – Charge kW ceiling converted from kVA to kW before applying INTERVAL_H
  F3 HIGH     – Round-trip battery efficiency modelled (default 0.95 one-way = ~90% RT)
  F4 MEDIUM   – Lookahead threshold now tied to peak_target_pct, not hardcoded 0.90
  F5 MEDIUM   – Pre-MD charge boost window (2 h before MD start) added
  F7 LOW      – intervals_load_reduced counts intervals, not individual actions
  F9 LOW      – Day-1 bootstrap commented and degraded gracefully
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import math, traceback
from io import StringIO

app = Flask(__name__, static_folder='.')
CORS(app)

try:
    from flask.json.provider import DefaultJSONProvider
    class _NP(DefaultJSONProvider):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_):    return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    app.json_provider_class = _NP
    app.json = _NP(app)
except ImportError:
    import json
    class _NP(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_):  return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    app.json_encoder = _NP


def calc_kva(kw, kvar):
    return math.sqrt(kw * kw + kvar * kvar)

def _normalize_cols(df):
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace('­', '', regex=False)
                  .str.replace(r'\s+', ' ', regex=True))
    return df

def _is_valid_header_row(cols):
    s = ' '.join(cols)
    has_time   = any(t in s for t in ('date','time','start','end','timestamp','datetime'))
    has_power  = any(t in s for t in ('kw','kvar','kwh','power','watt','energy','var'))
    return has_time and has_power

def _find_datetime_col(df):
    cols = list(df.columns)
    for col in cols:
        if ('date' in col and 'time' in col) or col in ('datetime','timestamp'):
            df[col] = pd.to_datetime(df[col], errors='coerce'); return col, df
    for col in cols:
        if col in ('end_time','end time','start_time','start time'):
            df[col] = pd.to_datetime(df[col], errors='coerce'); return col, df
    date_col = next((c for c in cols if c == 'date' or c.startswith('date')), None)
    time_col = next((c for c in cols if c in ('time','end time','end_time','start_time')
                                and c != date_col), None)
    if date_col and time_col:
        df['_dt'] = pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
        return '_dt', df
    for col in cols[:10]:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() / max(len(parsed), 1) >= 0.8:
                df[col] = parsed; return col, df
        except Exception:
            continue
    raise ValueError(f"No date/time column found. Columns: {cols}")

def _map_power_cols(df):
    col_map = {}
    for col in df.columns:
        is_kvar = 'kvar' in col or ('var' in col and 'kw' not in col)
        is_kw   = ('kw' in col or 'kwh' in col or 'watt' in col or 'active' in col
                   or 'power' in col or 'energy' in col) and not is_kvar
        is_imp, is_exp = 'import' in col, 'export' in col
        if is_kvar and is_imp and 'kvar_import' not in col_map: col_map['kvar_import'] = col
        elif is_kvar and is_exp and 'kvar_export' not in col_map: col_map['kvar_export'] = col
        elif is_kw   and is_imp and 'kw_import'   not in col_map: col_map['kw_import']   = col
        elif is_kw   and is_exp and 'kw_export'   not in col_map: col_map['kw_export']   = col
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
                            df = tmp; break
                    except Exception:
                        continue
                if df is not None: break
            if df is None: raise ValueError("Could not locate a valid header row in CSV.")
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
                        df = tmp; break
                except Exception:
                    continue
            if df is None: raise ValueError("Could not locate a valid header row in Excel.")
        else:
            raise ValueError(f"Unsupported format '{filename}'. Use CSV or Excel.")

        df = df.dropna(how='all').reset_index(drop=True)
        df = df.loc[:, df.columns.notna()]
        df = df.loc[:, df.columns.astype(str) != '']
        start_col, df = _find_datetime_col(df)
        df = df.dropna(subset=[start_col]).sort_values(start_col).reset_index(drop=True)

        col_map = _map_power_cols(df)
        def _s(key):
            col = col_map.get(key)
            if col and col in df.columns:
                return pd.to_numeric(df[col], errors='coerce').fillna(0)
            return pd.Series([0.0] * len(df), index=df.index)

        result = pd.DataFrame({
            'timestamp':     df[start_col].values,
            'kw_import':     _s('kw_import').values,
            'kw_export':     _s('kw_export').values,
            'kvar_import':   _s('kvar_import').values,
            'kvar_export':   _s('kvar_export').values,
        })
        result['kw_net']    = result['kw_import']    - result['kw_export']
        result['kvar_net']  = result['kvar_import']  - result['kvar_export']
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
    except Exception as e:
        raise ValueError(f"Data parsing error: {str(e)}")


def run_ai_manager(df, loads, battery_capacity_kwh, priority_order,
                   peak_target_pct, bat_charge_upper_pct,
                   c_rate=0.5, initial_soc_pct=0.50, bat_efficiency=0.95,
                   peak_reference_kva=None, lookahead_intervals=16,
                   md_start_hour=14, md_end_hour=22, pre_md_hours=2):
    INTERVAL_H                         = 0.5
    BAT_EMERGENCY_PCT                  = 0.15
    BAT_CHARGE_FULL_PCT                = 0.90
    BAT_DISCHARGE_MIN_PCT              = 0.15
    PROX_NORMAL                        = 0.80
    PROX_MD                            = 0.70
    CHARGE_GUARD_PCT                   = 0.92
    EMERG_GUARD_PCT                    = 1.00

    load_keys   = list(loads.keys())
    total_prop  = sum(loads[k].get('proportion', 0) for k in load_keys) or 1
    norm        = {k: loads[k].get('proportion', 0) / total_prop for k in load_keys}

    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['date']   = df['timestamp'].dt.date
    df['hour']   = df['timestamp'].dt.hour
    df['in_md']  = df['hour'].apply(lambda h: md_start_hour <= h < md_end_hour)
    pre_md_start = (md_start_hour - pre_md_hours) % 24
    df['in_pre_md'] = df['hour'].apply(
        lambda h: (pre_md_start <= h < md_start_hour) if pre_md_start < md_start_hour
                  else (h >= pre_md_start or h < md_start_hour))

    if peak_reference_kva is not None:
        df['_ref_peak'] = float(peak_reference_kva)
    else:
        daily_max     = df.groupby('date')['kva'].max().sort_index()
        rolling_ref   = daily_max.shift(1).rolling(30, min_periods=1).max()
        rolling_ref   = rolling_ref.fillna(daily_max.iloc[0] * 1.10)
        df['_ref_peak'] = df['date'].map(rolling_ref.to_dict())

    day_actual_peak = df.groupby('date')['kva'].max().to_dict()

    bat_max               = battery_capacity_kwh
    bat_min_abs           = battery_capacity_kwh * 0.05
    bat_emergency_abs     = battery_capacity_kwh * BAT_EMERGENCY_PCT
    bat_full              = battery_capacity_kwh * BAT_CHARGE_FULL_PCT
    bat_dis_min           = battery_capacity_kwh * BAT_DISCHARGE_MIN_PCT
    chg_rate_kw           = battery_capacity_kwh * c_rate
    dis_rate_kw           = battery_capacity_kwh * c_rate
    bat_soc               = battery_capacity_kwh * initial_soc_pct

    results = []
    n = len(df)

    for idx in range(n):
        row         = df.iloc[idx]
        kva_orig    = float(row['kva'])
        kw          = float(row['kw_net'])
        kvar        = float(row['kvar_net'])
        date        = row['date']
        ts          = row['timestamp']
        ref_peak    = float(row['_ref_peak'])
        in_md       = bool(row['in_md'])
        in_pre_md   = bool(row['in_pre_md'])

        discharge_trigger      = ref_peak * peak_target_pct
        discharge_proximity    = ref_peak * (PROX_MD if in_md else PROX_NORMAL)
        charge_upper           = ref_peak * bat_charge_upper_pct
        charge_kva_ceil        = discharge_trigger * CHARGE_GUARD_PCT
        emerg_kva_ceil         = discharge_trigger * EMERG_GUARD_PCT

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

        in_peak_zone = kva_orig >= discharge_proximity or upcoming_high
        if (in_peak_zone and mgd_kva >= discharge_trigger
            and bat_soc > bat_dis_min and mgd_kw > 0):

            kw_target = math.sqrt(max(discharge_trigger**2 - mgd_kvar**2, 0.0))
            dis_kw_load_needed = max(mgd_kw - kw_target, 0.0)
            dis_kwh_from_bat_needed = dis_kw_load_needed * INTERVAL_H / bat_efficiency
            dis_kwh_from_bat = min(
                dis_kwh_from_bat_needed,
                dis_rate_kw * INTERVAL_H,
                bat_soc - bat_dis_min,
            )
            dis_kwh_load   = dis_kwh_from_bat * bat_efficiency
            dis_kw_load    = dis_kwh_load / INTERVAL_H

            soc_before = bat_soc
            bat_soc   -= dis_kwh_from_bat
            mgd_kw    -= dis_kw_load
            mgd_kw     = max(mgd_kw, 0.0)
            mgd_kva    = calc_kva(mgd_kw, mgd_kvar)
            bat_dis_kw = dis_kw_load

            la_trig = bool(upcoming_high and kva_orig < discharge_proximity)
            actions.append({
                'type':                      'battery_discharge',
                'load':                      'Battery',
                'discharge_kw':              round(dis_kw_load, 2),
                'soc_before_kwh':            round(soc_before, 1),
                'soc_after_kwh':             round(bat_soc, 1),
                'kva_before':                round(kva_orig, 2),
                'kva_after':                 round(mgd_kva, 2),
                'lookahead_triggered':       la_trig,
                'md_hours':                  in_md,
            })

        remaining = max(0.0, mgd_kva - discharge_trigger)
        if remaining > 1e-3:
            for lk in priority_order:
                if remaining <= 1e-3: break
                if lk not in loads or load_kva.get(lk, 0) <= 0: continue
                max_possible = load_kva[lk] * loads[lk].get('max_cut_pct', 0.10)
                cut_kva      = min(remaining, max_possible)
                load_factor[lk] = 1.0 - cut_kva / load_kva[lk]
                pf_l   = mgd_kw  / mgd_kva if mgd_kva > 0 else 1.0
                qf_l   = mgd_kvar / mgd_kva if mgd_kva > 0 else 0.0
                mgd_kw   -= cut_kva * pf_l
                mgd_kvar -= cut_kva * qf_l
                mgd_kw    = max(mgd_kw, 0.0)
                mgd_kva   = calc_kva(mgd_kw, mgd_kvar)
                remaining = max(0.0, mgd_kva - discharge_trigger)
                if cut_kva > 0.05:
                    actions.append({
                        'type':           'load_reduction',
                        'load':           loads[lk].get('name', lk),
                        'load_key':       lk,
                        'cut_kva':        round(cut_kva, 2),
                        'factor_pct':     round(load_factor[lk] * 100, 1),
                    })

        if bat_dis_kw == 0 and bat_soc < bat_max:
            emergency  = bat_soc < bat_emergency_abs
            ceiling    = emerg_kva_ceil if emergency else charge_kva_ceil
            max_chg_kw_grid = (math.sqrt(max(ceiling**2 - mgd_kvar**2, 0.0)) - mgd_kw)
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
                    bat_chg_kw = chg_kw_grid
                    mgd_kw     += chg_kw_grid
                    mgd_kva    = calc_kva(mgd_kw, mgd_kvar)
                    actions.append({
                        'type':                  'battery_charge',
                        'load':                  'Battery',
                        'charge_kw':             round(chg_kw_grid, 2),
                        'soc_before_kwh':        round(soc_before, 1),
                        'soc_after_kwh':         round(bat_soc, 1),
                    })

        bat_soc = max(bat_min_abs, min(bat_max, bat_soc))

        load_managed  = {k: load_kva[k] * load_factor[k] for k in load_keys}
        load_kvah_cut = {k: (load_kva[k] - load_managed[k]) * INTERVAL_H for k in load_keys}

        row_out = {
            'timestamp':               ts.isoformat(),
            'date':                    str(date),
            'kva_original':            round(kva_orig, 2),
            'kw_original':             round(kw, 2),
            'kvar_original':           round(kvar, 2),
            'kw_managed':              round(mgd_kw, 2),
            'kvar_managed':            round(mgd_kvar, 2),
            'battery_action_kw':       round(bat_dis_kw - bat_chg_kw, 2),
            'battery_charge_kw':       round(bat_chg_kw, 2),
            'battery_discharge_kw':    round(bat_dis_kw, 2),
            'battery_soc_kwh':         round(bat_soc, 2),
            'battery_soc_pct':         round(bat_soc / bat_max * 100, 1) if bat_max else 0,
            'kva_managed':             round(mgd_kva, 2),
            'target_peak':             round(discharge_trigger, 2),
            'ref_peak':                round(ref_peak, 2),
            'in_md_hours':             in_md,
            'in_pre_md':               in_pre_md,
            'actions':                 actions,
        }
        for k in load_keys:
            row_out[f'{k}_kva']       = round(load_kva[k], 2)
            row_out[f'{k}_managed']   = round(load_managed[k], 2)
            row_out[f'{k}_factor']    = round(load_factor[k], 3)
            row_out[f'{k}_kvah_cut']  = round(load_kvah_cut[k], 3)
        results.append(row_out)

    return results


@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        f  = request.files['file']
        df = parse_uploaded_data(f.read(), f.filename)
        records = [{'timestamp': row['timestamp'].isoformat(),
                    'kva':         round(float(row['kva']),        2),
                    'kw_net':      round(float(row['kw_net']),     2),
                    'kvar_net':    round(float(row['kvar_net']),   2),
                    'kw_import':   round(float(row['kw_import']),  2),
                    'kw_export':   round(float(row['kw_export']),  2)}
                   for _, row in df.iterrows()]
        dates = sorted(df['timestamp'].dt.date.astype(str).unique().tolist())
        return jsonify({'success': True, 'records': records, 'dates': dates,
                        'total_intervals': len(records),
                        'peak_kva': round(float(df['kva'].max()), 2),
                        'avg_kva':  round(float(df['kva'].mean()), 2)})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/api/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json
        loads = data.get('loads', {
            'ev':    {'name': 'EV Charger',  'proportion': 0.3,  'max_cut_pct': 0.10},
            'hvac':  {'name': 'HVAC',        'proportion': 0.4,  'max_cut_pct': 0.15},
            'misc':  {'name': 'Misc',        'proportion': 0.3,  'max_cut_pct': 0.20},
        })
        for k in loads:
            loads[k]['proportion']  = float(loads[k].get('proportion',  0))
            loads[k]['max_cut_pct'] = float(loads[k].get('max_cut_pct', 0.10))

        records            = data.get('records', [])
        priority_order     = data.get('priority_order', list(loads.keys()))
        battery_capacity   = float(data.get('battery_capacity_kwh', 200))
        peak_target_pct    = float(data.get('peak_target_pct', 0.85))
        bat_charge_upper   = float(data.get('bat_charge_upper_pct', 0.70))
        c_rate             = float(data.get('c_rate', 0.5))
        initial_soc_pct    = float(data.get('initial_soc_pct', 0.50))
        bat_efficiency     = float(data.get('bat_efficiency', 0.95))
        peak_ref_kva       = data.get('peak_reference_kva', None)
        if peak_ref_kva is not None:
            peak_ref_kva = float(peak_ref_kva)
        lookahead          = int(data.get('lookahead_intervals', 16))
        md_start           = int(data.get('md_start_hour', 14))
        md_end             = int(data.get('md_end_hour', 22))
        pre_md_hours       = int(data.get('pre_md_hours', 2))

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        results = run_ai_manager(
            df, loads, battery_capacity, priority_order,
            peak_target_pct, bat_charge_upper,
            c_rate=c_rate, initial_soc_pct=initial_soc_pct,
            bat_efficiency=bat_efficiency,
            peak_reference_kva=peak_ref_kva,
            lookahead_intervals=lookahead,
            md_start_hour=md_start, md_end_hour=md_end,
            pre_md_hours=pre_md_hours
        )

        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


if __name__ == '__main__':
    print("=" * 55)
    print("  AI Energy Manager  v6  |  http://localhost:5050")
    print("=" * 55)
    app.run(debug=True, port=5050)
