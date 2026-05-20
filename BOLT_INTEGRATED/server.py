"""
BOLT Integrated — Unified Flask Backend  (server.py)

Endpoints
---------
GET  /                     → serve static/index.html
GET  /api/status           → pipeline stage flags + tariff list
POST /api/upload           → parse load-profile file (Manager-compatible response)
POST /api/optimize         → AI Manager peak-shaving  (Manager-compatible contract)
POST /api/predictor/train  → train LightGBM forecaster
POST /api/predictor/forecast → 48-step (24 h) forecast
POST /api/calculator/bill  → TNB bill before/after
POST /api/powerreco/run    → solar + battery sizing + 25-yr ROI

Run:  python server.py
Open: http://localhost:5000
"""
from __future__ import annotations
import io, json, math, traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── numpy→JSON serialisation ──────────────────────────────────────────────────
from flask.json.provider import DefaultJSONProvider

class _NP(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_):    return bool(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

BASE = Path(__file__).parent
app  = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="")
app.json_provider_class = _NP
app.json = _NP(app)
CORS(app)

# ── BOLT module imports ───────────────────────────────────────────────────────
from predictor.forecaster import DirectMultiStepForecaster
from predictor.cv import expanding_window_cv, format_cv_report
from predictor.solar_estimator import detect_has_solar, estimate_solar_capacity_kwp

from calculator.tnb_tariffs import (
    auto_detect_tariff, compute_monthly_stats, calculate_bill,
    compute_nem_credit, TARIFF_META,
)
from powerreco.solar_sizing import calculate_solar_sizing
from powerreco.battery_sizing import calculate_battery_sizing
from powerreco.roi_engine import calculate_roi

# ── In-memory state (single-user local server) ────────────────────────────────
S: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# MANAGER FUNCTIONS  (ported verbatim from Manager/app.py — keep exact contract)
# ══════════════════════════════════════════════════════════════════════════════

def _calc_kva(kw: float, kvar: float) -> float:
    return math.sqrt(kw * kw + kvar * kvar)

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace('﻿','', regex=False)
                  .str.replace(r'\s+', ' ', regex=True))
    return df

def _is_valid_header_row(cols: list) -> bool:
    s = ' '.join(cols)
    has_time  = any(t in s for t in ('date','time','start','end','timestamp','datetime'))
    has_power = any(t in s for t in ('kw','kvar','kwh','power','watt','energy','var'))
    return has_time and has_power

def _find_datetime_col(df: pd.DataFrame):
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
            df[date_col].astype(str)+' '+df[time_col].astype(str), errors='coerce')
        return '_dt', df
    for col in cols[:10]:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() / max(len(parsed),1) >= 0.8:
                df[col] = parsed; return col, df
        except Exception:
            continue
    raise ValueError(f"No date/time column found. Columns: {cols}")

def _map_power_cols(df: pd.DataFrame) -> dict:
    col_map: dict = {}
    for col in df.columns:
        is_kvar = 'kvar' in col or ('var' in col and 'kw' not in col)
        is_kw   = ('kw' in col or 'kwh' in col or 'watt' in col or 'active' in col
                   or 'power' in col or 'energy' in col) and not is_kvar
        is_imp, is_exp = 'import' in col, 'export' in col
        if is_kvar and is_imp and 'kvar_import' not in col_map: col_map['kvar_import'] = col
        elif is_kvar and is_exp and 'kvar_export' not in col_map: col_map['kvar_export'] = col
        elif is_kw   and is_imp and 'kw_import'   not in col_map: col_map['kw_import']   = col
        elif is_kw   and is_exp and 'kw_export'   not in col_map: col_map['kw_export']   = col
    for key in ('kw_import','kw_export','kvar_import','kvar_export'):
        col_map.setdefault(key, None)
    return col_map

def _mgr_parse_upload(file_content: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded load-profile file; returns DataFrame with kw_net/kvar_net/kva."""
    df = None
    if filename.lower().endswith('.csv'):
        for hrow in range(12):
            for enc in ('utf-8-sig','utf-8','latin-1','cp1252'):
                try:
                    tmp = pd.read_csv(io.StringIO(file_content.decode(enc)), header=hrow)
                    tmp = _normalize_cols(tmp)
                    if len(tmp.columns) >= 4 and _is_valid_header_row(list(tmp.columns)):
                        df = tmp; break
                except Exception:
                    continue
            if df is not None: break
        if df is None: raise ValueError("Could not locate a valid header row in CSV.")
    elif filename.lower().endswith(('.xlsx','.xls')):
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
        return pd.Series([0.0]*len(df), index=df.index)

    result = pd.DataFrame({
        'timestamp':   df[start_col].values,
        'kw_import':   _s('kw_import').values,
        'kw_export':   _s('kw_export').values,
        'kvar_import': _s('kvar_import').values,
        'kvar_export': _s('kvar_export').values,
    })
    result['kw_net']   = result['kw_import']  - result['kw_export']
    result['kvar_net'] = result['kvar_import'] - result['kvar_export']
    result['kva']      = result.apply(lambda r: _calc_kva(r['kw_net'], r['kvar_net']), axis=1)
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    result = (result.set_index('timestamp').resample('30min').mean()
              .dropna(how='all').reset_index())
    result['kva'] = result.apply(lambda r: _calc_kva(r['kw_net'], r['kvar_net']), axis=1)
    if result.empty:
        raise ValueError("Empty DataFrame after resampling.")
    return result


def _mgr_run(df: pd.DataFrame, loads: dict, battery_capacity_kwh: float,
             priority_order: list, peak_target_pct: float, bat_charge_upper_pct: float,
             c_rate=0.5, initial_soc_pct=0.50, bat_efficiency=0.95,
             peak_reference_kva=None, lookahead_intervals=16,
             md_start_hour=14, md_end_hour=22, pre_md_hours=2) -> list[dict]:
    """Manager v6 AI optimizer — exact replica of Manager/app.py run_ai_manager."""
    INTERVAL_H=0.5; BAT_EMERGENCY_PCT=0.15; BAT_CHARGE_FULL_PCT=0.90
    BAT_DISCHARGE_MIN_PCT=0.15; PROX_NORMAL=0.80; PROX_MD=0.70
    CHARGE_GUARD_PCT=0.92; EMERG_GUARD_PCT=1.00

    load_keys  = list(loads.keys())
    total_prop = sum(loads[k].get('proportion',0) for k in load_keys) or 1
    norm       = {k: loads[k].get('proportion',0)/total_prop for k in load_keys}

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
        rolling_ref = rolling_ref.fillna(daily_max.iloc[0]*1.10)
        df['_ref_peak'] = df['date'].map(rolling_ref.to_dict())

    day_actual_peak = df.groupby('date')['kva'].max().to_dict()

    bat_max=battery_capacity_kwh; bat_min_abs=battery_capacity_kwh*0.05
    bat_emergency_abs=battery_capacity_kwh*BAT_EMERGENCY_PCT
    bat_full=battery_capacity_kwh*BAT_CHARGE_FULL_PCT
    bat_dis_min=battery_capacity_kwh*BAT_DISCHARGE_MIN_PCT
    chg_rate_kw=battery_capacity_kwh*c_rate
    dis_rate_kw=battery_capacity_kwh*c_rate
    bat_soc=battery_capacity_kwh*initial_soc_pct

    results, n = [], len(df)
    for idx in range(n):
        row=df.iloc[idx]; kva_orig=float(row['kva']); kw=float(row['kw_net'])
        kvar=float(row['kvar_net']); date=row['date']; ts=row['timestamp']
        ref_peak=float(row['_ref_peak']); in_md=bool(row['in_md']); in_pre_md=bool(row['in_pre_md'])

        discharge_trigger=ref_peak*peak_target_pct
        discharge_proximity=ref_peak*(PROX_MD if in_md else PROX_NORMAL)
        charge_upper=ref_peak*bat_charge_upper_pct
        charge_kva_ceil=discharge_trigger*CHARGE_GUARD_PCT
        emerg_kva_ceil=discharge_trigger*EMERG_GUARD_PCT

        load_kva={k: kva_orig*norm[k] for k in load_keys}
        load_factor={k: 1.0 for k in load_keys}
        bat_chg_kw=bat_dis_kw=0.0; actions=[]
        mgd_kw=kw; mgd_kvar=kvar; mgd_kva=_calc_kva(mgd_kw,mgd_kvar)

        upcoming_high=False
        for fi in range(idx+1, min(idx+lookahead_intervals+1, n)):
            frow=df.iloc[fi]
            if float(frow['kva']) >= float(frow['_ref_peak'])*peak_target_pct:
                upcoming_high=True; break

        # Step 1: Discharge
        in_peak_zone = kva_orig >= discharge_proximity or upcoming_high
        if (in_peak_zone and mgd_kva >= discharge_trigger
                and bat_soc > bat_dis_min and mgd_kw > 0):
            kw_target=math.sqrt(max(discharge_trigger**2-mgd_kvar**2,0.0))
            dis_kw_load_needed=max(mgd_kw-kw_target,0.0)
            dis_kwh_from_bat_needed=dis_kw_load_needed*INTERVAL_H/bat_efficiency
            dis_kwh_from_bat=min(dis_kwh_from_bat_needed,dis_rate_kw*INTERVAL_H,bat_soc-bat_dis_min)
            dis_kwh_load=dis_kwh_from_bat*bat_efficiency; dis_kw_load=dis_kwh_load/INTERVAL_H
            soc_before=bat_soc; bat_soc-=dis_kwh_from_bat; mgd_kw-=dis_kw_load
            mgd_kw=max(mgd_kw,0.0); mgd_kva=_calc_kva(mgd_kw,mgd_kvar); bat_dis_kw=dis_kw_load
            la_trig=bool(upcoming_high and kva_orig<discharge_proximity)
            actions.append({'type':'battery_discharge','load':'Battery',
                'discharge_kw':round(dis_kw_load,2),'soc_before_kwh':round(soc_before,1),
                'soc_after_kwh':round(bat_soc,1),'kva_before':round(kva_orig,2),
                'kva_after':round(mgd_kva,2),'lookahead_triggered':la_trig,'md_hours':in_md,
                'note':(f'Bat discharged {round(dis_kw_load,1)}kW; SOC '
                        f'{round(soc_before,0):.0f}→{round(bat_soc,0):.0f}kWh; '
                        f'kVA {round(kva_orig,1)}→{round(mgd_kva,1)}'
                        +(' [look-ahead]' if la_trig else '')+(' [MD hrs]' if in_md else ''))})

        # Step 2: Load reduction
        remaining=max(0.0, mgd_kva-discharge_trigger)
        if remaining > 1e-3:
            for lk in priority_order:
                if remaining <= 1e-3: break
                if lk not in loads or load_kva.get(lk,0) <= 0: continue
                max_possible=load_kva[lk]*loads[lk].get('max_cut_pct',0.10)
                cut_kva=min(remaining,max_possible)
                load_factor[lk]=1.0-cut_kva/load_kva[lk]
                pf_l=mgd_kw/mgd_kva if mgd_kva>0 else 1.0
                qf_l=mgd_kvar/mgd_kva if mgd_kva>0 else 0.0
                mgd_kw-=cut_kva*pf_l; mgd_kvar-=cut_kva*qf_l
                mgd_kw=max(mgd_kw,0.0); mgd_kva=_calc_kva(mgd_kw,mgd_kvar)
                remaining=max(0.0,mgd_kva-discharge_trigger)
                if cut_kva>0.05:
                    actions.append({'type':'load_reduction','load':loads[lk].get('name',lk),
                        'load_key':lk,'cut_kva':round(cut_kva,2),
                        'factor_pct':round(load_factor[lk]*100,1),
                        'max_cut_pct':loads[lk].get('max_cut_pct',0.10)*100,
                        'note':(f'{loads[lk].get("name",lk)} cut {round(cut_kva,1)}kVA; '
                                f'factor {round(load_factor[lk]*100,1)}%')})

        # Step 3: Charge
        if bat_dis_kw == 0 and bat_soc < bat_max:
            emergency=bat_soc<bat_emergency_abs
            ceiling=emerg_kva_ceil if emergency else charge_kva_ceil
            max_chg_kw_grid=math.sqrt(max(ceiling**2-mgd_kvar**2,0.0))-mgd_kw
            max_chg_kwh_stored=max(max_chg_kw_grid,0.0)*bat_efficiency*INTERVAL_H
            off_peak_ok=(kva_orig<charge_upper) or in_pre_md
            normal=off_peak_ok and (bat_soc<bat_full) and not in_md
            if max_chg_kwh_stored>0.001 and (emergency or normal):
                chg_kwh_stored=min(max_chg_kwh_stored,chg_rate_kw*INTERVAL_H,bat_max-bat_soc)
                if chg_kwh_stored>0.01:
                    chg_kw_grid=chg_kwh_stored/bat_efficiency/INTERVAL_H
                    soc_before=bat_soc; bat_soc+=chg_kwh_stored; bat_chg_kw=chg_kw_grid
                    mgd_kw+=chg_kw_grid; mgd_kva=_calc_kva(mgd_kw,mgd_kvar)
                    trigger_str='emergency' if emergency else 'pre-MD boost' if in_pre_md else 'normal'
                    actions.append({'type':'battery_charge','load':'Battery',
                        'charge_kw':round(chg_kw_grid,2),'soc_before_kwh':round(soc_before,1),
                        'soc_after_kwh':round(bat_soc,1),'kva_ceiling':round(ceiling,2),
                        'kva_after_charge':round(mgd_kva,2),'charge_trigger':trigger_str,
                        'note':(f'{"Emergency" if emergency else "Pre-MD" if in_pre_md else "Normal"} charge '
                                f'{round(chg_kw_grid,1)}kW grid; stores {round(chg_kwh_stored,1)}kWh; '
                                f'SOC {round(soc_before,0):.0f}→{round(bat_soc,0):.0f}kWh')})

        bat_soc=max(bat_min_abs,min(bat_max,bat_soc))
        load_managed ={k: load_kva[k]*load_factor[k] for k in load_keys}
        load_kvah_cut={k: (load_kva[k]-load_managed[k])*INTERVAL_H for k in load_keys}

        row_out={'timestamp':ts.isoformat(),'date':str(date),
                 'kva_original':round(kva_orig,2),'kw_original':round(kw,2),
                 'kvar_original':round(kvar,2),'kw_managed':round(mgd_kw,2),
                 'kvar_managed':round(mgd_kvar,2),
                 'battery_action_kw':round(bat_dis_kw-bat_chg_kw,2),
                 'battery_charge_kw':round(bat_chg_kw,2),
                 'battery_discharge_kw':round(bat_dis_kw,2),
                 'battery_soc_kwh':round(bat_soc,2),
                 'battery_soc_pct':round(bat_soc/bat_max*100,1) if bat_max else 0,
                 'kva_managed':round(mgd_kva,2),
                 'target_peak':round(discharge_trigger,2),
                 'ref_peak':round(ref_peak,2),
                 'charge_threshold_upper':round(charge_upper,2),
                 'day_peak':round(day_actual_peak.get(date,kva_orig),2),
                 'bat_capacity':round(bat_max,2),
                 'in_md_hours':in_md,'in_pre_md':in_pre_md,'actions':actions}
        for k in load_keys:
            row_out[f'{k}_kva']      = round(load_kva[k],2)
            row_out[f'{k}_managed']  = round(load_managed[k],2)
            row_out[f'{k}_factor']   = round(load_factor[k],3)
            row_out[f'{k}_kvah_cut'] = round(load_kvah_cut[k],3)
        results.append(row_out)
    return results


# ── General helpers ───────────────────────────────────────────────────────────
def _sf(v, d=0.0) -> float:
    try:
        f = float(v)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return d

def _ok(d: dict):  return jsonify({'ok': True,  **d}), 200
def _err(msg: str, code=400): return jsonify({'ok': False, 'error': msg}), code

def _downsample(df: pd.DataFrame, x_col: str, *y_cols, max_pts=1200) -> dict:
    step = max(1, len(df)//max_pts)
    sub  = df.iloc[::step].copy()
    out  = {'labels': sub[x_col].astype(str).tolist()}
    for c in y_cols:
        if c in sub.columns:
            out[c] = [round(_sf(v),3) for v in
                      sub[c].replace([np.inf,-np.inf], np.nan).fillna(0)]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — Static
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — API
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/status')
def api_status():
    summ = S.get('file_summary') or {}
    return _ok({
        'stages': {
            'data':       S.get('df') is not None,
            'predictor':  S.get('forecaster') is not None,
            'forecast':   S.get('forecast_result') is not None,
            'manager':    S.get('manager_results') is not None,
            'calculator': S.get('bill_rows') is not None,
            'powerreco':  S.get('roi') is not None,
        },
        'file_summary': summ,
        'tariff_code':  S.get('tariff_code','C1'),
        'tariff_meta':  {k: {'name': v['name']} for k,v in TARIFF_META.items()},
        'solar_kwp':    S.get('solar_kwp', 0),
    })


# ── Upload (Manager-compatible) ───────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return _err('No file in request')
    f = request.files['file']
    raw = f.read()
    try:
        df = _mgr_parse_upload(raw, f.filename)
    except Exception as e:
        return _err(f'Parse error: {e}')

    S['df'] = df
    S['manager_results'] = None
    S['bill_rows']       = None
    S['forecast_result'] = None
    S['forecaster']      = None
    S['roi']             = None

    # Auto-detect tariff & solar using BOLT modules
    try:
        tc, tr, ts = auto_detect_tariff(df)
        S['tariff_code'] = tc
    except Exception:
        tc = 'C1'; S['tariff_code'] = tc

    try:
        has_solar, solar_reason = detect_has_solar(df)
        solar_kwp = 0.0
        if has_solar:
            sol = estimate_solar_capacity_kwp(df)
            solar_kwp = _sf(sol.get('capacity_kwp', 0))
        S['solar_kwp'] = solar_kwp
    except Exception:
        has_solar = False; solar_reason = ''; solar_kwp = 0.0; S['solar_kwp'] = 0.0

    # Build Manager-compatible records list
    records = []
    for _, r in df.iterrows():
        records.append({
            'timestamp':  r['timestamp'].isoformat(),
            'kva':        round(_sf(r['kva']),     2),
            'kw_net':     round(_sf(r['kw_net']),  2),
            'kvar_net':   round(_sf(r['kvar_net']),2),
            'kw_import':  round(_sf(r['kw_import']),2),
            'kw_export':  round(_sf(r['kw_export']),2),
        })
    dates = sorted(df['timestamp'].dt.date.astype(str).unique().tolist())

    S['file_summary'] = {
        'rows':           len(df),
        'days':           len(dates),
        'start':          dates[0]  if dates else '',
        'end':            dates[-1] if dates else '',
        'max_kw_import':  round(_sf(df['kw_import'].max()),  2),
        'mean_kw_import': round(_sf(df['kw_import'].mean()), 2),
        'peak_kva':       round(_sf(df['kva'].max()),         2),
    }

    return _ok({
        # Manager-compatible fields
        'success':         True,
        'records':         records,
        'dates':           dates,
        'total_intervals': len(records),
        'peak_kva':        round(_sf(df['kva'].max()),  2),
        'avg_kva':         round(_sf(df['kva'].mean()), 2),
        # Extra BOLT fields
        'summary':         S['file_summary'],
        'tariff':          tc,
        'tariff_meta':     {k: {'name': v['name']} for k,v in TARIFF_META.items()},
        'solar': {
            'has_solar':    has_solar,
            'reason':       solar_reason,
            'capacity_kwp': solar_kwp,
        },
    })


# ── Optimize (Manager-compatible contract) ────────────────────────────────────
@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.get_json(silent=True) or {}
    records = data.get('records', [])
    if not records and S.get('df') is not None:
        df = S['df']
    elif records:
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ensure kva column exists
        if 'kva' not in df.columns:
            kw   = df.get('kw_net',  df.get('kw_import',  pd.Series(0.0, index=df.index)))
            kvar = df.get('kvar_net',df.get('kvar_import', pd.Series(0.0, index=df.index)))
            df['kva']     = np.sqrt(kw**2 + kvar**2)
            df['kw_net']  = kw
            df['kvar_net']= kvar
    else:
        return _err('No data — upload a file first', 400)

    loads_raw = data.get('loads', {
        'ev':   {'name':'EV Charger','proportion':0.30,'max_cut_pct':0.10},
        'hvac': {'name':'HVAC',      'proportion':0.40,'max_cut_pct':0.15},
        'misc': {'name':'Misc',      'proportion':0.30,'max_cut_pct':0.20},
    })
    loads = {}
    for k, v in loads_raw.items():
        loads[k] = {
            'name':        v.get('name', k),
            'proportion':  float(v.get('proportion',  0)),
            'max_cut_pct': float(v.get('max_cut_pct', 0.10)),
            'color':       v.get('color','#00d4ff'),
        }

    priority_order  = data.get('priority_order', list(loads.keys()))
    battery_kwh     = float(data.get('battery_capacity_kwh', 200))
    peak_target     = float(data.get('peak_target_pct',      0.85))
    bat_charge_upper= float(data.get('bat_charge_upper_pct', 0.70))
    c_rate          = float(data.get('c_rate',               0.50))
    init_soc        = float(data.get('initial_soc_pct',      0.50))
    bat_eff         = float(data.get('bat_efficiency',        0.95))
    peak_ref_kva    = data.get('peak_reference_kva', None)
    if peak_ref_kva is not None:
        peak_ref_kva = float(peak_ref_kva)
    lookahead       = int(data.get('lookahead_intervals', 16))
    md_start        = int(data.get('md_start_hour',       14))
    md_end          = int(data.get('md_end_hour',         22))
    pre_md_h        = int(data.get('pre_md_hours',         2))

    try:
        results = _mgr_run(df, loads, battery_kwh, priority_order,
                           peak_target, bat_charge_upper,
                           c_rate=c_rate, initial_soc_pct=init_soc,
                           bat_efficiency=bat_eff,
                           peak_reference_kva=peak_ref_kva,
                           lookahead_intervals=lookahead,
                           md_start_hour=md_start, md_end_hour=md_end,
                           pre_md_hours=pre_md_h)
    except Exception as e:
        return _err(f'Optimization failed: {e}\n{traceback.format_exc()}')

    S['manager_results'] = results
    S['manager_loads']   = loads

    load_keys = list(loads.keys())
    day_summaries: dict = {}
    for r in results:
        d = r['date']
        if d not in day_summaries:
            day_summaries[d] = {
                'orig_peak':0,'managed_peak':0,'orig_md_peak':0,'managed_md_peak':0,
                'total_discharge_kwh':0,'total_charge_kwh':0,
                'intervals_battery_discharge':0,'intervals_battery_charge':0,
                'intervals_load_reduced':0,
            }
            for k in load_keys:
                day_summaries[d][f'{k}_total_kvah_cut'] = 0

        s = day_summaries[d]
        s['orig_peak']    = max(s['orig_peak'],    r['kva_original'])
        s['managed_peak'] = max(s['managed_peak'], r['kva_managed'])
        if r.get('in_md_hours'):
            s['orig_md_peak']    = max(s['orig_md_peak'],    r['kva_original'])
            s['managed_md_peak'] = max(s['managed_md_peak'], r['kva_managed'])
        s['total_discharge_kwh'] += r['battery_discharge_kw'] * 0.5
        s['total_charge_kwh']    += r['battery_charge_kw']    * 0.5
        for k in load_keys:
            s[f'{k}_total_kvah_cut'] += r.get(f'{k}_kvah_cut', 0)
        acts = r['actions']
        if any(a['type']=='battery_discharge' for a in acts): s['intervals_battery_discharge']+=1
        if any(a['type']=='battery_charge'    for a in acts): s['intervals_battery_charge']   +=1
        if any(a['type']=='load_reduction'    for a in acts): s['intervals_load_reduced']     +=1

    for s in day_summaries.values():
        for k in load_keys:
            s[f'{k}_total_kvah_cut'] = round(s[f'{k}_total_kvah_cut'],2)

    oo = max((s['orig_peak']    for s in day_summaries.values()), default=0)
    om = max((s['managed_peak'] for s in day_summaries.values()), default=0)
    mo = max((s['orig_md_peak']    for s in day_summaries.values()), default=0)
    mm = max((s['managed_md_peak'] for s in day_summaries.values()), default=0)
    pr  = (oo-om)/oo*100 if oo else 0
    mpr = (mo-mm)/mo*100 if mo else 0

    return jsonify({
        'success':True,'results':results,
        'day_summaries':day_summaries,'load_keys':load_keys,'loads':loads,
        'summary':{
            'original_peak_kva':           round(oo,2),
            'managed_peak_kva':            round(om,2),
            'peak_reduction_pct':          round(pr,1),
            'original_md_peak_kva':        round(mo,2),
            'managed_md_peak_kva':         round(mm,2),
            'md_peak_reduction_pct':       round(mpr,1),
            'total_battery_discharge_kwh': round(sum(s['total_discharge_kwh'] for s in day_summaries.values()),1),
            'total_battery_charge_kwh':    round(sum(s['total_charge_kwh']    for s in day_summaries.values()),1),
            'avg_original_kva': round(sum(r['kva_original'] for r in results)/len(results),2) if results else 0,
            'avg_managed_kva':  round(sum(r['kva_managed']  for r in results)/len(results),2) if results else 0,
        }
    })


# ── Predictor — Train ─────────────────────────────────────────────────────────
@app.route('/api/predictor/train', methods=['POST'])
def api_predictor_train():
    if S.get('df') is None:
        return _err('Upload a load profile first', 400)
    body = request.get_json(silent=True) or {}
    kwp  = float(body.get('capacity_kwp', S.get('solar_kwp', 0)))
    n_est= int(body.get('n_estimators', 250))
    lr   = float(body.get('learning_rate', 0.05))
    try:
        fc = DirectMultiStepForecaster(capacity_kwp=kwp,
                                       n_estimators=n_est, learning_rate=lr)
        metrics = fc.fit(S['df'], verbose=False)
        S['forecaster'] = fc
        S['forecast_result'] = None
        return _ok({
            'n_models':    metrics['n_models_trained'],
            'mean_mape':   round(_sf(metrics.get('mean_mape',  0)), 3),
            'mape_at_h24': round(_sf(metrics.get('mape_at_h24',0)), 3),
        })
    except Exception as e:
        return _err(f'Training failed: {e}\n{traceback.format_exc()}')


# ── Predictor — Forecast ──────────────────────────────────────────────────────
@app.route('/api/predictor/forecast', methods=['POST'])
def api_predictor_forecast():
    if S.get('forecaster') is None:
        return _err('Train the forecaster first', 400)
    try:
        fc = S['forecaster']
        fr = fc.forecast(output_steps=48)
        S['forecast_result'] = fr
        peaks = fc.detect_peaks(fr)
        hist  = fc.history.tail(96)  # last 2 days

        return _ok({
            'chart': {
                'hist_labels': hist['timestamp'].astype(str).tolist(),
                'hist_kw':     [round(_sf(v),2) for v in hist['kw_import']],
                'fc_labels':   [str(t) for t in fr.timestamps],
                'fc_median':   [round(_sf(v),2) for v in fr.median],
                'fc_p10':      [round(_sf(v),2) for v in fr.p10],
                'fc_p90':      [round(_sf(v),2) for v in fr.p90],
            },
            'peaks': peaks.to_dict(orient='records'),
            'metrics': {
                'n_horizons': len(fr.median),
                'horizon_h':  24,
            },
        })
    except Exception as e:
        return _err(f'Forecast failed: {e}\n{traceback.format_exc()}')


# ── Calculator — Bill ─────────────────────────────────────────────────────────
@app.route('/api/calculator/bill', methods=['POST'])
def api_calculator_bill():
    if S.get('df') is None:
        return _err('Upload a load profile first', 400)
    body     = request.get_json(silent=True) or {}
    tariff   = body.get('tariff',   S.get('tariff_code','C1'))
    icpt_sen = float(body.get('icpt_sen', 0.0))
    nem_rate = float(body.get('nem_rate', 0.31))

    # Use manager-optimized df if available, otherwise raw
    mgr = S.get('manager_results')
    if mgr:
        res_df = pd.DataFrame([{k:v for k,v in r.items() if k!='actions'} for r in mgr])
        res_df['timestamp'] = pd.to_datetime(res_df['timestamp'])
        # Build optimised bill df
        df_opt = res_df[['timestamp']].copy()
        df_opt['kw_import']   = res_df['kw_managed'].clip(lower=0)
        df_opt['kw_export']   = 0.0
        df_opt['kvar_import'] = res_df.get('kvar_managed', pd.Series(0.0, index=res_df.index)).clip(lower=0)
        df_opt['kvar_export'] = 0.0
        # Original
        df_orig = S['df'][['timestamp','kw_import','kw_export','kvar_import','kvar_export']].copy()
        show_comparison = True
    else:
        df_opt = S['df'][['timestamp','kw_import','kw_export','kvar_import','kvar_export']].copy()
        df_orig = df_opt.copy()
        show_comparison = False

    def _calc(df_in):
        stats = compute_monthly_stats(df_in)
        rows  = []
        for _, row in stats.iterrows():
            bill = calculate_bill(tariff,
                monthly_kwh=row['total_kwh'], peak_kwh=row['peak_kwh'],
                offpeak_kwh=row['offpeak_kwh'], max_demand_kw=row['max_demand_kw'],
                icpt_sen_per_kwh=icpt_sen)
            nem  = compute_nem_credit(row['export_kwh'], nem_rate)
            rows.append({
                'month':           str(row['month']),
                'total_kwh':       round(_sf(row['total_kwh']),0),
                'peak_kwh':        round(_sf(row['peak_kwh']),0),
                'offpeak_kwh':     round(_sf(row['offpeak_kwh']),0),
                'max_demand_kw':   round(_sf(row['max_demand_kw']),1),
                'energy_rm':       round(_sf(bill['energy_charge']),2),
                'md_rm':           round(_sf(bill['md_charge']),2),
                'icpt_rm':         round(_sf(bill['icpt_charge']),2),
                'kwtbb_rm':        round(_sf(bill['kwtbb_charge']),2),
                'svc_tax_rm':      round(_sf(bill['service_tax']),2),
                'nem_credit_rm':   round(_sf(nem['nem_credit_rm']),2),
                'gross_bill_rm':   round(_sf(bill['total_bill']),2),
                'net_bill_rm':     round(_sf(bill['total_bill'])-_sf(nem['nem_credit_rm']),2),
            })
        return rows

    try:
        rows_opt  = _calc(df_opt)
        rows_orig = _calc(df_orig) if show_comparison else rows_opt
    except Exception as e:
        return _err(f'Bill calculation failed: {e}\n{traceback.format_exc()}')

    total_opt  = sum(r['net_bill_rm'] for r in rows_opt)
    total_orig = sum(r['net_bill_rm'] for r in rows_orig)
    savings    = total_orig - total_opt
    n_months   = max(len(rows_opt), 1)
    S['bill_rows'] = rows_opt

    return _ok({
        'tariff':           tariff,
        'tariff_name':      TARIFF_META.get(tariff,{}).get('name',''),
        'show_comparison':  show_comparison,
        'totals': {
            'before_rm':      round(total_orig, 2),
            'after_rm':       round(total_opt,  2),
            'savings_rm':     round(savings,     2),
            'ann_savings_rm': round(savings*12/n_months, 2),
        },
        'rows_opt':  rows_opt,
        'rows_orig': rows_orig,
    })


# ── PowerRECO — Run ───────────────────────────────────────────────────────────
@app.route('/api/powerreco/run', methods=['POST'])
def api_powerreco_run():
    body        = request.get_json(silent=True) or {}
    roof_area   = float(body.get('roof_area',    500.0))
    panel_w     = int(  body.get('panel_w',      415))
    psh         = float(body.get('psh',          4.5))
    solar_cost  = float(body.get('solar_cost',   3500.0))
    batt_cost   = float(body.get('batt_cost',    2500.0))
    self_cons   = float(body.get('self_cons',    0.65))
    use_new_md  = bool( body.get('use_new_tariff', True))

    # Manual battery entry (used when no Manager results)
    manual_daily_kwh = float(body.get('manual_daily_kwh', 0.0))
    manual_md_kw     = float(body.get('manual_md_kw',    0.0))

    try:
        solar = calculate_solar_sizing(roof_area, panel_w, psh)
    except Exception as e:
        return _err(f'Solar sizing failed: {e}')

    # Battery sizing from Manager results, fallback to manual or heuristic
    mgr = S.get('manager_results')
    if mgr:
        try:
            # Build powerreco df from manager results
            res_df = pd.DataFrame([{k:v for k,v in r.items() if k!='actions'} for r in mgr])
            res_df['timestamp'] = pd.to_datetime(res_df['timestamp'])
            # battery_discharged_kwh per interval = discharge_kw * 0.5
            res_df['battery_discharged_kwh'] = res_df['battery_discharge_kw'] * 0.5
            batt = calculate_battery_sizing(res_df)
        except Exception:
            batt = _manual_battery(solar, manual_daily_kwh, manual_md_kw)
    elif manual_daily_kwh > 0:
        batt = _manual_battery(solar, manual_daily_kwh, manual_md_kw)
    else:
        daily = _sf(solar.get('daily_generation_kwh_avg', 0))
        batt  = {
            'min_capacity_kwh_commercial': max(10.0, round(daily*0.10/50)*50),
            'md_reduction_kw':  0.0,
            'n_days_analyzed':  0,
            'spike_note': 'Estimated (10% of daily solar) — run AI Manager for accuracy.',
        }

    try:
        roi = calculate_roi(
            solar_kwp=solar['system_kwp'],
            battery_kwh=float(batt['min_capacity_kwh_commercial']),
            monthly_generation_kwh=solar['monthly_generation_kwh_avg'],
            md_reduction_kw=_sf(batt.get('md_reduction_kw',0)),
            self_consumption_pct=self_cons,
            use_new_tariff=use_new_md,
            solar_cost_per_kwp=solar_cost,
            battery_cost_per_kwh=batt_cost,
        )
        S['roi'] = roi
    except Exception as e:
        return _err(f'ROI calculation failed: {e}')

    return _ok({
        'solar': {
            'system_kwp':               solar['system_kwp'],
            'n_panels':                 solar['n_panels'],
            'panel_wattage_w':          solar['panel_wattage_w'],
            'annual_generation_kwh':    solar['annual_generation_kwh'],
            'usable_area_m2':           solar['usable_area_m2'],
            'daily_generation_kwh_avg': round(_sf(solar['daily_generation_kwh_avg']),1),
            'month_labels':             solar['month_labels'],
            'monthly_breakdown_kwh':    [round(_sf(v),1) for v in solar['monthly_breakdown_kwh']],
        },
        'battery': {
            'min_kwh_commercial': float(batt['min_capacity_kwh_commercial']),
            'md_reduction_kw':    round(_sf(batt.get('md_reduction_kw',0)),1),
            'n_days_analyzed':    int(batt.get('n_days_analyzed',0)),
            'spike_note':         batt.get('spike_note',''),
        },
        'roi': {
            'total_capex_rm':           round(_sf(roi['total_capex_rm']),0),
            'simple_payback_years':     roi['simple_payback_years'],
            'npv_25yr_rm':              round(_sf(roi['npv_25yr_rm']),0),
            'irr_pct':                  roi.get('irr_pct'),
            'annual_energy_savings_rm': round(_sf(roi['annual_energy_savings_rm']),0),
            'annual_md_savings_rm':     round(_sf(roi['annual_md_savings_rm']),0),
            'annual_nem_credit_rm':     round(_sf(roi['annual_nem_credit_rm']),0),
            'co2_offset_tonnes_yr':     round(_sf(roi['co2_offset_tonnes_yr']),1),
            'md_rate_used':             roi.get('md_rate_used',''),
            'cumulative_npv':           [round(_sf(v),0) for v in roi['cumulative_npv']],
        },
    })


def _manual_battery(solar: dict, daily_kwh: float, md_kw: float) -> dict:
    from powerreco.battery_sizing import ROUNDTRIP_EFF, UPSIZE_FACTOR, _round_to_commercial
    base = daily_kwh if daily_kwh > 0 else _sf(solar.get('daily_generation_kwh_avg',0))*0.10
    cap  = base / ROUNDTRIP_EFF * UPSIZE_FACTOR
    return {
        'min_capacity_kwh_commercial': _round_to_commercial(cap),
        'md_reduction_kw':  md_kw,
        'n_days_analyzed':  0,
        'spike_note': f'Manual entry: {daily_kwh:.1f} kWh/day discharge target.',
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\nBOLT Integrated  |  http://localhost:5000\n')
    app.run(host='0.0.0.0', port=5000, debug=False)
