import React, { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import * as XLSX from 'xlsx';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, AreaChart, Area, Cell,
  ComposedChart, Legend
} from 'recharts';
import {
  Upload, Zap, AlertTriangle, TrendingDown, Activity, Clock, Calendar,
  FileSpreadsheet, Power, ChevronRight, Info, Settings2, Sun, Moon,
  CircleDollarSign, BarChart3, ArrowRight, Sparkles, Plug, Gauge
} from 'lucide-react';

/* ------------------------------------------------------------------ */
/*  TNB RP4 Tariff Schedule - Effective 1 July 2025 to 31 Dec 2027    */
/*  Source: Energy Commission (ST), via mytnb.com.my/tariff           */
/* ------------------------------------------------------------------ */

const TARIFFS = {
  LV_GENERAL: {
    code: 'LV_GENERAL',
    name: 'Low Voltage â General',
    sub: 'Replaces ex-Tariff B / D / F',
    voltage: 'LV',
    energy: 27.03,
    capacityKwh: 8.83,
    networkKwh: 14.82,
    retail: 20.00,
    mdBased: false,
  },
  LV_TOU: {
    code: 'LV_TOU',
    name: 'Low Voltage â Time-of-Use',
    sub: 'Smart-meter required',
    voltage: 'LV',
    energyPeak: 28.52,
    energyOffPeak: 24.43,
    capacityKwh: 8.83,
    networkKwh: 14.82,
    retail: 20.00,
    tou: true,
    mdBased: false,
  },
  MV_GENERAL: {
    code: 'MV_GENERAL',
    name: 'Medium Voltage â General',
    sub: 'Replaces ex-Tariff C1 / E1 / F1',
    voltage: 'MV',
    energy: 29.83,
    capacityKw: 29.43,    // RM/kW per month (24-hr MD)
    networkKw: 59.84,     // RM/kW per month (24-hr MD)
    retail: 200.00,
    mdBased: true,
    peakOnlyMD: false,
  },
  MV_TOU: {
    code: 'MV_TOU',
    name: 'Medium Voltage â Time-of-Use',
    sub: 'Replaces ex-Tariff C2 / E2 / F2',
    voltage: 'MV',
    energyPeak: 31.32,
    energyOffPeak: 27.23,
    capacityKw: 30.19,    // RM/kW peak-MD only
    networkKw: 66.87,     // RM/kW peak-MD only
    retail: 200.00,
    mdBased: true,
    peakOnlyMD: true,
    tou: true,
  },
  HV_GENERAL: {
    code: 'HV_GENERAL',
    name: 'High Voltage â General',
    sub: 'Replaces ex-Tariff C3',
    voltage: 'HV',
    energy: 43.03,
    capacityKw: 16.68,
    networkKw: 14.53,
    retail: 250.00,
    mdBased: true,
    peakOnlyMD: false,
  },
  HV_TOU: {
    code: 'HV_TOU',
    name: 'High Voltage â Time-of-Use',
    sub: 'Replaces ex-Tariff C4 / E3',
    voltage: 'HV',
    energyPeak: 44.52,
    energyOffPeak: 40.43,
    capacityKw: 21.76,
    networkKw: 23.06,
    retail: 250.00,
    mdBased: true,
    peakOnlyMD: true,
    tou: true,
  },
};

/* Malaysia public-holiday set used by TNB ToU (off-peak full day).     */
/* Approximate dates for 2025-2026; user can disable if desired.        */
const PUBLIC_HOLIDAYS = new Set([
  '2025-01-01', '2025-01-29', '2025-01-30', '2025-03-31', '2025-04-01',
  '2025-05-01', '2025-05-12', '2025-06-02', '2025-06-07', '2025-06-27',
  '2025-08-31', '2025-09-05', '2025-09-16', '2025-10-20', '2025-12-25',
  '2026-01-01', '2026-02-17', '2026-02-18', '2026-03-21', '2026-03-22',
  '2026-05-01', '2026-05-31', '2026-05-27', '2026-06-06', '2026-06-16',
  '2026-08-26', '2026-08-31', '2026-09-16', '2026-11-09', '2026-12-25',
]);

const ymd = (d) => {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
};

/* An interval "ends at" timestamp t, i.e. it covers (t-30min, t].      */
/* It is peak iff: weekday AND not a holiday AND end-time in (14:00,22:00] */
const isPeakInterval = (endTime, useHolidays = true) => {
  const dow = endTime.getDay();
  if (dow === 0 || dow === 6) return false;
  if (useHolidays && PUBLIC_HOLIDAYS.has(ymd(endTime))) return false;
  const mins = endTime.getHours() * 60 + endTime.getMinutes();
  return mins > 14 * 60 && mins <= 22 * 60;
};

/* ------------------------------------------------------------------ */
/*  Excel parser - robust to the four observed formats                */
/* ------------------------------------------------------------------ */

function parseWorkbook(arrayBuffer) {
  const wb = XLSX.read(arrayBuffer, { type: 'array', cellDates: true });
  const all = [];
  const sheetSummary = [];
  for (const name of wb.SheetNames) {
    const ws = wb.Sheets[name];
    const rows = XLSX.utils.sheet_to_json(ws, { header: 1, raw: true, defval: null });
    const parsed = parseSheetRows(rows);
    sheetSummary.push({ name, rows: parsed.length });
    all.push(...parsed);
  }
  // de-dup and sort
  const seen = new Set();
  const uniq = [];
  for (const r of all) {
    const k = r.timestamp.getTime();
    if (!seen.has(k)) { seen.add(k); uniq.push(r); }
  }
  uniq.sort((a, b) => a.timestamp - b.timestamp);
  return { records: uniq, sheets: sheetSummary };
}

function parseSheetRows(rows) {
  if (!rows || rows.length === 0) return [];
  let headerIdx = -1, dateCol = -1, kwImpCol = -1, kwExpCol = -1;

  for (let i = 0; i < Math.min(rows.length, 12); i++) {
    const r = (rows[i] || []).map(c => String(c == null ? '' : c).toLowerCase().trim());
    let d = -1, imp = -1, exp = -1;
    r.forEach((c, idx) => {
      if (d === -1 && (
        c === 'date / end time' || c === 'end_time' || c === 'end time' ||
        c === 'datetime' || c === 'timestamp' ||
        (c.includes('date') && c.includes('time')) ||
        (c.includes('end') && c.includes('time'))
      )) d = idx;
      if (imp === -1 && (
        c === 'kw_import' || c === 'kw import' || c === 'kwimport' ||
        c === 'import (kw)' || c === 'kw (import)'
      )) imp = idx;
      if (exp === -1 && (
        c === 'kw_export' || c === 'kw export' || c === 'kwexport'
      )) exp = idx;
    });
    if (d !== -1 && imp !== -1) {
      headerIdx = i; dateCol = d; kwImpCol = imp; kwExpCol = exp;
      break;
    }
  }
  if (headerIdx === -1) return [];

  const out = [];
  for (let i = headerIdx + 1; i < rows.length; i++) {
    const row = rows[i];
    if (!row) continue;
    const dv = row[dateCol];
    const iv = row[kwImpCol];
    if (dv == null || iv == null || iv === '') continue;
    let ts = null;
    if (dv instanceof Date) ts = dv;
    else if (typeof dv === 'number') {
      const obj = XLSX.SSF.parse_date_code(dv);
      if (obj) ts = new Date(obj.y, obj.m - 1, obj.d, obj.H || 0, obj.M || 0, obj.S || 0);
    } else {
      const s = String(dv).trim();
      const t = new Date(s.replace(' ', 'T'));
      if (!isNaN(t.getTime())) ts = t;
    }
    if (!ts || isNaN(ts.getTime())) continue;
    const kwI = parseFloat(iv);
    if (isNaN(kwI)) continue;
    const kwE = parseFloat(row[kwExpCol] || 0) || 0;
    out.push({ timestamp: ts, kwImport: kwI, kwExport: kwE });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/*  Bill calculation                                                  */
/* ------------------------------------------------------------------ */

function calculateBill(records, tariff, opts) {
  const { afaSenPerKwh = 0, applyKwtbb = true, applySst = true,
          useHolidays = true, intervalHours = 0.5 } = opts;

  let kwhTotal = 0, kwhPeak = 0, kwhOff = 0;
  let mdAll = 0, mdPeak = 0;
  let mdAllAt = null, mdPeakAt = null;
  let firstTs = null, lastTs = null;

  for (const r of records) {
    const energy = r.kwImport * intervalHours;
    kwhTotal += energy;
    const peak = isPeakInterval(r.timestamp, useHolidays);
    if (peak) {
      kwhPeak += energy;
      if (r.kwImport > mdPeak) { mdPeak = r.kwImport; mdPeakAt = r.timestamp; }
    } else {
      kwhOff += energy;
    }
    if (r.kwImport > mdAll) { mdAll = r.kwImport; mdAllAt = r.timestamp; }
    if (!firstTs || r.timestamp < firstTs) firstTs = r.timestamp;
    if (!lastTs || r.timestamp > lastTs) lastTs = r.timestamp;
  }

  // --- charges
  let energyCharge = 0;
  if (tariff.tou) {
    energyCharge = (kwhPeak * tariff.energyPeak + kwhOff * tariff.energyOffPeak) / 100;
  } else {
    energyCharge = (kwhTotal * tariff.energy) / 100;
  }
  const afaCharge = (kwhTotal * afaSenPerKwh) / 100;

  let capacityCharge = 0, networkCharge = 0;
  const billedMD = tariff.peakOnlyMD ? mdPeak : mdAll;
  if (tariff.mdBased) {
    capacityCharge = billedMD * tariff.capacityKw;
    networkCharge = billedMD * tariff.networkKw;
  } else {
    capacityCharge = (kwhTotal * tariff.capacityKwh) / 100;
    networkCharge = (kwhTotal * tariff.networkKwh) / 100;
  }
  const retail = tariff.retail;

  const subtotal = energyCharge + afaCharge + capacityCharge + networkCharge + retail;
  // KWTBB = 1.6% Ã (subtotal - AFA - retail)  per official formula
  const kwtbbBase = subtotal - afaCharge - retail;
  const kwtbb = applyKwtbb ? kwtbbBase * 0.016 : 0;
  // SST 8% on bill (electricity portion)
  const sst = applySst ? subtotal * 0.08 : 0;
  const total = subtotal + kwtbb + sst;

  return {
    kwhTotal, kwhPeak, kwhOff, mdAll, mdPeak, mdAllAt, mdPeakAt,
    billedMD, firstTs, lastTs,
    energyCharge, afaCharge, capacityCharge, networkCharge, retail,
    subtotal, kwtbb, sst, total,
    nIntervals: records.length,
    days: firstTs && lastTs ? Math.max(1, Math.round((lastTs - firstTs) / 86400000)) : 0,
  };
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

const fmtRM = (n) => 'RM ' + (n || 0).toLocaleString('en-MY', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtNum = (n, d = 0) => (n || 0).toLocaleString('en-MY', { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtTs = (ts) => ts ? ts.toLocaleString('en-MY', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) : 'â';

/* ------------------------------------------------------------------ */
/*  UI primitives                                                     */
/* ------------------------------------------------------------------ */

const Card = ({ children, className = '' }) => (
  <div className={`bg-[#131825] border border-[#252b3d] rounded ${className}`}>{children}</div>
);

const StatCard = ({ label, value, sub, accent, icon: Icon }) => (
  <div className="bg-[#131825] border border-[#252b3d] p-5 relative overflow-hidden group hover:border-[#3a4257] transition-colors">
    <div className="flex items-start justify-between mb-3">
      <span className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">{label}</span>
      {Icon && <Icon size={14} className="text-[#5a6478]" />}
    </div>
    <div className={`font-mono text-[28px] leading-none ${accent || 'text-[#e8edf5]'}`}>{value}</div>
    {sub && <div className="text-[11px] text-[#8a94a6] mt-2 font-mono">{sub}</div>}
  </div>
);

const Section = ({ title, kicker, children, right }) => (
  <div className="mb-8">
    <div className="flex items-end justify-between mb-4 border-b border-[#252b3d] pb-3">
      <div>
        {kicker && <div className="text-[10px] uppercase tracking-[0.22em] text-[#5eff7a] mb-1 font-medium">{kicker}</div>}
        <h2 className="text-2xl text-[#e8edf5]" style={{ fontFamily: 'Bricolage Grotesque, sans-serif', fontWeight: 600, letterSpacing: '-0.01em' }}>{title}</h2>
      </div>
      {right}
    </div>
    {children}
  </div>
);

/* ------------------------------------------------------------------ */
/*  Main App                                                          */
/* ------------------------------------------------------------------ */

export default function App() {
  const [records, setRecords] = useState([]);
  const [fileName, setFileName] = useState('');
  const [sheetSummary, setSheetSummary] = useState([]);
  const [tariffCode, setTariffCode] = useState('MV_GENERAL');
  const [afa, setAfa] = useState(0);
  const [applyKwtbb, setApplyKwtbb] = useState(true);
  const [applySst, setApplySst] = useState(true);
  const [useHolidays, setUseHolidays] = useState(true);
  const [includeExport, setIncludeExport] = useState(false); // NEM-style net offset
  const [parseError, setParseError] = useState('');
  const fileRef = useRef(null);

  const tariff = TARIFFS[tariffCode];

  /* derive net records based on toggle */
  const netRecords = useMemo(() => {
    if (!includeExport) return records;
    return records.map(r => ({
      ...r,
      kwImport: Math.max(0, r.kwImport - r.kwExport)
    }));
  }, [records, includeExport]);

  const bill = useMemo(() => {
    if (netRecords.length === 0) return null;
    return calculateBill(netRecords, tariff, { afaSenPerKwh: afa, applyKwtbb, applySst, useHolidays });
  }, [netRecords, tariff, afa, applyKwtbb, applySst, useHolidays]);

  /* compare against all tariffs of the same voltage */
  const tariffComparisons = useMemo(() => {
    if (netRecords.length === 0) return [];
    return Object.values(TARIFFS).map(t => ({
      tariff: t,
      bill: calculateBill(netRecords, t, { afaSenPerKwh: afa, applyKwtbb, applySst, useHolidays }),
    }));
  }, [netRecords, afa, applyKwtbb, applySst, useHolidays]);

  /* peak-shave sensitivity: simulate cap'ing MD at various levels */
  const peakShaveCurve = useMemo(() => {
    if (!bill || !tariff.mdBased) return [];
    const md = bill.billedMD;
    const out = [];
    for (let pct = 0; pct <= 30; pct += 5) {
      const newMD = md * (1 - pct / 100);
      const oldCharge = md * (tariff.capacityKw + tariff.networkKw);
      const newCharge = newMD * (tariff.capacityKw + tariff.networkKw);
      out.push({ reduction: pct, savedRM: oldCharge - newCharge, newMD: newMD });
    }
    return out;
  }, [bill, tariff]);

  /* daily aggregation for chart */
  const dailySeries = useMemo(() => {
    if (netRecords.length === 0) return [];
    const map = new Map();
    for (const r of netRecords) {
      const key = ymd(r.timestamp);
      const existing = map.get(key) || { day: key, kwh: 0, peakKw: 0, samplePeak: false };
      existing.kwh += r.kwImport * 0.5;
      if (r.kwImport > existing.peakKw) existing.peakKw = r.kwImport;
      map.set(key, existing);
    }
    return [...map.values()].sort((a, b) => a.day.localeCompare(b.day));
  }, [netRecords]);

  /* hourly profile - average kW by hour-of-day */
  const hourlyProfile = useMemo(() => {
    if (netRecords.length === 0) return [];
    const sums = Array(24).fill(0);
    const counts = Array(24).fill(0);
    const peaks = Array(24).fill(0);
    for (const r of netRecords) {
      const h = r.timestamp.getHours();
      sums[h] += r.kwImport;
      counts[h]++;
      if (r.kwImport > peaks[h]) peaks[h] = r.kwImport;
    }
    return Array.from({ length: 24 }, (_, h) => ({
      hour: h,
      label: String(h).padStart(2, '0'),
      avgKw: counts[h] ? sums[h] / counts[h] : 0,
      peakKw: peaks[h],
      isPeakWindow: h >= 14 && h < 22,
    }));
  }, [netRecords]);

  /* heatmap: avg kW by day-of-week Ã hour */
  const heatmap = useMemo(() => {
    if (netRecords.length === 0) return null;
    const sums = Array.from({ length: 7 }, () => Array(24).fill(0));
    const counts = Array.from({ length: 7 }, () => Array(24).fill(0));
    let maxV = 0;
    for (const r of netRecords) {
      const dow = r.timestamp.getDay();
      const h = r.timestamp.getHours();
      sums[dow][h] += r.kwImport;
      counts[dow][h]++;
    }
    const grid = sums.map((row, d) => row.map((s, h) => {
      const v = counts[d][h] ? s / counts[d][h] : 0;
      if (v > maxV) maxV = v;
      return v;
    }));
    return { grid, max: maxV };
  }, [netRecords]);

  /* ----------------- handlers ----------------- */
  const handleFile = useCallback(async (file) => {
    if (!file) return;
    setParseError('');
    try {
      const buf = await file.arrayBuffer();
      const { records, sheets } = parseWorkbook(buf);
      if (records.length === 0) {
        setParseError('No valid load profile data found. Expected columns: a date/end-time column and a "kW Import" column.');
        return;
      }
      setRecords(records);
      setSheetSummary(sheets);
      setFileName(file.name);
    } catch (e) {
      setParseError('Failed to parse file: ' + (e.message || e));
    }
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onPick = (e) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const reset = () => {
    setRecords([]); setFileName(''); setSheetSummary([]); setParseError('');
  };

  /* ----------------- styles ----------------- */
  return (
    <div className="min-h-screen bg-[#0a0d14] text-[#e8edf5]" style={{ fontFamily: 'Geist, ui-sans-serif, system-ui, sans-serif' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,600;12..96,700&family=Geist:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        body { font-family: 'Geist', system-ui, sans-serif; }
        .font-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-feature-settings: 'tnum'; }
        .font-display { font-family: 'Bricolage Grotesque', sans-serif; }
        input[type=range] { accent-color: #5eff7a; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0a0d14; }
        ::-webkit-scrollbar-thumb { background: #252b3d; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #3a4257; }
      `}</style>

      {/* HEADER ============================================================== */}
      <header className="border-b border-[#252b3d] bg-[#0a0d14] sticky top-0 z-30 backdrop-blur">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-[#5eff7a] flex items-center justify-center">
              <Zap size={20} strokeWidth={2.5} className="text-[#0a0d14]" />
            </div>
            <div>
              <div className="font-display font-semibold text-lg leading-none tracking-tight">TNB Bill Calculator</div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] mt-1 font-mono">RP4 / 1 Jul 2025 â 31 Dec 2027</div>
            </div>
          </div>
          <div className="flex items-center gap-6 text-xs text-[#8a94a6] font-mono">
            <span>NON-DOMESTIC TARIFF</span>
            <span className="text-[#5eff7a]">â</span>
          </div>
        </div>
      </header>

      <div className="max-w-[1400px] mx-auto px-6 py-8">
        {/* HERO ============================================================== */}
        <div className="mb-10 grid grid-cols-12 gap-6 items-end">
          <div className="col-span-12 lg:col-span-8">
            <div className="text-[10px] uppercase tracking-[0.22em] text-[#5eff7a] mb-3 font-mono">
              ESUM Ã RExharge â Load Shifting & Peak Shaving
            </div>
            <h1 className="font-display text-[44px] leading-[1.05] tracking-[-0.02em]">
              Compute the bill from any<br/>
              30-minute load profile.
            </h1>
            <p className="text-[#8a94a6] mt-4 max-w-2xl text-[15px] leading-relaxed">
              Drop your kW Import time-series. Pick a tariff. The calculator splits energy into peak / off-peak,
              detects the billed Maximum Demand window, and itemises the bill exactly the way TNB will issue it under RP4.
            </p>
          </div>
          <div className="col-span-12 lg:col-span-4 grid grid-cols-2 gap-3">
            <div className="bg-[#131825] border border-[#252b3d] p-4">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-mono">MV ToU MD</div>
              <div className="font-mono text-2xl text-[#ff3d8a] mt-1">RM 97.06</div>
              <div className="text-[10px] text-[#5a6478] font-mono mt-1">per kW Â· peak only</div>
            </div>
            <div className="bg-[#131825] border border-[#252b3d] p-4">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-mono">vs RP3 C1</div>
              <div className="font-mono text-2xl text-[#5eff7a] mt-1">+220%</div>
              <div className="text-[10px] text-[#5a6478] font-mono mt-1">RM 30.30 â RM 97.06</div>
            </div>
          </div>
        </div>

        {/* MAIN GRID ========================================================= */}
        <div className="grid grid-cols-12 gap-6">
          {/* LEFT: CONFIG ==================================================== */}
          <aside className="col-span-12 lg:col-span-4 xl:col-span-3 space-y-4">
            {/* Upload */}
            <Card className="p-5">
              <div className="flex items-center gap-2 mb-3">
                <Upload size={14} className="text-[#5eff7a]" />
                <span className="text-[10px] uppercase tracking-[0.2em] text-[#8a94a6] font-mono font-medium">01 â Load Profile</span>
              </div>
              {fileName ? (
                <div className="space-y-3">
                  <div className="bg-[#0a0d14] border border-[#252b3d] p-3">
                    <div className="flex items-start gap-2">
                      <FileSpreadsheet size={14} className="text-[#5eff7a] mt-0.5 flex-shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-mono truncate text-[#e8edf5]">{fileName}</div>
                        <div className="text-[10px] text-[#8a94a6] font-mono mt-1">{records.length.toLocaleString()} intervals Â· {sheetSummary.length} sheet{sheetSummary.length !== 1 ? 's' : ''}</div>
                      </div>
                    </div>
                  </div>
                  <button onClick={reset} className="w-full py-2 text-xs font-mono uppercase tracking-wider border border-[#252b3d] text-[#8a94a6] hover:text-[#e8edf5] hover:border-[#3a4257] transition-colors">
                    Clear & Upload New
                  </button>
                </div>
              ) : (
                <div
                  onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                  onDrop={onDrop}
                  onClick={() => fileRef.current?.click()}
                  className="border-2 border-dashed border-[#252b3d] hover:border-[#5eff7a] transition-colors p-8 text-center cursor-pointer group"
                >
                  <Upload size={28} className="mx-auto mb-3 text-[#5a6478] group-hover:text-[#5eff7a] transition-colors" />
                  <div className="text-sm text-[#e8edf5]">Drop xlsx file here</div>
                  <div className="text-[11px] text-[#8a94a6] mt-1 font-mono">or click to browse</div>
                  <input ref={fileRef} type="file" accept=".xlsx,.xls,.xlsm,.csv" onChange={onPick} className="hidden" />
                </div>
              )}
              {parseError && (
                <div className="mt-3 bg-[#2a0e1a] border border-[#ff3d8a] p-3 text-xs text-[#ff8fb3] flex items-start gap-2">
                  <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
                  <span>{parseError}</span>
                </div>
              )}
              <div className="mt-3 text-[10px] text-[#5a6478] font-mono leading-relaxed">
                Expects columns: <span className="text-[#8a94a6]">Date / End Time</span> Â· <span className="text-[#8a94a6]">kW Import</span> Â· <span className="text-[#8a94a6]">kW Export</span> (optional). 30-min intervals.
              </div>
            </Card>

            {/* Tariff selector */}
            <Card className="p-5">
              <div className="flex items-center gap-2 mb-4">
                <Plug size={14} className="text-[#5eff7a]" />
                <span className="text-[10px] uppercase tracking-[0.2em] text-[#8a94a6] font-mono font-medium">02 â Tariff</span>
              </div>
              <div className="space-y-2">
                {Object.values(TARIFFS).map(t => (
                  <button
                    key={t.code}
                    onClick={() => setTariffCode(t.code)}
                    className={`w-full text-left p-3 border transition-all ${
                      tariffCode === t.code
                        ? 'border-[#5eff7a] bg-[#0d1f1a]'
                        : 'border-[#252b3d] hover:border-[#3a4257] bg-[#0a0d14]'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="min-w-0 flex-1">
                        <div className={`text-xs font-medium ${tariffCode === t.code ? 'text-[#5eff7a]' : 'text-[#e8edf5]'}`}>
                          {t.name}
                        </div>
                        <div className="text-[10px] text-[#8a94a6] font-mono mt-0.5">{t.sub}</div>
                      </div>
                      <span className="text-[9px] font-mono uppercase tracking-wider text-[#5a6478] ml-2">{t.voltage}{t.tou ? 'Â·ToU' : ''}</span>
                    </div>
                  </button>
                ))}
              </div>
            </Card>

            {/* Adjustments */}
            <Card className="p-5">
              <div className="flex items-center gap-2 mb-4">
                <Settings2 size={14} className="text-[#5eff7a]" />
                <span className="text-[10px] uppercase tracking-[0.2em] text-[#8a94a6] font-mono font-medium">03 â Adjustments</span>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs text-[#8a94a6]">AFA (sen/kWh)</label>
                    <span className="font-mono text-xs text-[#e8edf5]">{afa.toFixed(2)}</span>
                  </div>
                  <input type="range" min="-3" max="3" step="0.1" value={afa} onChange={e => setAfa(parseFloat(e.target.value))} className="w-full" />
                  <div className="flex justify-between text-[9px] text-[#5a6478] font-mono mt-1">
                    <span>â3 (rebate)</span><span>+3 (surcharge)</span>
                  </div>
                </div>

                <Toggle label="KWTBB 1.6%" desc="Renewable Energy Fund (>300 kWh)" value={applyKwtbb} onChange={setApplyKwtbb} />
                <Toggle label="SST 8%" desc="Sales & Service Tax" value={applySst} onChange={setApplySst} />
                <Toggle label="Public Holidays" desc="15 days as off-peak (ToU)" value={useHolidays} onChange={setUseHolidays} />
                <Toggle label="Net of Solar Export" desc="Subtract kW Export (NEM-style)" value={includeExport} onChange={setIncludeExport} />
              </div>
            </Card>
          </aside>

          {/* RIGHT: RESULTS ================================================== */}
          <main className="col-span-12 lg:col-span-8 xl:col-span-9">
            {!bill ? (
              <EmptyState />
            ) : (
              <>
                {/* KPI ROW */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                  <StatCard
                    label="Total Bill"
                    value={fmtRM(bill.total)}
                    sub={`${bill.days} days Â· ${fmtNum(bill.nIntervals)} intervals`}
                    accent="text-[#5eff7a]"
                    icon={CircleDollarSign}
                  />
                  <StatCard
                    label="Energy (kWh)"
                    value={fmtNum(bill.kwhTotal, 0)}
                    sub={tariff.tou ? `${fmtNum(bill.kwhPeak, 0)} peak Â· ${fmtNum(bill.kwhOff, 0)} off` : 'no ToU split'}
                    icon={Activity}
                  />
                  <StatCard
                    label={tariff.peakOnlyMD ? 'Peak MD (kW)' : 'Max Demand (kW)'}
                    value={fmtNum(bill.billedMD, 1)}
                    sub={fmtTs(tariff.peakOnlyMD ? bill.mdPeakAt : bill.mdAllAt)}
                    accent="text-[#ff3d8a]"
                    icon={Gauge}
                  />
                  <StatCard
                    label="Effective Rate"
                    value={(bill.kwhTotal > 0 ? (bill.total / bill.kwhTotal * 100).toFixed(1) : '0.0') + ' sen'}
                    sub="per kWh, all-in"
                    icon={TrendingDown}
                  />
                </div>

                {/* BILL BREAKDOWN ================================================ */}
                <Section kicker="Itemised" title="Bill breakdown">
                  <div className="grid grid-cols-12 gap-6">
                    <Card className="col-span-12 lg:col-span-7 overflow-hidden">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-[#252b3d]">
                            <th className="text-left p-4 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Charge</th>
                            <th className="text-right p-4 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Basis</th>
                            <th className="text-right p-4 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Amount</th>
                          </tr>
                        </thead>
                        <tbody className="font-mono text-[13px]">
                          {tariff.tou ? (
                            <>
                              <BillRow label="Energy â Peak" basis={`${fmtNum(bill.kwhPeak, 0)} kWh Ã ${tariff.energyPeak} sen`} amount={(bill.kwhPeak * tariff.energyPeak / 100)} />
                              <BillRow label="Energy â Off-Peak" basis={`${fmtNum(bill.kwhOff, 0)} kWh Ã ${tariff.energyOffPeak} sen`} amount={(bill.kwhOff * tariff.energyOffPeak / 100)} />
                            </>
                          ) : (
                            <BillRow label="Energy" basis={`${fmtNum(bill.kwhTotal, 0)} kWh Ã ${tariff.energy} sen`} amount={bill.energyCharge} />
                          )}
                          <BillRow label="AFA" basis={`${fmtNum(bill.kwhTotal, 0)} kWh Ã ${afa.toFixed(2)} sen`} amount={bill.afaCharge} dim={afa === 0} />
                          {tariff.mdBased ? (
                            <>
                              <BillRow label="Capacity Charge" basis={`${fmtNum(bill.billedMD, 1)} kW Ã RM ${tariff.capacityKw}`} amount={bill.capacityCharge} accent />
                              <BillRow label="Network Charge" basis={`${fmtNum(bill.billedMD, 1)} kW Ã RM ${tariff.networkKw}`} amount={bill.networkCharge} accent />
                            </>
                          ) : (
                            <>
                              <BillRow label="Capacity Charge" basis={`${fmtNum(bill.kwhTotal, 0)} kWh Ã ${tariff.capacityKwh} sen`} amount={bill.capacityCharge} />
                              <BillRow label="Network Charge" basis={`${fmtNum(bill.kwhTotal, 0)} kWh Ã ${tariff.networkKwh} sen`} amount={bill.networkCharge} />
                            </>
                          )}
                          <BillRow label="Retail Charge" basis="fixed monthly" amount={bill.retail} />
                          <tr className="border-t border-[#252b3d] bg-[#0a0d14]">
                            <td className="p-4 text-[#8a94a6] uppercase text-[11px] tracking-wider">Subtotal</td>
                            <td></td>
                            <td className="p-4 text-right text-[#e8edf5]">{fmtRM(bill.subtotal)}</td>
                          </tr>
                          {applyKwtbb && <BillRow label="KWTBB" basis="1.6% Ã (subtotal â AFA â retail)" amount={bill.kwtbb} />}
                          {applySst && <BillRow label="SST" basis="8% Ã subtotal" amount={bill.sst} />}
                          <tr className="border-t-2 border-[#5eff7a] bg-[#0d1f1a]">
                            <td className="p-4 text-[#5eff7a] uppercase text-[11px] tracking-wider font-display font-semibold">Total Payable</td>
                            <td></td>
                            <td className="p-4 text-right text-[#5eff7a] text-lg">{fmtRM(bill.total)}</td>
                          </tr>
                        </tbody>
                      </table>
                    </Card>

                    {/* Composition donut-ish bar */}
                    <Card className="col-span-12 lg:col-span-5 p-5">
                      <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] mb-4 font-mono">Composition</div>
                      <CompositionBars bill={bill} tariff={tariff} />
                    </Card>
                  </div>
                </Section>

                {/* LOAD PROFILE CHART =========================================== */}
                <Section kicker="Time series" title="Daily energy & peak demand">
                  <Card className="p-5">
                    <ResponsiveContainer width="100%" height={300}>
                      <ComposedChart data={dailySeries}>
                        <CartesianGrid stroke="#252b3d" vertical={false} />
                        <XAxis dataKey="day" tick={{ fill: '#8a94a6', fontSize: 10, fontFamily: 'JetBrains Mono' }} stroke="#252b3d" />
                        <YAxis yAxisId="kwh" tick={{ fill: '#8a94a6', fontSize: 10, fontFamily: 'JetBrains Mono' }} stroke="#252b3d" label={{ value: 'kWh / day', angle: -90, position: 'insideLeft', fill: '#5a6478', fontSize: 10 }} />
                        <YAxis yAxisId="kw" orientation="right" tick={{ fill: '#8a94a6', fontSize: 10, fontFamily: 'JetBrains Mono' }} stroke="#252b3d" label={{ value: 'Peak kW', angle: 90, position: 'insideRight', fill: '#5a6478', fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: '#0a0d14', border: '1px solid #252b3d', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                        <Bar yAxisId="kwh" dataKey="kwh" fill="#5eff7a" opacity={0.7} name="Daily kWh" />
                        <Line yAxisId="kw" dataKey="peakKw" stroke="#ff3d8a" strokeWidth={2} dot={false} name="Peak kW" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </Card>
                </Section>

                {/* HOURLY PROFILE & HEATMAP ===================================== */}
                <Section kicker="Pattern" title="Where the peak lives">
                  <div className="grid grid-cols-12 gap-6">
                    <Card className="col-span-12 lg:col-span-7 p-5">
                      <div className="flex items-center justify-between mb-4">
                        <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-mono">Avg kW by hour-of-day</div>
                        <div className="flex items-center gap-3 text-[10px] font-mono">
                          <span className="flex items-center gap-1.5"><span className="w-2 h-2 bg-[#5eff7a]"></span><span className="text-[#8a94a6]">Off-Peak</span></span>
                          <span className="flex items-center gap-1.5"><span className="w-2 h-2 bg-[#ff3d8a]"></span><span className="text-[#8a94a6]">Peak (14-22h)</span></span>
                        </div>
                      </div>
                      <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={hourlyProfile}>
                          <CartesianGrid stroke="#252b3d" vertical={false} />
                          <XAxis dataKey="label" tick={{ fill: '#8a94a6', fontSize: 10, fontFamily: 'JetBrains Mono' }} stroke="#252b3d" />
                          <YAxis tick={{ fill: '#8a94a6', fontSize: 10, fontFamily: 'JetBrains Mono' }} stroke="#252b3d" />
                          <Tooltip contentStyle={{ background: '#0a0d14', border: '1px solid #252b3d', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                          <Bar dataKey="avgKw" name="Avg kW">
                            {hourlyProfile.map((d, i) => (
                              <Cell key={i} fill={d.isPeakWindow ? '#ff3d8a' : '#5eff7a'} fillOpacity={0.85} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Card>

                    <Card className="col-span-12 lg:col-span-5 p-5">
                      <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-mono mb-4">Heatmap Â· day Ã hour</div>
                      {heatmap && <Heatmap grid={heatmap.grid} max={heatmap.max} />}
                    </Card>
                  </div>
                </Section>

                {/* TARIFF COMPARISON ============================================ */}
                <Section kicker="What-if" title="Try every tariff option">
                  <Card className="p-5">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-[#252b3d]">
                            <th className="text-left p-3 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Tariff</th>
                            <th className="text-right p-3 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Energy</th>
                            <th className="text-right p-3 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Capacity</th>
                            <th className="text-right p-3 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Network</th>
                            <th className="text-right p-3 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Other</th>
                            <th className="text-right p-3 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">Total</th>
                            <th className="text-right p-3 text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-medium">vs current</th>
                          </tr>
                        </thead>
                        <tbody className="font-mono text-[12px]">
                          {tariffComparisons
                            .sort((a, b) => a.bill.total - b.bill.total)
                            .map(({ tariff: t, bill: b }, i) => {
                              const diff = b.total - bill.total;
                              const isCurrent = t.code === tariffCode;
                              const isCheapest = i === 0;
                              return (
                                <tr key={t.code} className={`border-b border-[#1a2030] ${isCurrent ? 'bg-[#0d1f1a]' : ''}`}>
                                  <td className="p-3">
                                    <div className="flex items-center gap-2">
                                      {isCheapest && <span className="text-[#5eff7a] text-xs">â</span>}
                                      <span className={isCurrent ? 'text-[#5eff7a]' : 'text-[#e8edf5]'}>{t.name}</span>
                                    </div>
                                  </td>
                                  <td className="p-3 text-right text-[#8a94a6]">{fmtRM(t.tou ? (b.kwhPeak * t.energyPeak + b.kwhOff * t.energyOffPeak) / 100 : b.energyCharge)}</td>
                                  <td className="p-3 text-right text-[#8a94a6]">{fmtRM(b.capacityCharge)}</td>
                                  <td className="p-3 text-right text-[#8a94a6]">{fmtRM(b.networkCharge)}</td>
                                  <td className="p-3 text-right text-[#8a94a6]">{fmtRM(b.afaCharge + b.retail + b.kwtbb + b.sst)}</td>
                                  <td className={`p-3 text-right font-semibold ${isCurrent ? 'text-[#5eff7a]' : 'text-[#e8edf5]'}`}>{fmtRM(b.total)}</td>
                                  <td className={`p-3 text-right ${diff < 0 ? 'text-[#5eff7a]' : diff > 0 ? 'text-[#ff3d8a]' : 'text-[#5a6478]'}`}>
                                    {isCurrent ? 'â' : (diff < 0 ? 'â' : '+') + fmtRM(Math.abs(diff)).replace('RM ', '') }
                                  </td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </Section>

                {/* PEAK SHAVE SIMULATOR ========================================= */}
                {tariff.mdBased && (
                  <Section kicker="Sensitivity" title="If we shave the peakâ¦">
                    <div className="grid grid-cols-12 gap-6">
                      <Card className="col-span-12 lg:col-span-7 p-5">
                        <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-mono mb-4">
                          MD reduction â MD-charge savings
                        </div>
                        <ResponsiveContainer width="100%" height={240}>
                          <AreaChart data={peakShaveCurve}>
                            <CartesianGrid stroke="#252b3d" vertical={false} />
                            <XAxis dataKey="reduction" tick={{ fill: '#8a94a6', fontSize: 10, fontFamily: 'JetBrains Mono' }} stroke="#252b3d" tickFormatter={(v) => v + '%'} />
                            <YAxis tick={{ fill: '#8a94a6', fontSize: 10, fontFamily: 'JetBrains Mono' }} stroke="#252b3d" tickFormatter={(v) => 'RM ' + (v / 1000).toFixed(0) + 'k'} />
                            <Tooltip contentStyle={{ background: '#0a0d14', border: '1px solid #252b3d', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                              formatter={(v, n) => n === 'savedRM' ? [fmtRM(v), 'Saved/month'] : [fmtNum(v, 1) + ' kW', 'New MD']} />
                            <Area type="monotone" dataKey="savedRM" stroke="#5eff7a" fill="#5eff7a" fillOpacity={0.15} strokeWidth={2} name="savedRM" />
                          </AreaChart>
                        </ResponsiveContainer>
                      </Card>
                      <Card className="col-span-12 lg:col-span-5 p-5">
                        <div className="text-[10px] uppercase tracking-[0.18em] text-[#8a94a6] font-mono mb-4">Quick-reference table</div>
                        <table className="w-full font-mono text-xs">
                          <thead>
                            <tr className="text-[#5a6478]">
                              <th className="text-left pb-2">Cut</th>
                              <th className="text-right pb-2">New MD</th>
                              <th className="text-right pb-2">Savings/mo</th>
                            </tr>
                          </thead>
                          <tbody>
                            {peakShaveCurve.map((p, i) => (
                              <tr key={i} className="border-t border-[#1a2030]">
                                <td className={`py-2 ${p.reduction === 0 ? 'text-[#5a6478]' : 'text-[#e8edf5]'}`}>â{p.reduction}%</td>
                                <td className="py-2 text-right text-[#8a94a6]">{fmtNum(p.newMD, 1)} kW</td>
                                <td className={`py-2 text-right ${p.reduction > 0 ? 'text-[#5eff7a]' : 'text-[#5a6478]'}`}>
                                  {p.reduction > 0 ? fmtRM(p.savedRM) : 'â'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <div className="mt-4 pt-4 border-t border-[#252b3d] text-[10px] text-[#5a6478] font-mono leading-relaxed">
                          A 10% MD cut on this period saves <span className="text-[#5eff7a]">{fmtRM(peakShaveCurve[2]?.savedRM || 0)}/month</span> â recurring as long as the new ceiling holds.
                        </div>
                      </Card>
                    </div>
                  </Section>
                )}

                {/* DISCLAIMER */}
                <div className="mt-8 p-4 border border-[#252b3d] bg-[#0a0d14] text-[11px] text-[#5a6478] font-mono leading-relaxed">
                  <strong className="text-[#8a94a6]">Notes.</strong> Rates per Energy Commission RP4 schedule (1 Jul 2025â31 Dec 2027).
                  ToU peak = MonâFri 14:00â22:00, excluding 15 federal public holidays (off-peak full day). MD = highest 30-min average kW
                  in the period; for ToU MV/HV, only peak-window MD is billed. KWTBB applied as 1.6% Ã (Subtotal â AFA â Retail). SST shown
                  is the 8% indicative rate; real exemption thresholds may apply to your account. AFA is published monthly and can be Â±3 sen/kWh.
                  Use this as a planning tool â your actual bill may include power-factor surcharges, late fees, NEM credits, GET premiums, etc.
                </div>
              </>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                    */
/* ------------------------------------------------------------------ */

function Toggle({ label, desc, value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="w-full flex items-start justify-between gap-3 text-left group"
    >
      <div className="min-w-0 flex-1">
        <div className="text-xs text-[#e8edf5]">{label}</div>
        <div className="text-[10px] text-[#5a6478] font-mono mt-0.5 leading-tight">{desc}</div>
      </div>
      <span className={`flex-shrink-0 mt-0.5 w-9 h-5 border transition-colors relative ${value ? 'bg-[#5eff7a] border-[#5eff7a]' : 'bg-[#0a0d14] border-[#252b3d] group-hover:border-[#3a4257]'}`}>
        <span className={`absolute top-[2px] w-3.5 h-3.5 transition-all ${value ? 'left-[18px] bg-[#0a0d14]' : 'left-[2px] bg-[#3a4257]'}`}></span>
      </span>
    </button>
  );
}

function BillRow({ label, basis, amount, accent, dim }) {
  return (
    <tr className="border-b border-[#1a2030]">
      <td className={`p-4 ${dim ? 'text-[#5a6478]' : 'text-[#e8edf5]'}`}>{label}</td>
      <td className={`p-4 text-right text-[11px] ${dim ? 'text-[#5a6478]' : 'text-[#8a94a6]'}`}>{basis}</td>
      <td className={`p-4 text-right ${accent ? 'text-[#ff3d8a]' : dim ? 'text-[#5a6478]' : 'text-[#e8edf5]'}`}>{fmtRM(amount)}</td>
    </tr>
  );
}

function CompositionBars({ bill, tariff }) {
  const items = [
    { label: 'Energy', value: bill.energyCharge, color: '#5eff7a' },
    { label: 'Capacity', value: bill.capacityCharge, color: '#ff3d8a' },
    { label: 'Network', value: bill.networkCharge, color: '#ffb547' },
    { label: 'Retail', value: bill.retail, color: '#a78bfa' },
    { label: 'AFA', value: Math.abs(bill.afaCharge), color: '#22d3ee' },
    { label: 'KWTBB', value: bill.kwtbb, color: '#64748b' },
    { label: 'SST', value: bill.sst, color: '#94a3b8' },
  ];
  const total = items.reduce((s, x) => s + x.value, 0) || 1;
  const sorted = items.filter(x => x.value > 0).sort((a, b) => b.value - a.value);

  return (
    <div className="space-y-3">
      <div className="h-2 flex w-full overflow-hidden bg-[#0a0d14]">
        {sorted.map((x, i) => (
          <div key={i} style={{ width: `${(x.value / total) * 100}%`, background: x.color }} />
        ))}
      </div>
      <div className="space-y-2.5">
        {sorted.map((x, i) => {
          const pct = (x.value / total) * 100;
          return (
            <div key={i} className="flex items-center justify-between text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2" style={{ background: x.color }}></span>
                <span className="text-[#8a94a6]">{x.label}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-[#e8edf5] tabular-nums">{fmtRM(x.value)}</span>
                <span className="text-[#5a6478] text-[10px] w-10 text-right">{pct.toFixed(1)}%</span>
              </div>
            </div>
          );
        })}
      </div>
      {tariff.mdBased && (
        <div className="pt-3 mt-3 border-t border-[#252b3d] text-[10px] text-[#5a6478] font-mono leading-relaxed">
          MD charges (Capacity + Network) account for <span className="text-[#ff3d8a] font-semibold">{(((bill.capacityCharge + bill.networkCharge) / total) * 100).toFixed(1)}%</span> of this bill.
        </div>
      )}
    </div>
  );
}

function Heatmap({ grid, max }) {
  const dows = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  return (
    <div className="font-mono text-[9px]">
      <div className="flex">
        <div className="w-8"></div>
        {Array.from({ length: 24 }, (_, h) => (
          <div key={h} className="flex-1 text-center text-[#5a6478]" style={{ minWidth: '12px' }}>
            {h % 6 === 0 ? h : ''}
          </div>
        ))}
      </div>
      {grid.map((row, dow) => (
        <div key={dow} className="flex items-center">
          <div className="w-8 text-[#8a94a6] py-0.5">{dows[dow]}</div>
          {row.map((v, h) => {
            const intensity = max > 0 ? v / max : 0;
            const isPeakWindow = (dow >= 1 && dow <= 5) && h >= 14 && h < 22;
            const baseColor = isPeakWindow ? [255, 61, 138] : [94, 255, 122];
            const bg = `rgba(${baseColor.join(',')}, ${0.08 + intensity * 0.7})`;
            return (
              <div key={h} className="flex-1 h-4 border-r border-[#0a0d14]" style={{ background: bg, minWidth: '12px' }} title={`${dows[dow]} ${h}:00 â ${fmtNum(v, 1)} kW`} />
            );
          })}
        </div>
      ))}
      <div className="flex items-center justify-between mt-3 text-[#5a6478]">
        <span>0 kW</span>
        <div className="flex h-1.5 w-32 mx-2">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="flex-1" style={{ background: `rgba(94,255,122,${0.08 + (i / 9) * 0.7})` }} />
          ))}
        </div>
        <span>{fmtNum(max, 0)} kW</span>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="border-2 border-dashed border-[#252b3d] p-16 text-center">
      <div className="w-16 h-16 mx-auto mb-6 border border-[#252b3d] flex items-center justify-center">
        <FileSpreadsheet size={28} className="text-[#3a4257]" strokeWidth={1.5} />
      </div>
      <h3 className="font-display text-2xl text-[#e8edf5] mb-2 tracking-tight">Awaiting load profile.</h3>
      <p className="text-[#8a94a6] text-sm max-w-md mx-auto leading-relaxed">
        Upload a 30-minute interval xlsx â date column plus a kW Import column is enough.
        Multiple sheets are concatenated. The calculator handles the four formats from the case-study dataset out of the box.
      </p>
      <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-xl mx-auto">
        {[
          ['SoL', 'with solar'],
          ['E', 'no solar'],
          ['SuN', 'no solar'],
          ['Mi2', 'with solar'],
        ].map(([code, lbl], i) => (
          <div key={i} className="border border-[#252b3d] p-3">
            <div className="text-xs font-mono text-[#5eff7a]">{code}</div>
            <div className="text-[10px] text-[#5a6478] font-mono mt-1">{lbl}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
