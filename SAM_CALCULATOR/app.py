"""
SAM Calculator — TNB Electricity Bill Estimator
Peninsular Malaysia · All tariff categories A / B / C1 / C2 / D / E1 / E2
Includes: full charge breakdown, solar NEM credit, auto-detect tariff by voltage/demand.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import auto_load, load_csv, summarize
from src.tnb_tariffs import (
    NEM_DEFAULT_RATE,
    TARIFF_META,
    TariffCode,
    auto_detect_tariff,
    calculate_bill,
    compute_monthly_stats,
    compute_nem_credit,
    get_md_rate,
    get_min_charge,
    is_peak_hour,
)

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAM Calculator — TNB Bill",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
/* --- Bill summary card --- */
.bill-card {
    border-radius: 14px;
    padding: 26px 32px 22px;
    color: white;
    margin-bottom: 8px;
}
.bill-card-gross { background: linear-gradient(135deg, #1a3c5e 0%, #0077b6 100%); }
.bill-card-net   { background: linear-gradient(135deg, #1a5e2e 0%, #2d9e50 100%); }
.bill-total  { font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; line-height: 1.1; }
.bill-sub    { font-size: 0.85rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.tariff-pill {
    display: inline-block;
    background: rgba(255,255,255,0.20);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.84rem;
    font-weight: 600;
    margin-top: 10px;
}
/* --- Detect badge --- */
.detect-badge {
    background: #e8f4fd;
    border-left: 4px solid #0077b6;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.9rem;
    margin-bottom: 12px;
}
/* --- Separator row in tables --- */
.separator { color: #aaa; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "df": None,
    "monthly_stats": None,
    "auto_tariff": None,
    "auto_reason": None,
    "auto_stats": None,
    "has_solar": False,
    "interval_min": 30,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Utilities ─────────────────────────────────────────────────────────────────

def _parse_upload(f) -> pd.DataFrame | None:
    suf = Path(f.name).suffix.lower()
    tmp = Path("data") / f"_upload_{f.name}"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(f.getvalue())
    try:
        return load_csv(tmp) if suf == ".csv" else auto_load(tmp)
    except Exception as e:
        st.error(f"Could not parse **{f.name}**: {e}")
        return None


def _detect_interval(df: pd.DataFrame) -> int:
    diffs = df["timestamp"].diff().dropna().dt.total_seconds() / 60
    m = diffs.mode()
    return int(m.iloc[0]) if not m.empty else 30


def _gross_card(bill: dict, label: str = ""):
    h = label or "Gross Bill (before solar NEM credit)"
    st.markdown(
        f'<div class="bill-card bill-card-gross">'
        f'<div class="bill-sub">{h}</div>'
        f'<div class="bill-total">RM {bill["total_bill"]:,.2f}</div>'
        f'<div class="tariff-pill">⚡ {bill["tariff_name"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _net_card(net_bill: float, nem: dict, label: str = ""):
    h = label or "Net Bill (after solar NEM credit)"
    st.markdown(
        f'<div class="bill-card bill-card-net">'
        f'<div class="bill-sub">{h}</div>'
        f'<div class="bill-total">RM {net_bill:,.2f}</div>'
        f'<div class="tariff-pill">☀️ NEM credit: RM {nem["nem_credit_rm"]:,.2f}'
        f'  ({nem["export_kwh"]:,.1f} kWh × RM {nem["nem_rate_rm"]:.3f})</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ SAM Calculator")
    st.caption("TNB Bill Estimator · Peninsular Malaysia")
    st.divider()

    # 1. File upload
    st.subheader("1. Upload Energy Usage File")
    st.caption("CSV, XLSX, or XLS — must contain `timestamp` and `kw_import` columns.")
    uploaded = st.file_uploader(
        "Drop load profile here", type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    if uploaded:
        df_raw = _parse_upload(uploaded)
        if df_raw is not None:
            intvl = _detect_interval(df_raw)
            code, reason, astats = auto_detect_tariff(df_raw, intvl)
            st.session_state.df            = df_raw
            st.session_state.interval_min  = intvl
            st.session_state.auto_tariff   = code
            st.session_state.auto_reason   = reason
            st.session_state.auto_stats    = astats
            st.session_state.monthly_stats = compute_monthly_stats(df_raw, intvl)
            st.session_state.has_solar     = bool((df_raw["kw_export"] > 0).any())
            s = summarize(df_raw)
            st.success(
                f"✅ **{s['rows']:,}** readings  ·  {s['days']} days  \n"
                f"Peak: **{s['max_kw_import']:.1f} kW**  ·  "
                f"Solar export: {'**Yes ☀️**' if s['has_solar'] else 'None'}"
            )

    st.divider()

    # 2. Tariff
    st.subheader("2. Tariff Category")
    t_keys   = list(TARIFF_META.keys())
    t_labels = [f"{k}  —  {TARIFF_META[k]['name'].split('—')[1].strip()}" for k in t_keys]
    auto_idx = t_keys.index(st.session_state.auto_tariff) if st.session_state.auto_tariff else 0
    use_auto = st.checkbox("Auto-detect from file", value=True)

    if use_auto and st.session_state.auto_tariff:
        selected: TariffCode = st.session_state.auto_tariff
        st.info(f"Detected: **{selected}**  \n{TARIFF_META[selected]['voltage']}")
    else:
        lbl = st.selectbox("Select tariff manually", t_labels, index=auto_idx)
        selected = t_keys[t_labels.index(lbl)]

    st.divider()

    # 3. Charges
    st.subheader("3. Charge Settings")
    icpt_sen = st.number_input(
        "ICPT (sen / kWh)", min_value=-10.0, max_value=20.0, value=0.0, step=0.1, format="%.2f",
        help="Quarterly surcharge (+) or rebate (−) set by Suruhanjaya Tenaga. 0 = exempt / not applicable.",
    )

    _ms = st.session_state.monthly_stats
    has_solar_flag = bool(
        st.session_state.has_solar or
        (_ms is not None and len(_ms) > 0 and bool((_ms["export_kwh"] > 0).any()))
    )
    nem_rate = NEM_DEFAULT_RATE
    if has_solar_flag:
        st.markdown("**☀️ Solar NEM Settings**")
        nem_rate = st.number_input(
            "NEM Buyback Rate (RM / kWh)",
            min_value=0.0, max_value=1.0, value=NEM_DEFAULT_RATE, step=0.01, format="%.3f",
            help="TNB NEM 3.0 default: RM 0.31/kWh. Check your NEM agreement for the exact rate.",
        )

    st.divider()
    with st.expander("⚙️ Advanced"):
        apply_st  = st.checkbox("Apply Service Tax (8%)", value=True,
            help="Non-domestic only. Tariff A is always exempt. Rate raised from 6% to 8% in March 2024.")
        apply_kb  = st.checkbox("Apply KWTBB (1.6%)", value=True,
            help="Kumpulan Wang Tenaga Boleh Baharu — non-domestic only.")
        intvl_ov  = st.number_input(
            "Sampling interval (minutes)", min_value=1, max_value=60,
            value=st.session_state.interval_min, step=1,
        )


# ─── Page header ──────────────────────────────────────────────────────────────
st.title("⚡ SAM Calculator — TNB Bill Estimator")
st.caption(
    "Upload your energy usage graph → auto-detect tariff category → "
    "get a fully itemised TNB bill with solar NEM credit breakdown."
)

# ─── No file yet ───────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("👈  Upload a load-profile file in the sidebar to begin.")
    st.divider()

    st.subheader("📋 TNB Tariff Quick Reference (Peninsular Malaysia)")
    st.dataframe(pd.DataFrame([
        {"Tariff": "A",  "Category": "Domestic",             "Voltage Supply": "Low (240 V / 415 V)",  "Energy Rate": "Tiered  RM 0.218 – 0.571 / kWh", "MD (RM/kW)": "—",      "Min Bill": "RM 3.00",   "Service Tax": "Exempt"},
        {"Tariff": "B",  "Category": "LV Commercial",        "Voltage Supply": "Low (240 V / 415 V)",  "Energy Rate": "RM 0.435 / kWh (flat)",           "MD (RM/kW)": "—",      "Min Bill": "RM 7.20",   "Service Tax": "8 %"},
        {"Tariff": "C1", "Category": "MV General",           "Voltage Supply": "Medium (6.6–22 kV)",   "Energy Rate": "RM 0.365 / kWh (flat)",           "MD (RM/kW)": "97.06",  "Min Bill": "RM 600.00", "Service Tax": "8 %"},
        {"Tariff": "C2", "Category": "MV Time-of-Use",       "Voltage Supply": "Medium (6.6–22 kV)",   "Energy Rate": "Peak 0.365 / Off-peak 0.219",     "MD (RM/kW)": "97.06",  "Min Bill": "RM 600.00", "Service Tax": "8 %"},
        {"Tariff": "D",  "Category": "HV General",           "Voltage Supply": "High (33–132 kV)",     "Energy Rate": "RM 0.337 / kWh (flat)",           "MD (RM/kW)": "97.06",  "Min Bill": "RM 600.00", "Service Tax": "8 %"},
        {"Tariff": "E1", "Category": "HV Peak/Off-Peak",     "Voltage Supply": "High (33–132 kV)",     "Energy Rate": "Peak 0.337 / Off-peak 0.202",     "MD (RM/kW)": "97.06",  "Min Bill": "RM 600.00", "Service Tax": "8 %"},
        {"Tariff": "E2", "Category": "HV TOU (MD ≥ 1500 kW)","Voltage Supply": "High (33–132 kV)",    "Energy Rate": "Peak 0.337 / Off-peak 0.202",     "MD (RM/kW)": "97.06",  "Min Bill": "RM 600.00", "Service Tax": "8 %"},
    ]), use_container_width=True, hide_index=True)

    st.caption(
        "Peak hours (TOU tariffs): 08:00–22:00, Monday–Saturday.  "
        "KWTBB 1.6% and Service Tax 8% apply to non-domestic tariffs.  "
        "Solar NEM credit applied after all charges."
    )
    st.stop()


# ─── Tabs ──────────────────────────────────────────────────────────────────────
df      = st.session_state.df
if df is None:
    st.info("👈  Upload a load-profile file in the sidebar to begin.")
    st.stop()

mstats  = st.session_state.monthly_stats
intvl   = intvl_ov
is_dom  = selected == "A"
use_st  = apply_st and not is_dom
use_kb  = apply_kb and not is_dom

tab_usage, tab_bill, tab_monthly, tab_ref = st.tabs([
    "📊 Energy Usage",
    "🧾 Bill Breakdown",
    "📅 Monthly Report",
    "ℹ️ Tariff Reference",
])


# ══════════════════════════ TAB 1 — ENERGY USAGE ══════════════════════════════
with tab_usage:
    st.header("Energy Usage Overview")

    total_kwh_all = float(df["kw_import"].sum() * (intvl / 60))
    peak_kw_all   = float(df["kw_import"].max())
    avg_kw_all    = float(df["kw_import"].mean())
    export_kwh_all = float(df["kw_export"].sum() * (intvl / 60))
    span_days     = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total kWh Imported",    f"{total_kwh_all:,.1f} kWh")
    c2.metric("Peak Demand",           f"{peak_kw_all:,.1f} kW")
    c3.metric("Average Demand",        f"{avg_kw_all:,.1f} kW")
    c4.metric("Solar Export",          f"{export_kwh_all:,.1f} kWh" if has_solar_flag else "—")
    c5.metric("Data Span",             f"{span_days} days")

    # Auto-detect result
    if st.session_state.auto_tariff:
        code_d = st.session_state.auto_tariff
        meta_d = TARIFF_META[code_d]
        st.markdown(
            f'<div class="detect-badge">'
            f'🔍 <strong>Auto-detected: {code_d} — {meta_d["name"].split("—")[1].strip()}</strong><br>'
            f'<span style="color:#555">{meta_d["voltage"]}  ·  Typical demand: {meta_d["typical_demand"]}</span><br>'
            f'<em>{st.session_state.auto_reason}</em>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Load profile chart ────────────────────────────────────────────────────
    st.subheader("Load Profile — Grid Import")
    fig = go.Figure()

    df_p = df.copy()
    df_p["is_peak"] = (df_p["timestamp"] - pd.Timedelta(minutes=intvl)).apply(is_peak_hour)
    peak_y    = df_p["kw_import"].where(df_p["is_peak"])
    offpeak_y = df_p["kw_import"].where(~df_p["is_peak"])
    fig.add_trace(go.Scatter(
        x=df_p["timestamp"], y=offpeak_y,
        mode="lines", name="Off-Peak  (22:00–08:00 daily, Sun & PH all day)",
        line=dict(color="#00b4d8", width=1.0),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.15)"))
    fig.add_trace(go.Scatter(
        x=df_p["timestamp"], y=peak_y,
        mode="lines", name="Peak  (08:00–22:00, Mon–Sat, excl. PH)",
        line=dict(color="#ff6b35", width=1.0),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.18)"))
    del df_p

    # Mark absolute peak
    pi = df["kw_import"].idxmax()
    fig.add_trace(go.Scatter(
        x=[df.loc[pi, "timestamp"]], y=[peak_kw_all],
        mode="markers+text",
        marker=dict(color="#e63946", size=13, symbol="star"),
        text=[f"Peak\n{peak_kw_all:.1f} kW"],
        textposition="top center",
        name="Peak Demand",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Date / Time", yaxis_title="Power  (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Solar export chart ────────────────────────────────────────────────────
    if has_solar_flag:
        st.subheader("☀️ Solar Export (kW fed back to TNB grid)")
        fig_sol = go.Figure()
        fig_sol.add_trace(go.Scatter(
            x=df["timestamp"], y=df["kw_import"],
            name="Grid Import  (kW)", line=dict(color="#0077b6", width=1.4)))
        fig_sol.add_trace(go.Scatter(
            x=df["timestamp"], y=df["kw_export"],
            name="Solar Export  (kW)", line=dict(color="#f4a261", width=1.4),
            fill="tozeroy", fillcolor="rgba(244,162,97,0.15)"))
        fig_sol.update_layout(
            template="plotly_white", height=300,
            yaxis_title="kW",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_sol, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.subheader("Average Demand Heatmap  (Hour × Day of Week)")
    df_hm = df.copy()
    df_hm["hour"] = df_hm["timestamp"].dt.hour
    df_hm["dow"]  = df_hm["timestamp"].dt.day_name()
    dow_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = (df_hm.groupby(["dow","hour"])["kw_import"]
               .mean().unstack("hour").reindex(dow_ord))
    fig_hm = px.imshow(
        pivot, color_continuous_scale="Blues",
        labels=dict(x="Hour of Day", y="Day", color="Avg kW"),
        aspect="auto",
    )
    fig_hm.update_layout(template="plotly_white", height=280,
                         margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════ TAB 2 — BILL BREAKDOWN ════════════════════════════
with tab_bill:
    st.header("TNB Bill Breakdown")

    if mstats is None or len(mstats) == 0:
        st.warning("No data to calculate.")
    else:
        # Month picker
        months = [str(m) for m in mstats["month"]]
        chosen = (st.selectbox("Billing month", months, index=len(months) - 1)
                  if len(months) > 1 else months[0])

        row    = mstats[mstats["month"].astype(str) == chosen].iloc[0]
        t_kwh  = float(row["total_kwh"])
        p_kwh  = float(row["peak_kwh"])
        o_kwh  = float(row["offpeak_kwh"])
        md_kw  = float(row["max_demand_kw"])
        ex_kwh = float(row["export_kwh"])
        avg_pf_val = float(row["avg_pf"])

        bill = calculate_bill(
            tariff=selected,
            monthly_kwh=t_kwh, peak_kwh=p_kwh, offpeak_kwh=o_kwh,
            max_demand_kw=md_kw, icpt_sen_per_kwh=icpt_sen,
            apply_service_tax=use_st, apply_kwtbb=use_kb,
            avg_pf=avg_pf_val,
        )

        # NEM credit
        has_export = ex_kwh > 0
        nem = compute_nem_credit(ex_kwh, nem_rate) if has_export else None
        net_bill_amt = round(max(0.0, bill["total_bill"] - (nem["nem_credit_rm"] if nem else 0.0)), 2)

        # ── Bill summary cards ────────────────────────────────────────────────
        if has_export:
            col_g, col_n = st.columns(2)
            with col_g:
                _gross_card(bill, f"Gross Bill — {chosen}  (before solar credit)")
            with col_n:
                _net_card(net_bill_amt, nem, f"Net Bill — {chosen}  (after solar NEM credit)")
        else:
            _gross_card(bill, f"Total Bill — {chosen}")

        # Input summary
        with st.expander("📥 Inputs used for this calculation"):
            ci1, ci2, ci3, ci4, ci5 = st.columns(5)
            ci1.metric("Total kWh", f"{t_kwh:,.1f}")
            ci2.metric("Peak kWh",  f"{p_kwh:,.1f}")
            ci3.metric("Off-Peak",  f"{o_kwh:,.1f}")
            ci4.metric("Max Demand", f"{md_kw:.1f} kW")
            ci5.metric("Avg Power Factor", f"{avg_pf_val:.4f}")
            if has_export:
                st.metric("Solar Export", f"{ex_kwh:,.1f} kWh")

        st.divider()

        # ── Itemised charges ──────────────────────────────────────────────────
        st.subheader("Itemised Charges")

        if bill["blocks"]:
            st.markdown("**Energy Charge Breakdown**")
            st.dataframe(pd.DataFrame(bill["blocks"]), use_container_width=True, hide_index=True)

        st.markdown("**Full Charge Summary**")
        md_r = get_md_rate(selected)
        rows: list[dict] = [{"Charge Component": "Energy Charges",
                              "Amount (RM)": f"{bill['energy_charge']:,.2f}"}]
        if bill["md_charge"] > 0:
            rows.append({"Charge Component": f"Maximum Demand Charge  ({md_kw:.1f} kW × RM {md_r:.2f}/kW)",
                         "Amount (RM)": f"{bill['md_charge']:,.2f}"})
        if bill["icpt_charge"] != 0:
            sign = "Surcharge" if bill["icpt_charge"] >= 0 else "Rebate"
            rows.append({"Charge Component": f"ICPT {sign}  ({icpt_sen:+.2f} sen/kWh × {t_kwh:,.1f} kWh)",
                         "Amount (RM)": f"{bill['icpt_charge']:,.2f}"})
        if bill["kwtbb_charge"] > 0:
            rows.append({"Charge Component": "KWTBB Levy  (1.6% × energy charges)",
                         "Amount (RM)": f"{bill['kwtbb_charge']:,.2f}"})
        if bill["pf_surcharge"] > 0:
            rows.append({"Charge Component":
                         f"PF Surcharge  ({bill['pf_surcharge_pct']:.1f}% × energy + MD + KWTBB)"
                         f"  [avg PF {avg_pf_val:.4f}]",
                         "Amount (RM)": f"{bill['pf_surcharge']:,.2f}"})
        if bill["service_tax"] > 0:
            st_base = "energy + MD + KWTBB + PF surcharge" if bill["pf_surcharge"] > 0 else "energy + MD + KWTBB"
            rows.append({"Charge Component": f"Service Tax  (8% × {st_base})",
                         "Amount (RM)": f"{bill['service_tax']:,.2f}"})

        rows.append({"Charge Component": "─" * 48, "Amount (RM)": "─" * 14})
        rows.append({"Charge Component": "GROSS BILL TOTAL",
                     "Amount (RM)": f"{bill['total_bill']:,.2f}"})

        if bill["minimum_charge_applied"]:
            rows.append({"Charge Component": f"⚠️  Minimum monthly charge of RM {get_min_charge(selected):.2f} applied (gross bill floored here)",
                         "Amount (RM)": ""})

        if has_export and nem:
            rows.append({"Charge Component": "─" * 48, "Amount (RM)": "─" * 14})
            rows.append({"Charge Component":
                         f"☀️ Solar NEM Credit  ({ex_kwh:,.1f} kWh × RM {nem['nem_rate_rm']:.3f}/kWh)"
                         "  [deducted after SST — from gross bill above]",
                         "Amount (RM)": f"− {nem['nem_credit_rm']:,.2f}"})
            rows.append({"Charge Component": "─" * 48, "Amount (RM)": "─" * 14})
            rows.append({"Charge Component": "NET BILL (after solar NEM credit)",
                         "Amount (RM)": f"{net_bill_amt:,.2f}"})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for note in bill["notes"]:
            st.caption(f"ℹ️  {note}")

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        col_pie, col_eff = st.columns(2)

        with col_pie:
            st.subheader("Bill Composition")
            pie_data = [
                ("Energy Charges",    bill["energy_charge"]),
                ("Max Demand Charge", bill["md_charge"]),
                ("ICPT",              bill["icpt_charge"]),
                ("KWTBB Levy",        bill["kwtbb_charge"]),
                ("PF Surcharge",      bill["pf_surcharge"]),
                ("Service Tax",       bill["service_tax"]),
            ]
            if has_export and nem:
                pie_data.append(("Solar NEM Credit (−)", -nem["nem_credit_rm"]))

            labels = [l for l, v in pie_data if abs(v) > 0]
            values = [abs(v) for _, v in pie_data if abs(v) > 0]
            colors = ["#003566","#0077b6","#00b4d8","#90e0ef","#e63946","#caf0f8","#2d9e50"]

            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.42,
                marker_colors=colors[:len(labels)],
                textinfo="label+percent",
            ))
            fig_pie.update_layout(template="plotly_white", height=340,
                showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_eff:
            st.subheader("Key Metrics")
            if t_kwh > 0:
                eff_gross = bill["total_bill"] / t_kwh
                eff_net   = net_bill_amt / t_kwh
                st.metric("Effective Rate (gross)", f"RM {eff_gross:.4f} / kWh")
                if has_export:
                    st.metric("Effective Rate (net)",   f"RM {eff_net:.4f} / kWh",
                        delta=f"−RM {eff_gross - eff_net:.4f} savings from solar")

            if selected == "A" and bill["blocks"]:
                st.markdown("**Tariff A — Block Rates Used**")
                bn = [b["Block"] for b in bill["blocks"]]
                br = [b["Rate (RM/kWh)"] for b in bill["blocks"]]
                bk = [b["kWh"] for b in bill["blocks"]]
                fig_blk = go.Figure(go.Bar(
                    x=[f"{n}\n({k:.0f} kWh)" for n, k in zip(bn, bk)],
                    y=br,
                    marker_color=["#003566","#0077b6","#0096c7","#00b4d8","#48cae4"][:len(br)],
                    text=[f"RM {r}" for r in br], textposition="outside",
                ))
                fig_blk.update_layout(template="plotly_white", height=270,
                    yaxis_title="RM / kWh", margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_blk, use_container_width=True)


# ══════════════════════════ TAB 3 — MONTHLY REPORT ════════════════════════════
with tab_monthly:
    st.header("Monthly Bill Report")

    if mstats is None or len(mstats) == 0:
        st.warning("No data available.")
    else:
        records: list[dict] = []
        for _, row in mstats.iterrows():
            b = calculate_bill(
                tariff=selected,
                monthly_kwh=float(row["total_kwh"]),
                peak_kwh=float(row["peak_kwh"]),
                offpeak_kwh=float(row["offpeak_kwh"]),
                max_demand_kw=float(row["max_demand_kw"]),
                icpt_sen_per_kwh=icpt_sen,
                apply_service_tax=use_st, apply_kwtbb=use_kb,
                avg_pf=float(row["avg_pf"]),
            )
            ex  = float(row["export_kwh"])
            nem_m = compute_nem_credit(ex, nem_rate) if ex > 0 else {"nem_credit_rm": 0.0}
            net = round(max(0.0, b["total_bill"] - nem_m["nem_credit_rm"]), 2)

            records.append({
                "Month":               str(row["month"]),
                "Total kWh":           round(float(row["total_kwh"]), 1),
                "Peak kWh":            round(float(row["peak_kwh"]), 1),
                "Off-Peak kWh":        round(float(row["offpeak_kwh"]), 1),
                "Max Demand (kW)":     round(float(row["max_demand_kw"]), 1),
                "Avg PF":              round(float(row["avg_pf"]), 4),
                "Export kWh":          round(ex, 1),
                "Energy Charge (RM)":  round(b["energy_charge"], 2),
                "MD Charge (RM)":      round(b["md_charge"], 2),
                "ICPT (RM)":           round(b["icpt_charge"], 2),
                "KWTBB (RM)":          round(b["kwtbb_charge"], 2),
                "PF Surcharge (RM)":   round(b["pf_surcharge"], 2),
                "Service Tax (RM)":    round(b["service_tax"], 2),
                "Gross Bill (RM)":     round(b["total_bill"], 2),
                "NEM Credit (RM)":     round(nem_m["nem_credit_rm"], 2),
                "Net Bill (RM)":       net,
            })

        rdf = pd.DataFrame(records)
        st.dataframe(rdf, use_container_width=True, hide_index=True)

        # ── Chart 1: Bill (RM) ────────────────────────────────────────────────
        has_nem = has_solar_flag and rdf["NEM Credit (RM)"].sum() > 0
        st.subheader("Monthly Bill  (RM)")
        fig_bill = go.Figure()
        fig_bill.add_trace(go.Bar(
            x=rdf["Month"], y=rdf["Gross Bill (RM)"],
            name="Gross Bill (RM)", marker_color="#0077b6",
            text=rdf["Gross Bill (RM)"].apply(lambda v: f"RM {v:,.0f}"),
            textposition="outside",
        ))
        if has_nem:
            fig_bill.add_trace(go.Bar(
                x=rdf["Month"], y=rdf["Net Bill (RM)"],
                name="Net Bill after NEM (RM)", marker_color="#2d9e50",
                text=rdf["Net Bill (RM)"].apply(lambda v: f"RM {v:,.0f}"),
                textposition="outside",
            ))
        fig_bill.update_layout(
            template="plotly_white", height=380,
            barmode="group",
            yaxis=dict(title="Bill (RM)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_bill, use_container_width=True)

        # ── Chart 2: Energy (kWh) ─────────────────────────────────────────────
        st.subheader("Monthly Energy  (kWh)")
        fig_kwh = go.Figure()
        fig_kwh.add_trace(go.Bar(
            x=rdf["Month"], y=rdf["Total kWh"],
            name="Import kWh", marker_color="#0096c7",
            text=rdf["Total kWh"].apply(lambda v: f"{v:,.0f}"),
            textposition="outside",
        ))
        if has_nem:
            fig_kwh.add_trace(go.Bar(
                x=rdf["Month"], y=rdf["Export kWh"],
                name="Export kWh (Solar)", marker_color="#f4a261",
                text=rdf["Export kWh"].apply(lambda v: f"{v:,.0f}"),
                textposition="outside",
            ))
        fig_kwh.update_layout(
            template="plotly_white", height=340,
            barmode="group",
            yaxis=dict(title="kWh"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_kwh, use_container_width=True)

        # Totals
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("Total Spend (gross)",   f"RM {rdf['Gross Bill (RM)'].sum():,.2f}")
        cs2.metric("Total NEM Savings",     f"RM {rdf['NEM Credit (RM)'].sum():,.2f}")
        cs3.metric("Total Spend (net)",     f"RM {rdf['Net Bill (RM)'].sum():,.2f}")
        cs4.metric("Avg Monthly (net)",     f"RM {rdf['Net Bill (RM)'].mean():,.2f}")

        # Download
        buf = io.StringIO()
        rdf.to_csv(buf, index=False)
        st.download_button(
            "⬇️  Download Monthly Report (CSV)",
            data=buf.getvalue(),
            file_name="sam_calculator_monthly_report.csv",
            mime="text/csv",
        )


# ══════════════════════════ TAB 4 — TARIFF REFERENCE ══════════════════════════
with tab_ref:
    st.header("TNB Tariff Reference — Peninsular Malaysia")
    st.caption("Official TNB tariff schedule, revised 2014 and updated July 2025. MD rate raised to RM 97.06/kW effective July 2025.")

    # Tariff A
    st.subheader("Tariff A — Domestic (Progressive / Tiered)")
    st.dataframe(pd.DataFrame([
        {"Block": "First 200 kWh",              "Rate (RM/kWh)": 0.218},
        {"Block": "Next 100 kWh  (201 – 300)",  "Rate (RM/kWh)": 0.334},
        {"Block": "Next 300 kWh  (301 – 600)",  "Rate (RM/kWh)": 0.516},
        {"Block": "Next 300 kWh  (601 – 900)",  "Rate (RM/kWh)": 0.546},
        {"Block": "Remaining kWh (> 900)",       "Rate (RM/kWh)": 0.571},
    ]), use_container_width=True, hide_index=True)
    st.caption("Min monthly: RM 3.00 · Service Tax: exempt (domestic) · ICPT: generally exempt for domestic.")

    # Commercial / Industrial
    st.subheader("Commercial & Industrial Tariffs")
    st.dataframe(pd.DataFrame([
        {"Tariff":"B",  "Voltage":"LV  (< 1 kV)",      "Flat Rate":0.435, "Peak Rate":"—",  "Off-Peak":"—",   "MD Rate (RM/kW)":"—",    "Min Bill":"RM 7.20",   "ST":"8%","KWTBB":"1.6%"},
        {"Tariff":"C1", "Voltage":"MV  (6.6–22 kV)",   "Flat Rate":0.365, "Peak Rate":"—",  "Off-Peak":"—",   "MD Rate (RM/kW)":"97.06","Min Bill":"RM 600.00", "ST":"8%","KWTBB":"1.6%"},
        {"Tariff":"C2", "Voltage":"MV  (6.6–22 kV)",   "Flat Rate":"TOU", "Peak Rate":0.365,"Off-Peak":0.219, "MD Rate (RM/kW)":"97.06","Min Bill":"RM 600.00", "ST":"8%","KWTBB":"1.6%"},
        {"Tariff":"D",  "Voltage":"HV  (33–132 kV)",   "Flat Rate":0.337, "Peak Rate":"—",  "Off-Peak":"—",   "MD Rate (RM/kW)":"97.06","Min Bill":"RM 600.00", "ST":"8%","KWTBB":"1.6%"},
        {"Tariff":"E1", "Voltage":"HV  (33–132 kV)",   "Flat Rate":"TOU", "Peak Rate":0.337,"Off-Peak":0.202, "MD Rate (RM/kW)":"97.06","Min Bill":"RM 600.00", "ST":"8%","KWTBB":"1.6%"},
        {"Tariff":"E2", "Voltage":"HV  (33–132 kV)",   "Flat Rate":"TOU", "Peak Rate":0.337,"Off-Peak":0.202, "MD Rate (RM/kW)":"97.06","Min Bill":"RM 600.00", "ST":"8%","KWTBB":"1.6%"},
    ]), use_container_width=True, hide_index=True)

    # Other charges
    st.subheader("Additional Charges")
    st.dataframe(pd.DataFrame([
        {"Charge":"ICPT",                    "Detail":"Imbalance Cost Pass-Through. Quarterly surcharge(+) or rebate(−) per kWh set by Suruhanjaya Tenaga. Domestic (A) is generally exempt."},
        {"Charge":"KWTBB  (1.6%)",           "Detail":"Kumpulan Wang Tenaga Boleh Baharu. Renewable energy fund levy, charged on non-domestic energy charges."},
        {"Charge":"PF Surcharge  (1.5%/unit)","Detail":"Power Factor surcharge for Tariff C1/C2/D/E1/E2. Applied when average monthly PF < 0.85. Each 0.01 below 0.85 = 1 unit; surcharge = units × 1.5% × (energy + MD + KWTBB). Service Tax applies to this surcharge. PF computed using TNB meter method: total kWh / √(total kWh² + total kVARh²) for the billing month."},
        {"Charge":"Service Tax  (8%)",       "Detail":"Charged on (energy + MD + KWTBB + PF surcharge) for Tariff B/C/D/E. Domestic (Tariff A) fully exempt. Rate raised from 6% to 8% in March 2024. ICPT and NEM credit are NOT in the SST base."},
        {"Charge":"Maximum Demand  (MD)",    "Detail":"Highest recorded 30-min average kW in the billing month × RM/kW rate. Applies to Tariff C1, C2, D, E1, E2."},
        {"Charge":"Peak Hours  (TOU)",       "Detail":"08:00–22:00, Monday–Saturday, excluding Public Holidays. Off-Peak = 22:00–08:00 daily, all day Sunday, all day Public Holidays."},
        {"Charge":"Minimum Monthly Charge",  "Detail":"Enforced when the computed bill is below RM 3.00 (A), RM 7.20 (B), or RM 600.00 (C/D/E)."},
        {"Charge":"Solar NEM Credit",        "Detail":"Net Energy Metering buyback for solar export. Applied AFTER all tariff charges, ICPT, KWTBB, and Service Tax. Default: RM 0.31/kWh (TNB NEM 3.0)."},
    ]), use_container_width=True, hide_index=True)

    # Auto-detection rules
    st.subheader("Auto-Detection Rules (Voltage Proxy via Peak Demand)")
    st.dataframe(pd.DataFrame([
        {"Peak Demand (kW)": "< 25 kW  (residential pattern)",           "Inferred Tariff": "A — Domestic",              "Voltage": "Low Voltage",    "Notes": "Evening-peak load shape, avg monthly usage ≤ 2,000 kWh, daytime ratio ≤ 2.5×."},
        {"Peak Demand (kW)": "< 25 kW  (commercial pattern)",            "Inferred Tariff": "B — LV Commercial",         "Voltage": "Low Voltage",    "Notes": "Daytime-heavy profile (business-hours ratio > 2.5×) or avg monthly usage > 2,000 kWh."},
        {"Peak Demand (kW)": "25 – 999 kW",                              "Inferred Tariff": "C1 or C2",                  "Voltage": "Medium Voltage", "Notes": "C2 suggested if off-peak fraction ≥ 35% or load factor ≥ 70%."},
        {"Peak Demand (kW)": "≥ 1,000 kW  (daytime-heavy)",             "Inferred Tariff": "D — HV General",            "Voltage": "High Voltage",   "Notes": "Flat-rate billing. Off-peak fraction < 35% and load factor < 70%."},
        {"Peak Demand (kW)": "≥ 1,000 kW  (off-peak heavy or high LF)", "Inferred Tariff": "E1 — HV Peak/Off-Peak",     "Voltage": "High Voltage",   "Notes": "TOU billing beneficial. Off-peak fraction ≥ 35% or load factor ≥ 70%."},
        {"Peak Demand (kW)": "≥ 1,500 kW  (off-peak heavy or high LF)", "Inferred Tariff": "E2 — HV TOU (MD ≥ 1500 kW)","Voltage": "High Voltage",  "Notes": "E2 incentive TOU rate. Same condition as E1 but peak demand ≥ 1,500 kW."},
    ]), use_container_width=True, hide_index=True)
