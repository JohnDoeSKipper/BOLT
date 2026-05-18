"""
PowerRECO — AI Solar & Battery Sizing Optimizer
Analyses BOLT Predictor and Manager outputs to recommend the optimal
solar PV size and battery capacity with a full 25-year ROI model.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Helper: KPI card renderer (defined early — called throughout the UI) ──────
def _kpi(col, label: str, value: str, caption: str, style: str = "big") -> None:
    col.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
    col.markdown(f'<div class="{style}-metric">{value}</div>', unsafe_allow_html=True)
    col.caption(caption)


def _manual_battery_result(daily_kwh: float, md_kw: float) -> dict:
    from src.battery_sizing import ROUNDTRIP_EFF, UPSIZE_FACTOR, _round_to_commercial
    usable = daily_kwh / ROUNDTRIP_EFF
    cap    = usable * UPSIZE_FACTOR
    return {
        "n_days_analyzed": 0,
        "peak_daily_discharge_kwh": daily_kwh,
        "n_top_days_used": 0,
        "avg_top_discharge_kwh": daily_kwh,
        "roundtrip_eff_pct": ROUNDTRIP_EFF * 100,
        "required_usable_kwh": round(usable, 2),
        "usable_dod_pct": 60.0,
        "soc_range": "20%–80%",
        "upsize_factor": round(UPSIZE_FACTOR, 4),
        "min_capacity_kwh": round(cap, 2),
        "min_capacity_kwh_commercial": _round_to_commercial(cap),
        "md_reduction_kw": md_kw,
        "discharge_col_used": "manual",
        "spike_note": "Manual entry — no AI analysis performed.",
    }


from src.solar_sizing import calculate_solar_sizing
from src.battery_sizing import calculate_battery_sizing, _round_to_commercial
from src.roi_engine import (
    calculate_roi,
    MD_CHARGE_NEW, MD_CHARGE_OLD,
    ENERGY_RATE_C1, NEM_BUYBACK,
)
from src.data_connector import (
    parse_manager_output,
    parse_predictor_output,
    extract_peak_demand_stats,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PowerRECO — Solar & Battery Optimizer",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
:root { --accent:#00d4ff; --green:#22c55e; --amber:#f59e0b; --red:#ef4444; }
.big-metric   { font-size:2.1rem; font-weight:700; color:#00d4ff; line-height:1.15; }
.green-metric { font-size:2.1rem; font-weight:700; color:#22c55e; line-height:1.15; }
.amber-metric { font-size:2.1rem; font-weight:700; color:#f59e0b; line-height:1.15; }
.red-metric   { font-size:2.1rem; font-weight:700; color:#ef4444; line-height:1.15; }
.metric-label { color:#888; font-size:0.78rem; text-transform:uppercase;
                letter-spacing:.06em; margin-bottom:2px; }
.reco-box  { background:rgba(0,212,255,.08); border:1px solid rgba(0,212,255,.35);
             border-radius:10px; padding:14px 18px; margin:8px 0; }
.good-box  { background:rgba(34,197,94,.08);  border:1px solid rgba(34,197,94,.35);
             border-radius:10px; padding:14px 18px; margin:8px 0; }
.warn-box  { background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.35);
             border-radius:10px; padding:14px 18px; margin:8px 0; }
.method-step { background:rgba(255,255,255,.04); border-radius:8px;
               padding:10px 14px; margin:4px 0; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    solar_result=None,
    battery_result=None,
    roi_result=None,
    manager_df=None,
    predictor_df=None,
    peak_stats=None,
    run_done=False,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([1, 11])
with hc1:
    st.markdown("<h1 style='margin:0;padding-top:8px;'>⚡</h1>", unsafe_allow_html=True)
with hc2:
    st.markdown("<h1 style='margin:0;'>PowerRECO</h1>", unsafe_allow_html=True)
    st.caption(
        "AI-Powered Solar & Battery Sizing Optimizer  ·  "
        "Analyses BOLT Predictor forecasts and Manager optimisation results  ·  "
        "Full ROI model with TNB NEM 3.0 rates"
    )
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    # ── 1. Data uploads ───────────────────────────────────────────────────────
    st.subheader("1 · BOLT Data Sources")
    st.caption("Upload outputs from the Manager and Predictor to enable AI-based battery sizing.")

    manager_file = st.file_uploader(
        "Manager Results (CSV / JSON / Excel)",
        type=["csv", "json", "xlsx", "xls"],
        key="mgr_upload",
        help="The optimisation results export from the BOLT AI Manager (Peak Load Optimizer).",
    )
    predictor_file = st.file_uploader(
        "Predictor Forecast (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="pred_upload",
        help="Load forecast CSV from the BOLT AI Energy Demand Forecaster.",
    )

    if manager_file:
        try:
            st.session_state.manager_df = parse_manager_output(manager_file)
            n = len(st.session_state.manager_df)
            st.success(f"✅ Manager: {n:,} intervals")
        except Exception as exc:
            st.error(f"Manager parse error: {exc}")
            st.session_state.manager_df = None

    if predictor_file:
        try:
            st.session_state.predictor_df = parse_predictor_output(predictor_file)
            st.session_state.peak_stats   = extract_peak_demand_stats(st.session_state.predictor_df)
            n = len(st.session_state.predictor_df)
            st.success(f"✅ Predictor: {n:,} rows")
        except Exception as exc:
            st.error(f"Predictor parse error: {exc}")
            st.session_state.predictor_df = None

    st.divider()

    # ── 2. Solar configuration ─────────────────────────────────────────────────
    st.subheader("2 · Solar Configuration")

    roof_area = st.number_input(
        "Roof Area (m²)",
        min_value=10.0, max_value=100_000.0, value=200.0, step=10.0,
        help="Total roof area available for solar. PowerRECO uses 85% of this.",
    )
    panel_wattage = st.selectbox(
        "Panel Wattage",
        options=[400, 415, 430, 450, 480, 500, 550, 600],
        index=1,
        format_func=lambda w: f"{w} W — monocrystalline PERC/TOPCon",
    )
    psh = st.slider(
        "Peak Sun Hours / day",
        min_value=3.5, max_value=5.5, value=4.5, step=0.1,
        help="Malaysia peninsular average 4.5 PSH/day. Coastal/highland sites vary ±0.5.",
    )

    st.divider()

    # ── 3. Financial parameters ────────────────────────────────────────────────
    st.subheader("3 · Financial Parameters")

    use_new_tariff = st.toggle(
        "New MD Tariff (RM 97.06 / kW)",
        value=True,
        help="Toggle off to use the legacy tariff of RM 30.30/kW.",
    )
    st.caption(
        f"MD charge: **RM {MD_CHARGE_NEW if use_new_tariff else MD_CHARGE_OLD:.2f}/kW** · "
        f"Energy: **RM {ENERGY_RATE_C1}/kWh** · "
        f"NEM: **RM {NEM_BUYBACK}/kWh**"
    )

    self_consumption_pct = st.slider(
        "Solar Self-Consumption %",
        min_value=30, max_value=95, value=65, step=5,
        help="% of solar generation consumed on-site; remainder exported under NEM 3.0.",
    ) / 100

    with st.expander("Advanced Cost Settings"):
        solar_cost = st.number_input(
            "Solar Installed Cost (RM / kWp)",
            min_value=1_000, max_value=10_000, value=3_500, step=100,
        )
        batt_cost = st.number_input(
            "Battery Installed Cost (RM / kWh)",
            min_value=500, max_value=8_000, value=2_500, step=100,
        )

    st.divider()

    # ── 4. Manual battery override (shown when no Manager data) ───────────────
    if st.session_state.manager_df is None:
        st.subheader("4 · Manual Battery Override")
        st.caption("Upload Manager data for AI sizing, or set manually below.")
        manual_daily_kwh = st.number_input(
            "Expected daily battery discharge (kWh)",
            min_value=1.0, max_value=2_000.0, value=50.0, step=5.0,
        )
        manual_md_kw = st.number_input(
            "Expected MD reduction (kW)",
            min_value=0.0, max_value=5_000.0, value=100.0, step=10.0,
        )
    else:
        manual_daily_kwh = 0.0
        manual_md_kw = 0.0

    st.divider()
    run_btn = st.button("🚀 Analyse & Recommend", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Running PowerRECO analysis…"):

        # ── Solar ─────────────────────────────────────────────────────────────
        st.session_state.solar_result = calculate_solar_sizing(
            roof_area_m2=roof_area,
            panel_wattage_w=int(panel_wattage),
            psh=psh,
        )

        # ── Battery ───────────────────────────────────────────────────────────
        if st.session_state.manager_df is not None:
            try:
                st.session_state.battery_result = calculate_battery_sizing(
                    manager_df=st.session_state.manager_df,
                )
            except Exception as exc:
                st.warning(f"AI battery sizing failed ({exc}). Falling back to manual values.")
                st.session_state.battery_result = _manual_battery_result(
                    manual_daily_kwh, manual_md_kw
                )
        else:
            st.session_state.battery_result = _manual_battery_result(
                manual_daily_kwh, manual_md_kw
            ) if (manual_daily_kwh > 0 or manual_md_kw > 0) else None

        # ── ROI ───────────────────────────────────────────────────────────────
        sr = st.session_state.solar_result
        br = st.session_state.battery_result
        battery_kwh = br["min_capacity_kwh_commercial"] if br else 0.0
        md_kw       = br["md_reduction_kw"]              if br else 0.0

        st.session_state.roi_result = calculate_roi(
            solar_kwp=sr["system_kwp"],
            battery_kwh=battery_kwh,
            monthly_generation_kwh=sr["monthly_generation_kwh_avg"],
            md_reduction_kw=md_kw,
            self_consumption_pct=self_consumption_pct,
            use_new_tariff=use_new_tariff,
            solar_cost_per_kwp=float(solar_cost),
            battery_cost_per_kwh=float(batt_cost),
        )
        st.session_state.run_done = True

    st.success("✅ Analysis complete — see tabs below.")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.run_done and st.session_state.solar_result:
    sr = st.session_state.solar_result
    br = st.session_state.battery_result
    rr = st.session_state.roi_result

    tab_solar, tab_batt, tab_roi, tab_data = st.tabs([
        "☀️ Solar Sizing", "🔋 Battery Sizing", "💰 ROI & Financials", "📊 Data Overview",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SOLAR
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_solar:
        st.header("☀️ Solar Panel Sizing Recommendation")

        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, "System Size",       f"{sr['system_kwp']} kWp",        f"{sr['n_panels']} × {sr['panel_wattage_w']} W panels", "big")
        _kpi(c2, "Usable Roof Area",  f"{sr['usable_area_m2']} m²",     f"85% of {sr['roof_area_m2']} m²",                     "big")
        _kpi(c3, "Annual Generation", f"{sr['annual_generation_kwh']:,} kWh", f"~{sr['daily_generation_kwh_avg']} kWh / day",   "green")
        _kpi(c4, "Monthly Average",   f"{sr['monthly_generation_kwh_avg']:,} kWh", "avg per month",                             "green")

        st.divider()
        cl, cr = st.columns(2)

        with cl:
            st.subheader("Roof Area Utilisation")
            fig = go.Figure(go.Pie(
                labels=["Solar Array (85%)", "Unusable — edges & obstructions (15%)"],
                values=[sr["usable_area_m2"], sr["roof_area_m2"] - sr["usable_area_m2"]],
                marker=dict(colors=["#00d4ff", "#1f2f4a"]),
                hole=0.48,
                textinfo="label+percent",
            ))
            fig.update_layout(
                template="plotly_dark", height=320, showlegend=False,
                annotations=[dict(
                    text=f"<b>{sr['system_kwp']}</b><br>kWp",
                    x=0.5, y=0.5, font_size=18, showarrow=False,
                )],
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.subheader("System Specifications")
            specs = {
                "Total Roof Area":         f"{sr['roof_area_m2']} m²",
                "Usable Area (85%)":       f"{sr['usable_area_m2']} m²",
                "Panel Count":             f"{sr['n_panels']} panels",
                "Panel Wattage":           f"{sr['panel_wattage_w']} W",
                "Area per Panel (w/ spacing)": f"{sr['effective_panel_area_m2']} m²",
                "System Capacity":         f"{sr['system_kwp']} kWp",
                "Peak Sun Hours":          f"{sr['psh']} hr/day",
                "System Derate Factor":    f"{sr['system_derate_pct']:.0f}%",
                "Annual Generation":       f"{sr['annual_generation_kwh']:,} kWh",
                "Monthly Average":         f"{sr['monthly_generation_kwh_avg']:,} kWh",
                "Daily Average":           f"{sr['daily_generation_kwh_avg']} kWh",
            }
            st.dataframe(
                pd.DataFrame(list(specs.items()), columns=["Parameter", "Value"]),
                use_container_width=True, hide_index=True, height=385,
            )

        st.subheader("Estimated Monthly Generation Profile (Malaysia Seasonal)")
        fig2 = go.Figure(go.Bar(
            x=sr["month_labels"],
            y=sr["monthly_breakdown_kwh"],
            marker_color="#00d4ff",
            text=[f"{v:,}" for v in sr["monthly_breakdown_kwh"]],
            textposition="outside",
        ))
        fig2.add_hline(
            y=sr["monthly_generation_kwh_avg"], line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Monthly avg  {sr['monthly_generation_kwh_avg']:,} kWh",
            annotation_position="top right",
        )
        fig2.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="Month", yaxis_title="kWh",
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
<div class="reco-box">
<b>Solar Recommendation:</b> Install a <b>{sr['system_kwp']} kWp</b> system comprising
<b>{sr['n_panels']} × {sr['panel_wattage_w']} W</b> monocrystalline panels across
<b>{sr['usable_area_m2']} m²</b> (85% of your {sr['roof_area_m2']} m² roof).
Estimated annual generation: <b>{sr['annual_generation_kwh']:,} kWh/year</b> at
{sr['psh']} PSH/day with 80% system derate.
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — BATTERY
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_batt:
        st.header("🔋 Battery Capacity Recommendation")

        if br is None:
            st.info("Upload Manager data or enter manual values in the sidebar to size the battery.")
        else:
            source_label = (
                "AI-derived from Manager data" if br["discharge_col_used"] != "manual"
                else "Manual entry (no Manager data)"
            )
            st.caption(f"Source: {source_label}")

            c1, c2, c3, c4 = st.columns(4)
            _kpi(c1, "Minimum Capacity",  f"{br['min_capacity_kwh_commercial']} kWh",
                 f"exact calc: {br['min_capacity_kwh']:.1f} kWh → rounded up", "big")
            _kpi(c2, "Usable Energy",     f"{br['required_usable_kwh']} kWh",
                 f"SOC window {br['soc_range']}", "green")
            _kpi(c3, "Upsize Factor",     f"×{br['upsize_factor']:.2f}",
                 "60% DOD → ×1.67 rule", "amber")
            _kpi(c4, "MD Reduction",      f"{br['md_reduction_kw']} kW",
                 "peak demand saved by battery", "green")

            st.divider()
            cl, cr = st.columns(2)

            with cl:
                st.subheader("Battery SOC Operating Window")
                cap    = br["min_capacity_kwh_commercial"]
                bot    = cap * 0.20
                usable = cap * 0.60
                top    = cap * 0.20
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=["Battery"], y=[bot],    base=[0],
                    name="Protected Reserve 0–20%", marker_color="#ef4444"))
                fig3.add_trace(go.Bar(x=["Battery"], y=[usable], base=[bot],
                    name=f"Usable Zone 20–80% = {usable:.0f} kWh", marker_color="#22c55e"))
                fig3.add_trace(go.Bar(x=["Battery"], y=[top],    base=[bot + usable],
                    name="Buffer Reserve 80–100%", marker_color="#f59e0b"))
                fig3.update_layout(
                    template="plotly_dark", barmode="stack", height=380,
                    yaxis_title="kWh",
                    title=f"Battery Bank: {cap} kWh Total Installed Capacity",
                    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
                    margin=dict(b=80),
                )
                st.plotly_chart(fig3, use_container_width=True)

            with cr:
                st.subheader("Sizing Methodology (step-by-step)")
                steps = [
                    ("Days analysed",                 f"{br['n_days_analyzed']}"),
                    ("Peak single-day discharge",     f"{br['peak_daily_discharge_kwh']} kWh"),
                    ("Spike protection",               br["spike_note"]),
                    ("Design discharge (avg top days)",f"{br['avg_top_discharge_kwh']} kWh"),
                    ("÷ Round-trip efficiency (90%)",  f"{br['required_usable_kwh']} kWh usable needed"),
                    ("60% DOD window",                 f"SOC {br['soc_range']} only"),
                    ("× Upsize factor 1.67",           f"{br['min_capacity_kwh']:.1f} kWh total"),
                    ("Round to commercial size",       f"→ {br['min_capacity_kwh_commercial']} kWh"),
                ]
                for label, val in steps:
                    st.markdown(
                        f'<div class="method-step"><b>{label}:</b> {val}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown(f"""
<div class="reco-box">
<b>Battery Recommendation:</b> Minimum <b>{br['min_capacity_kwh_commercial']} kWh</b> LFP battery bank.
Battery operates between 20–80% SOC to protect cycle life — effective usable capacity is
<b>{br['required_usable_kwh']} kWh</b>.  Sized to the <i>average of the top {br['n_top_days_used']}
highest-demand days</i> (not the single worst spike), giving a realistic and cost-effective capacity.
Expected MD reduction: <b>{br['md_reduction_kw']} kW</b>.
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — ROI
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_roi:
        st.header("💰 ROI & Financial Analysis")

        if rr is None:
            st.warning("Run the analysis first.")
        else:
            pb = rr["simple_payback_years"]
            npv_val = rr["npv_25yr_rm"]

            # KPI row
            c1, c2, c3, c4, c5 = st.columns(5)
            _kpi(c1, "Total CAPEX",    f"RM {rr['total_capex_rm']:,}",
                 f"Solar RM {rr['solar_capex_rm']:,} · Batt RM {rr['battery_capex_rm']:,}", "big")
            _kpi(c2, "Annual Savings", f"RM {rr['total_annual_benefit_rm']:,}",
                 f"RM {rr['monthly_net_benefit_rm']:,} / month", "green")
            pb_style = "green" if pb < 8 else "amber" if pb < 12 else "red"
            _kpi(c3, "Payback Period", f"{pb} yrs", "simple payback", pb_style)
            npv_style = "green" if npv_val > 0 else "red"
            _kpi(c4, "NPV (25 yr)",   f"RM {npv_val:,}", "@ 6% discount rate", npv_style)
            irr_txt = f"{rr['irr_pct']}%" if rr["irr_pct"] else "N/A"
            irr_style = "green" if (rr["irr_pct"] or 0) > 10 else "amber"
            _kpi(c5, "IRR", irr_txt, "internal rate of return", irr_style)

            st.divider()
            cl, cr = st.columns([3, 2])

            with cl:
                st.subheader("Cumulative NPV — 25-Year Forecast")
                years = list(range(1, 26))
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=years, y=rr["cumulative_npv"],
                    name="Cumulative NPV",
                    line=dict(color="#00d4ff", width=3),
                    fill="tozeroy", fillcolor="rgba(0,212,255,.10)",
                ))
                fig4.add_hline(y=0, line_dash="dash", line_color="#ef4444",
                               annotation_text="Breakeven", annotation_position="bottom right")
                pb_yr = int(math.ceil(pb))
                if 1 <= pb_yr <= 25:
                    fig4.add_vline(x=pb_yr, line_dash="dot", line_color="#f59e0b",
                                   annotation_text=f"Payback yr {pb_yr}",
                                   annotation_position="top left")
                fig4.update_layout(
                    template="plotly_dark", height=380,
                    xaxis_title="Year", yaxis_title="RM",
                    hovermode="x unified",
                    margin=dict(t=20),
                )
                st.plotly_chart(fig4, use_container_width=True)

            with cr:
                st.subheader("Annual Benefit Breakdown (Year 1)")
                labels = ["Energy Offset\n(Self-consumed Solar)",
                          "NEM Export Credit",
                          "MD Peak Shaving\n(Battery)"]
                values = [
                    rr["annual_energy_savings_rm"],
                    rr["annual_nem_credit_rm"],
                    rr["annual_md_savings_rm"],
                ]
                fig5 = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker=dict(colors=["#00d4ff", "#22c55e", "#f59e0b"]),
                    hole=0.42,
                    textinfo="label+percent",
                    texttemplate="%{label}<br>RM %{value:,}",
                ))
                fig5.update_layout(
                    template="plotly_dark", height=380,
                    showlegend=False, margin=dict(t=20),
                )
                st.plotly_chart(fig5, use_container_width=True)

            # Full detail table
            st.subheader("Full Financial Summary")
            summary_rows = [
                ("Solar system",             f"{sr['system_kwp']} kWp"),
                ("Battery bank",             f"{br['min_capacity_kwh_commercial']} kWh" if br else "—"),
                ("Solar CAPEX",              f"RM {rr['solar_capex_rm']:,}"),
                ("Battery CAPEX",            f"RM {rr['battery_capex_rm']:,}"),
                ("Total CAPEX",              f"RM {rr['total_capex_rm']:,}"),
                ("MD charge rate used",      f"RM {rr['md_rate_used']:.2f}/kW"),
                ("Energy tariff (C1)",        f"RM {rr['energy_rate_used']}/kWh"),
                ("NEM buyback rate",          f"RM {rr['nem_rate_used']}/kWh"),
                ("Annual energy savings",    f"RM {rr['annual_energy_savings_rm']:,}"),
                ("Annual NEM credit",         f"RM {rr['annual_nem_credit_rm']:,}"),
                ("Annual MD savings",         f"RM {rr['annual_md_savings_rm']:,}"),
                ("Total annual benefit (Y1)", f"RM {rr['total_annual_benefit_rm']:,}"),
                ("Monthly net benefit",       f"RM {rr['monthly_net_benefit_rm']:,}"),
                ("Simple payback",            f"{pb} years"),
                ("NPV over 25 yr @ 6%",       f"RM {npv_val:,}"),
                ("IRR",                       irr_txt),
                ("Self-consumed (annual)",    f"{rr['self_consumed_kwh_annual']:,} kWh"),
                ("NEM exported (annual)",     f"{rr['exported_kwh_annual']:,} kWh"),
                ("CO₂ offset",               f"{rr['co2_offset_tonnes_yr']:.1f} tonnes/year"),
            ]
            st.dataframe(
                pd.DataFrame(summary_rows, columns=["Parameter", "Value"]),
                use_container_width=True, hide_index=True,
            )

            # Verdict
            if pb < 7:
                box, verdict = "good-box", f"✅ <b>Excellent ROI</b> — {pb}-year payback is outstanding for a solar + battery system in Malaysia. Strongly recommended."
            elif pb < 10:
                box, verdict = "good-box", f"✅ <b>Good ROI</b> — {pb}-year payback is competitive. Recommended investment."
            elif pb < 13:
                box, verdict = "warn-box", f"⚠️ <b>Moderate ROI</b> — {pb}-year payback. Consider solar-only first or renegotiate installation cost."
            else:
                box, verdict = "warn-box", f"⚠️ <b>Long Payback</b> — {pb} years. Review input costs or reduce battery size. Solar-only may offer better returns."
            st.markdown(f'<div class="{box}">{verdict}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — DATA OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_data:
        st.header("📊 Loaded Data Overview")
        dl, dr = st.columns(2)

        with dl:
            st.subheader("Manager Data")
            df_m = st.session_state.manager_df
            if df_m is not None:
                st.caption(f"{len(df_m):,} intervals · {len(df_m.columns)} columns")
                st.dataframe(df_m.head(100), use_container_width=True)
            else:
                st.info("No Manager data uploaded.")

        with dr:
            st.subheader("Predictor Data")
            df_p = st.session_state.predictor_df
            ps   = st.session_state.peak_stats or {}
            if df_p is not None:
                st.caption(f"{len(df_p):,} rows · {len(df_p.columns)} columns")
                if ps:
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("Peak kW",      ps.get("peak_kw", "—"))
                    pc2.metric("Avg Top 5%",   ps.get("avg_top5pct_kw", "—"))
                    pc3.metric("Mean kW",      ps.get("mean_kw", "—"))
                st.dataframe(df_p.head(100), use_container_width=True)
            else:
                st.info("No Predictor data uploaded.")

# ═══════════════════════════════════════════════════════════════════════════════
# LANDING STATE (before first run)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
<div style="text-align:center;padding:50px 0 30px;">
  <h2 style="color:#00d4ff;margin-bottom:8px;">Configure inputs in the sidebar, then click Analyse & Recommend</h2>
  <p style="color:#888;font-size:1.05rem;max-width:680px;margin:0 auto;">
  PowerRECO reads your BOLT Manager and Predictor outputs, combines them with your roof area,
  and delivers a full solar + battery sizing recommendation with 25-year ROI.
  </p>
</div>
""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("""
#### ☀️ Solar Sizing
Enter roof area → PowerRECO places the maximum number of **415 W panels inside 85%** of
the usable roof area and estimates annual generation using Malaysia's seasonal irradiance profile.
        """)
    with fc2:
        st.markdown("""
#### 🔋 Battery Sizing
Upload Manager results → the engine derives daily discharge demand, applies the
**60% SOC window (×1.67 upsize)**, and uses the **average of the top discharge days**
(not the spike peak) to give a realistic, non-over-engineered battery capacity.
        """)
    with fc3:
        st.markdown("""
#### 💰 ROI Analysis
Full 25-year model: energy offset savings, **NEM 3.0 export credit (RM 0.31/kWh)**,
MD peak-shaving savings at **RM 97.06/kW**, battery replacement at year 10,
simple payback, NPV @ 6%, and IRR.
        """)
