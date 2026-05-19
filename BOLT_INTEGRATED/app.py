"""
BOLT Integrated — Unified AI Energy Management Platform
Combines: Predictor → Manager → SAM Calculator → PowerRECO

Run with:  streamlit run app.py
"""
from __future__ import annotations
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# ── Package imports ────────────────────────────────────────────────────────────
from predictor.data_loader import auto_load, load_csv, summarize
from predictor.solar_estimator import detect_has_solar, estimate_solar_capacity_kwp
from predictor.forecaster import DirectMultiStepForecaster
from predictor.cv import expanding_window_cv, format_cv_report

from manager.optimizer import run_ai_manager, parse_uploaded_data, calc_kva

from calculator.tnb_tariffs import (
    auto_detect_tariff, compute_monthly_stats, calculate_bill,
    compute_nem_credit, TARIFF_META,
)

from powerreco.solar_sizing import calculate_solar_sizing
from powerreco.battery_sizing import calculate_battery_sizing
from powerreco.roi_engine import calculate_roi

from pipeline.data_bridge import (
    forecast_to_manager_df,
    historical_to_manager_df,
    manager_results_to_sam_df,
    manager_results_to_original_df,
    manager_results_to_powerreco_df,
    manager_results_to_csv,
    forecast_result_to_csv,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BOLT Integrated",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────────
for key, default in [
    ("df", None),
    ("file_summary", None),
    ("solar_info", None),
    ("forecaster", None),
    ("forecast_result", None),
    ("manager_results", None),
    ("manager_df_optimized", None),
    ("manager_df_original", None),
    ("powerreco_df", None),
    ("solar_sizing", None),
    ("battery_sizing", None),
    ("roi", None),
    ("tariff_code", "C1"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_rm(v: float) -> str:
    return f"RM {v:,.2f}"


def _fmt_kw(v: float) -> str:
    return f"{v:,.1f} kW"


def _plot_load(df: pd.DataFrame, title: str = "Load Profile") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["kw_import"],
        mode="lines", name="kW Import", line=dict(color="#1f77b4"),
    ))
    if "kw_export" in df.columns and df["kw_export"].max() > 0:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["kw_export"],
            mode="lines", name="kW Export", line=dict(color="#ff7f0e"),
        ))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="kW",
                      hovermode="x unified", height=350)
    return fig


def _run_manager_on_df(df: pd.DataFrame, loads: dict, priority_order: list,
                        battery_kwh: float, peak_target_pct: float,
                        charge_upper_pct: float, c_rate: float,
                        init_soc_pct: float, bat_eff: float,
                        peak_ref_kva: float | None) -> list[dict]:
    mgr_df = historical_to_manager_df(df)
    return run_ai_manager(
        mgr_df, loads, battery_kwh, priority_order,
        peak_target_pct, charge_upper_pct,
        c_rate=c_rate, initial_soc_pct=init_soc_pct,
        bat_efficiency=bat_eff,
        peak_reference_kva=peak_ref_kva if peak_ref_kva and peak_ref_kva > 0 else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚡ BOLT Integrated")
    st.caption("AI Energy Management Platform")
    st.divider()

    if st.session_state.df is not None:
        summ = st.session_state.file_summary or {}
        st.success(f"Data loaded: {summ.get('rows', '?')} rows")
        st.caption(f"{summ.get('start', '?')} → {summ.get('end', '?')}")
        st.caption(f"Peak: {summ.get('max_kw_import', 0):.1f} kW")
        if st.button("Clear data", use_container_width=True):
            for key in ("df", "file_summary", "solar_info", "forecaster",
                        "forecast_result", "manager_results", "manager_df_optimized",
                        "manager_df_original", "powerreco_df",
                        "solar_sizing", "battery_sizing", "roi"):
                st.session_state[key] = None
            st.rerun()
    else:
        st.info("Upload a load profile to begin.")

    st.divider()
    st.caption("Pipeline status")
    _statuses = [
        ("Data", st.session_state.df is not None),
        ("Predictor", st.session_state.forecaster is not None),
        ("Manager", st.session_state.manager_results is not None),
        ("Calculator", st.session_state.manager_df_optimized is not None),
        ("PowerRECO", st.session_state.roi is not None),
    ]
    for label, done in _statuses:
        st.write(f"{'✅' if done else '⬜'} {label}")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Upload",
    "📈 Predictor",
    "⚙️ AI Manager",
    "💰 Bill Calculator",
    "🌞 PowerRECO",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DATA UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Load Profile Upload")
    st.markdown(
        "Upload an Excel or CSV file containing your site's half-hourly load data. "
        "Supported formats: all four TNB meter export formats (SOL, E, SUN, MI2) and plain CSV."
    )

    uploaded = st.file_uploader(
        "Choose a load profile file",
        type=["csv", "xlsx", "xls"],
        help="Half-hourly (30-min) interval data. Columns auto-detected.",
    )

    if uploaded is not None:
        try:
            with st.spinner("Parsing file…"):
                content = uploaded.read()
                fname   = uploaded.name
                # parse_uploaded_data handles all formats via bytes — avoids Path() issues
                df = parse_uploaded_data(content, fname)
                # Trim to canonical predictor columns (drop kw_net / kvar_net / kva)
                for col in ("kw_net", "kvar_net", "kva"):
                    if col in df.columns:
                        df = df.drop(columns=[col])

            st.session_state.df = df
            summ = summarize(df)
            summ["max_kw_import"] = float(df["kw_import"].max())
            st.session_state.file_summary = summ

            has_solar, solar_reason = detect_has_solar(df)
            solar_info = {"has_solar": has_solar, "reason": solar_reason}
            if has_solar:
                solar_est = estimate_solar_capacity_kwp(df)
                solar_info.update(solar_est)
            st.session_state.solar_info = solar_info

            tariff_code, tariff_reason, tariff_stats = auto_detect_tariff(df)
            st.session_state.tariff_code = tariff_code

            st.success(f"Loaded {summ['rows']:,} intervals  |  "
                       f"{summ['days']} days  |  "
                       f"Peak {summ['max_kw_import']:.1f} kW")

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean kW", f"{summ['mean_kw_import']:.1f}")
            col2.metric("Peak kW", f"{summ['max_kw_import']:.1f}")
            col3.metric("Days of data", summ["days"])

            st.plotly_chart(_plot_load(df, "Raw Load Profile"), use_container_width=True)

            with st.expander("Tariff auto-detection"):
                st.write(f"**Detected tariff:** {tariff_code} — {TARIFF_META[tariff_code]['name']}")
                st.write(tariff_reason)
                st.json(tariff_stats)

            if has_solar:
                with st.expander("Solar detection"):
                    st.warning(solar_reason)
                    if "capacity_kwp" in solar_info:
                        st.write(f"Estimated capacity: **{solar_info['capacity_kwp']} kWp**")

        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    elif st.session_state.df is not None:
        st.info("File already loaded. See sidebar for summary.")
        st.plotly_chart(_plot_load(st.session_state.df, "Loaded Load Profile"),
                        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("AI Load Predictor")

    if st.session_state.df is None:
        st.warning("Upload a load profile in the Data Upload tab first.")
        st.stop()

    df = st.session_state.df
    solar_info = st.session_state.solar_info or {}
    default_kwp = solar_info.get("capacity_kwp", 0.0)

    with st.expander("Model configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        capacity_kwp = col1.number_input(
            "Solar capacity (kWp)", min_value=0.0, value=float(default_kwp), step=10.0,
            help="0 if no solar. Auto-estimated from load profile if solar detected.")
        n_estimators = col2.number_input("LightGBM rounds", 100, 500, 250, step=50)
        lr           = col3.number_input("Learning rate", 0.01, 0.2, 0.05, step=0.01)

    col_train, col_cv = st.columns([2, 1])
    with col_train:
        if st.button("Train Forecaster", type="primary", use_container_width=True):
            try:
                with st.spinner("Training LightGBM models for all horizons…"):
                    fc = DirectMultiStepForecaster(
                        capacity_kwp=capacity_kwp,
                        n_estimators=int(n_estimators),
                        learning_rate=lr,
                    )
                    metrics = fc.fit(df, verbose=False)
                st.session_state.forecaster = fc
                st.success(
                    f"Trained {metrics['n_models_trained']} models. "
                    f"Mean MAPE: {metrics['mean_mape']:.2f}%  |  "
                    f"MAPE@24h: {metrics.get('mape_at_h24', 0):.2f}%"
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.code(traceback.format_exc())

    with col_cv:
        if st.button("Run Cross-Validation", use_container_width=True):
            if st.session_state.forecaster is None:
                st.warning("Train the forecaster first.")
            else:
                cap = st.session_state.forecaster.capacity_kwp
                with st.spinner("Running expanding-window CV…"):
                    try:
                        cv_summary = expanding_window_cv(
                            df,
                            forecaster_factory=lambda: DirectMultiStepForecaster(
                                capacity_kwp=cap, n_estimators=int(n_estimators), learning_rate=lr),
                            n_splits=4, min_train_days=21, val_block_days=5,
                        )
                        st.code(format_cv_report(cv_summary))
                    except Exception as e:
                        st.error(f"CV failed: {e}")

    if st.session_state.forecaster is not None:
        st.divider()
        st.subheader("24-Hour Forecast")
        fc = st.session_state.forecaster

        try:
            fr = fc.forecast(output_steps=48)
            st.session_state.forecast_result = fr

            fig = go.Figure()
            # Historical tail (last 48 readings)
            hist_tail = fc.history.tail(48)
            fig.add_trace(go.Scatter(
                x=hist_tail["timestamp"], y=hist_tail["kw_import"],
                mode="lines", name="Historical", line=dict(color="#666"),
            ))
            # Forecast bands
            fig.add_trace(go.Scatter(
                x=list(fr.timestamps) + list(fr.timestamps[::-1]),
                y=list(fr.p90) + list(fr.p10[::-1]),
                fill="toself", fillcolor="rgba(31,119,180,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="P10–P90 band",
            ))
            fig.add_trace(go.Scatter(
                x=fr.timestamps, y=fr.median,
                mode="lines", name="Median forecast",
                line=dict(color="#1f77b4", width=2),
            ))
            fig.update_layout(title="48-Step Ahead Forecast (24 hours)",
                              xaxis_title="Time", yaxis_title="kW",
                              hovermode="x unified", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Peak table
            peaks = fc.detect_peaks(fr)
            st.write("**Top 3 Predicted Peaks**")
            st.dataframe(peaks.style.format({
                "predicted_kw": "{:.1f}",
                "lower_bound_kw": "{:.1f}",
                "upper_bound_kw": "{:.1f}",
            }), use_container_width=True)

            # Download forecast CSV
            csv_str = forecast_result_to_csv(fr)
            st.download_button(
                "Download Forecast CSV",
                data=csv_str.encode(),
                file_name="forecast_output.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: AI MANAGER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("AI Load Manager")
    st.caption(
        "Runs peak-shaving and load-shifting optimization on the loaded data. "
        "Uses the exact discharge formula: dis_kw = kW − √(trigger² − kVAR²)."
    )

    if st.session_state.df is None:
        st.warning("Upload a load profile first.")
        st.stop()

    df = st.session_state.df

    with st.expander("Battery & optimization settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        battery_kwh   = c1.number_input("Battery capacity (kWh)", 50.0, 5000.0, 200.0, step=50.0)
        c_rate        = c2.number_input("C-rate", 0.1, 1.0, 0.5, step=0.1,
                                         help="Max charge/discharge rate as fraction of capacity per hour")
        bat_eff       = c3.number_input("One-way efficiency", 0.80, 1.00, 0.95, step=0.01)

        c4, c5, c6 = st.columns(3)
        peak_target   = c4.slider("Peak target (% of ref peak)", 70, 95, 85) / 100
        charge_upper  = c5.slider("Charge upper threshold (% of ref peak)", 50, 90, 70) / 100
        init_soc      = c6.slider("Initial SOC (%)", 20, 80, 50) / 100

        c7, c8 = st.columns(2)
        peak_ref_kva  = c7.number_input(
            "Reference peak kVA (0 = auto from rolling 30-day)", 0.0, 10000.0, 0.0, step=10.0)
        pre_md_hours  = c8.number_input("Pre-MD boost window (hours)", 0, 4, 2)

    st.subheader("Load configuration")
    st.caption("Define controllable loads and their priority order.")
    with st.expander("Load settings"):
        n_loads = st.number_input("Number of load groups", 1, 6, 3)
        loads = {}
        load_names_raw = []
        cols = st.columns(n_loads)
        for i, col in enumerate(cols):
            with col:
                default_names   = ["EV Chargers", "HVAC", "Misc Loads", "Lighting", "Process", "Other"]
                default_props   = [0.3, 0.4, 0.3, 0.1, 0.2, 0.1]
                default_cuts    = [10, 15, 20, 30, 10, 20]
                lname = st.text_input(f"Load {i+1} name", default_names[i] if i < len(default_names) else f"Load{i+1}")
                lprop = st.number_input(f"Proportion {i+1}", 0.0, 1.0, default_props[i] if i < len(default_props) else 0.2, step=0.05)
                lcut  = st.number_input(f"Max cut % {i+1}", 0, 50, default_cuts[i] if i < len(default_cuts) else 10)
                key   = f"load_{i}"
                loads[key] = {"name": lname, "proportion": lprop, "max_cut_pct": lcut / 100}
                load_names_raw.append(key)

        priority_order = load_names_raw  # order = priority

    if st.button("Run AI Manager", type="primary", use_container_width=True):
        try:
            with st.spinner("Optimizing load profile…"):
                results = _run_manager_on_df(
                    df, loads, priority_order,
                    battery_kwh, peak_target, charge_upper,
                    c_rate, init_soc, bat_eff,
                    peak_ref_kva if peak_ref_kva > 0 else None,
                )
            st.session_state.manager_results    = results
            st.session_state.manager_df_optimized = manager_results_to_sam_df(results)
            st.session_state.manager_df_original  = manager_results_to_original_df(results)
            st.session_state.powerreco_df         = manager_results_to_powerreco_df(results)
            st.success(f"Optimization complete — {len(results):,} intervals processed.")
        except Exception as e:
            st.error(f"Manager failed: {e}")
            st.code(traceback.format_exc())

    if st.session_state.manager_results:
        results = st.session_state.manager_results
        res_df  = pd.DataFrame([
            {k: v for k, v in r.items() if k != "actions"} for r in results
        ])
        res_df["timestamp"] = pd.to_datetime(res_df["timestamp"])

        # KPI metrics
        orig_peak    = float(res_df["kva_original"].max())
        managed_peak = float(res_df["kva_managed"].max())
        peak_red_pct = (orig_peak - managed_peak) / orig_peak * 100 if orig_peak else 0
        total_dis    = float(res_df["battery_discharge_kw"].sum()) * 0.5
        total_chg    = float(res_df["battery_charge_kw"].sum()) * 0.5

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Peak", f"{orig_peak:.1f} kVA")
        c2.metric("Managed Peak", f"{managed_peak:.1f} kVA", delta=f"-{peak_red_pct:.1f}%", delta_color="inverse")
        c3.metric("Total Discharge", f"{total_dis:.1f} kWh")
        c4.metric("Total Charge", f"{total_chg:.1f} kWh")

        # Before/after chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Load Profile kVA", "Battery SOC"),
                            row_heights=[0.65, 0.35])
        fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["kva_original"],
                                  mode="lines", name="Original kVA", line=dict(color="#d62728")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["kva_managed"],
                                  mode="lines", name="Managed kVA", line=dict(color="#1f77b4")), row=1, col=1)
        if "target_peak" in res_df.columns:
            fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["target_peak"],
                                      mode="lines", name="Target", line=dict(color="#2ca02c", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["battery_soc_pct"],
                                  mode="lines", name="SOC %", line=dict(color="#9467bd")), row=2, col=1)
        fig.update_layout(height=550, hovermode="x unified")
        fig.update_yaxes(title_text="kVA", row=1, col=1)
        fig.update_yaxes(title_text="SOC %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv_str = manager_results_to_csv(results)
        st.download_button(
            "Download Manager Results CSV",
            data=csv_str.encode(),
            file_name="manager_results.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: BILL CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("TNB Bill Calculator")
    st.caption("Compares monthly electricity bills before and after AI Manager optimization.")

    if st.session_state.df is None:
        st.warning("Upload a load profile first.")
        st.stop()

    if st.session_state.manager_df_optimized is None:
        st.info("Run the AI Manager first to see before/after bill comparison. "
                "The calculator will use raw uploaded data for now.")
        df_for_bill  = st.session_state.df
        df_orig_bill = st.session_state.df
        show_comparison = False
    else:
        df_for_bill  = st.session_state.manager_df_optimized
        df_orig_bill = st.session_state.manager_df_original
        show_comparison = True

    with st.expander("Tariff & billing settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        tariff_options = list(TARIFF_META.keys())
        default_idx    = tariff_options.index(st.session_state.tariff_code) if st.session_state.tariff_code in tariff_options else 2
        tariff         = c1.selectbox("Tariff", tariff_options, index=default_idx,
                                       format_func=lambda t: f"{t} — {TARIFF_META[t]['name']}")
        icpt_sen       = c2.number_input("ICPT (sen/kWh)", -10.0, 20.0, 0.0, step=0.5,
                                          help="Positive = surcharge, negative = rebate")
        nem_rate       = c3.number_input("NEM buyback rate (RM/kWh)", 0.20, 0.50, 0.31, step=0.01)

    try:
        monthly_stats_opt  = compute_monthly_stats(df_for_bill)
        monthly_stats_orig = compute_monthly_stats(df_orig_bill) if show_comparison else monthly_stats_opt

        bill_rows_opt, bill_rows_orig = [], []
        for _, row in monthly_stats_opt.iterrows():
            bill = calculate_bill(
                tariff,
                monthly_kwh=row["total_kwh"],
                peak_kwh=row["peak_kwh"],
                offpeak_kwh=row["offpeak_kwh"],
                max_demand_kw=row["max_demand_kw"],
                icpt_sen_per_kwh=icpt_sen,
            )
            nem = compute_nem_credit(row["export_kwh"], nem_rate)
            bill_rows_opt.append({
                "Month": str(row["month"]),
                "Total kWh": row["total_kwh"],
                "Peak kW (MD)": round(row["max_demand_kw"], 1),
                "Energy (RM)": bill["energy_charge"],
                "MD Charge (RM)": bill["md_charge"],
                "ICPT (RM)": bill["icpt_charge"],
                "KWTBB (RM)": bill["kwtbb_charge"],
                "Svc Tax (RM)": bill["service_tax"],
                "NEM Credit (RM)": nem["nem_credit_rm"],
                "Net Bill (RM)": round(bill["total_bill"] - nem["nem_credit_rm"], 2),
            })

        if show_comparison:
            for _, row in monthly_stats_orig.iterrows():
                bill = calculate_bill(
                    tariff,
                    monthly_kwh=row["total_kwh"],
                    peak_kwh=row["peak_kwh"],
                    offpeak_kwh=row["offpeak_kwh"],
                    max_demand_kw=row["max_demand_kw"],
                    icpt_sen_per_kwh=icpt_sen,
                )
                nem = compute_nem_credit(row["export_kwh"], nem_rate)
                bill_rows_orig.append({
                    "Month": str(row["month"]),
                    "Net Bill (RM)": round(bill["total_bill"] - nem["nem_credit_rm"], 2),
                })

        opt_df  = pd.DataFrame(bill_rows_opt)
        total_opt  = opt_df["Net Bill (RM)"].sum()

        if show_comparison:
            orig_df   = pd.DataFrame(bill_rows_orig)
            total_orig = orig_df["Net Bill (RM)"].sum()
            savings    = total_orig - total_opt

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Bill (Before)", _fmt_rm(total_orig))
            c2.metric("Total Bill (After)",  _fmt_rm(total_opt), delta=f"-{_fmt_rm(savings)}", delta_color="inverse")
            c3.metric("Annual Savings",       _fmt_rm(savings * 12 / max(len(monthly_stats_opt), 1)))

            # Chart
            if len(opt_df) > 0 and "Month" in opt_df.columns:
                fig = go.Figure()
                if len(orig_df):
                    fig.add_trace(go.Bar(name="Before", x=orig_df["Month"], y=orig_df["Net Bill (RM)"],
                                         marker_color="#d62728"))
                fig.add_trace(go.Bar(name="After",  x=opt_df["Month"],  y=opt_df["Net Bill (RM)"],
                                     marker_color="#1f77b4"))
                fig.update_layout(barmode="group", title="Monthly Bill Comparison",
                                  yaxis_title="RM", height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Total Bill (period)", _fmt_rm(total_opt))
            c2.metric("Avg Monthly",         _fmt_rm(total_opt / max(len(opt_df), 1)))

        st.subheader("Monthly Bill Breakdown (After Optimization)")
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

        # Download
        buf = io.StringIO()
        opt_df.to_csv(buf, index=False)
        st.download_button("Download Bill CSV", buf.getvalue().encode(),
                           "bill_breakdown.csv", "text/csv")

    except Exception as e:
        st.error(f"Bill calculation failed: {e}")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: POWERRECO
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("PowerRECO — Solar & Battery Sizing")
    st.caption(
        "Sizes solar PV and battery storage using Manager optimization results, "
        "then calculates a 25-year financial return."
    )

    with st.expander("Site & system parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        roof_area   = c1.number_input("Roof area (m²)", 50.0, 10000.0, 500.0, step=50.0)
        panel_w     = c2.number_input("Panel wattage (W)", 300, 700, 415, step=5)
        psh         = c3.number_input("Peak sun hours/day", 3.5, 6.0, 4.5, step=0.1,
                                       help="Malaysia average: 4.5 h/day")

        c4, c5, c6 = st.columns(3)
        solar_cost  = c4.number_input("Solar cost (RM/kWp)", 2000.0, 6000.0, 3500.0, step=100.0)
        batt_cost   = c5.number_input("Battery cost (RM/kWh)", 1500.0, 5000.0, 2500.0, step=100.0)
        self_cons   = c6.slider("Self-consumption (%)", 40, 90, 65) / 100.0

        use_new_tariff = st.checkbox("Use new MD tariff (RM 97.06/kW)", value=True,
                                      help="Uncheck to use legacy rate RM 30.30/kW")

    if st.button("Run PowerRECO Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Sizing solar and battery…"):
                solar = calculate_solar_sizing(roof_area, int(panel_w), psh)
                st.session_state.solar_sizing = solar

                # Battery sizing from Manager results if available
                if st.session_state.powerreco_df is not None:
                    batt = calculate_battery_sizing(st.session_state.powerreco_df)
                    md_reduction_kw = batt["md_reduction_kw"]
                else:
                    st.warning(
                        "No Manager results — battery sized conservatively to 10% of daily solar. "
                        "Run the AI Manager for a data-driven battery recommendation."
                    )
                    daily_kwh = solar["daily_generation_kwh_avg"]
                    batt = {
                        "min_capacity_kwh_commercial": max(10.0, round(daily_kwh * 0.10 / 50) * 50),
                        "md_reduction_kw": 0.0,
                        "n_days_analyzed": 0,
                        "spike_note": "Estimated — no Manager data.",
                    }
                    md_reduction_kw = 0.0
                st.session_state.battery_sizing = batt

                battery_kwh_rec = float(batt["min_capacity_kwh_commercial"])

                roi = calculate_roi(
                    solar_kwp=solar["system_kwp"],
                    battery_kwh=battery_kwh_rec,
                    monthly_generation_kwh=solar["monthly_generation_kwh_avg"],
                    md_reduction_kw=float(batt.get("md_reduction_kw", 0.0)),
                    self_consumption_pct=self_cons,
                    use_new_tariff=use_new_tariff,
                    solar_cost_per_kwp=solar_cost,
                    battery_cost_per_kwh=batt_cost,
                )
                st.session_state.roi = roi

            st.success("Analysis complete.")
        except Exception as e:
            st.error(f"PowerRECO failed: {e}")
            st.code(traceback.format_exc())

    if st.session_state.solar_sizing and st.session_state.roi:
        solar = st.session_state.solar_sizing
        batt  = st.session_state.battery_sizing
        roi   = st.session_state.roi

        st.divider()
        st.subheader("Solar System")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("System Size",     f"{solar['system_kwp']} kWp")
        c2.metric("Panels",          f"{solar['n_panels']} × {solar['panel_wattage_w']}W")
        c3.metric("Annual Output",   f"{solar['annual_generation_kwh']:,} kWh")
        c4.metric("Usable Roof",     f"{solar['usable_area_m2']} m²")

        # Monthly generation chart
        fig = go.Figure(go.Bar(
            x=solar["month_labels"],
            y=solar["monthly_breakdown_kwh"],
            marker_color="#ff7f0e",
            name="Monthly kWh",
        ))
        fig.update_layout(title="Monthly Solar Generation (Malaysia seasonal factors)",
                          yaxis_title="kWh", height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Battery Sizing")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended",     f"{batt['min_capacity_kwh_commercial']} kWh")
        c2.metric("MD Reduction",    f"{batt.get('md_reduction_kw', 0):.1f} kW")
        c3.metric("Days Analysed",   batt.get("n_days_analyzed", "N/A"))
        if "spike_note" in batt:
            st.caption(batt["spike_note"])

        st.subheader("25-Year Financial Return")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total CAPEX",     _fmt_rm(roi["total_capex_rm"]))
        c2.metric("Simple Payback",  f"{roi['simple_payback_years']} yrs")
        c3.metric("NPV (25 yr)",     _fmt_rm(roi["npv_25yr_rm"]))
        c4.metric("IRR",             f"{roi['irr_pct']}%" if roi["irr_pct"] else "N/A")

        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Energy Savings", _fmt_rm(roi["annual_energy_savings_rm"]))
        c2.metric("Annual MD Savings",     _fmt_rm(roi["annual_md_savings_rm"]))
        c3.metric("Annual NEM Credit",     _fmt_rm(roi["annual_nem_credit_rm"]))

        # Cumulative NPV chart
        years = list(range(1, len(roi["cumulative_npv"]) + 1))
        fig2 = go.Figure()
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.add_trace(go.Scatter(
            x=years, y=roi["cumulative_npv"],
            mode="lines+markers", name="Cumulative NPV",
            line=dict(color="#2ca02c", width=2),
            fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
        ))
        fig2.update_layout(title="Cumulative NPV over 25 Years",
                           xaxis_title="Year", yaxis_title="RM",
                           height=380)
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            f"CO₂ offset: **{roi['co2_offset_tonnes_yr']:.1f} tonnes/year**  |  "
            f"Self-consumption: {self_cons*100:.0f}%  |  "
            f"MD rate: RM {roi['md_rate_used']}/kW/month"
        )

        # Summary download
        summary = {
            "solar_kwp": solar["system_kwp"],
            "battery_kwh_recommended": batt["min_capacity_kwh_commercial"],
            **{k: v for k, v in roi.items() if k != "cumulative_npv"},
        }
        import json
        st.download_button(
            "Download ROI Summary (JSON)",
            data=json.dumps(summary, indent=2).encode(),
            file_name="powerreco_roi.json",
            mime="application/json",
        )
