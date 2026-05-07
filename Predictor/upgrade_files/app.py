"""
ESUM × RExharge — AI Energy Demand Forecaster (v3)

v3 adds: Live Simulation tab with auto-replay of future data, live
forecast-vs-actual tracking, running accuracy metrics, and periodic
warm-start retrains.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from src.data_loader import auto_load, load_csv, summarize
from src.solar_estimator import estimate_solar_capacity_kwp, detect_has_solar
from src.forecaster import DirectMultiStepForecaster, DEFAULT_HORIZONS
from src.cv import expanding_window_cv, format_cv_report
from src.simulation import (
    initialize_simulation,
    advance_one_tick,
    compute_running_accuracy,
    build_forecast_vs_actual_df,
    split_historical_for_simulation,
)


st.set_page_config(page_title="RExharge Load Forecaster", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.big-metric { font-size: 2.4rem; font-weight: 700; color: #00d4aa; }
.metric-label { color: #999; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.live-indicator {
    display: inline-block;
    width: 12px; height: 12px;
    background: #ff4444;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

defaults = {
    "forecaster": None, "history": None, "last_metrics": None,
    "forecast_result": None, "update_log": [], "cv_report": None,
    "site_name": "Demo Site",
    "sim_state": None, "sim_running": False,
    "sim_tick_interval_s": 1.0, "sim_last_advance_time": 0.0,
    "sim_future_data": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =================================================================== SIDEBAR
with st.sidebar:
    st.title("⚡ Setup")
    st.session_state.site_name = st.text_input("Site name", value=st.session_state.site_name)

    st.divider()
    st.subheader("Solar PV Configuration")
    solar_mode = st.radio(
        "Does this site have solar?",
        ["No solar", "Yes — I know the capacity", "Yes — please estimate it"],
    )
    manual_capacity = 0.0
    if solar_mode == "Yes — I know the capacity":
        manual_capacity = st.number_input("Installed capacity (kWp)",
            min_value=0.0, max_value=5000.0, value=500.0, step=10.0)

    st.divider()
    st.subheader("1. Historical Data")
    st.caption("Tip: use a CSV from `data/synthetic/` for richer training.")
    hist_file = st.file_uploader("Upload load profile",
        type=["xlsx", "xls", "csv"], key="hist_uploader")

    st.divider()
    st.subheader("2. Live Update (one-shot)")
    st.caption("Batch upload newer data. For auto-replay, use Live Simulation tab.")
    live_file = st.file_uploader("Upload new readings",
        type=["xlsx", "xls", "csv"], key="live_uploader")

    st.divider()
    with st.expander("⚙️ Advanced"):
        ws_rounds = st.slider("Warm-start rounds", 20, 200, 50, 10)
        n_estimators = st.slider("Initial training rounds", 100, 500, 250, 50)

# ================================================================== HEADER
st.title(f"{st.session_state.site_name} — Energy Demand Forecaster")
st.caption("Direct multi-step LightGBM with quantile bands + live-adapting online learning.")

col1, col2, col3, col4 = st.columns(4)
fc = st.session_state.forecaster
metrics = st.session_state.last_metrics

with col1:
    st.markdown('<div class="metric-label">Status</div>', unsafe_allow_html=True)
    if fc is None:
        st.markdown('<div class="big-metric">—</div>', unsafe_allow_html=True)
        st.caption("Awaiting training")
    else:
        st.markdown(f'<div class="big-metric">Live</div>', unsafe_allow_html=True)
        st.caption(f"{len(fc.boosters)} models • {fc.n_train_rows:,} samples")

with col2:
    st.markdown('<div class="metric-label">Mean MAPE</div>', unsafe_allow_html=True)
    if metrics:
        st.markdown(f'<div class="big-metric">{metrics["mean_mape"]:.1f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="big-metric">—</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-label">MAPE @ 24h</div>', unsafe_allow_html=True)
    v = metrics.get("mape_at_h24") if metrics else None
    st.markdown(f'<div class="big-metric">{v:.1f}%</div>' if v else '<div class="big-metric">—</div>',
                unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-label">Updates</div>', unsafe_allow_html=True)
    total = len(st.session_state.update_log)
    if st.session_state.sim_state is not None:
        total += len(st.session_state.sim_state.retrain_log)
    st.markdown(f'<div class="big-metric">{total}</div>', unsafe_allow_html=True)

# ================================================================ TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Setup & Train", "📈 Live Forecast",
    "🎬 Live Simulation", "🧪 Cross-Validation", "🔬 Diagnostics"
])

# ======================================================= TAB 1: SETUP
with tab1:
    st.header("Step 1 — Train the initial model")
    if hist_file is not None:
        tmp = Path("data") / f"_upload_{hist_file.name}"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(hist_file.getvalue())
        try:
            df = load_csv(tmp) if tmp.suffix.lower() == ".csv" else auto_load(tmp)
        except Exception as e:
            st.error(f"Could not parse: {e}")
            st.stop()

        s = summarize(df)
        st.success(f"✅ Loaded **{s['rows']:,}** samples ({s['days']} days).")
        cA, cB, cC = st.columns(3)
        cA.metric("Period", f"{s['start'].date()} → {s['end'].date()}")
        cB.metric("Mean kW", f"{s['mean_kw_import']:.0f}")
        cC.metric("Peak kW", f"{s['max_kw_import']:.0f}")

        if solar_mode == "Yes — please estimate it":
            has, why = detect_has_solar(df)
            est = estimate_solar_capacity_kwp(df)
            st.info(f"**Solar:** {'✅' if has else '❌'} {why} • **Estimate:** {est['capacity_kwp']:.0f} kWp")
            capacity = est["capacity_kwp"]
        elif solar_mode == "Yes — I know the capacity":
            capacity = manual_capacity
            st.info(f"Using **{capacity:.0f} kWp**")
        else:
            capacity = 0.0

        st.divider()
        st.subheader("Reserve data for Live Simulation (recommended)")
        reserve_sim = st.checkbox(
            "Split this dataset — train on past, simulate on future",
            value=True,
        )
        train_frac = 0.7
        if reserve_sim:
            train_frac = st.slider("Fraction for training", 0.3, 0.95, 0.7, 0.05)
            pt, pf = split_historical_for_simulation(df, train_fraction=train_frac)
            cA, cB = st.columns(2)
            cA.metric("Training rows", f"{len(pt):,}")
            cB.metric("Simulation rows", f"{len(pf):,}")
            st.caption(
                f"Train cutoff: {pt['timestamp'].max()}  →  "
                f"Sim starts: {pf['timestamp'].min()}"
            )

        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            if reserve_sim:
                training_df, future_df = split_historical_for_simulation(df, train_fraction=train_frac)
                st.session_state.sim_future_data = future_df
            else:
                training_df = df
                st.session_state.sim_future_data = None

            t0 = time.time()
            with st.spinner(f"Training {len(DEFAULT_HORIZONS) * 3} quantile models…"):
                forecaster = DirectMultiStepForecaster(capacity_kwp=capacity, n_estimators=n_estimators)
                m = forecaster.fit(training_df)
                st.session_state.forecaster = forecaster
                st.session_state.history = training_df
                st.session_state.last_metrics = m
                st.session_state.forecast_result = forecaster.forecast(output_steps=48)
                st.session_state.update_log.append({
                    "time": pd.Timestamp.now(), "action": "Initial fit",
                    "added_rows": len(training_df), "mean_mape": m["mean_mape"],
                    "duration_s": round(time.time() - t0, 1),
                })
            st.session_state.sim_state = None
            st.session_state.sim_running = False

            msg = f"✅ Trained {m['n_models_trained']} models in {time.time()-t0:.1f}s. Mean MAPE = {m['mean_mape']:.2f}%."
            if reserve_sim:
                msg += f" {len(future_df):,} rows reserved for Live Simulation."
            st.success(msg)
            st.rerun()
    else:
        st.info("⬅️ Upload a load profile in the sidebar to begin.")

# ======================================================= TAB 2: LIVE FORECAST
with tab2:
    st.header("Step 2 — Forecast & Live Adaptation (batch)")
    if fc is None:
        st.warning("Train a model in Setup first.")
    else:
        if live_file is not None:
            tmp = Path("data") / f"_live_{live_file.name}"
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(live_file.getvalue())
            try:
                new_df = load_csv(tmp) if tmp.suffix.lower() == ".csv" else auto_load(tmp)
            except Exception as e:
                st.error(f"Could not parse: {e}")
                st.stop()

            cutoff = fc.history["timestamp"].max()
            new_only = new_df[new_df["timestamp"] > cutoff]

            if len(new_only) == 0:
                st.warning(f"No new data (latest history: {cutoff}).")
            else:
                t0 = time.time()
                with st.spinner(f"Warm-starting on {len(new_only)} new samples…"):
                    m = fc.update(new_only, n_rounds=ws_rounds)
                    st.session_state.last_metrics = m
                    st.session_state.forecast_result = fc.forecast(output_steps=48)
                    st.session_state.update_log.append({
                        "time": pd.Timestamp.now(), "action": "Warm-start",
                        "added_rows": len(new_only), "mean_mape": m["mean_mape"],
                        "duration_s": round(time.time() - t0, 1),
                    })
                st.success(f"✅ Warm-started in {time.time()-t0:.1f}s. New MAPE = {m['mean_mape']:.2f}%.")

        result = st.session_state.forecast_result
        if result is not None:
            hist = fc.history.tail(48 * 3)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["kw_import"],
                name="Actual", line=dict(color="#888", width=2)))
            fig.add_trace(go.Scatter(
                x=list(result.timestamps) + list(result.timestamps[::-1]),
                y=list(result.p90) + list(result.p10[::-1]),
                fill="toself", fillcolor="rgba(0,212,170,0.15)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="80% confidence band",
            ))
            fig.add_trace(go.Scatter(x=result.timestamps, y=result.median,
                name="Forecast (median)", line=dict(color="#00d4aa", width=3)))
            pi = int(np.argmax(result.median))
            fig.add_trace(go.Scatter(x=[result.timestamps[pi]], y=[result.median[pi]],
                mode="markers+text", marker=dict(color="#ff6b6b", size=14, symbol="star"),
                text=[f"PEAK<br>{result.median[pi]:.0f} kW"], textposition="top center",
                name="Predicted Peak"))
            fig.update_layout(title=f"24-hour Forecast — {st.session_state.site_name}",
                xaxis_title="Time", yaxis_title="kW Import",
                template="plotly_dark", height=540, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)

            peaks = fc.detect_peaks(result, top_n=3)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("🔥 Top 3 Predicted Peaks")
                pp = peaks.copy()
                pp["timestamp"] = pp["timestamp"].dt.strftime("%a %d %b, %H:%M")
                for col in ["predicted_kw", "lower_bound_kw", "upper_bound_kw"]:
                    pp[col] = pp[col].round(0).astype(int)
                st.dataframe(pp, use_container_width=True, hide_index=True)
            with c2:
                mp = float(result.median.max())
                st.metric("Forecast Peak", f"{mp:.0f} kW")
                st.metric("Est. Monthly MD Charge", f"RM {mp * 97.06:,.0f}",
                    delta=f"+RM {mp * (97.06 - 30.30):,.0f} vs old tariff",
                    delta_color="inverse")

# ======================================================= TAB 3: LIVE SIMULATION
with tab3:
    st.header("🎬 Live Data Simulation")
    st.caption(
        "Auto-replays 'future' readings one at a time, comparing forecasts against actuals "
        "live, with periodic warm-start retrains."
    )

    if fc is None:
        st.warning("Train a model in Setup first.")
    else:
        sim_state = st.session_state.sim_state

        # ---------------- CONFIGURATION PANEL ----------------
        if sim_state is None:
            st.subheader("Configure Simulation")

            # Always show the uploader so the user can supply their own data
            # at any time, regardless of whether data was reserved at training.
            sim_upload = st.file_uploader(
                "Upload simulation data (optional — overrides reserved split)",
                type=["xlsx", "xls", "csv"], key="sim_upload",
            )

            future_df = st.session_state.sim_future_data

            if sim_upload is not None:
                tmp = Path("data") / f"_sim_{sim_upload.name}"
                tmp.parent.mkdir(exist_ok=True)
                tmp.write_bytes(sim_upload.getvalue())
                try:
                    fd = load_csv(tmp) if tmp.suffix.lower() == ".csv" else auto_load(tmp)
                    cutoff = fc.history["timestamp"].max()
                    fd_new = fd[fd["timestamp"] > cutoff].reset_index(drop=True)
                    if len(fd_new) == 0:
                        st.error(f"No rows after training cutoff ({cutoff}). "
                                 "Upload data that continues from where training ended.")
                    else:
                        future_df = fd_new
                        st.success(f"✅ Loaded **{len(future_df):,}** simulation rows from upload "
                                   f"({future_df['timestamp'].min().date()} → "
                                   f"{future_df['timestamp'].max().date()}).")
                except Exception as e:
                    st.error(f"Could not parse uploaded file: {e}")
            elif future_df is not None:
                st.success(f"✅ Using **{len(future_df):,}** rows reserved during training.")
                st.caption(f"Simulation period: "
                           f"{future_df['timestamp'].min()} → {future_df['timestamp'].max()}")
            else:
                st.info("Upload a file above, or go back to Setup and tick "
                        "'Split this dataset' when training.")

            if future_df is not None and len(future_df) > 0:
                c1, c2, c3 = st.columns(3)
                with c1:
                    sim_speed = st.select_slider(
                        "Playback speed",
                        options=["0.3s (fast)", "0.6s", "1.0s (normal)", "2.0s", "3.0s (slow)"],
                        value="1.0s (normal)",
                    )
                    sim_interval = float(sim_speed.split("s")[0])
                with c2:
                    retrain_every = st.number_input("Retrain every N readings",
                        min_value=4, max_value=96, value=12, step=4,
                        help="12 = every 6 hours of simulated time")
                with c3:
                    sim_ws_rounds = st.number_input("Warm-start rounds",
                        min_value=10, max_value=200, value=30, step=10,
                        help="Lower = faster retrains. 30 is a good balance.")

                max_ticks = st.slider("Max simulation length (readings)",
                    min_value=48, max_value=min(len(future_df), 2016),
                    value=min(len(future_df), 336),  # default 1 week
                    step=48,
                    help="48 = 1 day, 336 = 1 week, 2016 = 6 weeks")

                if st.button("▶️ Initialize Simulation", type="primary", use_container_width=True):
                    subset = future_df.iloc[:max_ticks].copy()
                    st.session_state.sim_state = initialize_simulation(
                        forecaster=fc, future_data=subset,
                        retrain_every_n=int(retrain_every),
                        tick_interval_s=sim_interval,
                        warm_start_rounds=int(sim_ws_rounds),
                    )
                    st.session_state.sim_tick_interval_s = sim_interval
                    st.session_state.sim_running = False
                    st.session_state.sim_last_advance_time = 0.0
                    st.rerun()

        # ---------------- RUNNING SIMULATION ----------------
        else:
            is_finished = sim_state.is_finished
            is_running = st.session_state.sim_running

            # Control bar
            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 3])
            with ctrl1:
                play_label = "⏸️ Pause" if is_running else "▶️ Play"
                if st.button(play_label, use_container_width=True, disabled=is_finished):
                    st.session_state.sim_running = not is_running
                    st.session_state.sim_last_advance_time = time.time()
                    st.rerun()
            with ctrl2:
                if st.button("⏭️ Step", use_container_width=True,
                             disabled=(is_running or is_finished)):
                    advance_one_tick(fc, sim_state)
                    st.rerun()
            with ctrl3:
                if st.button("🔁 Reset", use_container_width=True):
                    st.session_state.sim_state = None
                    st.session_state.sim_running = False
                    st.rerun()
            with ctrl4:
                if is_finished:
                    st.markdown("✅ **Simulation finished**")
                elif is_running:
                    st.markdown('<span class="live-indicator"></span>**LIVE**',
                        unsafe_allow_html=True)
                else:
                    st.markdown("⏸️ Paused")

            # Progress
            st.progress(sim_state.progress,
                text=f"Tick {sim_state.tick} / {sim_state.total_ticks}  "
                     f"({sim_state.progress*100:.1f}%)")

            # Live metrics
            acc = compute_running_accuracy(sim_state)
            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                st.markdown('<div class="metric-label">Simulated time</div>', unsafe_allow_html=True)
                revealed = sim_state.revealed_data()
                if len(revealed) > 0:
                    cur_ts = revealed["timestamp"].iloc[-1]
                    st.markdown(f'<div class="big-metric" style="font-size: 1.4rem;">'
                                f'{cur_ts.strftime("%a %H:%M")}</div>', unsafe_allow_html=True)
                    st.caption(cur_ts.strftime("%d %b %Y"))
                else:
                    st.markdown('<div class="big-metric">—</div>', unsafe_allow_html=True)

            with m2:
                st.markdown('<div class="metric-label">Live MAPE</div>', unsafe_allow_html=True)
                if acc["overall_mape"] is not None:
                    st.markdown(f'<div class="big-metric">{acc["overall_mape"]:.1f}%</div>',
                        unsafe_allow_html=True)
                    st.caption(f"{acc['n_verified']} verified")
                else:
                    st.markdown('<div class="big-metric">—</div>', unsafe_allow_html=True)

            with m3:
                st.markdown('<div class="metric-label">Live MAE</div>', unsafe_allow_html=True)
                if acc["overall_mae"] is not None:
                    st.markdown(f'<div class="big-metric">{acc["overall_mae"]:.0f} kW</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown('<div class="big-metric">—</div>', unsafe_allow_html=True)

            with m4:
                st.markdown('<div class="metric-label">In 80% CI</div>', unsafe_allow_html=True)
                if acc["within_80ci_pct"] is not None:
                    color = "#00d4aa" if acc["within_80ci_pct"] >= 70 else "#ff9a3c"
                    st.markdown(f'<div class="big-metric" style="color: {color};">'
                                f'{acc["within_80ci_pct"]:.0f}%</div>', unsafe_allow_html=True)
                    st.caption("target: ~80%")
                else:
                    st.markdown('<div class="big-metric">—</div>', unsafe_allow_html=True)

            with m5:
                st.markdown('<div class="metric-label">Retrains</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="big-metric">{len(sim_state.retrain_log)}</div>',
                    unsafe_allow_html=True)
                st.caption(f"every {sim_state.retrain_every_n} ticks")

            # ---------------- MAIN CHART ----------------
            st.subheader("📈 Forecast vs Actual (live)")
            revealed = sim_state.revealed_data()
            cur_fc   = sim_state.current_forecast

            # 24-hour sliding window: last 48 revealed readings + 48-step forecast
            window_actual = revealed.tail(48)  # last 24 h of actuals

            # x-axis bounds: 24 h back from now, 24 h forward
            if len(window_actual) > 0:
                x_now   = window_actual["timestamp"].iloc[-1]
                x_start = x_now - pd.Timedelta(hours=24)
                x_end   = x_now + pd.Timedelta(hours=24)
            elif cur_fc is not None:
                x_now   = cur_fc.timestamps[0]
                x_start = x_now - pd.Timedelta(hours=24)
                x_end   = x_now + pd.Timedelta(hours=24)
            else:
                x_start = x_end = None

            fig = go.Figure()

            # A few hours of training context before the simulation window
            sim_first_ts = sim_state.future_data["timestamp"].iloc[0]
            if x_start is not None:
                train_tail = fc.history[
                    (fc.history["timestamp"] >= x_start) &
                    (fc.history["timestamp"] < sim_first_ts)
                ]
            else:
                train_tail = fc.history[fc.history["timestamp"] < sim_first_ts].tail(48)

            if len(train_tail) > 0:
                fig.add_trace(go.Scatter(
                    x=train_tail["timestamp"], y=train_tail["kw_import"],
                    name="Training history",
                    line=dict(color="#555", width=1.5),
                ))

            # Revealed actuals (24-h window)
            if len(window_actual) > 0:
                fig.add_trace(go.Scatter(
                    x=window_actual["timestamp"], y=window_actual["kw_import"],
                    name="Actual (live)",
                    line=dict(color="#00d4aa", width=3),
                    mode="lines+markers", marker=dict(size=5),
                ))

            # Current 24-h forecast
            if cur_fc is not None:
                fig.add_trace(go.Scatter(
                    x=list(cur_fc.timestamps) + list(cur_fc.timestamps[::-1]),
                    y=list(cur_fc.p90) + list(cur_fc.p10[::-1]),
                    fill="toself", fillcolor="rgba(255,154,60,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                    name="80% CI (forecast)",
                ))
                fig.add_trace(go.Scatter(
                    x=cur_fc.timestamps, y=cur_fc.median,
                    name="Forecast (median)",
                    line=dict(color="#ff9a3c", width=2.5, dash="dot"),
                ))

            # Retrain markers — add_vline with annotation_text is broken in
            # newer plotly (its internal _mean calls sum([x]) on non-numeric x).
            # Use add_shape + add_annotation separately to bypass that code path.
            shapes      = []
            annotations = []
            for rt in sim_state.retrain_log:
                rt_ts = pd.Timestamp(rt["reveal_ts"])
                # Only draw markers within the visible window
                if x_start is not None and not (x_start <= rt_ts <= x_end):
                    continue
                rt_str = rt_ts.isoformat()
                shapes.append(dict(
                    type="line",
                    x0=rt_str, x1=rt_str, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
                ))
                annotations.append(dict(
                    x=rt_str, y=1, xref="x", yref="paper",
                    text="🔄", showarrow=False,
                    font=dict(size=11), yanchor="bottom",
                ))

            fig.update_layout(
                template="plotly_dark", height=480,
                xaxis_title="Time", yaxis_title="kW Import",
                hovermode="x unified",
                xaxis=dict(range=[x_start, x_end] if x_start else None),
                shapes=shapes,
                annotations=annotations,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"sim_main_{sim_state.tick}")

            # ---------------- SECONDARY CHARTS ----------------
            fva = build_forecast_vs_actual_df(sim_state)
            verified = fva[fva["actual"].notna()]

            if len(verified) > 0:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📊 Forecast vs Actual @ verified timestamps")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=verified["target_ts"], y=verified["median"],
                        name="Forecast", line=dict(color="#ff9a3c", width=2, dash="dot")))
                    fig2.add_trace(go.Scatter(x=verified["target_ts"], y=verified["actual"],
                        name="Actual", line=dict(color="#00d4aa", width=2)))
                    fig2.update_layout(template="plotly_dark", height=320,
                        xaxis_title="Time", yaxis_title="kW", hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    st.plotly_chart(fig2, use_container_width=True, key=f"sim_vs_{sim_state.tick}")

                with c2:
                    st.subheader("📉 MAPE improvement (retrain events)")
                    if len(sim_state.retrain_log) > 0:
                        log = pd.DataFrame(sim_state.retrain_log)
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(
                            x=log["reveal_ts"], y=log["mean_mape"],
                            mode="lines+markers+text",
                            text=[f"{v:.1f}%" for v in log["mean_mape"]],
                            textposition="top center",
                            line=dict(color="#00d4aa"), marker=dict(size=10)))
                        fig3.update_layout(template="plotly_dark", height=320,
                            xaxis_title="Retrain event", yaxis_title="MAPE (%)",
                            showlegend=False)
                        st.plotly_chart(fig3, use_container_width=True,
                            key=f"sim_mape_{sim_state.tick}")
                    else:
                        st.info("Waiting for first retrain event…")

            # ---------------- AUTO-ADVANCE ----------------
            # Streamlit doesn't have background loops, so we use this pattern:
            # if playing, sleep for tick_interval, advance once, rerun.
            # Each rerun re-executes the whole script, landing here and
            # advancing again.
            if st.session_state.sim_running and not sim_state.is_finished:
                now = time.time()
                elapsed = now - st.session_state.sim_last_advance_time
                wait = max(0.0, sim_state.tick_interval_s - elapsed)
                if wait > 0:
                    time.sleep(wait)
                advance_one_tick(fc, sim_state)
                st.session_state.sim_last_advance_time = time.time()
                st.rerun()

# ======================================================= TAB 4: CV
with tab4:
    st.header("Time-Series Cross-Validation")
    if fc is None or st.session_state.history is None:
        st.warning("Train a model first.")
    else:
        cA, cB, cC = st.columns(3)
        n_splits = cA.number_input("CV folds", 2, 8, 4)
        min_train_days = cB.number_input("Min training days", 14, 90, 21)
        val_block = cC.number_input("Val block (days)", 3, 14, 5)

        if st.button("🧪 Run CV", type="primary"):
            t0 = time.time()
            with st.spinner(f"Running {n_splits}-fold CV…"):
                report = expanding_window_cv(
                    st.session_state.history,
                    forecaster_factory=lambda: DirectMultiStepForecaster(
                        capacity_kwp=fc.capacity_kwp, n_estimators=150),
                    n_splits=int(n_splits),
                    min_train_days=int(min_train_days),
                    val_block_days=int(val_block),
                    forecast_origins_per_fold=5,
                )
                st.session_state.cv_report = report
            st.success(f"✅ CV done in {time.time()-t0:.1f}s")

        report = st.session_state.cv_report
        if report:
            st.subheader("Aggregate")
            rows = []
            for h, m in sorted(report["aggregate"].items()):
                hl = f"+{h*30}min" if h*30 < 60 else f"+{h/2:.1f}h"
                rows.append({"Horizon": hl, "Mean MAPE": f"{m['mean_mape']:.2f}%",
                    "± Std": f"{m['std_mape']:.2f}%", "Mean MAE": f"{m['mean_mae']:.1f} kW",
                    "n folds": m["n_folds"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with st.expander("📋 Plain-text report"):
                st.code(format_cv_report(report))

# ======================================================= TAB 5: DIAGNOSTICS
with tab5:
    st.header("Model Diagnostics")
    if fc is None or metrics is None:
        st.warning("Train a model first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Per-Horizon Accuracy")
            ph = metrics.get("per_horizon", {})
            rows = []
            for h, m in sorted(ph.items()):
                hl = f"+{h*30}min" if h*30 < 60 else f"+{h/2:.1f}h"
                rows.append({"Horizon": hl, "MAE (kW)": f"{m['mae']:.1f}",
                    "MAPE (%)": f"{m['mape']:.2f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Update History")
            if st.session_state.update_log:
                ldf = pd.DataFrame(st.session_state.update_log)
                ldf["time"] = ldf["time"].dt.strftime("%H:%M:%S")
                st.dataframe(ldf, use_container_width=True, hide_index=True)

        st.subheader("Save / Load Model")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save"):
                Path("models").mkdir(exist_ok=True)
                p = Path("models") / f"{st.session_state.site_name.replace(' ', '_')}.joblib"
                fc.save(p)
                st.success(f"Saved to {p}")
        with c2:
            saved = list(Path("models").glob("*.joblib")) if Path("models").exists() else []
            if saved:
                pick = st.selectbox("Load", [str(p) for p in saved])
                if st.button("📂 Load"):
                    st.session_state.forecaster = DirectMultiStepForecaster.load(pick)
                    st.session_state.forecast_result = st.session_state.forecaster.forecast(48)
                    st.success(f"Loaded {pick}")
                    st.rerun()
