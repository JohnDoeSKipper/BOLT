"""
PowerRECO — AI Solar & Battery Sizing Optimizer
Battery-first (mandatory), Solar optional.
ACCEPT / CONDITIONAL / REJECT recommendation based on user ROI and IRR thresholds.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Helpers ───────────────────────────────────────────────────────────────────
def _kpi(col, label: str, value: str, caption: str, style: str = "big") -> None:
    col.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
    col.markdown(f'<div class="{style}-metric">{value}</div>', unsafe_allow_html=True)
    col.caption(caption)


def _manual_battery_result(daily_kwh: float, md_kw: float, note: str = "") -> dict:
    from src.battery_sizing import ROUNDTRIP_EFF, UPSIZE_FACTOR, _round_to_commercial
    usable = daily_kwh / ROUNDTRIP_EFF
    cap    = usable * UPSIZE_FACTOR
    return {
        "n_days_analyzed":            0,
        "peak_daily_discharge_kwh":   round(daily_kwh, 2),
        "n_top_days_used":            0,
        "avg_top_discharge_kwh":      round(daily_kwh, 2),
        "roundtrip_eff_pct":          ROUNDTRIP_EFF * 100,
        "required_usable_kwh":        round(usable, 2),
        "usable_dod_pct":             60.0,
        "soc_range":                  "20%–80%",
        "upsize_factor":              round(UPSIZE_FACTOR, 4),
        "min_capacity_kwh":           round(cap, 2),
        "min_capacity_kwh_commercial": _round_to_commercial(cap),
        "md_reduction_kw":            round(md_kw, 1),
        "discharge_col_used":         "manual",
        "spike_note":                 note or "Manual entry — no AI analysis performed.",
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
.reco-box { background:rgba(0,212,255,.08); border:1px solid rgba(0,212,255,.35);
            border-radius:10px; padding:14px 18px; margin:8px 0; }
.good-box { background:rgba(34,197,94,.08);  border:1px solid rgba(34,197,94,.35);
            border-radius:10px; padding:14px 18px; margin:8px 0; }
.warn-box { background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.35);
            border-radius:10px; padding:14px 18px; margin:8px 0; }
.bad-box  { background:rgba(239,68,68,.08);  border:1px solid rgba(239,68,68,.35);
            border-radius:10px; padding:14px 18px; margin:8px 0; }
.method-step { background:rgba(255,255,255,.04); border-radius:8px;
               padding:10px 14px; margin:4px 0; font-size:.9rem; }
.pass-badge { background:#22c55e; color:#000; font-weight:700;
              padding:2px 10px; border-radius:12px; font-size:.82rem; }
.fail-badge { background:#ef4444; color:#fff; font-weight:700;
              padding:2px 10px; border-radius:12px; font-size:.82rem; }
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
    include_solar_run=True,
    peak_demand_kw_run=500.0,
    desired_md_pct_run=15.0,
    required_irr_run=12.0,
    required_roi_run=200.0,
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
        "TNB NEM 3.0 rates  ·  25-year ROI  ·  ACCEPT / CONDITIONAL / REJECT recommendation"
    )
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    # ── 1. Manager Data (Optional — separate section) ─────────────────────────
    st.subheader("1 · Manager Data (Optional)")
    st.caption("Upload BOLT outputs for AI-powered battery sizing and demand forecasting.")

    manager_file = st.file_uploader(
        "Manager Results (CSV / JSON / Excel)",
        type=["csv", "json", "xlsx", "xls"],
        key="mgr_upload",
        help="Optimisation results from the BOLT AI Manager (Peak Load Optimizer).",
    )
    predictor_file = st.file_uploader(
        "Predictor Forecast (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="pred_upload",
        help="Load forecast from the BOLT AI Energy Demand Forecaster.",
    )

    if manager_file:
        try:
            st.session_state.manager_df = parse_manager_output(manager_file)
            st.success(f"✅ Manager: {len(st.session_state.manager_df):,} intervals")
        except Exception as exc:
            st.error(f"Manager parse error: {exc}")
            st.session_state.manager_df = None

    if predictor_file:
        try:
            st.session_state.predictor_df = parse_predictor_output(predictor_file)
            st.session_state.peak_stats   = extract_peak_demand_stats(st.session_state.predictor_df)
            st.success(f"✅ Predictor: {len(st.session_state.predictor_df):,} rows")
        except Exception as exc:
            st.error(f"Predictor parse error: {exc}")
            st.session_state.predictor_df = None

    st.divider()

    # ── 2. Battery Configuration (Mandatory) ──────────────────────────────────
    st.subheader("2 · Battery Configuration")
    st.caption("Battery sizing is mandatory. Manager data enables AI sizing; otherwise the MD target is used.")

    _ps = st.session_state.peak_stats or {}
    _default_peak = float(_ps.get("peak_kw", 500.0))

    peak_demand_kw = st.number_input(
        "Current Peak Demand (kW)",
        min_value=1.0, max_value=100_000.0, value=_default_peak, step=10.0,
        help="Your facility's highest monthly recorded demand (kW). "
             "Used to compute the MD reduction target.",
    )
    desired_md_pct = st.slider(
        "Desired MD Reduction (%)",
        min_value=1, max_value=50, value=15, step=1,
        help="Percentage of peak demand to shave via the battery. Default 15%.",
    )
    md_target_kw = peak_demand_kw * desired_md_pct / 100
    st.info(
        f"MD target: **{md_target_kw:.1f} kW**  "
        f"({desired_md_pct}% × {peak_demand_kw:.0f} kW)"
    )

    if st.session_state.manager_df is not None:
        st.caption(
            "AI sizing from Manager data is active. "
            "MD target above is used as the performance validation threshold."
        )
    else:
        st.caption(
            "No Manager data — battery will be sized for the MD target "
            "over a 1-hour discharge window."
        )

    st.divider()

    # ── 3. Solar Configuration (Optional) ─────────────────────────────────────
    st.subheader("3 · Solar Configuration (Optional)")

    include_solar = st.toggle(
        "Include Solar PV System",
        value=True,
        help="Toggle off for a battery-only analysis with no solar generation.",
    )

    # Always-defined defaults so later code never raises NameError
    roof_area     = 200.0
    panel_area_m2 = 2.0
    panel_wattage = 415
    psh           = 4.5
    solar_cost    = 3_500

    if include_solar:
        roof_area = st.number_input(
            "Roof Area (m²)",
            min_value=10.0, max_value=100_000.0, value=200.0, step=10.0,
            help="Total roof area available. PowerRECO uses 85% for panels.",
        )
        panel_area_m2 = st.number_input(
            "Panel Size (m²)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.05,
            help="Physical surface area of one solar panel in m². Standard monocrystalline panels: 1.6–2.2 m².",
        )
        panel_wattage = st.number_input(
            "Panel Rated Power (W)",
            min_value=100, max_value=700, value=415, step=5,
            help="Rated output per panel in watts. Common: 415 W, 450 W, 500 W, 550 W.",
        )
        psh = st.slider(
            "Peak Sun Hours / day",
            min_value=3.5, max_value=5.5, value=4.5, step=0.25,
            help="Malaysia peninsular average 4.5 PSH/day. Step = 0.25 hr (15 min).",
        )

    st.divider()

    # ── 4. Financial Parameters ───────────────────────────────────────────────
    st.subheader("4 · Financial Parameters")

    use_new_tariff = st.toggle(
        "New MD Tariff (RM 97.06 / kW)",
        value=True,
        help="Toggle off to use the legacy tariff of RM 30.30/kW.",
    )
    _md_display = MD_CHARGE_NEW if use_new_tariff else MD_CHARGE_OLD
    st.caption(
        f"MD: **RM {_md_display:.2f}/kW** · "
        f"Energy: **RM {ENERGY_RATE_C1}/kWh** · "
        f"NEM: **RM {NEM_BUYBACK}/kWh**"
    )

    self_consumption_pct = 0.65   # irrelevant when solar is off (generation = 0)
    if include_solar:
        self_consumption_pct = st.slider(
            "Solar Self-Consumption %",
            min_value=30, max_value=95, value=65, step=5,
            help="% of solar used on-site; remainder exported under NEM 3.0.",
        ) / 100

    required_irr = st.number_input(
        "Required IRR (%)",
        min_value=1.0, max_value=50.0, value=12.0, step=0.5,
        help="Your minimum acceptable Internal Rate of Return (hurdle rate). Default 12%.",
    )
    required_roi = st.number_input(
        "Required ROI (%)",
        min_value=50.0, max_value=1000.0, value=200.0, step=25.0,
        help="Minimum 25-year ROI threshold (net cashflows ÷ CAPEX × 100%). Default 200%.",
    )

    batt_cost = 2_500   # default; always overwritten by the widget below
    with st.expander("Advanced Cost Settings"):
        if include_solar:
            solar_cost = st.number_input(
                "Solar Installed Cost (RM / kWp)",
                min_value=1_000, max_value=10_000, value=3_500, step=100,
            )
        batt_cost = st.number_input(
            "Battery Installed Cost (RM / kWh)",
            min_value=500, max_value=8_000, value=2_500, step=100,
        )

    st.divider()
    run_btn = st.button("🚀 Analyse & Recommend", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Running PowerRECO analysis…"):

        # ── Solar (optional) ──────────────────────────────────────────────────
        if include_solar:
            sr = calculate_solar_sizing(
                roof_area_m2=roof_area,
                panel_area_m2=float(panel_area_m2),
                panel_wattage_w=int(panel_wattage),
                psh=psh,
            )
            solar_kwp   = sr["system_kwp"]
            monthly_gen = sr["monthly_generation_kwh_avg"]
        else:
            sr          = None
            solar_kwp   = 0.0
            monthly_gen = 0.0

        # ── Battery (mandatory) ───────────────────────────────────────────────
        _md_kw_target = peak_demand_kw * desired_md_pct / 100
        _disch_kwh    = _md_kw_target * 1.0   # 1-hour discharge window assumption

        if st.session_state.manager_df is not None:
            try:
                br = calculate_battery_sizing(st.session_state.manager_df)
            except Exception as exc:
                st.warning(
                    f"AI battery sizing failed ({exc}). Falling back to MD target."
                )
                br = _manual_battery_result(
                    _disch_kwh, _md_kw_target,
                    note=(
                        f"Fallback: {desired_md_pct}% MD reduction "
                        f"→ {_md_kw_target:.1f} kW over 1-hr discharge window."
                    ),
                )
        else:
            br = _manual_battery_result(
                _disch_kwh, _md_kw_target,
                note=(
                    f"Manual: {desired_md_pct}% of {peak_demand_kw:.0f} kW "
                    f"= {_md_kw_target:.1f} kW MD target, 1-hr discharge window."
                ),
            )

        # ── ROI ───────────────────────────────────────────────────────────────
        rr = calculate_roi(
            solar_kwp=solar_kwp,
            battery_kwh=br["min_capacity_kwh_commercial"],
            monthly_generation_kwh=monthly_gen,
            md_reduction_kw=br["md_reduction_kw"],
            self_consumption_pct=self_consumption_pct,
            use_new_tariff=use_new_tariff,
            solar_cost_per_kwp=float(solar_cost),
            battery_cost_per_kwh=float(batt_cost),
        )

        # ── Persist ───────────────────────────────────────────────────────────
        st.session_state.solar_result        = sr
        st.session_state.battery_result      = br
        st.session_state.roi_result          = rr
        st.session_state.include_solar_run   = include_solar
        st.session_state.peak_demand_kw_run  = peak_demand_kw
        st.session_state.desired_md_pct_run  = desired_md_pct
        st.session_state.required_irr_run    = required_irr
        st.session_state.required_roi_run    = required_roi
        st.session_state.run_done            = True

    st.success("✅ Analysis complete — see the Recommendation tab below.")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.run_done and st.session_state.roi_result:
    sr       = st.session_state.solar_result
    br       = st.session_state.battery_result
    rr       = st.session_state.roi_result
    _solar   = st.session_state.include_solar_run
    _peak_kw = st.session_state.peak_demand_kw_run
    _md_pct  = st.session_state.desired_md_pct_run
    _req_irr = st.session_state.required_irr_run
    _req_roi = st.session_state.required_roi_run

    # ── Recommendation logic ──────────────────────────────────────────────────
    _roi_pct  = rr.get("roi_pct")
    _irr_pct  = rr.get("irr_pct")
    _roi_pass = _roi_pct is not None and _roi_pct >= _req_roi
    _irr_pass = _irr_pct is not None and _irr_pct >= _req_irr

    if _roi_pass and _irr_pass:
        _verdict = "ACCEPT"
    elif _roi_pass or _irr_pass:
        _verdict = "CONDITIONAL"
    else:
        _verdict = "REJECT"

    # ── Build tab list (Solar tab only when solar was included) ───────────────
    _tab_labels = ["⭐ Recommendation"]
    _has_solar_tab = _solar and sr is not None
    if _has_solar_tab:
        _tab_labels.append("☀️ Solar Sizing")
    _tab_labels += ["🔋 Battery Sizing", "💰 ROI & Financials", "📊 Data Overview"]

    _tabs = st.tabs(_tab_labels)
    _ti = 0
    tab_reco  = _tabs[_ti]; _ti += 1
    tab_solar = _tabs[_ti] if _has_solar_tab else None
    if _has_solar_tab:
        _ti += 1
    tab_batt  = _tabs[_ti]; _ti += 1
    tab_roi   = _tabs[_ti]; _ti += 1
    tab_data  = _tabs[_ti]

    # ═════════════════════════════════════════════════════════════════════════
    # TAB — RECOMMENDATION
    # ═════════════════════════════════════════════════════════════════════════
    with tab_reco:
        st.header("⭐ Investment Recommendation")

        # Verdict colours and copy
        if _verdict == "ACCEPT":
            _badge_color = "#22c55e"
            _badge_icon  = "✅"
            _badge_text  = "ACCEPT"
            _badge_sub   = "Both ROI and IRR targets are met. Investment is strongly recommended."
        elif _verdict == "CONDITIONAL":
            _badge_color = "#f59e0b"
            _badge_icon  = "⚠️"
            _badge_text  = "CONDITIONAL"
            _badge_sub   = "One target is met. Review the details below before deciding to proceed."
        else:
            _badge_color = "#ef4444"
            _badge_icon  = "❌"
            _badge_text  = "REJECT"
            _badge_sub   = "Neither target is met. Investment is not recommended at current parameters."

        st.markdown(
            f"<div style='text-align:center;padding:24px 0 8px;'>"
            f"<div style='font-size:3rem;'>{_badge_icon}</div>"
            f"<div style='font-size:2.6rem;font-weight:800;color:{_badge_color};'>{_badge_text}</div>"
            f"<div style='color:#aaa;font-size:1rem;margin-top:6px;'>{_badge_sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # KPI row — four columns: calculated ROI | required ROI | calculated IRR | required IRR
        _roi_str = f"{_roi_pct:.1f}%" if _roi_pct is not None else "N/A"
        _irr_str = f"{_irr_pct:.1f}%" if _irr_pct is not None else "N/A"
        rc1, rc2, rc3, rc4 = st.columns(4)
        _kpi(rc1, "Calculated ROI",  _roi_str,
             "25-yr cashflows ÷ CAPEX", "green" if _roi_pass else "red")
        _kpi(rc2, "Required ROI",    f"≥ {_req_roi:.0f}%",
             "your ROI threshold", "big")
        _kpi(rc3, "Calculated IRR",  _irr_str,
             "internal rate of return", "green" if _irr_pass else "red")
        _kpi(rc4, "Required IRR",    f"≥ {_req_irr:.1f}%",
             "your hurdle rate", "big")

        # Pass / Fail badges directly under the KPIs
        pb1, pb2, pb3, pb4 = st.columns(4)
        _pass_html = '<span class="pass-badge">PASS</span>'
        _fail_html = '<span class="fail-badge">FAIL</span>'
        pb1.markdown(_pass_html if _roi_pass else _fail_html, unsafe_allow_html=True)
        pb3.markdown(_pass_html if _irr_pass else _fail_html, unsafe_allow_html=True)

        st.divider()

        # ── Detailed explanation ───────────────────────────────────────────────
        if _verdict == "ACCEPT":
            st.markdown(f"""
<div class="good-box">
<b>Recommendation: Proceed with Investment</b><br><br>
The system achieves a <b>{_roi_str} ROI</b> over 25 years (threshold: ≥{_req_roi:.0f}%)
and an <b>IRR of {_irr_str}</b> (required: ≥{_req_irr:.1f}%).<br><br>
Total CAPEX of <b>RM {rr['total_capex_rm']:,}</b> is expected to generate
<b>RM {rr['total_25yr_cashflows']:,}</b> in net savings over 25 years.
Simple payback: <b>{rr['simple_payback_years']} years</b>.
NPV @ 6%: <b>RM {rr['npv_25yr_rm']:,}</b>.
</div>
""", unsafe_allow_html=True)

        elif _verdict == "CONDITIONAL":
            if _roi_pass and not _irr_pass:
                _explain = (
                    f"The 25-year cumulative return is strong ({_roi_str}), but the annual rate of "
                    f"return ({_irr_str}) does not reach your {_req_irr:.1f}% hurdle rate. "
                    f"This is common when the simple payback period is long relative to the hurdle rate. "
                    f"Consider: increasing the MD reduction target, adding solar (if not included), "
                    f"or reviewing your IRR requirement."
                )
                _passing, _failing = "ROI", "IRR"
            else:
                _explain = (
                    f"The annual rate of return ({_irr_str}) exceeds your hurdle rate, but the "
                    f"cumulative 25-year return ({_roi_str}) does not yet reach {_req_roi:.0f}%. "
                    f"This typically means the payback is fast but long-tail benefits are limited. "
                    f"Consider: adding solar (if not included), increasing the battery scope, "
                    f"or using a higher MD reduction target."
                )
                _passing, _failing = "IRR", "ROI"

            st.markdown(f"""
<div class="warn-box">
<b>Recommendation: Review Before Proceeding</b><br><br>
<b>{_passing}</b> target is met, but the <b>{_failing}</b> target is not.<br><br>
{_explain}<br><br>
<b>You may choose to proceed</b> — the investment is partially viable.
Review the ROI &amp; Financials tab for the full 25-year breakdown.
</div>
""", unsafe_allow_html=True)

        else:   # REJECT
            _reasons = []
            if not _roi_pass:
                _reasons.append(
                    f"ROI is {_roi_str} (need ≥{_req_roi:.0f}%). "
                    f"25-year net cashflows of RM {rr['total_25yr_cashflows']:,} "
                    f"do not meet the {_req_roi:.0f}% ROI threshold on CAPEX of RM {rr['total_capex_rm']:,}."
                )
            if not _irr_pass:
                _reasons.append(
                    f"IRR is {_irr_str} (need ≥{_req_irr:.1f}%). "
                    f"The annual rate of return does not meet your hurdle rate."
                )
            _suggestions = []
            if not _solar:
                _suggestions.append("Enable Solar PV — it adds energy savings and NEM export income at low marginal cost.")
            _suggestions.append(f"Increase the Desired MD Reduction % (currently {_md_pct}%) to capture more demand savings.")
            _suggestions.append("Negotiate a lower battery installed cost (RM/kWh).")
            _suggestions.append("Review your Required IRR — 12% is aggressive for infrastructure investments.")

            _reason_html    = "<br>".join(f"• {r}" for r in _reasons)
            _suggest_html   = "<br>".join(f"• {s}" for s in _suggestions)
            st.markdown(f"""
<div class="bad-box">
<b>Recommendation: Do Not Proceed at Current Parameters</b><br><br>
{_reason_html}<br><br>
<b>Suggestions to improve viability:</b><br>
{_suggest_html}
</div>
""", unsafe_allow_html=True)

        # ── MD reduction target check ──────────────────────────────────────────
        st.divider()
        st.subheader("MD Reduction Target vs Achieved")
        _md_target_kw = _peak_kw * _md_pct / 100
        _md_achieved  = br["md_reduction_kw"]
        _md_met       = _md_achieved >= _md_target_kw * 0.95   # 5% tolerance

        mc1, mc2 = st.columns(2)
        _kpi(mc1, "MD Reduction Achieved", f"{_md_achieved:.1f} kW",
             "from battery sizing result", "green" if _md_met else "amber")
        _kpi(mc2, "MD Reduction Target",   f"{_md_target_kw:.1f} kW",
             f"{_md_pct}% of {_peak_kw:.0f} kW peak demand", "big")

        if _md_met:
            st.markdown(
                '<div class="good-box">MD reduction target is met.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="warn-box">MD target not fully met — achieved {_md_achieved:.1f} kW '
                f'vs target {_md_target_kw:.1f} kW. '
                f'Consider increasing the battery capacity or raising the MD reduction %.</div>',
                unsafe_allow_html=True,
            )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB — SOLAR SIZING (conditional)
    # ═════════════════════════════════════════════════════════════════════════
    if tab_solar is not None:
        with tab_solar:
            st.header("☀️ Solar Panel Sizing Recommendation")

            c1, c2, c3, c4 = st.columns(4)
            _kpi(c1, "System Size",       f"{sr['system_kwp']} kWp",
                 f"{sr['n_panels']} × {sr['panel_wattage_w']} W panels", "big")
            _kpi(c2, "Usable Roof Area",  f"{sr['usable_area_m2']} m²",
                 f"85% of {sr['roof_area_m2']} m²", "big")
            _kpi(c3, "Annual Generation", f"{sr['annual_generation_kwh']:,} kWh",
                 f"~{sr['daily_generation_kwh_avg']} kWh / day", "green")
            _kpi(c4, "Monthly Average",   f"{sr['monthly_generation_kwh_avg']:,} kWh",
                 "avg per month", "green")

            st.divider()
            cl, cr = st.columns(2)

            with cl:
                st.subheader("Roof Area Utilisation")
                fig = go.Figure(go.Pie(
                    labels=["Solar Array (85%)", "Unusable — edges & obstructions (15%)"],
                    values=[sr["usable_area_m2"], sr["roof_area_m2"] - sr["usable_area_m2"]],
                    marker=dict(colors=["#00d4ff", "#1f2f4a"]),
                    hole=0.48, textinfo="label+percent",
                ))
                fig.update_layout(
                    template="plotly_dark", height=300, showlegend=False,
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
                    "Total Roof Area":             f"{sr['roof_area_m2']} m²",
                    "Usable Area (85%)":           f"{sr['usable_area_m2']} m²",
                    "Panel Count":                 f"{sr['n_panels']} panels",
                    "Panel Size":                  f"{sr['panel_area_m2']} m²",
                    "Panel Rated Power":           f"{sr['panel_wattage_w']} W",
                    "Area per Panel (w/ spacing)": f"{sr['effective_panel_area_m2']} m²",
                    "System Capacity":             f"{sr['system_kwp']} kWp",
                    "Peak Sun Hours":              f"{sr['psh']} hr/day",
                    "System Derate Factor":        f"{sr['system_derate_pct']:.0f}%",
                    "Annual Generation":           f"{sr['annual_generation_kwh']:,} kWh",
                    "Monthly Average":             f"{sr['monthly_generation_kwh_avg']:,} kWh",
                    "Daily Average":               f"{sr['daily_generation_kwh_avg']} kWh",
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
<b>Solar Recommendation:</b> Install a <b>{sr['system_kwp']} kWp</b> system —
<b>{sr['n_panels']} × {sr['panel_wattage_w']} W</b> monocrystalline panels across
<b>{sr['usable_area_m2']} m²</b> (85% of your {sr['roof_area_m2']} m² roof).
Annual generation: <b>{sr['annual_generation_kwh']:,} kWh/year</b> at
{sr['psh']} PSH/day with 80% system derate.
</div>
""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB — BATTERY SIZING
    # ═════════════════════════════════════════════════════════════════════════
    with tab_batt:
        st.header("🔋 Battery Capacity Recommendation")

        _src = (
            "AI-derived from Manager data"
            if br["discharge_col_used"] != "manual"
            else "Manual — MD reduction target"
        )
        st.caption(f"Source: {_src}")

        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, "Minimum Capacity", f"{br['min_capacity_kwh_commercial']} kWh",
             f"exact: {br['min_capacity_kwh']:.1f} kWh → rounded up", "big")
        _kpi(c2, "Usable Energy",    f"{br['required_usable_kwh']} kWh",
             f"SOC {br['soc_range']}", "green")
        _kpi(c3, "Upsize Factor",    f"×{br['upsize_factor']:.2f}",
             "60% DOD → ×1.67", "amber")
        _kpi(c4, "MD Reduction",     f"{br['md_reduction_kw']} kW",
             "peak demand shaved by battery", "green")

        st.divider()
        cl, cr = st.columns(2)

        with cl:
            st.subheader("Battery SOC Operating Window")
            cap    = br["min_capacity_kwh_commercial"]
            bot    = cap * 0.20
            usable = cap * 0.60
            top    = cap * 0.20
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=["Battery"], y=[bot], base=[0],
                name="Protected Reserve 0–20%", marker_color="#ef4444"))
            fig3.add_trace(go.Bar(x=["Battery"], y=[usable], base=[bot],
                name=f"Usable Zone 20–80% = {usable:.0f} kWh", marker_color="#22c55e"))
            fig3.add_trace(go.Bar(x=["Battery"], y=[top], base=[bot + usable],
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
            _days_label = (
                str(br["n_days_analyzed"])
                if br["n_days_analyzed"] > 0 else "Manual (MD target)"
            )
            steps = [
                ("Days analysed",                _days_label),
                ("Design discharge",             f"{br['avg_top_discharge_kwh']} kWh"),
                ("Spike protection",             br["spike_note"]),
                ("÷ Round-trip efficiency (90%)", f"{br['required_usable_kwh']} kWh usable needed"),
                ("60% DOD window",               f"SOC {br['soc_range']} only"),
                ("× Upsize factor 1.67",         f"{br['min_capacity_kwh']:.1f} kWh total"),
                ("Round to commercial size",     f"→ {br['min_capacity_kwh_commercial']} kWh"),
            ]
            for label, val in steps:
                st.markdown(
                    f'<div class="method-step"><b>{label}:</b> {val}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(f"""
<div class="reco-box">
<b>Battery Recommendation:</b> Minimum <b>{br['min_capacity_kwh_commercial']} kWh</b> LFP battery bank.
Operates 20–80% SOC — effective usable capacity is <b>{br['required_usable_kwh']} kWh</b>.
Expected MD reduction: <b>{br['md_reduction_kw']} kW</b>.
</div>
""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB — ROI & FINANCIALS
    # ═════════════════════════════════════════════════════════════════════════
    with tab_roi:
        st.header("💰 ROI & Financial Analysis")

        pb          = rr["simple_payback_years"]
        npv_val     = rr["npv_25yr_rm"]
        _roi_str    = f"{_roi_pct:.1f}%" if _roi_pct is not None else "N/A"
        _irr_str_r  = f"{_irr_pct:.1f}%" if _irr_pct is not None else "N/A"

        # Two rows of 3 KPI cards each
        row1 = st.columns(3)
        row2 = st.columns(3)

        _kpi(row1[0], "Total CAPEX",    f"RM {rr['total_capex_rm']:,}",
             f"Solar RM {rr['solar_capex_rm']:,} · Batt RM {rr['battery_capex_rm']:,}", "big")
        _kpi(row1[1], "Annual Savings", f"RM {rr['total_annual_benefit_rm']:,}",
             f"RM {rr['monthly_net_benefit_rm']:,} / month", "green")
        pb_style = "green" if pb < 10 else "amber" if pb < 13 else "red"
        _kpi(row1[2], "Payback Period", f"{pb} yrs", "simple payback", pb_style)

        npv_style = "green" if npv_val > 0 else "red"
        _kpi(row2[0], "NPV (25 yr)",  f"RM {npv_val:,}", "@ 6% discount rate", npv_style)
        _kpi(row2[1], "IRR",          _irr_str_r,
             f"required ≥{_req_irr:.1f}%", "green" if _irr_pass else "red")
        _kpi(row2[2], "25-yr ROI",    _roi_str,
             f"required ≥{_req_roi:.0f}%", "green" if _roi_pass else "red")

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
            pb_yr = int(math.ceil(pb)) if pb != float("inf") else 999
            if 1 <= pb_yr <= 25:
                fig4.add_vline(x=pb_yr, line_dash="dot", line_color="#f59e0b",
                               annotation_text=f"Payback yr {pb_yr}",
                               annotation_position="top left")
            fig4.update_layout(
                template="plotly_dark", height=380,
                xaxis_title="Year", yaxis_title="RM",
                hovermode="x unified", margin=dict(t=20),
            )
            st.plotly_chart(fig4, use_container_width=True)

        with cr:
            st.subheader("Annual Benefit Breakdown (Year 1)")
            _pie_labels, _pie_values, _pie_colors = [], [], []
            _color_pool = ["#00d4ff", "#22c55e", "#f59e0b"]
            _ci = 0
            for _lbl, _val in [
                ("Energy Offset\n(Self-consumed Solar)", rr["annual_energy_savings_rm"]),
                ("NEM Export Credit",                    rr["annual_nem_credit_rm"]),
                ("MD Peak Shaving\n(Battery)",           rr["annual_md_savings_rm"]),
            ]:
                if _val > 0:
                    _pie_labels.append(_lbl)
                    _pie_values.append(_val)
                    _pie_colors.append(_color_pool[_ci])
                _ci += 1

            if _pie_values:
                fig5 = go.Figure(go.Pie(
                    labels=_pie_labels, values=_pie_values,
                    marker=dict(colors=_pie_colors),
                    hole=0.42,
                    textinfo="label+percent",
                    texttemplate="%{label}<br>RM %{value:,}",
                ))
                fig5.update_layout(
                    template="plotly_dark", height=380,
                    showlegend=False, margin=dict(t=20),
                )
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No annual benefits to display.")

        # Full financial summary table
        st.subheader("Full Financial Summary")
        _system_type = "Solar + Battery" if _solar else "Battery Only"
        _sum_rows = [("System type", _system_type)]
        if _solar and sr:
            _sum_rows.append(("Solar system", f"{sr['system_kwp']} kWp"))
        _sum_rows += [
            ("Battery bank",              f"{br['min_capacity_kwh_commercial']} kWh"),
            ("Solar CAPEX",               f"RM {rr['solar_capex_rm']:,}"),
            ("Battery CAPEX",             f"RM {rr['battery_capex_rm']:,}"),
            ("Total CAPEX",               f"RM {rr['total_capex_rm']:,}"),
            ("MD charge rate used",       f"RM {rr['md_rate_used']:.2f}/kW"),
            ("Energy tariff (C1)",        f"RM {rr['energy_rate_used']}/kWh"),
            ("NEM buyback rate",          f"RM {rr['nem_rate_used']}/kWh"),
            ("Annual energy savings",     f"RM {rr['annual_energy_savings_rm']:,}"),
            ("Annual NEM credit",         f"RM {rr['annual_nem_credit_rm']:,}"),
            ("Annual MD savings",         f"RM {rr['annual_md_savings_rm']:,}"),
            ("Total annual benefit (Y1)", f"RM {rr['total_annual_benefit_rm']:,}"),
            ("Monthly net benefit",       f"RM {rr['monthly_net_benefit_rm']:,}"),
            ("Simple payback",            f"{pb} years"),
            ("NPV over 25 yr @ 6%",       f"RM {npv_val:,}"),
            ("IRR",                       _irr_str_r),
            ("25-yr ROI",                 _roi_str),
            ("25-yr net cashflows",       f"RM {rr['total_25yr_cashflows']:,}"),
            ("CO₂ offset",               f"{rr['co2_offset_tonnes_yr']:.1f} tonnes/year"),
        ]
        if _solar and sr:
            _sum_rows += [
                ("Self-consumed (annual)", f"{rr['self_consumed_kwh_annual']:,} kWh"),
                ("NEM exported (annual)",  f"{rr['exported_kwh_annual']:,} kWh"),
            ]

        st.dataframe(
            pd.DataFrame(_sum_rows, columns=["Parameter", "Value"]),
            use_container_width=True, hide_index=True,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB — DATA OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════
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
                    pc1.metric("Peak kW",    ps.get("peak_kw", "—"))
                    pc2.metric("Avg Top 5%", ps.get("avg_top5pct_kw", "—"))
                    pc3.metric("Mean kW",    ps.get("mean_kw", "—"))
                st.dataframe(df_p.head(100), use_container_width=True)
            else:
                st.info("No Predictor data uploaded.")


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING STATE (before first run)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
<div style="text-align:center;padding:50px 0 30px;">
  <h2 style="color:#00d4ff;margin-bottom:8px;">Configure inputs in the sidebar, then click Analyse &amp; Recommend</h2>
  <p style="color:#888;font-size:1.05rem;max-width:700px;margin:0 auto;">
  PowerRECO sizes battery and solar systems, calculates a 25-year ROI, and delivers a clear
  <b>ACCEPT / CONDITIONAL / REJECT</b> verdict based on your ROI and IRR thresholds.
  </p>
</div>
""", unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown("""
#### 🔋 Battery (Mandatory)
Set your peak demand and desired MD reduction %. Upload Manager data for AI sizing,
or PowerRECO sizes automatically from the MD target using a 1-hour discharge window.
        """)
    with fc2:
        st.markdown("""
#### ☀️ Solar (Optional)
Toggle on to include a solar PV system. Enter roof area, panel size (m²), and rated power —
PowerRECO optimises the array using Malaysia's seasonal irradiance profile (PSH step = 15 min).
        """)
    with fc3:
        st.markdown("""
#### 💰 25-Year ROI
Full model: MD peak shaving, energy offset, NEM 3.0 export, battery replacement at year 10,
panel & battery degradation, simple payback, NPV @ 6%, and IRR.
        """)
    with fc4:
        st.markdown("""
#### ⭐ Recommendation
Set your Required ROI (default 200%) and Required IRR (default 12%). PowerRECO checks both,
then delivers **ACCEPT**, **CONDITIONAL**, or **REJECT** with a full explanation.
        """)
