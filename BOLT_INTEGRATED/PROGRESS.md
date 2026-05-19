# BOLT Integrated — Build Progress & Documentation

## Status: COMPLETE ✅

All four BOLT AI modules have been merged into a single Streamlit application.

---

## Directory Structure

```
BOLT_INTEGRATED/
├── app.py                        ← Unified Streamlit application (5 tabs)
├── requirements.txt
├── PROGRESS.md                   ← This file
│
├── predictor/                    ← LightGBM load forecasting
│   ├── __init__.py
│   ├── data_loader.py            ← Multi-format Excel/CSV parser
│   ├── features.py               ← Feature engineering (time, lags, solar, temp)
│   ├── forecaster.py             ← DirectMultiStepForecaster (quantile LightGBM)
│   ├── cv.py                     ← Expanding-window cross-validation
│   ├── simulation.py             ← Live replay simulation engine
│   └── solar_estimator.py        ← Solar capacity estimator from load profile
│
├── manager/                      ← AI load optimizer (extracted from Flask)
│   ├── __init__.py
│   └── optimizer.py              ← run_ai_manager(), parse_uploaded_data()
│
├── calculator/                   ← TNB tariff billing engine
│   ├── __init__.py
│   └── tnb_tariffs.py            ← calculate_bill(), auto_detect_tariff(), etc.
│
├── powerreco/                    ← Solar + battery sizing & ROI
│   ├── __init__.py
│   ├── solar_sizing.py           ← calculate_solar_sizing()
│   ├── battery_sizing.py         ← calculate_battery_sizing()
│   ├── roi_engine.py             ← calculate_roi() — 25-year DCF, IRR
│   └── data_connector.py        ← parse_manager_output(), parse_predictor_output()
│
├── pipeline/                     ← Inter-module data converters
│   ├── __init__.py
│   └── data_bridge.py            ← forecast_to_manager_df(), manager_results_to_*()
│
├── data/                         ← Place input Excel/CSV files here
├── models/                       ← Saved forecaster .joblib files
├── outputs/                      ← Downloaded CSVs from the app
│
└── Backup/                       ← Originals from GitHub (untouched)
    ├── Manager/app.py
    ├── Predictor/src/
    ├── SAM_CALCULATOR/src/
    └── PowerRECO/src/
```

---

## How to Run

```bash
cd C:\Users\A\BOLT_INTEGRATED
pip install -r requirements.txt
streamlit run app.py
```

---

## Pipeline Data Flow

```
Upload Load Profile (Excel/CSV)
  │
  ▼
predictor.data_loader.auto_load()  — normalises all 4 TNB meter formats
  │
  ├─→ Tab 2: Predictor
  │     predictor.forecaster.DirectMultiStepForecaster.fit()
  │     .forecast(output_steps=48)  →  MultiStepForecastResult
  │     pipeline.data_bridge.forecast_to_manager_df()  [if feeding to Manager]
  │
  └─→ Tab 3: AI Manager
        manager.optimizer.run_ai_manager()
          │
          ├─→ Tab 4: Bill Calculator
          │     pipeline.data_bridge.manager_results_to_sam_df()
          │     calculator.tnb_tariffs.compute_monthly_stats()
          │     calculator.tnb_tariffs.calculate_bill()
          │
          └─→ Tab 5: PowerRECO
                pipeline.data_bridge.manager_results_to_powerreco_df()
                powerreco.battery_sizing.calculate_battery_sizing()
                powerreco.solar_sizing.calculate_solar_sizing()
                powerreco.roi_engine.calculate_roi()
```

---

## Known Assumptions & Fabrications

| # | Category | Detail |
|---|----------|--------|
| F1 | **kVAR estimation** | When historical kVAR is unavailable (e.g., forecast output), kVAR = kW × tan(arccos(PF)) where PF = 0.85 (typical commercial/industrial). This affects Manager discharge accuracy — the exact formula `dis_kw = kW − √(T²−kVAR²)` is sensitive to kVAR errors. Mitigation: `pipeline.data_bridge.forecast_to_manager_df()` derives PF from historical data if provided. |
| F2 | **Solar export in forecast** | The Predictor only forecasts kW import. `forecast_to_manager_df()` sets kw_export = 0. For solar sites, feed historical data (with export readings) directly to the Manager instead of the forecast. |
| F3 | **30-min intervals assumed** | All modules assume 30-minute (half-hourly) data. Hourly or 15-min data will be resampled to 30-min by `parse_uploaded_data()`. |
| F4 | **TNB tariff rates** | Rates are from the official TNB schedule (revised 2014, updated 2022). The new MD rate of RM 97.06/kW in roi_engine.py is from publicly available TNB press releases — verify against your actual bill. The original `tnb_tariffs.py` uses RM 30.30/kW (legacy rate). |
| F5 | **Battery replacement cost** | Assumed 20% discount at year 10 (industry norm for LFP). Actual replacement cost will depend on 2034 market prices. |
| F6 | **Solar degradation** | 0.5%/year (manufacturer warranty basis). Real-world degradation for Malaysian climate (high humidity, heat) may be slightly higher. |
| F7 | **PowerRECO battery sizing from forecast** | If Manager results are unavailable, battery is sized to 10% of daily solar generation as a conservative fallback. This is a rough heuristic — always run the Manager first. |
| F8 | **Seasonal irradiance factors** | The 12 monthly multipliers in `solar_sizing.py` are approximate values for Peninsular Malaysia. Exact values vary by location (Perlis vs Johor). |

---

## Import Changes Made to Original Code

| File | Change | Reason |
|------|--------|--------|
| `predictor/forecaster.py` | `from src.features import` → `from predictor.features import` | Both Predictor and SAM_CALCULATOR have `src/data_loader.py` — adding both to `sys.path` would cause name conflicts. Renamed packages resolve this. |
| `predictor/simulation.py` | `from src.forecaster import` (TYPE_CHECKING block) → `from predictor.forecaster import` | Same namespace conflict resolution. |
| `manager/optimizer.py` | All Flask imports and routes removed. `run_ai_manager()` and `parse_uploaded_data()` extracted as pure Python functions. | Manager was a Flask backend — merged into Streamlit without HTTP layer. |

---

## Session History

- **Session 1**: Fetched all source files from GitHub. Resolved import conflicts. Created directory structure and Backup files. Wrote `predictor/` package.  
- **Session 2**: Wrote `manager/`, `calculator/`, `powerreco/`, `pipeline/` packages. Wrote unified `app.py` and `requirements.txt`. System complete.
