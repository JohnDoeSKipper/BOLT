import sys
sys.path.insert(0, '.')

from src.solar_sizing import calculate_solar_sizing
from src.battery_sizing import calculate_battery_sizing, _round_to_commercial, ROUNDTRIP_EFF, UPSIZE_FACTOR
from src.roi_engine import calculate_roi, MD_CHARGE_NEW, MD_CHARGE_OLD, ENERGY_RATE_C1, NEM_BUYBACK
from src.data_connector import parse_manager_output, parse_predictor_output, extract_peak_demand_stats

print("All imports OK")

sr = calculate_solar_sizing(200.0, 415, 4.5)
print(f"Solar: {sr['system_kwp']} kWp, {sr['n_panels']} panels, {sr['annual_generation_kwh']:,} kWh/yr")
print(f"  usable_area={sr['usable_area_m2']} m2, monthly_avg={sr['monthly_generation_kwh_avg']:,} kWh")

rr = calculate_roi(sr['system_kwp'], 50.0, sr['monthly_generation_kwh_avg'], 80.0)
print(f"ROI: payback={rr['simple_payback_years']}yr, NPV=RM{rr['npv_25yr_rm']:,}, IRR={rr['irr_pct']}%")
print(f"  energy_save=RM{rr['annual_energy_savings_rm']:,}/yr, MD_save=RM{rr['annual_md_savings_rm']:,}/yr")

print(f"Battery factor: 1/{ROUNDTRIP_EFF}/0.60 -> upsize={UPSIZE_FACTOR:.4f}")

manual_kwh = 50.0
usable = manual_kwh / ROUNDTRIP_EFF
cap = usable * UPSIZE_FACTOR
commercial = _round_to_commercial(cap)
print(f"Battery sizing (manual 50 kWh/day): usable={usable:.1f} kWh, cap={cap:.1f} kWh -> {commercial} kWh commercial")

print("\nAll checks passed.")
