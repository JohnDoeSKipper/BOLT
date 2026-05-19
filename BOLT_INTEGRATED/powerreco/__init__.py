"""
PowerRECO package — solar and battery sizing with ROI analysis.
"""
from powerreco.solar_sizing import calculate_solar_sizing
from powerreco.battery_sizing import calculate_battery_sizing
from powerreco.roi_engine import calculate_roi
from powerreco.data_connector import parse_manager_output, parse_predictor_output, extract_peak_demand_stats

__all__ = [
    "calculate_solar_sizing",
    "calculate_battery_sizing",
    "calculate_roi",
    "parse_manager_output",
    "parse_predictor_output",
    "extract_peak_demand_stats",
]
