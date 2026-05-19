"""
Calculator package — TNB tariff billing engine.
"""
from calculator.tnb_tariffs import (
    calculate_bill,
    compute_nem_credit,
    auto_detect_tariff,
    compute_monthly_stats,
    TARIFF_META,
)

__all__ = [
    "calculate_bill",
    "compute_nem_credit",
    "auto_detect_tariff",
    "compute_monthly_stats",
    "TARIFF_META",
]
