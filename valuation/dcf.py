"""Simple 5-year DCF model using FCF projections."""
from __future__ import annotations

from typing import Optional

from config import DCF_DISCOUNT_RATE, DCF_PROJECTION_YEARS, DCF_TERMINAL_GROWTH


def dcf_per_share(
    fcf_bn: float,
    shares_millions: float,
    fcf_growth_rate: float = 0.08,
    discount_rate: float = DCF_DISCOUNT_RATE,
    terminal_growth: float = DCF_TERMINAL_GROWTH,
    projection_years: int = DCF_PROJECTION_YEARS,
) -> Optional[float]:
    """5-year DCF → intrinsic value per share in VND.

    fcf_bn: trailing FCF in VND billions (use last 4-quarter sum for TTM)
    shares_millions: shares outstanding in millions
    fcf_growth_rate: projected annual FCF growth rate (default 8%)
    discount_rate: WACC proxy — VN 10Y bond ~4.5% + ERP ~4% = 8.5%
    terminal_growth: long-run perpetuity growth (default 3%)

    Returns None if FCF is non-positive or shares are zero.
    """
    if not fcf_bn or fcf_bn <= 0 or not shares_millions or shares_millions <= 0:
        return None
    if discount_rate <= terminal_growth:
        return None

    # Project FCF for each year and discount to PV
    pv_fcfs = 0.0
    fcf = fcf_bn
    for year in range(1, projection_years + 1):
        fcf *= (1 + fcf_growth_rate)
        pv_fcfs += fcf / (1 + discount_rate) ** year

    # Terminal value (Gordon Growth Model) discounted back
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** projection_years

    total_value_bn = pv_fcfs + pv_terminal
    # VND billions / millions shares = VND thousands per share → × 1000
    return (total_value_bn / shares_millions) * 1_000


def upside_pct(intrinsic_value: float, current_price: float) -> Optional[float]:
    """(intrinsic_value - price) / price. Returns None if price is zero."""
    if not current_price or current_price <= 0:
        return None
    return (intrinsic_value - current_price) / current_price
