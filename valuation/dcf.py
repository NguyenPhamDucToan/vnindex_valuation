"""FCFF-based DCF valuation model.

FCFF = NOPAT + D&A − CapEx − ΔNWC
     where NOPAT = EBIT × (1 − tax_rate)

Enterprise Value = Σ PV(FCFF_1..n) + PV(Terminal Value)
Equity Value     = EV − Net Debt
Intrinsic Price  = Equity Value / Shares Outstanding
"""
from __future__ import annotations

from typing import Optional

from config import DCF_PROJECTION_YEARS, DCF_TERMINAL_GROWTH, TAX_RATE
from valuation.wacc import wacc as calc_wacc, DEFAULT_BETA, DEFAULT_COD


def nopat(ebit: float, tax_rate: float = TAX_RATE) -> float:
    """Net Operating Profit After Tax = EBIT × (1 - T)."""
    return ebit * (1 - tax_rate)


def fcff_from_components(
    ebit: float,
    depreciation: float,
    capex: float,
    delta_nwc: float,
    tax_rate: float = TAX_RATE,
) -> float:
    """FCFF = NOPAT + D&A − CapEx − ΔNWC.

    delta_nwc: positive = WC increased (cash consumed), negative = WC released.
    All values in VND billions.
    """
    return nopat(ebit, tax_rate) + depreciation - capex - delta_nwc


def dcf_valuation(
    fcff_base: float,
    net_debt_bn: float,
    shares_millions: float,
    fcff_growth_rate: float = 0.12,
    terminal_growth: float = DCF_TERMINAL_GROWTH,
    projection_years: int = DCF_PROJECTION_YEARS,
    beta: float = DEFAULT_BETA,
    cost_of_debt: float = DEFAULT_COD,
    debt_bn: float = 0.0,
    equity_bn: float = 1.0,
    wacc_override: float | None = None,
) -> dict:
    """Full FCFF DCF returning a results dict.

    Args:
        fcff_base:        TTM FCFF in VND billions (starting point for projection)
        net_debt_bn:      Total debt − Cash in VND billions
        shares_millions:  Shares outstanding in millions
        fcff_growth_rate: Projected annual FCFF growth rate (default 12%)
        terminal_growth:  Long-run perpetuity growth (default 3% = VN GDP)
        projection_years: Explicit forecast horizon (default 5)
        beta / cost_of_debt / debt_bn / equity_bn: WACC inputs

    Returns dict with keys:
        wacc, pv_fcffs, terminal_value, pv_terminal,
        enterprise_value, equity_value, price_per_share, fcff_projections
    """
    if not shares_millions or shares_millions <= 0:
        return _empty_result()

    w = wacc_override if wacc_override is not None else calc_wacc(beta, cost_of_debt, debt_bn, equity_bn)

    if w <= terminal_growth:
        return _empty_result()

    # Project and discount FCFF
    pv_fcffs = 0.0
    fcff_projections = []
    fcff = fcff_base
    for year in range(1, projection_years + 1):
        fcff *= (1 + fcff_growth_rate)
        pv = fcff / (1 + w) ** year
        pv_fcffs += pv
        fcff_projections.append({"year": year, "fcff": round(fcff, 2), "pv": round(pv, 2)})

    # Terminal value (Gordon Growth Model)
    terminal_fcff = fcff * (1 + terminal_growth)
    tv = terminal_fcff / (w - terminal_growth)
    pv_tv = tv / (1 + w) ** projection_years

    ev = pv_fcffs + pv_tv
    equity_val = ev - net_debt_bn
    price = (equity_val / shares_millions) * 1_000  # billions/millions → VND per share

    return {
        "wacc": round(w, 4),
        "pv_fcffs": round(pv_fcffs, 2),
        "terminal_value": round(tv, 2),
        "pv_terminal": round(pv_tv, 2),
        "enterprise_value": round(ev, 2),
        "equity_value": round(equity_val, 2),
        "price_per_share": round(price, 0),
        "fcff_projections": fcff_projections,
    }


def upside_pct(intrinsic_price: float, current_price: float) -> Optional[float]:
    """(intrinsic_price - current_price) / current_price."""
    if not current_price or current_price <= 0:
        return None
    return (intrinsic_price - current_price) / current_price


def sensitivity_grid(
    fcff_base: float,
    net_debt_bn: float,
    shares_millions: float,
    wacc_range: list[float] = (0.12, 0.146, 0.16),
    growth_range: list[float] = (0.08, 0.12, 0.18),
) -> list[dict]:
    """3×3 sensitivity table varying WACC and revenue/FCFF growth rate."""
    results = []
    for w in wacc_range:
        for g in growth_range:
            if w <= DCF_TERMINAL_GROWTH:
                continue
            res = dcf_valuation(
                fcff_base=fcff_base,
                net_debt_bn=net_debt_bn,
                shares_millions=shares_millions,
                fcff_growth_rate=g,
                wacc_override=w,          # directly vary the discount rate
            )
            results.append({
                "wacc": w,
                "growth": g,
                "price": res.get("price_per_share"),
            })
    return results


def _empty_result() -> dict:
    return {
        "wacc": None, "pv_fcffs": None, "terminal_value": None,
        "pv_terminal": None, "enterprise_value": None,
        "equity_value": None, "price_per_share": None,
        "fcff_projections": [],
    }
