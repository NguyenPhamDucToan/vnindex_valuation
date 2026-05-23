"""P/E, P/B, and EV/EBITDA calculations."""
from __future__ import annotations

from typing import Optional


def pe_ratio(price: float, eps_ttm: float) -> Optional[float]:
    """Price / TTM EPS. Returns None for negative or zero EPS."""
    if not eps_ttm or eps_ttm <= 0:
        return None
    return price / eps_ttm


def pb_ratio(price: float, equity: float, shares_outstanding: float) -> Optional[float]:
    """Price / Book value per share. equity in VND billions, shares in millions."""
    if not shares_outstanding or shares_outstanding <= 0:
        return None
    bvps = (equity * 1_000) / shares_outstanding   # convert billions→millions VND
    if bvps <= 0:
        return None
    return price / bvps


def ev_ebitda(
    market_cap: float,
    debt: float,
    cash: float,
    ebitda: float,
) -> Optional[float]:
    """(Market cap + Debt - Cash) / EBITDA.
    market_cap in VND billions, debt/cash/ebitda in VND billions.
    Best for capital-intensive companies: REE, GAS, HPG.
    """
    if not ebitda or ebitda <= 0:
        return None
    enterprise_value = market_cap + debt - cash
    return enterprise_value / ebitda
