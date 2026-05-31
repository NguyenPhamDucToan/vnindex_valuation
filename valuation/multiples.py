"""Multiple-based and alternative intrinsic value estimates.

All functions return a price in VND/share, or None if inputs are insufficient.
All monetary inputs are in VND billions; shares in millions.

VN market target multiples (HOSE historical medians):
  P/B      1.5×   — VN market median book multiple
  P/S      1.2×   — conservative; higher for tech/consumer
  EV/EBITDA 8×    — industrials, real estate, utilities
  P/OCF   10×     — similar to P/E but uses cash, not accrual
"""
from __future__ import annotations

from typing import Optional

from config import TAX_RATE
from valuation.wacc import cost_of_equity, default_wacc, DEFAULT_BETA

TARGET_PB        = 1.5
TARGET_PS        = 1.2
TARGET_EV_EBITDA = 8.0
TARGET_POCF      = 10.0


def pb_implied(
    equity_bn: float,
    shares_millions: float,
    target_pb: float = TARGET_PB,
) -> Optional[float]:
    """P/B implied: BVPS × target P/B multiple."""
    if not equity_bn or not shares_millions or shares_millions <= 0:
        return None
    bvps = equity_bn * 1_000 / shares_millions
    if bvps <= 0:
        return None
    return round(bvps * target_pb, 0)


def ev_ebitda_implied(
    ebitda_bn: float,
    net_debt_bn: float,
    shares_millions: float,
    target_multiple: float = TARGET_EV_EBITDA,
) -> Optional[float]:
    """EV/EBITDA implied: (EBITDA × multiple − net debt) ÷ shares."""
    if not ebitda_bn or ebitda_bn <= 0 or not shares_millions or shares_millions <= 0:
        return None
    ev = ebitda_bn * target_multiple
    equity_val = ev - (net_debt_bn or 0.0)
    if equity_val <= 0:
        return None
    return round(equity_val * 1_000 / shares_millions, 0)


def epv_implied(
    ebit_bn: float,
    net_debt_bn: float,
    shares_millions: float,
    wacc: Optional[float] = None,
    tax_rate: float = TAX_RATE,
) -> Optional[float]:
    """Earnings Power Value: NOPAT ÷ WACC with zero growth (conservative floor).

    EPV strips out any growth premium — good baseline to compare against DCF.
    """
    if not ebit_bn or ebit_bn <= 0 or not shares_millions or shares_millions <= 0:
        return None
    w = wacc if wacc else default_wacc()
    if w <= 0:
        return None
    nopat = ebit_bn * (1.0 - tax_rate)
    epv_bn = nopat / w
    equity_val = epv_bn - (net_debt_bn or 0.0)
    if equity_val <= 0:
        return None
    return round(equity_val * 1_000 / shares_millions, 0)


def ps_implied(
    revenue_bn: float,
    shares_millions: float,
    target_ps: float = TARGET_PS,
) -> Optional[float]:
    """Price/Sales implied: (Revenue ÷ shares) × target P/S multiple.

    Useful when earnings are negative or volatile.
    """
    if not revenue_bn or revenue_bn <= 0 or not shares_millions or shares_millions <= 0:
        return None
    rps = revenue_bn * 1_000 / shares_millions
    return round(rps * target_ps, 0)


def residual_income_implied(
    equity_bn: float,
    shares_millions: float,
    net_income_bn: float,
    g: float = 0.12,
    beta: float = DEFAULT_BETA,
) -> Optional[float]:
    """Residual Income (RI) implied price.

    Value = BVPS + (ROE − CoE) × BVPS ÷ (CoE − g)

    If ROE > CoE the company earns above cost of equity → premium to book.
    If ROE < CoE it destroys value → trades at a discount to book.
    """
    if not equity_bn or not shares_millions or shares_millions <= 0:
        return None
    if not net_income_bn:
        return None

    bvps = equity_bn * 1_000 / shares_millions
    coe  = cost_of_equity(beta)

    if coe <= g:
        return None

    roe_val = net_income_bn / equity_bn
    ri_price = bvps + (roe_val - coe) * bvps / (coe - g)

    if ri_price <= 0:
        return None
    return round(ri_price, 0)


def pocf_implied(
    operating_cf_bn: float,
    shares_millions: float,
    target_pocf: float = TARGET_POCF,
) -> Optional[float]:
    """Price/OCF implied: (Operating CF ÷ shares) × target multiple.

    Cross-checks P/E using actual cash instead of accrual earnings.
    """
    if not operating_cf_bn or operating_cf_bn <= 0 or not shares_millions or shares_millions <= 0:
        return None
    ocf_ps = operating_cf_bn * 1_000 / shares_millions
    return round(ocf_ps * target_pocf, 0)
