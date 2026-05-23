"""WACC calculation via CAPM for Vietnamese (HOSE) companies."""
from __future__ import annotations

from config import DCF_DISCOUNT_RATE

# --- Vietnamese market parameters ---
RF = 0.05          # Risk-free rate: VN 10-year government bond
ERP = 0.08         # Equity risk premium for Vietnam (higher than developed markets)
DEFAULT_BETA = 1.2 # Market-average beta for HOSE; use company-specific when available
DEFAULT_COD = 0.08 # Average cost of debt on HOSE
TAX_RATE = 0.20    # Standard Vietnamese corporate income tax rate


def cost_of_equity(beta: float = DEFAULT_BETA, rf: float = RF, erp: float = ERP) -> float:
    """CAPM: CoE = Rf + β × ERP."""
    return rf + beta * erp


def wacc(
    beta: float = DEFAULT_BETA,
    cost_of_debt: float = DEFAULT_COD,
    debt_bn: float = 0.0,
    equity_bn: float = 1.0,
    tax_rate: float = TAX_RATE,
) -> float:
    """WACC = CoE × [E/(D+E)] + CoD × (1-T) × [D/(D+E)].

    debt_bn / equity_bn in VND billions (same unit, ratio is what matters).
    Falls back to DCF_DISCOUNT_RATE from config when equity is zero.
    """
    total = debt_bn + equity_bn
    if total <= 0:
        return DCF_DISCOUNT_RATE

    e_weight = equity_bn / total
    d_weight = debt_bn / total
    coe = cost_of_equity(beta)
    after_tax_cod = cost_of_debt * (1 - tax_rate)

    return coe * e_weight + after_tax_cod * d_weight


def default_wacc() -> float:
    """WACC for a typical HOSE company with no debt/equity breakdown available."""
    return wacc(
        beta=DEFAULT_BETA,
        cost_of_debt=DEFAULT_COD,
        debt_bn=1.0,
        equity_bn=1.0,
    )
