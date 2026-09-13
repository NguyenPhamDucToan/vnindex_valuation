"""Composite quality score and 7-level Buy/Sell signal classification.

Used by the Valuation Screen, Watchlist, and the Day-18 historical backtest.
"""
from __future__ import annotations

from typing import Optional

# Canonical signal order, best → worst
SIGNAL_ORDER = ["Strong Buy", "Buy", "Watch", "Neutral", "Reduce", "Sell", "Strong Sell"]


# Banks, brokers and insurers do not report the line items the industrial
# thresholds below are calibrated against, so three of the six sub-scores stop
# measuring anything for them. Measured across the market on 2026-09-13:
#
#   * net margin -- a bank's "revenue" is total operating income, so the sector
#     median margin is 37% against a 20% bar: 19 of 21 banks scored a perfect
#     10/10 and the metric separated nobody. Retail managed 1 of 12. The bar
#     moves to 40% for financials, which is where the sector's own range tops
#     out, so it discriminates again.
#   * profit quality and FCF margin -- both divide by operating cash flow, which
#     for a bank is driven by deposit and loan flows rather than by trading
#     well, and 18 of 21 banks maxed out FCF margin too. The scorecard in the
#     dashboard already drops exactly these two for brokers, for exactly this
#     reason; the score now agrees with it.
#   * D/E -- for a bank this is interbank borrowing against equity, and the
#     industrial 0-3 band scored the sector 2.1/10 for being a bank at all. The
#     band becomes 0-4, the one the bank scorecard itself displays (<=2x good,
#     >4x danger).
#
# Leaving `sector` unset keeps the original behaviour, so callers that have no
# sector to hand are unaffected.
FINANCIAL_SECTORS = frozenset({"Ngân hàng", "Chứng khoán", "Bảo hiểm"})


def compute_quality_score(roe_v, net_margin_v, profit_quality_v,
                           fcf_margin_v, current_ratio_v, debt_to_equity_v,
                           sector=None) -> float:
    """Composite quality score 0-100 from up to 6 metrics.

    Thresholds (maps raw value to 0-10 sub-score):
      ROE           0 % -> 0,  25 %+ -> 10
      Net margin    0 % -> 0,  20 %+ -> 10   (40 %+ for financials)
      Profit quality 0 -> 0,  1.5+  -> 10    (skipped for financials)
      FCF margin    0 % -> 0,  15 %+ -> 10   (skipped for financials)
      Current ratio 0.5 -> 0,  2.5+  -> 10
      D/E           3.0+ -> 0,  0     -> 10  (4.0+ -> 0 for financials)
    """
    financial = sector in FINANCIAL_SECTORS
    scores = []
    if roe_v is not None:
        scores.append(max(0.0, min(10.0, max(roe_v, 0.0) / 0.25 * 10.0)))
    if net_margin_v is not None:
        nm_bar = 0.40 if financial else 0.20
        scores.append(max(0.0, min(10.0, max(net_margin_v, 0.0) / nm_bar * 10.0)))
    if profit_quality_v is not None and not financial:
        scores.append(max(0.0, min(10.0, max(profit_quality_v, 0.0) / 1.5 * 10.0)))
    if fcf_margin_v is not None and not financial:
        scores.append(max(0.0, min(10.0, max(fcf_margin_v, 0.0) / 0.15 * 10.0)))
    if current_ratio_v is not None:
        scores.append(max(0.0, min(10.0, (current_ratio_v - 0.5) / 2.0 * 10.0)))
    if debt_to_equity_v is not None:
        de_bar = 4.0 if financial else 3.0
        scores.append(max(0.0, min(10.0, (de_bar - min(debt_to_equity_v, de_bar)) / de_bar * 10.0)))
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores) * 10.0, 1)


def classify_signal(avg_upside: float, quality_score: float) -> str:
    """7-level signal from average valuation upside and quality score.

    avg_upside: (avg intrinsic value - price) / price, e.g. 0.20 = +20%.
    quality_score: 0-100, from compute_quality_score().
    Returns one of SIGNAL_ORDER.
    """
    u, q = avg_upside, quality_score
    if   u >=  0.20 and q >= 60: return "Strong Buy"
    elif u >=  0.10 and q >= 45: return "Buy"
    elif u >=  0.00:             return "Watch"
    elif u >= -0.10:             return "Neutral"
    elif u >= -0.30:             return "Reduce"
    elif u >= -0.50:             return "Sell"
    else:                        return "Strong Sell"
