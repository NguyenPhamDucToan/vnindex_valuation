"""Composite quality score and 7-level Buy/Sell signal classification.

Used by the Valuation Screen, Watchlist, and the Day-18 historical backtest.
"""
from __future__ import annotations

from typing import Optional

# Canonical signal order, best → worst
SIGNAL_ORDER = ["Strong Buy", "Buy", "Watch", "Neutral", "Reduce", "Sell", "Strong Sell"]


def compute_quality_score(roe_v, net_margin_v, profit_quality_v,
                           fcf_margin_v, current_ratio_v, debt_to_equity_v) -> float:
    """Composite quality score 0-100 from 6 metrics.

    Thresholds (maps raw value to 0-10 sub-score):
      ROE           0 % -> 0,  25 %+ -> 10
      Net margin    0 % -> 0,  20 %+ -> 10
      Profit quality 0 -> 0,  1.5+  -> 10
      FCF margin    0 % -> 0,  15 %+ -> 10
      Current ratio 0.5 -> 0,  2.5+  -> 10
      D/E           3.0+ -> 0,  0     -> 10
    """
    scores = []
    if roe_v is not None:
        scores.append(max(0.0, min(10.0, max(roe_v, 0.0) / 0.25 * 10.0)))
    if net_margin_v is not None:
        scores.append(max(0.0, min(10.0, max(net_margin_v, 0.0) / 0.20 * 10.0)))
    if profit_quality_v is not None:
        scores.append(max(0.0, min(10.0, max(profit_quality_v, 0.0) / 1.5 * 10.0)))
    if fcf_margin_v is not None:
        scores.append(max(0.0, min(10.0, max(fcf_margin_v, 0.0) / 0.15 * 10.0)))
    if current_ratio_v is not None:
        scores.append(max(0.0, min(10.0, (current_ratio_v - 0.5) / 2.0 * 10.0)))
    if debt_to_equity_v is not None:
        scores.append(max(0.0, min(10.0, (3.0 - min(debt_to_equity_v, 3.0)) / 3.0 * 10.0)))
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
