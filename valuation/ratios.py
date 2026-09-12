"""All financial ratios across 4 dimensions:
  1. Profitability (Income Statement)
  2. Cash Flow Quality
  3. Balance Sheet (Liquidity + Capital Structure)
  4. Working Capital / Cash Conversion Cycle
"""
from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(numerator, denominator) -> Optional[float]:
    """Divide safely; return None on zero/None denominator."""
    try:
        if not denominator or denominator == 0:
            return None
        return numerator / denominator
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# 1. PROFITABILITY (Income Statement dimension)
# ---------------------------------------------------------------------------

def gross_margin(gross_profit: float, revenue: float) -> Optional[float]:
    """Gross Profit / Net Revenue. Benchmark ≥ 25%."""
    return _safe(gross_profit, revenue)


def operating_margin(ebit: float, revenue: float) -> Optional[float]:
    """EBIT / Net Revenue. Benchmark ≥ 15%."""
    return _safe(ebit, revenue)


def ebitda_margin(ebitda: float, revenue: float) -> Optional[float]:
    """EBITDA / Net Revenue. Benchmark ≥ 15%."""
    return _safe(ebitda, revenue)


def net_margin(net_income: float, revenue: float) -> Optional[float]:
    """PAT / Net Revenue. Industry benchmark ≈ 8%; ≥ 15% = excellent."""
    return _safe(net_income, revenue)


def roe(net_income: float, equity: float) -> Optional[float]:
    """Return on Equity = PAT / Equity. Benchmark 25–35%."""
    return _safe(net_income, equity)


def roa(net_income: float, total_assets: float) -> Optional[float]:
    """Return on Assets = PAT / Total Assets. Benchmark 12–18%."""
    return _safe(net_income, total_assets)


def roic(ebit: float, tax_rate: float, debt: float, equity: float, cash: float) -> Optional[float]:
    """Return on Invested Capital = NOPAT / (Debt + Equity − Cash).

    NOPAT = EBIT × (1 − tax_rate). Compare against WACC: ROIC > WACC means
    the company creates economic value; ROIC < WACC means it destroys value
    even if accounting profit is positive.
    """
    if ebit is None:
        return None
    nopat = ebit * (1 - tax_rate)
    invested_capital = (debt or 0) - (cash or 0) + (equity or 0)
    return _safe(nopat, invested_capital)


def asset_turnover(revenue: float, total_assets: float) -> Optional[float]:
    """Revenue / Total Assets. Benchmark 1.2–2.0×."""
    return _safe(revenue, total_assets)


def ebit_interest_coverage(ebit: float, interest_expense: float) -> Optional[float]:
    """EBIT / Interest Expense. ≥ 5× safe, ≥ 10× very safe."""
    return _safe(ebit, interest_expense)


def revenue_growth(revenue_now: float, revenue_prev: float) -> Optional[float]:
    """YoY revenue growth. Benchmark ≥ 12%."""
    if not revenue_prev or revenue_prev <= 0:
        return None
    return (revenue_now - revenue_prev) / revenue_prev


def profit_growth(net_income_now: float, net_income_prev: float) -> Optional[float]:
    """YoY PAT growth. Should exceed revenue_growth (operating leverage)."""
    if not net_income_prev or net_income_prev <= 0:
        return None
    return (net_income_now - net_income_prev) / net_income_prev


# ---------------------------------------------------------------------------
# 2. CASH FLOW QUALITY (CF dimension — FCF is the primary signal)
# ---------------------------------------------------------------------------

def profit_quality(operating_cf: float, net_income: float) -> Optional[float]:
    """OCF / Net income. ≥ 1.0 = good; < 0.8 = earnings quality risk.

    Undefined when the company lost money: "how much of the profit is backed by
    cash" has no answer when there is no profit, and dividing two negatives
    manufactures a flattering positive. TCI (OCF -666bn, net income -2bn) scored
    328.74x, which compute_quality_score clamps to a perfect 10/10 on this
    metric -- the most cash-destroying names in the market came out looking like
    the highest earnings quality. Five of them, ST8 and TVB included, were
    pushed over the quality-45 line and shown as "Buy". None returns them to
    "not measurable", which the score already handles by averaging the rest.
    """
    if net_income is None or net_income <= 0:
        return None
    return _safe(operating_cf, net_income)


def ocf_to_revenue(operating_cf: float, revenue: float) -> Optional[float]:
    """OCF / Revenue. ≥ 10% = good; ≥ 15% = excellent."""
    return _safe(operating_cf, revenue)


def fcf_margin(fcf: float, revenue: float) -> Optional[float]:
    """FCF / Revenue. ≥ 5% = good; ≥ 10% = excellent."""
    return _safe(fcf, revenue)


def fcf_yield(fcf: float, total_assets: float) -> Optional[float]:
    """FCF / Total Assets. ≥ 6% = good."""
    return _safe(fcf, total_assets)


def capex_coverage(operating_cf: float, capex: float) -> Optional[float]:
    """OCF / CapEx. ≥ 1.5 = self-financing; ≥ 2.9 = excellent."""
    return _safe(operating_cf, capex)


def fcf_conversion(fcf: float, operating_cf: float) -> Optional[float]:
    """FCF / OCF. ≥ 60% = good — most operating cash survives CapEx."""
    return _safe(fcf, operating_cf)


def ocf_to_current_liabilities(operating_cf: float, current_liabilities: float) -> Optional[float]:
    """OCF / Current Liabilities. < 0.2 = liquidity alarm; ≥ 0.4 = good."""
    return _safe(operating_cf, current_liabilities)


def cash_interest_coverage(operating_cf: float, interest_expense: float) -> Optional[float]:
    """OCF / Interest Expense. < 3 = debt alarm; ≥ 5 = safe; ≥ 10 = very safe."""
    return _safe(operating_cf, interest_expense)


def cash_coverage(operating_cf: float, total_debt: float) -> Optional[float]:
    """OCF / Total Debt. ≥ 0.2 = good; ≥ 0.5 = strong."""
    return _safe(operating_cf, total_debt)


def ocf_growth(ocf_now: float, ocf_prev: float) -> Optional[float]:
    """YoY OCF growth. ≥ 10% = good; ≥ 25% = strong."""
    if not ocf_prev or ocf_prev <= 0:
        return None
    return (ocf_now - ocf_prev) / ocf_prev


def fcf_growth(fcf_now: float, fcf_prev: float) -> Optional[float]:
    """YoY FCF growth. ≥ 15% = good; ≥ 50% = excellent."""
    if not fcf_prev or fcf_prev <= 0:
        return None
    return (fcf_now - fcf_prev) / fcf_prev


# ---------------------------------------------------------------------------
# 3. BALANCE SHEET — Liquidity + Capital Structure
# ---------------------------------------------------------------------------

def current_ratio(current_assets: float, current_liabilities: float) -> Optional[float]:
    """Current Assets / Current Liabilities. Benchmark ≥ 2.0."""
    return _safe(current_assets, current_liabilities)


def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> Optional[float]:
    """(Current Assets - Inventory) / Current Liabilities. Benchmark ≥ 1.0."""
    if current_assets is None:
        return None
    return _safe(current_assets - (inventory or 0), current_liabilities)


def absolute_liquidity(cash: float, current_liabilities: float) -> Optional[float]:
    """Cash / Current Liabilities."""
    return _safe(cash, current_liabilities)


def debt_to_assets(total_debt: float, total_assets: float) -> Optional[float]:
    """Total Debt / Total Assets. Benchmark ≤ 0.50; acceptable up to 0.58."""
    return _safe(total_debt, total_assets)


def debt_to_equity(total_debt: float, equity: float) -> Optional[float]:
    """Total Debt / Equity. Benchmark ≤ 1.5."""
    return _safe(total_debt, equity)


def financial_leverage(total_assets: float, equity: float) -> Optional[float]:
    """Total Assets / Equity. Benchmark ~2.5×."""
    return _safe(total_assets, equity)


def equity_ratio(equity: float, total_assets: float) -> Optional[float]:
    """Equity / Total Assets. Target ≥ 40%."""
    return _safe(equity, total_assets)


# ---------------------------------------------------------------------------
# 4. WORKING CAPITAL / CASH CONVERSION CYCLE
# ---------------------------------------------------------------------------

def dso(receivables: float, revenue: float) -> Optional[float]:
    """Days Sales Outstanding = Receivables × 365 / Revenue. Target 20–30 days."""
    if receivables is None:
        return None
    return _safe(receivables * 365, revenue)


def dio(inventory: float, cogs: float) -> Optional[float]:
    """Days Inventory Outstanding = Inventory × 365 / COGS. Target 45–50 days."""
    if inventory is None:
        return None
    return _safe(inventory * 365, cogs)


def dpo(payables: float, cogs: float) -> Optional[float]:
    """Days Payable Outstanding = Payables × 365 / COGS. Target 27–30 days."""
    if payables is None:
        return None
    return _safe(payables * 365, cogs)


def ccc(dso_days: float, dio_days: float, dpo_days: float) -> Optional[float]:
    """Cash Conversion Cycle = DSO + DIO - DPO. Target < 50 days; lower is better."""
    if any(v is None for v in (dso_days, dio_days, dpo_days)):
        return None
    return dso_days + dio_days - dpo_days


# ---------------------------------------------------------------------------
# 5. VALUATION MULTIPLES
# ---------------------------------------------------------------------------

def pe_ratio(price: float, eps_ttm: float) -> Optional[float]:
    """Price / TTM EPS. Returns None for non-positive EPS."""
    if not eps_ttm or eps_ttm <= 0:
        return None
    return _safe(price, eps_ttm)


def pb_ratio(price: float, equity_bn: float, shares_millions: float) -> Optional[float]:
    """Price / Book Value per Share.
    equity_bn in VND billions, shares_millions in millions.
    """
    if not shares_millions or shares_millions <= 0:
        return None
    bvps = (equity_bn * 1_000) / shares_millions  # VND per share
    if bvps <= 0:
        return None
    return _safe(price, bvps)


def ev_ebitda(market_cap_bn: float, debt_bn: float, cash_bn: float, ebitda_bn: float) -> Optional[float]:
    """(Market Cap + Debt - Cash) / EBITDA. All in VND billions.
    Best for capital-intensive: REE, GAS, HPG.
    """
    if not ebitda_bn or ebitda_bn <= 0:
        return None
    return _safe(market_cap_bn + debt_bn - cash_bn, ebitda_bn)


# ---------------------------------------------------------------------------
# 6. RISK FLAGS
# ---------------------------------------------------------------------------

RISK_FLAGS = {
    "liquidity":       lambda ocf, cl:    ocf / cl < 0.2 if cl else False,
    "debt_dependency": lambda ocf, int_:  ocf / int_ < 3.0 if int_ else False,
    "over_investment": lambda ocf, capex: ocf < capex,
    # ni > 0, not just truthy: a loss makes the ratio meaningless, and the sign
    # flip turned cash-burning loss-makers into "no risk here".
    "profit_quality":  lambda ocf, ni:    ocf / ni < 0.8 if (ni and ni > 0) else False,
}


def check_risk_flags(
    operating_cf: float,
    current_liabilities: float,
    interest_expense: float,
    capex: float,
    net_income: float,
) -> dict[str, bool]:
    """Return a dict of risk flags; True = warning triggered."""
    return {
        "liquidity_risk":      bool(current_liabilities and operating_cf / current_liabilities < 0.2),
        "debt_dependency_risk": bool(interest_expense and operating_cf / interest_expense < 3.0),
        "over_investment_risk": bool(capex and operating_cf < capex),
        "profit_quality_risk":  bool(net_income and net_income > 0
                                     and operating_cf / net_income < 0.8),
    }


# ---------------------------------------------------------------------------
# Scorecard coloring + DuPont decomposition (used by the Company Analysis UI)
# ---------------------------------------------------------------------------

def rating_color(val: Optional[float], good: float, ok: float, higher_better: bool = True) -> str:
    """Map a value to a green/amber/red hex color against two thresholds.

    `good`/`ok` must be in the same unit/scale as `val` (e.g. both raw fractions
    like 0.10, or both ratios like 2.0 — never mix a raw fraction against a
    percentage-point threshold).
    """
    if val is None:
        # #94a3b8 (slate-400) only reached 2.53:1 against the app's card
        # background -- below the 3:1 floor even for the large bold numbers
        # these colours are used on, so a missing value rendered as
        # near-illegible grey. This is the app's muted-text token, 5.03:1.
        return "#666f7c"
    above_good = val >= good if higher_better else val <= good
    above_ok   = val >= ok   if higher_better else val <= ok
    return "#16a34a" if above_good else ("#d97706" if above_ok else "#dc2626")


def _margin_note(level: str, value: float) -> str:
    pct = value * 100
    if level == "cao":
        return (f"Biên lợi nhuận ròng cao ({pct:.1f}%) — doanh nghiệp giữ được phần lớn doanh thu "
                 "thành lợi nhuận, thường nhờ định giá tốt hoặc kiểm soát chi phí hiệu quả.")
    if level == "trung bình":
        return f"Biên lợi nhuận ròng ở mức trung bình ({pct:.1f}%) — không nổi bật nhưng cũng không đáng lo."
    return (f"Biên lợi nhuận ròng thấp ({pct:.1f}%) — doanh nghiệp giữ lại rất ít lợi nhuận trên mỗi đồng "
             "doanh thu, dễ bị ảnh hưởng khi chi phí tăng hoặc giá bán giảm.")


def _turnover_note(level: str, value: float) -> str:
    if level == "cao":
        return (f"Hiệu suất sử dụng tài sản cao ({value:.2f}x) — mỗi đồng tài sản tạo ra nhiều doanh thu, "
                 "cho thấy tài sản được sử dụng hiệu quả.")
    if level == "trung bình":
        return f"Hiệu suất sử dụng tài sản ở mức trung bình ({value:.2f}x)."
    return (f"Hiệu suất sử dụng tài sản thấp ({value:.2f}x) — mỗi đồng tài sản chỉ tạo ra ít doanh thu, "
             "có thể do tài sản dư thừa hoặc đặc thù ngành có vòng quay vốn chậm (ngân hàng, BĐS, hạ tầng).")


def _leverage_note(level: str, value: float) -> str:
    if level == "cao":
        return (f"Đòn bẩy tài chính cao ({value:.2f}x) — công ty dùng nhiều nợ vay để tài trợ tài sản, "
                 "khuếch đại ROE nhưng cũng làm tăng rủi ro khi lãi suất tăng hoặc kinh doanh khó khăn.")
    if level == "trung bình":
        return (f"Đòn bẩy tài chính ở mức trung bình ({value:.2f}x) — cơ cấu vốn cân bằng "
                 "giữa nợ và vốn chủ sở hữu.")
    return (f"Đòn bẩy tài chính thấp ({value:.2f}x) — công ty dùng ít nợ vay, an toàn hơn nhưng "
             "cũng không tận dụng đòn bẩy để khuếch đại ROE.")


def dupont_analysis(margin: float, turnover: float, leverage: float) -> dict:
    """Decompose ROE = Net Margin x Asset Turnover x Financial Leverage.

    Classifies each factor and the resulting ROE into "cao"/"trung bình"/"thấp",
    and builds a Vietnamese commentary explaining what drives ROE — and, when ROE
    is high, whether that's margin-driven (sustainable) or leverage-driven (riskier).
    """
    roe_val = margin * turnover * leverage

    margin_lvl   = "cao" if margin   >= 0.10 else "trung bình" if margin   >= 0.05 else "thấp"
    turnover_lvl = "cao" if turnover >= 1.0  else "trung bình" if turnover >= 0.5  else "thấp"
    leverage_lvl = "cao" if leverage >= 3.0  else "trung bình" if leverage >= 2.0  else "thấp"
    roe_lvl      = "cao" if roe_val  >= 0.15 else "trung bình" if roe_val  >= 0.10 else "thấp"

    drivers = []
    if margin_lvl == "cao":
        drivers.append("biên lợi nhuận tốt")
    if turnover_lvl == "cao":
        drivers.append("quay vòng tài sản hiệu quả")
    if leverage_lvl == "cao":
        drivers.append("sử dụng nhiều vay nợ (đòn bẩy tài chính)")

    weaknesses = []
    if margin_lvl == "thấp":
        weaknesses.append("biên lợi nhuận thấp")
    if turnover_lvl == "thấp":
        weaknesses.append("hiệu suất sử dụng tài sản thấp")
    if leverage_lvl == "thấp":
        weaknesses.append("ít dùng vay nợ nên đòn bẩy không hỗ trợ thêm cho ROE")

    if roe_lvl == "cao":
        comment = (f"ROE cao chủ yếu được thúc đẩy bởi: {', '.join(drivers)}."
                   if drivers else "ROE cao nhưng không có yếu tố nào nổi bật rõ ràng.")
        if leverage_lvl == "cao" and margin_lvl != "cao":
            comment += (" ⚠️ Lưu ý: ROE cao phần lớn đến từ vay nợ chứ không phải lợi nhuận kinh doanh — "
                        "đây là tín hiệu kém bền vững hơn, vì rủi ro tăng khi lãi suất tăng hoặc kinh doanh sa sút.")
        elif margin_lvl == "cao" and leverage_lvl != "cao":
            comment += " ✅ Đây là dạng ROE cao bền vững — đến từ hiệu quả kinh doanh thực sự, không phải vay nợ nhiều."
    elif roe_lvl == "thấp":
        comment = (f"ROE thấp, chủ yếu do: {', '.join(weaknesses)}."
                   if weaknesses else
                   "ROE thấp dù không có yếu tố thành phần nào yếu rõ ràng — có thể do biến động bất thường trong kỳ.")
    else:
        comment = "ROE ở mức trung bình, không có yếu tố nào nổi bật rõ theo hướng tốt hay xấu."

    return {
        "roe": roe_val,
        "margin_level": margin_lvl,
        "turnover_level": turnover_lvl,
        "leverage_level": leverage_lvl,
        "roe_level": roe_lvl,
        "drivers": drivers,
        "weaknesses": weaknesses,
        "comment": comment,
        "margin_note": _margin_note(margin_lvl, margin),
        "turnover_note": _turnover_note(turnover_lvl, turnover),
        "leverage_note": _leverage_note(leverage_lvl, leverage),
    }
