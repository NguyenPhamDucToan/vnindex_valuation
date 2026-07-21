import pytest

from valuation import ratios as r


def test_safe_division_handles_zero_and_none():
    assert r._safe(10, 0) is None
    assert r._safe(10, None) is None
    assert r._safe(10, 2) == pytest.approx(5.0)


def test_profitability_ratios():
    assert r.gross_margin(40, 100) == pytest.approx(0.40)
    assert r.operating_margin(15, 100) == pytest.approx(0.15)
    assert r.net_margin(8, 100) == pytest.approx(0.08)
    assert r.roe(25, 100) == pytest.approx(0.25)
    assert r.roa(12, 100) == pytest.approx(0.12)
    assert r.asset_turnover(150, 100) == pytest.approx(1.5)


def test_roic():
    # NOPAT = 100*(1-0.20) = 80; Invested Capital = 200+300-50 = 450
    assert r.roic(ebit=100, tax_rate=0.20, debt=200, equity=300, cash=50) == pytest.approx(80 / 450)
    assert r.roic(ebit=None, tax_rate=0.20, debt=200, equity=300, cash=50) is None
    # Zero invested capital -> None (via _safe), not a ZeroDivisionError
    assert r.roic(ebit=100, tax_rate=0.20, debt=0, equity=0, cash=0) is None
    # Missing debt/cash default to 0 rather than raising
    assert r.roic(ebit=100, tax_rate=0.20, debt=None, equity=300, cash=None) == pytest.approx(80 / 300)


def test_financial_leverage():
    assert r.financial_leverage(250, 100) == pytest.approx(2.5)
    assert r.financial_leverage(100, 0) is None
    assert r.financial_leverage(100, None) is None


def test_revenue_and_profit_growth():
    assert r.revenue_growth(110, 100) == pytest.approx(0.10)
    assert r.revenue_growth(100, 0) is None
    assert r.profit_growth(120, 100) == pytest.approx(0.20)
    assert r.profit_growth(100, -10) is None


def test_cash_flow_quality_ratios():
    assert r.profit_quality(120, 100) == pytest.approx(1.2)
    assert r.ocf_to_revenue(15, 100) == pytest.approx(0.15)
    assert r.fcf_margin(10, 100) == pytest.approx(0.10)
    assert r.capex_coverage(150, 100) == pytest.approx(1.5)
    assert r.fcf_conversion(60, 100) == pytest.approx(0.60)


def test_balance_sheet_ratios():
    assert r.current_ratio(200, 100) == pytest.approx(2.0)
    assert r.quick_ratio(200, 50, 100) == pytest.approx(1.5)
    assert r.debt_to_equity(150, 100) == pytest.approx(1.5)
    assert r.equity_ratio(40, 100) == pytest.approx(0.40)


def test_working_capital_cycle():
    dso = r.dso(receivables=20, revenue=365)   # 20*365/365 = 20 days
    dio = r.dio(inventory=30, cogs=365)        # 30 days
    dpo = r.dpo(payables=10, cogs=365)         # 10 days
    assert dso == pytest.approx(20.0)
    assert dio == pytest.approx(30.0)
    assert dpo == pytest.approx(10.0)
    assert r.ccc(dso, dio, dpo) == pytest.approx(40.0)
    assert r.ccc(None, dio, dpo) is None


def test_valuation_multiples():
    assert r.pe_ratio(100, 10) == pytest.approx(10.0)
    assert r.pe_ratio(100, -1) is None
    assert r.pe_ratio(100, 0) is None
    # equity 100bn, shares 1000M -> bvps = 100*1000/1000 = 100 VND, P/B = 50/100
    assert r.pb_ratio(50, equity_bn=100, shares_millions=1000) == pytest.approx(0.5)
    assert r.pb_ratio(50, equity_bn=100, shares_millions=0) is None
    # ev_ebitda: (mcap+debt-cash)/ebitda
    assert r.ev_ebitda(market_cap_bn=800, debt_bn=200, cash_bn=100, ebitda_bn=100) == pytest.approx(9.0)
    assert r.ev_ebitda(market_cap_bn=800, debt_bn=200, cash_bn=100, ebitda_bn=0) is None


def test_check_risk_flags():
    flags = r.check_risk_flags(
        operating_cf=10, current_liabilities=100,  # 0.1 < 0.2 -> liquidity risk
        interest_expense=5,                         # 10/5=2 < 3 -> debt dependency risk
        capex=20,                                    # 10 < 20 -> over-investment risk
        net_income=20,                               # 10/20=0.5 < 0.8 -> profit quality risk
    )
    assert flags == {
        "liquidity_risk": True,
        "debt_dependency_risk": True,
        "over_investment_risk": True,
        "profit_quality_risk": True,
    }

    healthy = r.check_risk_flags(
        operating_cf=100, current_liabilities=100,
        interest_expense=20, capex=20, net_income=80,
    )
    assert all(v is False for v in healthy.values())


def test_rating_color_higher_better():
    # missing data -> muted grey (#666f7c, the app's muted-text token; the
    # previous #94a3b8 only hit 2.53:1 contrast on the card background)
    assert r.rating_color(None, 0.10, 0.05) == "#666f7c"
    assert r.rating_color(0.15, 0.10, 0.05) == "#16a34a"           # above good -> green
    assert r.rating_color(0.07, 0.10, 0.05) == "#d97706"           # between ok/good -> amber
    assert r.rating_color(0.02, 0.10, 0.05) == "#dc2626"           # below ok -> red
    # Boundary values are inclusive ("good"/"ok" themselves count as passing)
    assert r.rating_color(0.10, 0.10, 0.05) == "#16a34a"
    assert r.rating_color(0.05, 0.10, 0.05) == "#d97706"


def test_rating_color_lower_better():
    # e.g. Debt/Assets: lower is safer, so the comparison direction flips
    assert r.rating_color(0.20, 0.30, 0.60, higher_better=False) == "#16a34a"
    assert r.rating_color(0.45, 0.30, 0.60, higher_better=False) == "#d97706"
    assert r.rating_color(0.80, 0.30, 0.60, higher_better=False) == "#dc2626"


def test_rating_color_thresholds_must_share_value_scale():
    """Regression test for the bug where thresholds were written in percentage
    points (e.g. 25 for 25%) but compared against a raw fraction (0.25) — every
    margin/ROE/ROA card always fell through to red because 0.05-0.50 is always
    less than 5-60. Catches any future reintroduction of that unit mismatch."""
    healthy_margin = 0.417  # 41.7% gross margin - clearly good
    assert r.rating_color(healthy_margin, 0.25, 0.15) == "#16a34a"
    # The old buggy call would have been rating_color(healthy_margin, 25, 15),
    # which incorrectly falls to red:
    assert r.rating_color(healthy_margin, 25, 15) == "#dc2626"


def test_dupont_analysis_high_roe_margin_driven_is_sustainable():
    # VNM-like: strong margin & turnover, modest leverage -> high, sustainable ROE
    result = r.dupont_analysis(margin=0.154, turnover=1.21, leverage=1.51)
    assert result["roe"] == pytest.approx(0.154 * 1.21 * 1.51)
    assert result["roe_level"] == "cao"
    assert result["margin_level"] == "cao"
    assert "biên lợi nhuận tốt" in result["comment"]
    assert "✅" in result["comment"]
    assert "⚠️" not in result["comment"]


def test_dupont_analysis_high_roe_leverage_driven_is_flagged_risky():
    # Weak margin, propped up by heavy leverage -> high ROE but flagged as risky
    result = r.dupont_analysis(margin=0.04, turnover=1.0, leverage=4.0)
    assert result["roe_level"] == "cao"
    assert result["leverage_level"] == "cao"
    assert result["margin_level"] != "cao"
    assert "⚠️" in result["comment"]
    assert "✅" not in result["comment"]


def test_dupont_analysis_low_roe_explains_the_weak_factor():
    # Regression test: VSC-like case (margin 14.3% good, turnover 0.24x weak)
    # previously mislabeled "ROE cao ben vung" even though ROE itself was only 7.1%.
    result = r.dupont_analysis(margin=0.143, turnover=0.24, leverage=2.09)
    assert result["roe"] == pytest.approx(0.143 * 0.24 * 2.09)
    assert result["roe_level"] == "thấp"
    assert result["margin_level"] == "cao"
    assert result["turnover_level"] == "thấp"
    assert "ROE thấp" in result["comment"]
    assert "hiệu suất sử dụng tài sản thấp" in result["comment"]
    assert "cao bền vững" not in result["comment"]
    assert "✅" not in result["comment"]


def test_dupont_analysis_medium_roe_is_neutral():
    result = r.dupont_analysis(margin=0.07, turnover=0.8, leverage=2.2)
    assert 0.10 <= result["roe"] < 0.15
    assert result["roe_level"] == "trung bình"
    assert "trung bình" in result["comment"]
