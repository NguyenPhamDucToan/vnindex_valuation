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
