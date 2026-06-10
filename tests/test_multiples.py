import pytest

from valuation.multiples import (
    pb_implied, ev_ebitda_implied, epv_implied, ps_implied,
    residual_income_implied, pocf_implied,
)
from valuation.wacc import cost_of_equity, DEFAULT_BETA


def test_pb_implied():
    # equity 1000bn, shares 1000M -> bvps = 1000*1000/1000 = 1000 VND; *1.5 = 1500
    assert pb_implied(equity_bn=1000, shares_millions=1000) == pytest.approx(1500.0)
    assert pb_implied(equity_bn=0, shares_millions=1000) is None
    assert pb_implied(equity_bn=1000, shares_millions=0) is None


def test_ev_ebitda_implied():
    # ev = 100 * 8 = 800; equity = 800 - net_debt(50) = 750
    # price = 750 * 1000 / shares(1000) = 750
    assert ev_ebitda_implied(ebitda_bn=100, net_debt_bn=50, shares_millions=1000) == pytest.approx(750.0)
    assert ev_ebitda_implied(ebitda_bn=0, net_debt_bn=50, shares_millions=1000) is None
    # equity value <= 0 -> None
    assert ev_ebitda_implied(ebitda_bn=10, net_debt_bn=1000, shares_millions=1000) is None


def test_epv_implied():
    w = cost_of_equity(DEFAULT_BETA)
    # nopat = 100 * 0.8 = 80; epv = 80/w; equity = epv - net_debt(0)
    expected_equity = (80.0 / w)
    expected_price = round(expected_equity * 1000 / 1000, 0)
    assert epv_implied(ebit_bn=100, net_debt_bn=0, shares_millions=1000, wacc=w) == pytest.approx(expected_price)
    assert epv_implied(ebit_bn=0, net_debt_bn=0, shares_millions=1000) is None


def test_ps_implied():
    # rps = 1000*1000/1000 = 1000; * 1.2 = 1200
    assert ps_implied(revenue_bn=1000, shares_millions=1000) == pytest.approx(1200.0)
    assert ps_implied(revenue_bn=0, shares_millions=1000) is None


def test_residual_income_implied_premium_when_roe_above_coe():
    coe = cost_of_equity(DEFAULT_BETA)
    # ROE well above CoE -> price > BVPS
    bvps = 1000 * 1000 / 1000  # = 1000
    price = residual_income_implied(equity_bn=1000, shares_millions=1000, net_income_bn=300, g=0.05)
    assert price is not None
    assert price > bvps


def test_residual_income_implied_returns_none_when_no_income():
    assert residual_income_implied(equity_bn=1000, shares_millions=1000, net_income_bn=0) is None
    assert residual_income_implied(equity_bn=0, shares_millions=1000, net_income_bn=100) is None


def test_pocf_implied():
    # ocf_ps = 100*1000/1000 = 100; *10 = 1000
    assert pocf_implied(operating_cf_bn=100, shares_millions=1000) == pytest.approx(1000.0)
    assert pocf_implied(operating_cf_bn=-5, shares_millions=1000) is None
