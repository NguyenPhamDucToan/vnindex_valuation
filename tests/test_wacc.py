import pytest

from valuation.wacc import cost_of_equity, wacc, default_wacc, RF, ERP, DEFAULT_BETA


def test_cost_of_equity_default():
    # CAPM: Rf + beta * ERP = 0.05 + 1.2 * 0.08
    assert cost_of_equity() == pytest.approx(RF + DEFAULT_BETA * ERP)


def test_cost_of_equity_custom_beta():
    assert cost_of_equity(beta=1.0, rf=0.04, erp=0.06) == pytest.approx(0.10)


def test_wacc_all_equity_equals_cost_of_equity():
    w = wacc(beta=1.2, debt_bn=0.0, equity_bn=100.0)
    assert w == pytest.approx(cost_of_equity(1.2))


def test_wacc_blends_debt_and_equity():
    w = wacc(beta=1.2, cost_of_debt=0.08, debt_bn=50.0, equity_bn=50.0, tax_rate=0.20)
    coe = cost_of_equity(1.2)
    after_tax_cod = 0.08 * (1 - 0.20)
    expected = coe * 0.5 + after_tax_cod * 0.5
    assert w == pytest.approx(expected)


def test_wacc_falls_back_to_config_when_no_capital():
    from config import DCF_DISCOUNT_RATE
    assert wacc(debt_bn=0.0, equity_bn=0.0) == DCF_DISCOUNT_RATE


def test_default_wacc_is_between_cod_and_coe():
    w = default_wacc()
    coe = cost_of_equity(DEFAULT_BETA)
    assert 0 < w < coe
