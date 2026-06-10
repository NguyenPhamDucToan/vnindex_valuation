import pytest

from valuation.dcf import (
    nopat, fcff_from_components, dcf_valuation, upside_pct, sensitivity_grid,
)
from config import DCF_TERMINAL_GROWTH


def test_nopat():
    assert nopat(100, tax_rate=0.20) == pytest.approx(80.0)


def test_fcff_from_components():
    # NOPAT(100*0.8=80) + D&A(10) - CapEx(20) - dNWC(5) = 65
    assert fcff_from_components(ebit=100, depreciation=10, capex=20,
                                 delta_nwc=5, tax_rate=0.20) == pytest.approx(65.0)


def test_dcf_valuation_basic_shape():
    res = dcf_valuation(
        fcff_base=100.0, net_debt_bn=50.0, shares_millions=100.0,
        fcff_growth_rate=0.10, wacc_override=0.15,
    )
    assert res["wacc"] == pytest.approx(0.15)
    assert len(res["fcff_projections"]) == 5
    assert res["enterprise_value"] > 0
    assert res["equity_value"] == pytest.approx(res["enterprise_value"] - 50.0)
    # price = equity_value(bn) / shares(M) * 1000 -> VND/share
    assert res["price_per_share"] == pytest.approx(
        round(res["equity_value"] / 100.0 * 1000, 0)
    )


def test_dcf_valuation_zero_shares_returns_empty():
    res = dcf_valuation(fcff_base=100.0, net_debt_bn=0.0, shares_millions=0)
    assert res["price_per_share"] is None
    assert res["fcff_projections"] == []


def test_dcf_valuation_wacc_below_terminal_growth_returns_empty():
    res = dcf_valuation(
        fcff_base=100.0, net_debt_bn=0.0, shares_millions=100.0,
        wacc_override=DCF_TERMINAL_GROWTH - 0.01,
    )
    assert res["price_per_share"] is None


def test_dcf_higher_growth_gives_higher_price():
    base = dict(fcff_base=100.0, net_debt_bn=0.0, shares_millions=100.0, wacc_override=0.15)
    low = dcf_valuation(**base, fcff_growth_rate=0.05)
    high = dcf_valuation(**base, fcff_growth_rate=0.10)
    assert high["price_per_share"] > low["price_per_share"]


def test_upside_pct():
    assert upside_pct(120, 100) == pytest.approx(0.20)
    assert upside_pct(80, 100) == pytest.approx(-0.20)
    assert upside_pct(100, 0) is None
    assert upside_pct(100, None) is None


def test_sensitivity_grid_shape():
    grid = sensitivity_grid(
        fcff_base=100.0, net_debt_bn=0.0, shares_millions=100.0,
        wacc_range=[0.12, 0.15], growth_range=[0.08, 0.12],
    )
    assert len(grid) == 4
    for row in grid:
        assert "wacc" in row and "growth" in row and "price" in row
        assert row["price"] > 0
