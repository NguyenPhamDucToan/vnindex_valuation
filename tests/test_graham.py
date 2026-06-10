import math

import pytest

from valuation.graham import graham_number, bvps_from_financials


def test_graham_number_basic():
    # sqrt(22.5 * 5 * 50000) = sqrt(5,625,000) ~= 2371.7
    val = graham_number(eps=5, bvps=50000)
    assert val == pytest.approx(math.sqrt(22.5 * 5 * 50000))


def test_graham_number_negative_inputs_return_none():
    assert graham_number(eps=-1, bvps=50000) is None
    assert graham_number(eps=5, bvps=0) is None
    assert graham_number(eps=None, bvps=50000) is None


def test_bvps_from_financials():
    # equity 1000 bn VND, 100M shares -> bvps = 1000*1000/100 = 10,000 VND/share
    assert bvps_from_financials(equity_bn=1000, shares_millions=100) == pytest.approx(10_000.0)
    assert bvps_from_financials(equity_bn=1000, shares_millions=0) is None
    assert bvps_from_financials(equity_bn=1000, shares_millions=None) is None
