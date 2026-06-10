import pytest

from valuation.inputs import compute_fcff_ttm
from config import TAX_RATE


def test_compute_fcff_ttm_ebit_based():
    ttm = {
        "ebit": 100.0, "depreciation": 10.0, "capex": 20.0, "delta_nwc": 5.0,
    }
    expected = 100.0 * (1 - TAX_RATE) + 10.0 - 20.0 - 5.0
    assert compute_fcff_ttm(ttm) == pytest.approx(expected)


def test_compute_fcff_ttm_falls_back_to_ocf_minus_capex():
    ttm = {"ebit": None, "operating_cf": 80.0, "capex": 30.0}
    assert compute_fcff_ttm(ttm) == pytest.approx(50.0)


def test_compute_fcff_ttm_falls_back_to_stored_fcf():
    ttm = {"ebit": None, "operating_cf": None, "fcf": 42.0}
    assert compute_fcff_ttm(ttm) == pytest.approx(42.0)


def test_compute_fcff_ttm_missing_delta_nwc_defaults_to_zero():
    ttm = {"ebit": 100.0, "depreciation": 10.0, "capex": 20.0}  # no delta_nwc key
    expected = 100.0 * (1 - TAX_RATE) + 10.0 - 20.0 - 0.0
    assert compute_fcff_ttm(ttm) == pytest.approx(expected)


def test_compute_fcff_ttm_returns_none_when_no_data():
    assert compute_fcff_ttm({}) is None
