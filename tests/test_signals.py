import pytest

from valuation.signals import compute_quality_score, classify_signal, SIGNAL_ORDER


def test_compute_quality_score_perfect():
    score = compute_quality_score(
        roe_v=0.30, net_margin_v=0.25, profit_quality_v=2.0,
        fcf_margin_v=0.20, current_ratio_v=2.5, debt_to_equity_v=0.0,
    )
    assert score == pytest.approx(100.0)


def test_compute_quality_score_worst():
    score = compute_quality_score(
        roe_v=0.0, net_margin_v=0.0, profit_quality_v=0.0,
        fcf_margin_v=0.0, current_ratio_v=0.5, debt_to_equity_v=3.0,
    )
    assert score == pytest.approx(0.0)


def test_compute_quality_score_no_data_returns_zero():
    assert compute_quality_score(None, None, None, None, None, None) == 0.0


def test_compute_quality_score_partial_data_averages_available():
    # only ROE provided at max -> sub-score 10 -> avg 10 -> *10 = 100
    score = compute_quality_score(roe_v=0.25, net_margin_v=None, profit_quality_v=None,
                                   fcf_margin_v=None, current_ratio_v=None, debt_to_equity_v=None)
    assert score == pytest.approx(100.0)


@pytest.mark.parametrize("upside,quality,expected", [
    (0.25, 70, "Strong Buy"),   # u>=20%, q>=60
    (0.20, 60, "Strong Buy"),   # boundary
    (0.15, 50, "Buy"),          # u>=10%, q>=45
    (0.10, 45, "Buy"),          # boundary
    (0.20, 50, "Buy"),          # high upside but quality too low for Strong Buy
    (0.05, 0, "Watch"),         # u>=0%
    (0.00, 0, "Watch"),         # boundary
    (-0.05, 0, "Neutral"),      # u>=-10%
    (-0.10, 0, "Neutral"),      # boundary
    (-0.20, 0, "Reduce"),       # u>=-30%
    (-0.40, 0, "Sell"),         # u>=-50%
    (-0.60, 0, "Strong Sell"),  # u<-50%
])
def test_classify_signal(upside, quality, expected):
    assert classify_signal(upside, quality) == expected


def test_classify_signal_returns_value_in_signal_order():
    for u in [-0.8, -0.4, -0.2, -0.05, 0.05, 0.15, 0.30]:
        assert classify_signal(u, 50) in SIGNAL_ORDER
