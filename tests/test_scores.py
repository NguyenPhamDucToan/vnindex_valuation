"""Tests for valuation/scores.py -- pure functions, no database."""
import pytest

from valuation.scores import (
    altman_z_em, altman_zone, cagr, compute_scores, dividend_yield,
    interest_cover, net_debt_ebitda, peg, piotroski_f,
)

PREV = dict(net_income=80, total_assets=1000, operating_cf=70, debt=300,
            current_assets=400, current_liabilities=300, shares_outstanding=100,
            gross_profit=200, revenue=1000)
CUR = dict(net_income=120, total_assets=1100, operating_cf=150, debt=250,
           current_assets=500, current_liabilities=300, shares_outstanding=100,
           gross_profit=260, revenue=1200)


def test_piotroski_all_pass():
    f = piotroski_f(CUR, PREV)
    assert f["tested"] == 9 and f["score"] == 9


def test_piotroski_all_fail_mirror():
    # Swap the years and every "improved" test turns into a deterioration;
    # ROA>0, OCF>0 and the unchanged share count still pass; accrual
    # (OCF 70 < NI 80) fails.
    f = piotroski_f(PREV, CUR)
    assert f["score"] == 3
    assert f["checks"]["roa_pos"] and f["checks"]["ocf_pos"]
    assert f["checks"]["accrual"] is False


def test_piotroski_skips_financials_and_thin_data():
    assert piotroski_f(CUR, PREV, "Ngân hàng") is None
    thin = {"net_income": 1, "total_assets": 10}
    assert piotroski_f(thin, thin) is None          # fewer than 7 testable


def test_piotroski_debt_free_both_years_passes_leverage():
    c = dict(CUR, debt=0); p = dict(PREV, debt=0)
    assert piotroski_f(c, p)["checks"]["lev_down"] is True


def test_piotroski_bonus_issue_counts_as_dilution():
    c = dict(CUR, shares_outstanding=120)
    assert piotroski_f(c, PREV)["checks"]["no_dilution"] is False


def test_altman_z_formula_and_zones():
    bal = dict(total_assets=1000, equity=500, current_assets=400,
               current_liabilities=200, retained_earnings=100)
    z = altman_z_em(bal, 100)
    # 6.56*.2 + 3.26*.1 + 6.72*.1 + 1.05*1
    assert z == pytest.approx(1.312 + 0.326 + 0.672 + 1.05)
    assert altman_zone(z) == "safe"
    assert altman_zone(1.5) == "grey" and altman_zone(0.5) == "distress"
    assert altman_z_em(bal, 100, "Chứng khoán") is None
    assert altman_z_em(dict(bal, equity=1000), 100) is None   # no liabilities


def test_net_debt_ebitda_and_cover():
    assert net_debt_ebitda(300, 100, 100) == 2.0
    assert net_debt_ebitda(100, 300, 100) == -2.0      # net cash
    assert net_debt_ebitda(300, 100, -5) is None
    assert net_debt_ebitda(300, 100, 100, "Bảo hiểm") is None
    assert interest_cover(100, -20) == 5.0
    assert interest_cover(100, 0) is None


def test_peg_and_cagr():
    g = cagr(100, 172.8, 3)
    assert g == pytest.approx(0.2)
    assert peg(12, g) == pytest.approx(0.6)
    assert peg(-5, 0.2) is None and peg(12, -0.1) is None
    assert cagr(-10, 100, 3) is None


def test_dividend_yield_units():
    # 1,000bn paid, 50,000 VND x 400m shares = 20,000bn market cap -> 5%
    assert dividend_yield(1000, 50_000, 400) == pytest.approx(0.05)
    assert dividend_yield(-1000, 50_000, 400) == pytest.approx(0.05)   # cash-flow sign
    assert dividend_yield(1000, 0, 400) is None


def _q(period, **kw):
    base = dict(period=period, period_type="Q", ebit=25, ebitda=30, interest_expense=5,
                total_assets=1000, equity=500, current_assets=400, current_liabilities=200,
                retained_earnings=100, debt=200, cash=80, shares_outstanding=400)
    base.update(kw)
    return base


def test_compute_scores_end_to_end():
    fin = [_q(p) for p in ("2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1")]
    fin += [dict(PREV, period="2022", period_type="Y", net_income=50),
            dict(PREV, period="2023", period_type="Y"),
            dict(PREV, period="2024", period_type="Y"),
            dict(CUR, period="2025", period_type="Y")]
    divs = [{"year": "2024", "value": 800}, {"year": "2025", "value": 1000},
            {"year": "2026", "value": 300}]
    s = compute_scores(fin, "Bán lẻ", 50_000, pe=12, dividends=divs, today_year=2026)
    assert s["piotroski"] == 9 and s["piotroski_year"] == "2025"
    assert s["altman_zone"] == "safe"
    assert s["net_debt_ebitda"] == 1.0                 # (200-80)/120
    assert s["interest_cover"] == 5.0
    assert s["ni_cagr_3y"] == pytest.approx((120 / 50) ** (1 / 3) - 1, abs=1e-4)
    assert s["peg"] == pytest.approx(12 / (s["ni_cagr_3y"] * 100), abs=0.01)
    # 2026 is the current, unfinished year -- the yield uses 2025.
    assert s["div_year"] == "2025" and s["div_yield"] == pytest.approx(0.05)


def test_compute_scores_bank_gets_only_peg_and_yield():
    fin = [_q(p) for p in ("2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1")]
    s = compute_scores(fin, "Ngân hàng", 50_000)
    assert not {"piotroski", "altman_z", "net_debt_ebitda", "interest_cover"} & set(s)
