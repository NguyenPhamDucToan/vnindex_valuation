"""Composite health scores and a few ratios the scorecard did not yet carry.

Pure functions over plain dicts shaped like `Financial` rows (VND billions,
shares in millions, prices in VND). Each returns None rather than a number it
cannot stand behind: a missing field, a zero denominator, or a sector the score
was never designed for.

  piotroski_f      9-point F-score from two consecutive annual reports
  altman_z_em      Altman Z''-score (1995, non-manufacturers / emerging markets)
  net_debt_ebitda  (debt - cash) / TTM EBITDA
  interest_cover   TTM EBIT / TTM interest expense
  peg              P/E over the 3-year net-income CAGR in percent
  dividend_yield   last full year's cash dividends paid / market cap

Banks, brokers and insurers are excluded from Piotroski, Altman and net
debt/EBITDA: their debt is their raw material, they have no current/non-current
split, and the scores were fitted on industrial companies.
"""
from __future__ import annotations

from typing import Optional

FINANCIAL_SECTORS = ("Ngân hàng", "Chứng khoán", "Bảo hiểm")

# Z'' zones (Altman 1995, without the +3.25 emerging-market constant).
ALTMAN_SAFE = 2.6
ALTMAN_DISTRESS = 1.1


def _num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f else None          # NaN -> None


def _div(a, b) -> Optional[float]:
    a, b = _num(a), _num(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _is_financial(sector: str | None) -> bool:
    return (sector or "") in FINANCIAL_SECTORS


def piotroski_f(cur: dict, prev: dict, sector: str | None = None) -> Optional[dict]:
    """Piotroski (2000) F-score on annual rows `cur` (year t) and `prev` (t-1).

    Returns {"score": passed, "tested": n, "checks": {name: bool|None}} or None
    when fewer than 7 of the 9 tests can be run. `score` counts passes among the
    tests that could be run; `tested` says how many that was.

    Test 7 ("no new shares") fails on a bonus issue too, which in Vietnam is
    common and raises no money. The figure is the textbook one; the caller's
    legend says so.
    """
    if _is_financial(sector) or not cur or not prev:
        return None

    roa_c = _div(cur.get("net_income"), cur.get("total_assets"))
    roa_p = _div(prev.get("net_income"), prev.get("total_assets"))
    ocf_c = _num(cur.get("operating_cf"))
    ni_c = _num(cur.get("net_income"))
    lev_c = _div(cur.get("debt") or 0.0, cur.get("total_assets"))
    lev_p = _div(prev.get("debt") or 0.0, prev.get("total_assets"))
    cr_c = _div(cur.get("current_assets"), cur.get("current_liabilities"))
    cr_p = _div(prev.get("current_assets"), prev.get("current_liabilities"))
    sh_c, sh_p = _num(cur.get("shares_outstanding")), _num(prev.get("shares_outstanding"))
    gm_c = _div(cur.get("gross_profit"), cur.get("revenue"))
    gm_p = _div(prev.get("gross_profit"), prev.get("revenue"))
    at_c = _div(cur.get("revenue"), cur.get("total_assets"))
    at_p = _div(prev.get("revenue"), prev.get("total_assets"))

    def _cmp(a, b, op):
        if a is None or b is None:
            return None
        return op(a, b)

    checks = {
        "roa_pos": None if roa_c is None else roa_c > 0,
        "ocf_pos": None if ocf_c is None else ocf_c > 0,
        "roa_up": _cmp(roa_c, roa_p, lambda a, b: a > b),
        "accrual": _cmp(ocf_c, ni_c, lambda a, b: a > b),
        # Debt-free both years counts as a pass: leverage did not rise.
        "lev_down": _cmp(lev_c, lev_p, lambda a, b: a < b or (a == 0 and b == 0)),
        "cr_up": _cmp(cr_c, cr_p, lambda a, b: a > b),
        # Half a percent of slack for rounding in the reported share count.
        "no_dilution": _cmp(sh_c, sh_p, lambda a, b: a <= b * 1.005),
        "gm_up": _cmp(gm_c, gm_p, lambda a, b: a > b),
        "at_up": _cmp(at_c, at_p, lambda a, b: a > b),
    }
    tested = [v for v in checks.values() if v is not None]
    if len(tested) < 7:
        return None
    return {"score": sum(1 for v in tested if v), "tested": len(tested), "checks": checks}


def altman_z_em(bal: dict, ebit_ttm, sector: str | None = None) -> Optional[float]:
    """Altman Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4.

    X1 working capital / assets, X2 retained earnings / assets, X3 EBIT / assets,
    X4 book equity / total liabilities. The book-equity form, so it needs no
    share price and suits companies that are not manufacturers. Zones:
    >= 2.6 safe, 1.1-2.6 grey, < 1.1 distress.
    """
    if _is_financial(sector) or not bal:
        return None
    ta = _num(bal.get("total_assets"))
    eq = _num(bal.get("equity"))
    ca, cl = _num(bal.get("current_assets")), _num(bal.get("current_liabilities"))
    re_ = _num(bal.get("retained_earnings"))
    ebit = _num(ebit_ttm)
    if not ta or ta <= 0 or eq is None or ca is None or cl is None or re_ is None or ebit is None:
        return None
    liab = ta - eq
    if liab <= 0:
        return None
    x1 = (ca - cl) / ta
    x2 = re_ / ta
    x3 = ebit / ta
    x4 = eq / liab
    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def altman_zone(z: Optional[float]) -> Optional[str]:
    if z is None:
        return None
    if z >= ALTMAN_SAFE:
        return "safe"
    if z >= ALTMAN_DISTRESS:
        return "grey"
    return "distress"


def net_debt_ebitda(debt, cash, ebitda_ttm, sector: str | None = None) -> Optional[float]:
    """(debt - cash) / EBITDA. Negative = net cash. None if EBITDA <= 0."""
    if _is_financial(sector):
        return None
    e = _num(ebitda_ttm)
    if e is None or e <= 0:
        return None
    return ((_num(debt) or 0.0) - (_num(cash) or 0.0)) / e


def interest_cover(ebit_ttm, interest_ttm, sector: str | None = None) -> Optional[float]:
    """EBIT / interest expense (interest taken as a positive cost)."""
    if _is_financial(sector):
        return None
    i = _num(interest_ttm)
    e = _num(ebit_ttm)
    if e is None or i is None or i == 0:
        return None
    return e / abs(i)


def cagr(first, last, years: int) -> Optional[float]:
    a, b = _num(first), _num(last)
    if a is None or b is None or a <= 0 or b <= 0 or years <= 0:
        return None
    return (b / a) ** (1.0 / years) - 1.0


def peg(pe, growth) -> Optional[float]:
    """P/E divided by growth in percent (0.15 -> 15). None unless both positive."""
    p, g = _num(pe), _num(growth)
    if p is None or g is None or p <= 0 or g <= 0:
        return None
    return p / (g * 100.0)


def dividend_yield(cash_dividends_bn, price_vnd, shares_millions) -> Optional[float]:
    """Cash dividends paid in a year / market cap. Dividends in VND billions."""
    d, p, s = _num(cash_dividends_bn), _num(price_vnd), _num(shares_millions)
    if d is None or p is None or s is None or p <= 0 or s <= 0:
        return None
    mcap_bn = p * s / 1_000.0          # VND x millions of shares -> billions
    return abs(d) / mcap_bn


def _sum4(rows: list[dict], key: str) -> Optional[float]:
    vals = [_num(r.get(key)) for r in rows[-4:]]
    if len(vals) < 4 or any(v is None for v in vals):
        return None
    return sum(vals)


def compute_scores(financials: list[dict], sector: str | None, price_vnd,
                   pe=None, dividends: list[dict] | None = None,
                   today_year: int | None = None) -> dict:
    """All of the above from one ticker's exported rows.

    `financials` is the list of Financial-shaped dicts (period, period_type Q/Y);
    `dividends` the [{"year": "2025", "value": bn}] cash-dividend series.
    Missing inputs leave a key out instead of writing None everywhere.
    """
    qs = sorted((r for r in financials if r.get("period_type") == "Q"), key=lambda r: r["period"])
    ys = sorted((r for r in financials if r.get("period_type") == "Y"), key=lambda r: str(r["period"]))
    out: dict = {}

    if len(ys) >= 2:
        f = piotroski_f(ys[-1], ys[-2], sector)
        if f:
            out["piotroski"] = f["score"]
            out["piotroski_tested"] = f["tested"]
            out["piotroski_checks"] = f["checks"]
            out["piotroski_year"] = str(ys[-1]["period"])

    last_q = qs[-1] if qs else None
    ebit = _sum4(qs, "ebit")
    ebitda = _sum4(qs, "ebitda")
    interest = _sum4(qs, "interest_expense")
    if last_q:
        z = altman_z_em(last_q, ebit, sector)
        if z is not None:
            out["altman_z"] = round(z, 2)
            out["altman_zone"] = altman_zone(z)
        nd = net_debt_ebitda(last_q.get("debt"), last_q.get("cash"), ebitda, sector)
        if nd is not None:
            out["net_debt_ebitda"] = round(nd, 2)
        ic = interest_cover(ebit, interest, sector)
        if ic is not None:
            out["interest_cover"] = round(ic, 2)

    # Growth over three annual reports, on net income: EPS is distorted by
    # bonus issues, which VCI does not restate backwards.
    if len(ys) >= 4:
        g = cagr(ys[-4].get("net_income"), ys[-1].get("net_income"), 3)
        if g is not None:
            out["ni_cagr_3y"] = round(g, 4)
            p = peg(pe, g)
            if p is not None:
                out["peg"] = round(p, 2)

    if dividends and last_q:
        full = [d for d in dividends
                if today_year is None or str(d.get("year")) < str(today_year)]
        if full:
            latest = max(full, key=lambda d: str(d.get("year")))
            y = dividend_yield(latest.get("value"), price_vnd, last_q.get("shares_outstanding"))
            if y is not None:
                out["div_yield"] = round(y, 4)
                out["div_year"] = str(latest.get("year"))
    return out
