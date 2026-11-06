"""quality/audit.py rules against a small in-memory database (never a real one)."""
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.schema import Base, Company, Financial, Price, Valuation
from quality.audit import run_checks

TODAY = date(2026, 11, 6)   # a Friday; Q3 ended 37 days ago, under the 40-day lag, so 2026-Q2 is expected


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _sessions(n):
    """The last n weekdays up to TODAY, oldest first."""
    out, d = [], TODAY
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _market(s, days):
    """100 filler shares trading every session, so each date counts as a session."""
    for i in range(100):
        t = f"Z{i // 26 % 26 + 65:c}{i % 26 + 65:c}"[:3]
        s.add(Company(ticker=t, name=t, exchange="HOSE", is_active=False))
        for d in days:
            s.add(Price(ticker=t, date=d, open=10, high=10, low=10, close=10, volume=1, adjusted_close=10))


def _quarters(s, t, rows):
    for period, revenue, assets, equity, shares in rows:
        s.add(Financial(ticker=t, period=period, period_type="Q", revenue=revenue,
                        total_assets=assets, equity=equity, shares_outstanding=shares))


def _rules(findings, ticker=None):
    return {f[0] for f in findings if ticker is None or f[2] == ticker}


def test_each_rule_fires_on_its_case_and_not_on_a_clean_share(session):
    days = _sessions(25)
    _market(session, days)
    good = [("2026-Q2", 100, 1000, 500, 10), ("2026-Q3", 110, 1050, 520, 10)]

    def share(t, closes, quarters=good):
        session.add(Company(ticker=t, name=t, exchange="HOSE", is_active=True))
        for d, c in zip(days[-len(closes):], closes):
            session.add(Price(ticker=t, date=d, open=c, high=c, low=c, close=c, volume=1, adjusted_close=c))
        if quarters:
            _quarters(session, t, quarters)

    flat = [10.0] * 25
    share("AAA", flat)                                              # clean
    share("NOP", [])                                                # no prices
    share("OLD", [10.0] * 10)                                       # stops 15 sessions early
    session.query(Price).filter(Price.ticker == "OLD").delete()
    for d in days[:10]:
        session.add(Price(ticker="OLD", date=d, open=10, high=10, low=10, close=10, volume=1, adjusted_close=10))
    share("UNT", flat[:-1] + [10000.0])                             # x1000 close
    share("GAP", flat[:-1] + [6.7])                                 # -33%, a bonus issue
    share("NOF", flat, quarters=None)                               # no statements
    share("LAG", flat, quarters=[("2025-Q4", 100, 1000, 500, 10), ("2026-Q1", 100, 1000, 500, 10)])
    share("BSU", flat, quarters=[("2026-Q2", 100, 1000, 500, 10), ("2026-Q3", 100, 1_000_000, 500, 10)])
    share("REV", flat, quarters=[("2026-Q2", 1, 1000, 500, 10), ("2026-Q3", 40, 1000, 500, 10)])
    share("BAD", flat, quarters=[("2026-Q2", 100, 1000, 500, 10), ("2026-Q3", 100, 300, 500, 0)])
    share("NEG", flat, quarters=[("2026-Q2", 100, 1000, 50, 10), ("2026-Q3", 100, 1000, -20, 10)])
    session.add(Valuation(ticker="AAA", calc_date=days[-1]))
    session.commit()

    f = run_checks(session, TODAY)
    assert _rules(f, "AAA") == set()
    assert _rules(f, "NOP") == {"no_prices"}
    assert "stale_price" in _rules(f, "OLD")
    assert "price_unit_jump" in _rules(f, "UNT")
    assert _rules(f, "GAP") == {"price_gap"}
    assert _rules(f, "NOF") == {"no_financials"}
    assert _rules(f, "LAG") == {"missing_quarter"}
    assert _rules(f, "BSU") == {"fin_unit_jump"}
    # A 40x revenue swing is a lumpy quarter, not a unit error.
    assert _rules(f, "REV") == set()
    assert _rules(f, "BAD") == {"bad_balance"}
    assert _rules(f, "NEG") == {"negative_equity"}
    assert "valuations_stale" not in _rules(f)
    # Inactive tickers and non-share symbols are never audited.
    assert not any(t and t.startswith("Z") for _, _, t, _, _ in f)


def test_valuations_older_than_prices_is_an_error(session):
    days = _sessions(5)
    _market(session, days)
    session.add(Valuation(ticker="AAA", calc_date=days[-2]))
    session.commit()
    f = run_checks(session, TODAY)
    assert ("valuations_stale", "error") in {(r, sev) for r, sev, *_ in f}


def test_write_appends_one_run_and_prunes_old_runs(session):
    from datetime import datetime
    from models.schema import DataQualityLog
    from quality.audit import KEEP_DAYS, _write
    now = datetime(2026, 11, 6, 10, 0)
    _write(session, [("stale_price", "warn", "OLD", "x", 5.0)], now - timedelta(days=KEEP_DAYS + 1))
    _write(session, [("no_prices", "error", "NOP", "y", None), ("valuations_stale", "error", None, "z", None)], now)
    session.commit()
    rows = session.query(DataQualityLog).all()
    # The old run was pruned; the new one carries its marker row.
    assert {r.rule for r in rows} == {"audit_run", "no_prices", "valuations_stale"}
    assert {r.run_at for r in rows} == {now}


def test_a_clean_run_still_leaves_a_marker(session):
    from datetime import datetime
    from models.schema import DataQualityLog
    from quality.audit import _write
    _write(session, [], datetime(2026, 11, 6, 10, 0))
    session.commit()
    (row,) = session.query(DataQualityLog).all()
    assert (row.rule, row.severity, row.value) == ("audit_run", "info", 0.0)
