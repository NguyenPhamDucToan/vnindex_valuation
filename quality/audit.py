"""Data quality audit: stale prices, missing statements, unit outliers.

Runs after the weekday price + valuation refresh (refresh.yml). Every rule
returns findings -- (rule, severity, ticker, detail, value) -- which are
printed, appended to the `data_quality_log` table under one run_at, and
summarised for the static site by the web exporter (data/quality.json), which
shows a badge.

Rules (ordinary shares only -- three-letter tickers; covered warrants, ETFs and
funds file no statements and trade too thinly for a staleness test):

  no_prices        error  an active share with no price at all
  stale_price      warn   last close 3+ sessions behind the market
  price_unit_jump  error  a close 3x / one-third of the session before, last
                          30 days -- a scale error (VND vs thousands), since no
                          HOSE session moves more than 7%
  price_gap        warn   a 15-200% one-day move in the last 30 days -- usually
                          a bonus issue the raw series was never adjusted for
  no_financials    warn   an active share with no statements
  missing_quarter  warn   newest stored quarter older than the one whose
                          filing deadline has passed
  fin_unit_jump    error  total assets or equity 20x / one-twentieth of the
                          quarter before -- a balance sheet does not move that
                          much in a quarter, so it is a unit error
  revenue_unit     warn   revenue 1000x / one-thousandth of the quarter before.
                          Revenue alone is lumpy -- a developer books a year's
                          handovers in one quarter (DRH x600, CRV x31 in 2026-Q3)
                          -- so only a jump the size of a unit slip is flagged
  bad_balance      error  shares outstanding missing or <= 0, or total assets
                          below equity in the newest quarter
  negative_equity  warn   equity <= 0 in the newest quarter (can be genuine)
  valuations_stale error  the newest valuation run predates the newest prices

    python -m quality.audit [--dry-run]

Never exits non-zero on findings: an audit that fails the job would stop the
deploy it is meant to inform. It exits 1 only if the audit itself cannot run.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date, datetime, timedelta

from sqlalchemy import delete, func, select

from models.database import get_session
from models.schema import Company, DataQualityLog, Financial, Price, Valuation

STALE_SESSIONS = 3
RECENT_DAYS = 30
UNIT_JUMP = 3.0            # one-day close ratio that can only be a scale error
GAP = 0.15                 # one-day move beyond this is a corporate action or an error
FIN_UNIT_JUMP = 20.0
REVENUE_UNIT_JUMP = 1000.0
KEEP_DAYS = 60             # runs older than this are deleted
_MIN_TICKERS_FOR_A_SESSION = 100   # a date counts as a session once this many shares traded


def _is_share(ticker: str) -> bool:
    return len(ticker) == 3 and ticker.isalpha()


def _expected_quarter(today: date) -> str:
    from run_financials_refresh import DEFAULT_LAG_DAYS, expected_quarter
    return expected_quarter(today, DEFAULT_LAG_DAYS)


def _ratio_bad(cur, prev, limit) -> bool:
    try:
        cur, prev = float(cur), float(prev)
    except (TypeError, ValueError):
        return False
    if cur <= 0 or prev <= 0:
        return False
    r = cur / prev
    return r >= limit or r <= 1 / limit


def run_checks(session, today: date | None = None) -> list[tuple]:
    today = today or date.today()
    out: list[tuple] = []

    active = sorted(t for (t,) in session.execute(
        select(Company.ticker).where(Company.is_active == True)).all()  # noqa: E712
        if _is_share(t))

    # ── prices ───────────────────────────────────────────────────────────
    since = today - timedelta(days=90)
    sessions = sorted(d for d, n in session.execute(
        select(Price.date, func.count()).where(Price.date >= since)
        .group_by(Price.date)).all() if n >= _MIN_TICKERS_FOR_A_SESSION)
    last_px = dict(session.execute(
        select(Price.ticker, func.max(Price.date)).group_by(Price.ticker)).all())
    market_day = sessions[-1] if sessions else None

    for t in active:
        d = last_px.get(t)
        if d is None:
            out.append(("no_prices", "error", t, "active share with no price rows", None))
        elif market_day and d < market_day:
            behind = sum(1 for s in sessions if s > d)
            if behind >= STALE_SESSIONS:
                out.append(("stale_price", "warn", t,
                            f"last close {d}, {behind} sessions behind {market_day}", behind))

    recent = today - timedelta(days=RECENT_DAYS + 10)
    rows = session.execute(
        select(Price.ticker, Price.date, Price.close)
        .where(Price.date >= recent).order_by(Price.ticker, Price.date)).all()
    prev = {}
    cutoff = today - timedelta(days=RECENT_DAYS)
    for t, d, c in rows:
        p = prev.get(t)
        prev[t] = (d, c)
        if not p or not _is_share(t) or d < cutoff or not c or not p[1]:
            continue
        r = float(c) / float(p[1])
        if r >= UNIT_JUMP or r <= 1 / UNIT_JUMP:
            out.append(("price_unit_jump", "error", t,
                        f"{p[0]} {p[1]} -> {d} {c} (x{r:.2f})", round(r, 3)))
        elif abs(r - 1) > GAP:
            out.append(("price_gap", "warn", t,
                        f"{p[0]} {p[1]} -> {d} {c} ({(r - 1) * 100:+.0f}%): unadjusted action?",
                        round(r, 3)))

    # ── statements ───────────────────────────────────────────────────────
    target = _expected_quarter(today)
    quarters = session.execute(
        select(Financial.ticker, Financial.period, Financial.revenue, Financial.total_assets,
               Financial.equity, Financial.shares_outstanding)
        .where(Financial.period_type == "Q").order_by(Financial.ticker, Financial.period)).all()
    by_ticker: dict[str, list] = {}
    for row in quarters:
        by_ticker.setdefault(row[0], []).append(row)

    for t in active:
        qs = by_ticker.get(t)
        if not qs:
            out.append(("no_financials", "warn", t, "active share with no statements", None))
            continue
        newest = qs[-1]
        if newest[1] < target:
            out.append(("missing_quarter", "warn", t,
                        f"newest quarter {newest[1]}, expected {target}", None))
        if len(qs) >= 2:
            before = qs[-2]
            for idx, name, limit, rule, sev in (
                    (2, "revenue", REVENUE_UNIT_JUMP, "revenue_unit", "warn"),
                    (3, "total_assets", FIN_UNIT_JUMP, "fin_unit_jump", "error"),
                    (4, "equity", FIN_UNIT_JUMP, "fin_unit_jump", "error")):
                if _ratio_bad(newest[idx], before[idx], limit):
                    out.append((rule, sev, t,
                                f"{name} {before[1]} {before[idx]} -> {newest[1]} {newest[idx]}",
                                round(float(newest[idx]) / float(before[idx]), 3)))
        _, period, _rev, assets, equity, shares = newest
        if not shares or float(shares) <= 0:
            out.append(("bad_balance", "error", t, f"{period}: shares outstanding {shares}", None))
        if equity is not None and float(equity) <= 0:
            out.append(("negative_equity", "warn", t, f"{period}: equity {equity}", float(equity)))
        elif assets is not None and equity is not None and float(assets) < float(equity):
            out.append(("bad_balance", "error", t,
                        f"{period}: total assets {assets} below equity {equity}", None))

    # ── valuations ───────────────────────────────────────────────────────
    last_val = session.execute(select(func.max(Valuation.calc_date))).scalar()
    if market_day and (last_val is None or last_val < market_day):
        out.append(("valuations_stale", "error", None,
                    f"newest valuation run {last_val}, newest prices {market_day}", None))
    return out


def _write(session, findings: list[tuple], run_at: datetime) -> None:
    # Created on first run; additive, so it needs no migration.
    DataQualityLog.__table__.create(bind=session.get_bind(), checkfirst=True)
    # One marker row per run, so a clean run (no findings) is still a run the
    # site can show as "no issues" rather than as "never audited".
    marker = ("audit_run", "info", None, f"{len(findings)} findings", float(len(findings)))
    for rule, severity, ticker, detail, value in [marker, *findings]:
        session.add(DataQualityLog(run_at=run_at, rule=rule, severity=severity,
                                   ticker=ticker, detail=(detail or "")[:300], value=value))
    session.execute(delete(DataQualityLog).where(
        DataQualityLog.run_at < run_at - timedelta(days=KEEP_DAYS)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print findings, write nothing")
    args = ap.parse_args(argv)
    run_at = datetime.utcnow().replace(microsecond=0)
    try:
        with get_session() as session:
            findings = run_checks(session)
            if not args.dry_run:
                _write(session, findings, run_at)
    except Exception as e:                  # the audit itself broke -- say so loudly
        print(f"::error::data quality audit could not run: {e}")
        return 1

    counts = Counter((f[0], f[1]) for f in findings)
    n_err = sum(n for (_, sev), n in counts.items() if sev == "error")
    print(f"Data quality audit {run_at:%Y-%m-%d %H:%M} UTC: {len(findings)} findings, {n_err} errors"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    for (rule, sev), n in sorted(counts.items(), key=lambda kv: (kv[0][1] != "error", kv[0][0])):
        sample = [f[2] or "-" for f in findings if f[0] == rule][:12]
        print(f"  {sev:5} {rule:17} {n:4}  {', '.join(sample)}")
        if sev == "error":
            print(f"::warning::data quality: {n} x {rule}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
