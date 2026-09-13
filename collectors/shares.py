"""Keep the share count current between quarterly reports.

`Financial.shares_outstanding` is whatever the newest quarterly report stated,
so it is correct only on that report's balance-sheet date. A bonus issue or a
split after that date doubles the real share count while the stored one stays
put -- and the price halves the same day. Every per-share figure computed from
the pair is then wrong by the issue ratio: intrinsic value, DCF, Graham and EPS
come out too high, market cap too low.

Measured on 2026-09-13, with the reported count against VCI's live issue_share:

    TRA   41.5m -> 82.9m  (x2.000)      PHR  135.5m -> 243.9m  (x1.800)
    PVD  556.3m -> 927.7m (x1.668)      VHM  4107m  -> 8215m   (x2.000)

TRA's published upside was +88% purely because of it. The exchange's own count
is exact -- deriving a factor from the price drop is not, because the drop also
carries any cash dividend going ex on the same day (TRA's price ratio was
2.0985 against a true share ratio of 2.000).

    python -m collectors.shares            # refresh every active ticker
    python -m collectors.shares VNM TRA    # just these
"""
from __future__ import annotations

import sys
import time
from datetime import date

import pandas as pd
from loguru import logger
from sqlalchemy import select

from models.database import get_session
from models.schema import Company, Financial

_DELAY = 1.5
_RATE_LIMIT_SLEEP = 90
# A real corporate action can multiply the count several times over; a unit or
# parsing error moves it by orders of magnitude. Anything outside this is
# rejected rather than written, so a bad fetch cannot poison every valuation.
_MIN_RATIO, _MAX_RATIO = 0.5, 20.0


def fetch_issue_share(ticker: str) -> float | None:
    """Shares outstanding today, in millions, or None if unavailable."""
    from vnstock import Company as VnCompany

    ov = VnCompany(symbol=ticker, source="VCI").overview()
    if ov is None or ov.empty or "issue_share" not in ov.columns:
        return None
    # The overview frame carries several duplicate `issue_share` columns; take
    # the first that parses as a number.
    block = ov.loc[:, ov.columns == "issue_share"]
    vals = pd.to_numeric(block.iloc[0], errors="coerce").dropna()
    if vals.empty:
        return None
    raw = float(vals.iloc[0])
    if raw <= 0:
        return None
    return raw / 1e6          # Financial stores millions of shares


def _reported(session, ticker: str) -> float | None:
    row = session.execute(
        select(Financial.shares_outstanding)
        .where(Financial.ticker == ticker, Financial.period_type == "Q")
        .order_by(Financial.period.desc())
        .limit(1)
    ).first()
    return float(row[0]) if row and row[0] else None


def refresh_share_counts(tickers: list[str] | None = None) -> None:
    """Fetch and store the live share count for each ticker."""
    with get_session() as session:
        if tickers:
            names = tickers
        else:
            names = [r[0] for r in session.execute(
                select(Company.ticker).where(Company.is_active == True)  # noqa: E712
                .order_by(Company.ticker)).all()]
            # Warrants have no financial statements and no per-share valuation,
            # so their share count is never used -- do not spend calls on them.
            with_fin = {r[0] for r in session.execute(
                select(Financial.ticker).distinct()
                .where(Financial.period_type == "Q")).all()}
            names = [t for t in names if t in with_fin]

    logger.info(f"=== Refreshing share counts for {len(names)} tickers ===")
    today = date.today()
    ok = same = skip = fail = 0
    changed = []

    for i, ticker in enumerate(names, 1):
        prefix = f"[{i}/{len(names)}] {ticker}"
        value = None
        for attempt in range(1, 4):
            try:
                value = fetch_issue_share(ticker)
                break
            except SystemExit:
                logger.warning(f"{prefix}: rate limit (attempt {attempt}) — "
                               f"sleeping {_RATE_LIMIT_SLEEP}s")
                time.sleep(_RATE_LIMIT_SLEEP)
            except Exception as e:
                logger.error(f"{prefix}: FAILED — {e}")
                fail += 1
                break
        else:
            logger.error(f"{prefix}: gave up after 3 rate-limit retries")
            fail += 1

        if value is None:
            skip += 1
        else:
            with get_session() as session:
                rep = _reported(session, ticker)
                ratio = (value / rep) if rep else None
                if ratio is not None and not (_MIN_RATIO <= ratio <= _MAX_RATIO):
                    logger.warning(f"{prefix}: rejected {value:,.1f}m vs reported "
                                   f"{rep:,.1f}m (x{ratio:.2f}) — outside sanity band")
                    skip += 1
                else:
                    company = session.get(Company, ticker)
                    if company is not None:
                        company.shares_outstanding_current = value
                        company.shares_updated_at = today
                        ok += 1
                        if ratio is not None and abs(ratio - 1.0) > 0.01:
                            changed.append((ticker, rep, value, ratio))
                            logger.info(f"{prefix}: {rep:,.1f}m -> {value:,.1f}m (x{ratio:.3f})")
                        else:
                            same += 1
                    else:
                        skip += 1

        if i < len(names):
            time.sleep(_DELAY)

    logger.info(f"Share counts: {ok} stored ({same} unchanged, {len(changed)} moved), "
                f"{skip} skipped, {fail} failed")
    for t, rep, cur, ratio in changed:
        logger.info(f"  moved: {t} {rep:,.1f}m -> {cur:,.1f}m  x{ratio:.3f}")


if __name__ == "__main__":
    refresh_share_counts(sys.argv[1:] or None)
