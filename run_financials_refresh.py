"""Pull in newly published quarterly reports.

The daily job (run_prices_then_valuations.py) only refreshes prices and
recomputes valuations -- its docstring says "financials already done". Nothing
ever re-fetched the statements, and step2_load_financials skips any ticker that
already has rows unless forced, so once a ticker was loaded its newest quarter
was frozen forever. That is why the database sat on 2026-Q1 through September
2026 while VCI had been serving 2026-Q2 for weeks.

This closes that gap without paying for a full sweep every week: it asks which
tickers are behind the quarter that should have been filed by now, and refetches
only those. Right after earnings season that is most of the market; the rest of
the year it is nobody, and the job finishes in seconds. A company that files
late is simply still behind next week, so it gets picked up then -- no list of
stragglers to maintain.

    python run_financials_refresh.py            # only tickers behind
    python run_financials_refresh.py --all      # every ticker, force refetch
    python run_financials_refresh.py --lag 30   # expect filings 30 days out
"""
from __future__ import annotations

import argparse
from datetime import date

from loguru import logger
from sqlalchemy import func, select

from collectors.bulk_load import step2_load_financials
from collectors.compute_valuations import run as compute_all
from collectors.shares import refresh_share_counts
from models.database import get_session
from models.schema import Financial

# Vietnamese issuers file quarterly reports within 20 days of quarter end (30
# for consolidated, 45 with an extension). 40 days is past the ordinary
# deadlines without chasing a quarter the whole market has yet to publish.
DEFAULT_LAG_DAYS = 40

# A quarter is not final when it is first filed. Vietnamese issuers publish
# unaudited quarterly figures and correct them afterwards, and the half-year and
# annual reports go through review: BMI restated 2026-Q2 total assets from
# 8,500.4bn to 8,647.5bn four days after we fetched them. Chasing only the
# tickers that are MISSING a quarter never picks that up, because they are not
# missing anything. So while a quarter is still young, the tickers that already
# have it get refetched too. Measured rate on a 19-ticker sample: one restated,
# so this is a small correction that is nonetheless invisible without it.
DEFAULT_RESTATE_DAYS = 120

_QUARTER_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def expected_quarter(today: date, lag_days: int = DEFAULT_LAG_DAYS) -> str:
    """The newest quarter whose filing deadline has passed, as '2026-Q2'."""
    year, quarter = today.year, (today.month - 1) // 3 + 1
    for _ in range(8):
        month, day = _QUARTER_END[quarter]
        if (today - date(year, month, day)).days >= lag_days:
            return f"{year}-Q{quarter}"
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
    raise RuntimeError("could not resolve an expected quarter")


def tickers_behind(target: str) -> list[str]:
    """Tickers whose newest stored quarter is older than `target`.

    Only tickers that already have quarterly rows are considered. The companies
    table also holds 238 covered warrants (CHPG2523 and the like), which have no
    financial statements at all -- including them would spend an hour a week
    fetching nothing. A genuinely new listing arrives through the full bulk load.
    """
    with get_session() as session:
        rows = session.execute(
            select(Financial.ticker, func.max(Financial.period))
            .where(Financial.period_type == "Q")
            .group_by(Financial.ticker)
            .order_by(Financial.ticker)
        ).all()
    # Periods are 'YYYY-Qn', so lexicographic order is chronological order.
    return [ticker for ticker, newest in rows if (newest or "") < target]


def _quarter_age_days(period: str):
    """Days since the balance-sheet date of a '2026-Q2' label."""
    try:
        year, quarter = period.split("-Q")
        # _QUARTER_END is keyed by int, as expected_quarter uses it -- looking it
        # up with the string straight out of split() returned None silently and
        # disabled the sweep this function gates.
        month, day = _QUARTER_END[int(quarter)]
        return (date.today() - date(int(year), month, day)).days
    except (ValueError, KeyError):
        return None


def _tickers_with_target(target: str) -> list[str]:
    """Tickers whose newest stored quarter is exactly `target`."""
    with get_session() as session:
        rows = session.execute(
            select(Financial.ticker, func.max(Financial.period))
            .where(Financial.period_type == "Q")
            .group_by(Financial.ticker)
            .order_by(Financial.ticker)
        ).all()
    return [ticker for ticker, newest in rows if newest == target]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="refetch every ticker that has financials, not just the stale ones")
    ap.add_argument("--lag", type=int, default=DEFAULT_LAG_DAYS,
                    help=f"days after quarter end before a quarter is expected (default {DEFAULT_LAG_DAYS})")
    ap.add_argument("--skip-shares", action="store_true",
                    help="do not refresh share counts (they move independently of filings)")
    ap.add_argument("--restate-days", type=int, default=DEFAULT_RESTATE_DAYS,
                    help=("refetch tickers that already have the expected quarter while it is "
                          f"younger than this many days (default {DEFAULT_RESTATE_DAYS}; 0 disables)"))
    args = ap.parse_args()

    # Share counts first, and unconditionally: a bonus issue changes every
    # per-share figure the moment it happens, with no filing to signal it. The
    # reported count only catches up at the next quarter end, and until then
    # the price is on the new base while the divisor is on the old one.
    if not args.skip_shares:
        refresh_share_counts()

    target = expected_quarter(date.today(), args.lag)
    logger.info(f"Quarter expected to be published by now: {target}")

    if args.all:
        with get_session() as session:
            rows = session.execute(
                select(Financial.ticker).distinct()
                .where(Financial.period_type == "Q")
                .order_by(Financial.ticker)
            ).all()
        todo = [r[0] for r in rows]
        logger.info(f"--all: refetching all {len(todo)} tickers")
    else:
        todo = tickers_behind(target)
        logger.info(f"{len(todo)} tickers are missing {target}")

    if todo:
        # force=True is the whole point: without it step2 skips every ticker
        # that already has any rows, which is all of them.
        step2_load_financials(todo, force=True)
    else:
        logger.info("No ticker is missing a quarter.")

    # Restatement sweep, quarterly report only -- annual figures do not move
    # between the quarterly filings this is checking, and skipping them halves
    # the API calls.
    age = _quarter_age_days(target)
    if args.restate_days and age is not None and age <= args.restate_days:
        current = [t for t in _tickers_with_target(target) if t not in set(todo)]
        if current:
            logger.info(f"{target} is {age} days old — refetching {len(current)} tickers "
                        f"that already have it, in case they have been restated")
            step2_load_financials(current, force=True, quarterly_only=True)
    elif args.restate_days:
        logger.info(f"{target} is {age} days old, past the {args.restate_days}-day "
                    f"restatement window — no refetch")

    # Always recompute: even with no new filing, a refreshed share count moves
    # every per-share figure -- TRA's DCF halved from 76,882 to 38,444 and its
    # published upside fell from +88% to +3.6% on nothing else.
    logger.info("Recomputing valuations...")
    compute_all()

    still = tickers_behind(target)
    if still:
        logger.warning(f"{len(still)} tickers still missing {target} "
                       f"(late filers or no data): {', '.join(still[:15])}"
                       + (" ..." if len(still) > 15 else ""))
    else:
        logger.info(f"Every ticker now has {target}")
    logger.info("=== Financials refresh complete ===")


if __name__ == "__main__":
    main()
