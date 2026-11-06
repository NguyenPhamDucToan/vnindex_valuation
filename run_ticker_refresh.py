"""Weekly: refresh the HOSE ticker list and load statements for new listings.

The ticker list was only ever loaded by hand (collectors.bulk_load
--tickers-only), so a company that listed after the first load never entered
the database, and an expired covered warrant or a delisted share stayed active
and was price-fetched every day. run_financials_refresh.py cannot fill the gap:
it only re-fetches tickers that already have quarterly rows.

This runs before the weekly financials refresh:
  1. upsert the current HOSE list; tickers no longer listed are marked inactive
     (collectors.ticker_list.upsert_companies, guarded against a short reply);
  2. fetch statements for active ordinary shares that have none yet. Only
     three-letter symbols: covered warrants (CHPG2523), ETFs and funds file no
     statements, and asking VCI for 200+ empty reports a week is the kind of
     volume that gets a caller blocked.
Prices for a new ticker need nothing here -- the daily refresh loads five
years for any active ticker it has no prices for.

    python run_ticker_refresh.py
"""
from __future__ import annotations

from loguru import logger
from sqlalchemy import select

from collectors.bulk_load import _active_tickers, step1_load_tickers, step2_load_financials
from models.database import get_session
from models.schema import Financial


def _without_statements(tickers: list[str]) -> list[str]:
    with get_session() as session:
        have = {r[0] for r in session.execute(select(Financial.ticker).distinct()).all()}
    return [t for t in tickers if t not in have and len(t) == 3 and t.isalpha()]


def _short(tickers: list[str], n: int = 20) -> str:
    return ", ".join(tickers[:n]) + (f" … (+{len(tickers) - n})" if len(tickers) > n else "")


def main() -> None:
    step1_load_tickers()
    new = _without_statements(_active_tickers())
    if not new:
        logger.info("No listed share is missing statements")
        return
    logger.info(f"{len(new)} listed shares have no statements yet: {_short(new)}")
    step2_load_financials(new)
    still = _without_statements(new)
    if still:
        # Not a failure: a fresh listing may not have filed a quarter VCI serves.
        logger.warning(f"{len(still)} still without statements after fetching: {_short(still)}")


if __name__ == "__main__":
    main()
