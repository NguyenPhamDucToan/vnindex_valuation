"""Bulk data loader for all HOSE tickers.

Run this once (or periodically) to populate the database with:
  1. All HOSE ticker symbols → companies table
  2. Financials (20 quarterly + 8 annual) → financials table
  3. Price history (5 years, incremental) → prices table

Usage:
    python -m collectors.bulk_load               # full pipeline
    python -m collectors.bulk_load --tickers-only # step 1 only
    python -m collectors.bulk_load --skip-prices  # skip price fetch

Rate limiting: 1.5 s between financial API calls, 1 s between price calls.
Already-loaded tickers are skipped (incremental by default).
Use --force to re-fetch tickers that already have financial data.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

from loguru import logger
from sqlalchemy import func, select

from collectors.ticker_list import fetch_ticker_list, upsert_companies
from collectors.financials import fetch_financials, upsert_financials
from collectors.prices import fetch_prices, upsert_prices, incremental_start_date
from models.database import get_session
from models.schema import Company, Financial

# VCI Guest: 20 requests/minute limit
# Financials: 2 calls/ticker + each call takes ~3s → effective ~7 req/min (safe)
# Prices: 1 call/ticker takes ~5-6s already → 2s delay keeps it ~7-8s/req (~8 req/min, safe)
# Do NOT run financials + prices in parallel — combined rate exceeds limit.
_FINANCIALS_DELAY = 1.5
_PRICES_DELAY = 2.0
_RATE_LIMIT_SLEEP = 90  # seconds to sleep when vnstock fires sys.exit() on rate limit

# Number of periods to fetch per ticker
_N_QUARTERS = 20
_N_ANNUAL = 8


def _tickers_with_financials() -> set[str]:
    """Return set of tickers that already have at least one quarterly financial row."""
    with get_session() as session:
        rows = session.execute(
            select(Financial.ticker).distinct()
            .where(Financial.period_type == "Q")
        ).all()
    return {r[0] for r in rows}


def _active_tickers() -> list[str]:
    """Return sorted list of all active HOSE company tickers."""
    with get_session() as session:
        rows = session.execute(
            select(Company.ticker)
            .where(Company.is_active == True)
            .order_by(Company.ticker)
        ).all()
    return [r[0] for r in rows]


def step1_load_tickers() -> list[str]:
    """Fetch HOSE ticker list and upsert into companies table."""
    logger.info("=== Step 1: Loading HOSE ticker list ===")
    df = fetch_ticker_list()
    upsert_companies(df)
    tickers = _active_tickers()
    logger.info(f"Companies table now has {len(tickers)} active HOSE tickers")
    return tickers


def step2_load_financials(tickers: list[str], force: bool = False) -> None:
    """Fetch financials for all tickers. Skips tickers already loaded unless force=True."""
    logger.info("=== Step 2: Loading financials ===")
    already_loaded = _tickers_with_financials()
    to_fetch = tickers if force else [t for t in tickers if t not in already_loaded]

    logger.info(f"Fetching financials for {len(to_fetch)} tickers "
                f"({len(already_loaded)} already loaded, force={force})")

    ok = skip = fail = 0
    for i, ticker in enumerate(to_fetch, 1):
        prefix = f"[{i}/{len(to_fetch)}] {ticker}"
        try:
            # Quarterly
            df_q = fetch_financials(ticker, n_periods=_N_QUARTERS, freq="quarter")
            n_q = upsert_financials(ticker, df_q)

            # Annual
            df_y = fetch_financials(ticker, n_periods=_N_ANNUAL, freq="year")
            n_y = upsert_financials(ticker, df_y)

            if n_q == 0 and n_y == 0:
                logger.warning(f"{prefix}: no data returned")
                skip += 1
            else:
                logger.info(f"{prefix}: {n_q}Q + {n_y}Y rows stored")
                ok += 1

        except Exception as e:
            logger.error(f"{prefix}: FAILED — {e}")
            fail += 1

        if i < len(to_fetch):
            time.sleep(_FINANCIALS_DELAY)

    logger.info(f"Financials done: {ok} ok, {skip} empty, {fail} failed out of {len(to_fetch)}")


def step3_load_prices(tickers: list[str], force: bool = False) -> None:
    """Fetch price history for all tickers (incremental unless force=True)."""
    logger.info("=== Step 3: Loading price history ===")
    today = date.today()

    ok = skip = fail = 0
    for i, ticker in enumerate(tickers, 1):
        prefix = f"[{i}/{len(tickers)}] {ticker}"
        start = today - timedelta(days=365 * 5) if force else incremental_start_date(ticker)
        if start > today:
            logger.debug(f"{prefix}: prices up to date, skipping")
            skip += 1
            continue

        fetched = False
        for attempt in range(1, 4):  # up to 3 attempts
            try:
                df = fetch_prices(ticker, start, today)
                count = upsert_prices(ticker, df)
                logger.info(f"{prefix}: {count} price rows stored (from {start})"
                            + (f" [attempt {attempt}]" if attempt > 1 else ""))
                ok += 1
                fetched = True
                break
            except SystemExit:
                # vnstock rate limiter calls sys.exit() — sleep and retry
                logger.warning(f"{prefix}: rate limit hit (attempt {attempt}) — "
                               f"sleeping {_RATE_LIMIT_SLEEP}s")
                time.sleep(_RATE_LIMIT_SLEEP)
            except Exception as e:
                logger.error(f"{prefix}: price fetch FAILED — {e}")
                fail += 1
                fetched = True  # don't retry non-rate-limit errors
                break

        if not fetched:
            logger.error(f"{prefix}: gave up after 3 rate-limit retries")
            fail += 1

        if i < len(tickers):
            time.sleep(_PRICES_DELAY)

    logger.info(f"Prices done: {ok} ok, {skip} skipped, {fail} failed out of {len(tickers)}")


def run(
    tickers_only: bool = False,
    skip_prices: bool = False,
    force: bool = False,
) -> None:
    """Full bulk load pipeline."""
    tickers = step1_load_tickers()

    if tickers_only:
        logger.info("--tickers-only flag set, stopping after step 1")
        return

    step2_load_financials(tickers, force=force)

    if not skip_prices:
        step3_load_prices(tickers, force=force)
    else:
        logger.info("--skip-prices flag set, skipping step 3")

    logger.info("=== Bulk load complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-load HOSE data into SQLite")
    parser.add_argument("--tickers-only", action="store_true",
                        help="Only load ticker list (step 1)")
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip price history fetch (step 3)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch tickers even if already loaded")
    args = parser.parse_args()

    run(
        tickers_only=args.tickers_only,
        skip_prices=args.skip_prices,
        force=args.force,
    )
