"""Full data collection pipeline for HOSE universe.

Runs steps in sequence (never parallel — VCI Guest = 20 req/min):
  1. Load ticker list        (~5s)
  2. Fetch financials        (~90 min for 637 tickers, skips already-loaded)
  3. Fetch price history     (~40 min for 641 tickers, incremental, 3.5s delay)
  4. Compute valuations      (~30s, reads from DB)

Usage:
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --from-prices    # skip financials, start from prices
    python run_pipeline.py --valuations-only
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from loguru import logger

from collectors.bulk_load import (
    step1_load_tickers, step2_load_financials,
    step3_load_prices, _active_tickers,
)
from collectors.compute_valuations import run as compute_all


def _banner(msg: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {msg}")
    logger.info("=" * 60)


def run_full(from_prices: bool = False, valuations_only: bool = False) -> None:
    t0 = datetime.now()

    if valuations_only:
        _banner("Step 4: Computing valuations")
        compute_all()
        _banner(f"Done in {(datetime.now()-t0).seconds}s")
        return

    if from_prices:
        tickers = _active_tickers()
        logger.info(f"--from-prices: using {len(tickers)} existing tickers from DB")
    else:
        _banner("Step 1: Ticker list")
        tickers = step1_load_tickers()

        _banner("Step 2: Financials  (20Q + 8Y per ticker, ~90 min)")
        step2_load_financials(tickers, force=False)

    _banner("Step 3: Price history  (incremental, 5-year, ~40 min)")
    step3_load_prices(tickers, force=False)

    _banner("Step 4: Computing valuations")
    compute_all()

    elapsed = (datetime.now() - t0).seconds
    _banner(f"Pipeline complete in {elapsed//60}m {elapsed%60}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-prices", action="store_true",
                        help="Skip financials — start from price fetch")
    parser.add_argument("--valuations-only", action="store_true",
                        help="Recompute valuations only (no API calls)")
    args = parser.parse_args()
    run_full(from_prices=args.from_prices, valuations_only=args.valuations_only)
