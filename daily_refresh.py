"""Daily auto-refresh: fetch latest prices + recompute valuations.

Runs Mon–Fri at 18:30 (after HOSE closes at 15:00).
Pass --now to trigger immediately in addition to the schedule.

Usage:
    python daily_refresh.py          # start scheduler (blocks)
    python daily_refresh.py --now    # run once immediately, then schedule
"""
import subprocess
import sys
import os
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

logger.add("daily_refresh.log", rotation="7 days", retention="30 days", encoding="utf-8")

SCRIPT = "run_prices_then_valuations.py"
DONE_MARKER = "=== Pipeline complete ==="
RETRY_DELAY = 120   # seconds between retry attempts
MAX_ATTEMPTS = 3


def run_pipeline() -> None:
    logger.info("=== Daily refresh triggered ===")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Attempt {attempt}/{MAX_ATTEMPTS}...")
        proc = subprocess.run(
            [sys.executable, SCRIPT],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        try:
            with open("run_prices.log", encoding="utf-8", errors="ignore") as f:
                if DONE_MARKER in f.read():
                    logger.info("Daily refresh complete.")
                    return
        except FileNotFoundError:
            pass

        if attempt < MAX_ATTEMPTS:
            logger.warning(
                f"Pipeline incomplete (exit {proc.returncode}) — "
                f"retrying in {RETRY_DELAY}s..."
            )
            time.sleep(RETRY_DELAY)

    logger.error(f"Daily refresh FAILED after {MAX_ATTEMPTS} attempts.")


def run_macro_refresh() -> None:
    """Check nso.gov.vn for new CPI/GDP/trade releases and upsert into macro_indicators.

    Only the newest 1-2 listing pages are scanned each run — published macro data
    rarely changes, so this just picks up the latest monthly/quarterly release.
    The upsert is idempotent, so re-checking already-seen articles is harmless.
    """
    from collectors.macro_collector import (
        collect_cpi, collect_gdp, collect_trade, collect_exchange_rate, collect_sbv_interest_rates,
    )

    logger.info("=== Macro data refresh triggered ===")
    try:
        n_cpi = collect_cpi(max_pages=2)
        n_gdp = collect_gdp(max_pages=2)
        n_trade = collect_trade(max_pages=1)
        logger.info(f"Macro refresh complete: cpi={n_cpi}, gdp/fdi/retail={n_gdp}, trade={n_trade}")
    except Exception:
        logger.exception("Macro refresh failed")

    # vnstock/SBV-sourced replacements for stale World Bank indicators —
    # independent try/except so a failure here doesn't block the NSO refresh above.
    try:
        n_fx = collect_exchange_rate(days=14)  # weekly job, only need to top up recent days
        n_rates = collect_sbv_interest_rates()
        logger.info(f"Rate refresh complete: exchange_rate={n_fx}, sbv_rates={n_rates}")
    except Exception:
        logger.exception("Exchange rate / SBV rate refresh failed")


if __name__ == "__main__":
    if "--now" in sys.argv:
        run_pipeline()
        run_macro_refresh()

    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")
    scheduler.add_job(
        run_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=30),
        id="daily_refresh",
        name="Daily price + valuation refresh",
        misfire_grace_time=3600,    # allow up to 1h late start
    )
    # NSO publishes CPI monthly (~end of month) and GDP/trade quarterly;
    # checking weekly is plenty and cheap (1-2 pages per source).
    scheduler.add_job(
        run_macro_refresh,
        CronTrigger(day_of_week="mon", hour=3, minute=0),
        id="weekly_macro_refresh",
        name="Weekly macro data refresh (CPI/GDP/trade/FDI/retail)",
        misfire_grace_time=3600,
    )
    logger.info(
        "Scheduler started — prices Mon–Fri 18:30 ICT, macro data Mon 03:00 ICT. Ctrl+C to stop."
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
