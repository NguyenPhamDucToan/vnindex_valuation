"""APScheduler jobs for automated data refresh."""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import PRICE_REFRESH_HOUR, FINANCIAL_REFRESH_DAY, TZ


def refresh_prices() -> None:
    from collectors.ticker_list import fetch_ticker_list
    from collectors.prices import run as run_prices

    logger.info("Scheduled price refresh starting...")
    df = fetch_ticker_list()
    tickers = df["ticker"].tolist()
    run_prices(tickers)
    logger.info("Scheduled price refresh complete")


def refresh_financials() -> None:
    from collectors.ticker_list import fetch_ticker_list
    from collectors.financials import run as run_financials

    logger.info("Scheduled financial refresh starting...")
    df = fetch_ticker_list()
    tickers = df["ticker"].tolist()
    run_financials(tickers)
    logger.info("Scheduled financial refresh complete")


def start() -> None:
    scheduler = BlockingScheduler(timezone=TZ)

    # Daily price refresh at 5pm ICT (after market close at 3pm + settlement buffer)
    scheduler.add_job(
        refresh_prices,
        CronTrigger(hour=PRICE_REFRESH_HOUR, minute=0, timezone=TZ),
        id="daily_prices",
        name="Daily price refresh",
    )

    # Weekly financial refresh on Sunday
    scheduler.add_job(
        refresh_financials,
        CronTrigger(day_of_week=FINANCIAL_REFRESH_DAY, hour=2, minute=0, timezone=TZ),
        id="weekly_financials",
        name="Weekly financial refresh",
    )

    logger.info("Scheduler started. Press Ctrl+C to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start()
