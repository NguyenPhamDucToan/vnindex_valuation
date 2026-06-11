"""Run step3 prices + compute_valuations directly (financials already done)."""
from collectors.bulk_load import _active_tickers, step3_load_prices
from collectors.compute_valuations import run as compute_all
from loguru import logger

logger.info("Starting price fetch for all tickers...")
tickers = _active_tickers()
logger.info(f"Tickers to fetch prices: {len(tickers)}")
step3_load_prices(tickers, force=False)

logger.info("Prices complete — computing valuations...")
compute_all()

logger.info("=== Pipeline complete ===")
