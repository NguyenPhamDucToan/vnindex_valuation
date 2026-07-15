"""Collect all macro indicators: SBV rates, credit growth, exchange rate, CPI, GDP, trade, labor + World Bank."""
from loguru import logger
from collectors.macro_collector import (
    collect_cpi,
    collect_gdp,
    collect_trade,
    collect_labor,
    collect_exchange_rate,
    collect_sbv_interest_rates,
    collect_sbv_homepage_stats,
)
from collectors.worldbank_collector import collect_worldbank_macro as collect_worldbank

steps = [
    ("SBV interest rates", collect_sbv_interest_rates),
    ("SBV credit growth", collect_sbv_homepage_stats),
    ("Exchange rate", lambda: collect_exchange_rate(days=90)),
    ("CPI", collect_cpi),
    ("GDP", collect_gdp),
    ("Trade", collect_trade),
    ("Labor", collect_labor),
    ("World Bank", collect_worldbank),
]

total = 0
for name, fn in steps:
    try:
        n = fn()
        logger.info(f"{name}: {n} rows upserted")
        total += n or 0
    except Exception as e:
        logger.warning(f"{name} failed: {e}")

logger.info(f"=== Macro collection complete: {total} total rows ===")
