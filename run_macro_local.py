"""Collect the nso.gov.vn macro series from THIS machine.

nso.gov.vn blocks cloud IPs, so the GitHub workflow (run_macro.py) only covers
SBV, exchange rate and World Bank. CPI, GDP, investment, retail, FDI, trade and
labour come from here, run daily by a Windows scheduled task
("vnindex-macro-nso", 18:00 ICT) so the web deploy at 19:30 ICT picks them up.

Each step only scans the newest listing pages and the upsert is idempotent,
so a daily run costs a few dozen requests and re-reading old articles is
harmless.

    .venv/Scripts/python.exe run_macro_local.py
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from loguru import logger

logger.add("run_macro_local.log", rotation="2 MB", retention=5, encoding="utf-8", level="INFO")

from collectors.macro_collector import collect_cpi, collect_gdp, collect_labor, collect_trade

steps = [
    ("NSO CPI", lambda: collect_cpi(max_pages=2)),
    ("NSO GDP/investment/FDI/retail", lambda: collect_gdp(max_pages=2)),
    ("NSO trade", lambda: collect_trade(max_pages=1)),
    ("NSO labour", lambda: collect_labor(max_pages=1)),
]

failed = 0
for name, fn in steps:
    try:
        logger.info(f"{name}: {fn()} rows upserted")
    except Exception as e:  # one source failing shouldn't stop the others
        failed += 1
        logger.warning(f"{name} failed: {e}")
logger.info(f"=== Local macro collection done, {failed} step(s) failed ===")
sys.exit(1 if failed == len(steps) else 0)
