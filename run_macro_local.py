"""Collect the nso.gov.vn macro series: CPI, GDP, investment, retail, FDI,
trade and labour. (SBV, exchange rate and World Bank are run_macro.py.)

Runs in .github/workflows/macro_refresh.yml, job "nso", weekdays 11:00 UTC,
before the web deploy at 12:30 UTC reads the database. nso.gov.vn refused
cloud IPs in July 2026; the job probes first, and if the runner is refused
again this can still be run by hand from a home connection.

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

from collectors.macro_collector import (collect_cpi, collect_gdp, collect_labor, collect_trade,
                                        collect_bank_deposit_rates, collect_sjc_gold,
                                        collect_ktxh_activity, collect_exchange_rate)

steps = [
    ("NSO CPI", lambda: collect_cpi(max_pages=2)),
    ("NSO GDP/investment/FDI/retail", lambda: collect_gdp(max_pages=2)),
    ("NSO trade", lambda: collect_trade(max_pages=1)),
    ("NSO labour", lambda: collect_labor(max_pages=1)),
    ("NSO monthly report: IIP/retail/arrivals/new firms", collect_ktxh_activity),
    # Daily rather than only in the Sunday job, so the FX chart is never a week behind.
    ("USD/VND (VNDirect)", lambda: collect_exchange_rate(days=14)),
    ("Bank deposit rates (CafeF)", collect_bank_deposit_rates),
    ("SJC gold", collect_sjc_gold),
]

failed = rows = 0
for name, fn in steps:
    try:
        n = fn() or 0
        rows += n
        logger.info(f"{name}: {n} rows upserted")
    except Exception as e:  # one source failing shouldn't stop the others
        failed += 1
        logger.warning(f"{name} failed: {e}")
logger.info(f"=== NSO macro collection done: {rows} rows, {failed} step(s) failed ===")
# Each run re-reads the newest articles, so a healthy run always upserts rows.
# Zero means every page was refused or unparseable -- fail so CI shows it red.
sys.exit(1 if failed == len(steps) or rows == 0 else 0)
