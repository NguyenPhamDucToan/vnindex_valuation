"""Re-fetch financials for all Ngan hang (banking) tickers in the DB.

Rate limit: vnstock guest = 20 req/min. Each ticker needs 2 freq × 3 reports = 6 calls.
Sleep 5s between each (ticker, freq) to stay safe (~12 calls/min).
"""
import sys
import time
sys.path.insert(0, ".")
import sqlalchemy as sa
from models.database import get_session
from models.schema import Company
from collectors.financials import fetch_financials, upsert_financials
from loguru import logger

with get_session() as s:
    rows = s.execute(
        sa.select(Company.ticker)
        .where(Company.sector == "Ngân hàng")
    ).all()
    tickers = [r[0] for r in rows]

print(f"Banking tickers ({len(tickers)}): {tickers}")

DELAY = 5  # seconds between each (ticker, freq) fetch

done = 0
for ticker in tickers:
    for freq, n in [("quarter", 20), ("year", 8)]:
        attempt = 0
        while attempt < 3:
            try:
                df = fetch_financials(ticker, n_periods=n, freq=freq)
                if df.empty:
                    logger.warning(f"{ticker}/{freq}: no data")
                else:
                    count = upsert_financials(ticker, df)
                    logger.success(f"{ticker}/{freq}: {count} rows")
                break
            except SystemExit:
                # vnstock raises SystemExit on rate-limit
                wait = 45
                logger.warning(f"{ticker}/{freq}: rate-limited — waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                attempt += 1
            except Exception as e:
                logger.error(f"{ticker}/{freq}: {e}")
                break
        time.sleep(DELAY)
    done += 1
    print(f"Progress: {done}/{len(tickers)} tickers done")

print("Done.")
