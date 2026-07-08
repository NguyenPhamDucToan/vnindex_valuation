"""Fetch all HOSE-listed tickers and upsert into the companies table."""
from __future__ import annotations

import pandas as pd
from loguru import logger
from vnstock import Listing

from config import EXCHANGE, VNSTOCK_SOURCE
from models.database import get_session
from models.schema import Company


def fetch_ticker_list() -> pd.DataFrame:
    """Return a DataFrame of all active HOSE tickers.

    Columns: symbol, organ_name, exchange, sector (industry_name from ICB)
    """
    logger.info("Fetching ticker list from vnstock...")
    listing = Listing(source=VNSTOCK_SOURCE)

    df = listing.symbols_by_exchange()
    df = df[df["exchange"] == EXCHANGE].copy()
    logger.info(f"Found {len(df)} tickers on {EXCHANGE}")

    try:
        ind = listing.symbols_by_industries()[["symbol", "industry_name"]].copy()
        df = df.merge(ind, on="symbol", how="left")
        logger.info(f"Merged industry data: {df['industry_name'].notna().sum()} tickers have sector")
    except Exception as e:
        logger.warning(f"Could not fetch industry data: {e}")
        df["industry_name"] = None

    return df


def upsert_companies(df: pd.DataFrame) -> int:
    """Upsert ticker rows into the companies table. Returns count of upserted rows."""
    upserted = 0
    with get_session() as session:
        for _, row in df.iterrows():
            ticker = row["symbol"]  # vnstock uses 'symbol', not 'ticker'
            company = session.get(Company, ticker)
            if company is None:
                company = Company(ticker=ticker)
                session.add(company)
            company.name = row.get("organ_name") or row.get("short_name") or ticker
            company.exchange = row.get("exchange", EXCHANGE)
            company.sector = row.get("industry_name") or row.get("sector") or None
            company.industry = row.get("industry_name") or row.get("industry") or None
            company.is_active = True
            upserted += 1
    logger.info(f"Upserted {upserted} companies")
    return upserted


def run() -> None:
    df = fetch_ticker_list()
    upsert_companies(df)


if __name__ == "__main__":
    run()
