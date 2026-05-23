"""Fetch all HOSE-listed tickers and upsert into the companies table."""
from __future__ import annotations

import pandas as pd
from loguru import logger
from vnstock import Vnstock

from config import EXCHANGE, VNSTOCK_SOURCE
from models.database import get_session
from models.schema import Company


def fetch_ticker_list() -> pd.DataFrame:
    """Return a DataFrame of all active HOSE tickers.

    Columns: ticker, name, exchange, sector, industry
    """
    logger.info("Fetching ticker list from vnstock...")
    stock = Vnstock().stock(symbol="VNM", source=VNSTOCK_SOURCE)
    df = stock.listing.symbols_by_exchange()
    df = df[df["exchange"] == EXCHANGE].copy()
    logger.info(f"Found {len(df)} tickers on {EXCHANGE}")
    return df


def upsert_companies(df: pd.DataFrame) -> int:
    """Upsert ticker rows into the companies table. Returns count of upserted rows."""
    upserted = 0
    with get_session() as session:
        for _, row in df.iterrows():
            company = session.get(Company, row["ticker"])
            if company is None:
                company = Company(ticker=row["ticker"])
                session.add(company)
            company.name = row.get("organ_name", row.get("short_name", ""))
            company.exchange = row.get("exchange", EXCHANGE)
            company.sector = row.get("sector", None)
            company.industry = row.get("industry", None)
            company.is_active = True
            upserted += 1
    logger.info(f"Upserted {upserted} companies")
    return upserted


def run() -> None:
    df = fetch_ticker_list()
    upsert_companies(df)


if __name__ == "__main__":
    run()
