"""Fetch all HOSE-listed tickers and upsert into the companies table."""
from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import select
from vndata import Listing

from config import EXCHANGE, VNSTOCK_SOURCE
from models.database import get_session
from models.schema import Company


def fetch_ticker_list() -> pd.DataFrame:
    """Return a DataFrame of all active HOSE tickers.

    Columns: symbol, organ_name, exchange, sector (industry_name from ICB)
    """
    logger.info("Fetching ticker list from VCI...")
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


# A listing fetch this short is a broken reply, not a market that delisted
# half its stocks; deactivating against it would stop every price refresh.
# HOSE lists ~400 ordinary shares.
_MIN_LISTED_STOCKS = 300


# VCI now serves ICB industry names, but every stored ticker and the dashboard's
# sector logic use the older house names. A new listing would otherwise sit
# alone in an ICB sector (ECO in "Hóa chất", DSH in "Xây dựng và Vật liệu")
# with no peers to compare against. Map each ICB name to its house sector;
# names with no clear house equivalent ("Truyền thông", "Dịch vụ tài chính")
# stay as they are.
ICB_TO_HOUSE = {
    "Hóa chất": "SX Nhựa - Hóa chất",
    "Xây dựng và Vật liệu": "Xây dựng",
    "Tài nguyên Cơ bản": "Vật liệu xây dựng",      # steel and sheet, as HPG/HSG/NKG
    "Hàng cá nhân & Gia dụng": "SX Hàng gia dụng",
    "Thực phẩm và đồ uống": "Thực phẩm - Đồ uống",
    "Y tế": "Chăm sóc sức khỏe",
    "Du lịch và Giải trí": "Dịch vụ lưu trú, ăn uống, giải trí",
    "Công nghệ Thông tin": "Công nghệ và thông tin",
    "Viễn thông": "Công nghệ và thông tin",
    "Bảo hiểm nhân thọ": "Bảo hiểm",
    "Bảo hiểm phi nhân thọ": "Bảo hiểm",
    "Điện, nước & xăng dầu khí đốt": "Tiện ích",
}
# Where the ICB default is wrong for the business.
TICKER_SECTOR = {
    "MZG": "SX Phụ trợ",   # Miza: paper packaging, as DHC
    "CLC": "SX Phụ trợ",   # Cát Lợi: cigarette filter materials
}


def house_sector(ticker: str, icb: str | None) -> str | None:
    if ticker in TICKER_SECTOR:
        return TICKER_SECTOR[ticker]
    return ICB_TO_HOUSE.get(icb, icb) if icb else None


def upsert_companies(df: pd.DataFrame) -> int:
    """Upsert ticker rows into the companies table. Returns count of upserted rows.

    Tickers that have left the exchange (delisted shares, expired covered
    warrants) are marked inactive, so the daily price refresh stops fetching
    them; one that comes back is reactivated. That only happens when the
    fetched list looks complete -- see _MIN_LISTED_STOCKS.
    """
    upserted = added = 0
    listed = set(df["symbol"])
    n_stocks = int((df["type"] == "STOCK").sum()) if "type" in df.columns else len(df)
    with get_session() as session:
        for _, row in df.iterrows():
            ticker = row["symbol"]  # vnstock uses 'symbol', not 'ticker'
            company = session.get(Company, ticker)
            if company is None:
                company = Company(ticker=ticker)
                session.add(company)
                added += 1
            company.name = row.get("organ_name") or row.get("short_name") or ticker
            company.exchange = row.get("exchange", EXCHANGE)
            # Only fill a blank sector, or one still holding a raw ICB name
            # that has a house equivalent. The stored house names predate the
            # ICB names and the dashboard keys on them; overwriting those would
            # re-sector tickers that are already right.
            if (not company.sector or company.sector in ICB_TO_HOUSE
                    or (ticker in TICKER_SECTOR and company.sector != TICKER_SECTOR[ticker])):
                company.sector = house_sector(
                    ticker, company.sector if company.sector in ICB_TO_HOUSE
                    else row.get("industry_name") or row.get("sector") or None)
            if not company.industry:
                company.industry = row.get("industry_name") or row.get("industry") or None
            company.is_active = True
            upserted += 1
        gone = []
        if n_stocks >= _MIN_LISTED_STOCKS:
            for company in session.execute(
                    select(Company).where(Company.is_active == True)).scalars():  # noqa: E712
                if company.ticker not in listed:
                    company.is_active = False
                    gone.append(company.ticker)
        else:
            logger.warning(f"Listing has only {n_stocks} stocks (< {_MIN_LISTED_STOCKS}); "
                           "not deactivating anything")
    logger.info(f"Upserted {upserted} companies ({added} new); "
                f"{len(gone)} no longer listed, marked inactive"
                + (f": {', '.join(sorted(gone)[:20])}" + (" …" if len(gone) > 20 else "")
                   if gone else ""))
    return upserted


def run() -> None:
    df = fetch_ticker_list()
    upsert_companies(df)


if __name__ == "__main__":
    run()
