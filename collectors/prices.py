"""Fetch daily OHLCV price history and upsert into the prices table."""
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
from loguru import logger
from sqlalchemy import select, func
from vnstock import Vnstock

from config import VNSTOCK_SOURCE
from models.database import get_session
from models.schema import Price


_DEFAULT_HISTORY_YEARS = 5


def fetch_prices(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Return OHLCV DataFrame for a single ticker between start_date and end_date.

    Columns: date, open, high, low, close, volume, adjusted_close
    All prices in VND (not billions).
    """
    stock = Vnstock().stock(symbol=ticker, source=VNSTOCK_SOURCE)
    df = stock.quote.history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1D",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={
        "time": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # vnstock doesn't always provide adjusted close; fall back to close
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]
    return df[["date", "open", "high", "low", "close", "volume", "adjusted_close"]]


def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    """Upsert price rows for a ticker. Returns count of upserted rows."""
    if df.empty:
        return 0
    upserted = 0
    with get_session() as session:
        # Build set of existing dates to decide insert vs update
        existing = {
            r[0]
            for r in session.execute(
                select(Price.date).where(Price.ticker == ticker)
            )
        }
        for _, row in df.iterrows():
            if row["date"] in existing:
                session.execute(
                    Price.__table__.update()
                    .where(Price.ticker == ticker, Price.date == row["date"])
                    .values(
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                        adjusted_close=row["adjusted_close"],
                    )
                )
            else:
                session.add(Price(ticker=ticker, **row.to_dict()))
            upserted += 1
    return upserted


def incremental_start_date(ticker: str) -> date:
    """Return the day after the latest stored price date, or 5 years ago if none."""
    with get_session() as session:
        max_date = session.scalar(
            select(func.max(Price.date)).where(Price.ticker == ticker)
        )
    if max_date:
        return max_date + timedelta(days=1)
    return date.today() - timedelta(days=365 * _DEFAULT_HISTORY_YEARS)


def run(tickers: list[str]) -> None:
    end = date.today()
    for ticker in tickers:
        start = incremental_start_date(ticker)
        if start > end:
            logger.debug(f"{ticker}: prices up to date, skipping")
            continue
        try:
            df = fetch_prices(ticker, start, end)
            count = upsert_prices(ticker, df)
            logger.info(f"{ticker}: upserted {count} price rows")
        except Exception as e:
            logger.error(f"{ticker}: price fetch failed — {e}")


if __name__ == "__main__":
    run(["VNM", "FPT", "VIC"])
