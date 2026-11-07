"""Fetch daily OHLCV price history and upsert into the prices table."""
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
from loguru import logger
from sqlalchemy import select, func
from vndata import Quote

from config import VNSTOCK_SOURCE
from models.database import get_session
from models.schema import Price


_DEFAULT_HISTORY_YEARS = 5


def fetch_prices(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Return OHLCV DataFrame for a single ticker between start_date and end_date.

    Columns: date, open, high, low, close, volume, adjusted_close
    All prices in VND (not billions).
    """
    df = Quote(symbol=ticker, source=VNSTOCK_SOURCE).history(
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


_UPSERT_CHUNK = 500


def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    """Upsert price rows for a ticker. Returns count of upserted rows."""
    if df.empty:
        return 0
    upserted = 0
    with get_session() as session:
        if session.get_bind().dialect.name == "postgresql":
            # One INSERT .. ON CONFLICT per chunk: a full-history rebase is
            # ~1,200 rows, which row-by-row UPDATEs turned into minutes per ticker.
            from sqlalchemy.dialects.postgresql import insert
            rows = [{"ticker": ticker, **r} for r in df.to_dict("records")]
            for k in range(0, len(rows), _UPSERT_CHUNK):
                stmt = insert(Price.__table__).values(rows[k:k + _UPSERT_CHUNK])
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_price_ticker_date",
                    set_={c: stmt.excluded[c] for c in
                          ("open", "high", "low", "close", "volume", "adjusted_close")})
                session.execute(stmt)
            return len(rows)
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


# VCI's daily history is back-adjusted for cash dividends and share issues as
# of the day it is fetched. Appending only new days therefore leaves a seam:
# every row stored before an entitlement stays on the old scale, everything
# after is on the new one. A stretch of missed runs makes it worse -- the
# catch-up after the late-September 2026 outage put GMD's 2026-10-07 bonus
# (x1.5) at 2026-09-25, its first day, as a fake -33% session.
# So each run re-reads a short overlap, and when VCI's view of those days no
# longer matches what is stored, the ticker's whole history is reloaded.
OVERLAP_DAYS = 14
REBASE_TOL = 0.003        # a 0.3% cash dividend or more; below that the seam is noise


def stored_closes(ticker: str, since: date) -> dict:
    with get_session() as session:
        rows = session.execute(
            select(Price.date, Price.close).where(Price.ticker == ticker, Price.date >= since)
        ).all()
    return {d: float(c) for d, c in rows if c}


def rebase_ratio(stored: dict, fetched: pd.DataFrame) -> float | None:
    """Median stored/fetched close over the dates both have; None if < 3 overlap.

    1.0 means VCI still prices those days as we stored them; anything else is
    an adjustment VCI has applied since, and the stored history is stale."""
    if fetched is None or fetched.empty or not stored:
        return None
    r = [stored[d] / float(c) for d, c in zip(fetched["date"], fetched["close"])
         if d in stored and c and float(c) > 0]
    if len(r) < 3:
        return None
    r.sort()
    return r[len(r) // 2]


def needs_rebase(ratio: float | None) -> bool:
    return ratio is not None and abs(ratio - 1) > REBASE_TOL


def stored_range(ticker: str) -> tuple[date | None, date | None]:
    """(first, last) stored price date. A rebase reloads from the first, not
    from five years back, or the rows older than that keep the old scale."""
    with get_session() as session:
        lo, hi = session.execute(
            select(func.min(Price.date), func.max(Price.date)).where(Price.ticker == ticker)
        ).one()
    return lo, hi


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
