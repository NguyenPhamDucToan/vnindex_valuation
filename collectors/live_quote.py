"""Live (intraday) price snapshot via vnstock's price board.

Used by the dashboard to overlay the current market price on top of the
last stored EOD bar during trading hours.
"""
from __future__ import annotations

from datetime import datetime, time as _time, timedelta, timezone

from config import VNSTOCK_SOURCE

_ICT = timezone(timedelta(hours=7))


def is_market_hours_ict(now: datetime | None = None) -> bool:
    """True during HOSE trading hours (9:00-15:05 ICT, Mon-Fri).

    `now` may be passed (any tz-aware datetime, or naive treated as UTC)
    for testing; defaults to the current time.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_ict = now.astimezone(_ICT)
    if now_ict.weekday() >= 5:  # Sat/Sun
        return False
    return _time(9, 0) <= now_ict.time() <= _time(15, 5)


def apply_live_overlay(
    current_price: float,
    prev_close: float,
    high: float,
    low: float,
    live_quote: dict | None,
) -> dict:
    """Overlay a live quote on top of EOD price/prev-close/high/low.

    `current_price`/`prev_close`/`high`/`low` are the EOD-derived values
    (raw VND). `live_quote` is the dict returned by `fetch_live_quote`
    (values in thousands VND), or None.

    Returns a dict with keys: current_price, prev_close, high, low, as_of
    (as_of is None when no live quote is applied).
    """
    if not live_quote:
        return {
            "current_price": current_price,
            "prev_close": prev_close,
            "high": high,
            "low": low,
            "as_of": None,
        }
    return {
        "current_price": live_quote["price"] * 1000,
        "prev_close": live_quote["ref"] * 1000,
        "high": live_quote["high"] * 1000,
        "low": live_quote["low"] * 1000,
        "as_of": live_quote["as_of"],
    }


def fetch_live_quote(ticker: str) -> dict | None:
    """Return a live price snapshot for `ticker`, or None on error/no data.

    Values are in thousands VND (same convention as Price.close).
    """
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=ticker, source=VNSTOCK_SOURCE)
        df = stock.trading.price_board(symbols_list=[ticker])
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        price = float(row[("match", "match_price")])
        if not price:
            return None
        return {
            "price":  price,
            "open":   float(row[("match", "open_price")] or price),
            "high":   float(row[("match", "highest")] or price),
            "low":    float(row[("match", "lowest")] or price),
            "ref":    float(row[("match", "reference_price")] or price),
            "volume": float(row[("match", "accumulated_volume")] or 0),
            "as_of":  datetime.now(_ICT).strftime("%H:%M:%S"),
        }
    except Exception:
        return None
