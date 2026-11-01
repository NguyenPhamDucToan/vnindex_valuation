"""Daily and intraday OHLCV: VCI first, SSI iBoard when VCI fails or is empty.

Units match what vnstock returned, which is what the `prices` table stores:
stock prices in thousands of VND (VCI sends 56300 for 56.3), index levels in
points, untouched. vnstock 5 also divided indices by 1000 and rounded them to
two decimals (VNINDEX 1,741.62 -> 1.74); that was a bug, and this does not
reproduce it.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ._http import SourceError, get_json, post_json

_VCI_CHART = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
_SSI_CHART = "https://iboard-api.ssi.com.vn/statistics/charts/history"

_ICT = timezone(timedelta(hours=7))

# Accepted spelling -> the code VCI's chart API uses.
INDEX_CODES = {
    "VNINDEX": "VNINDEX", "VNI": "VNINDEX",
    "HNX": "HNXIndex", "HNXINDEX": "HNXIndex",
    "UPCOM": "HNXUpcomIndex", "UPCOMINDEX": "HNXUpcomIndex",
    "VN30": "VN30", "VN100": "VN100", "VNMID": "VNMIDCAP", "VNMIDCAP": "VNMIDCAP",
    "VNSML": "VNSMALLCAP", "VNSMALLCAP": "VNSMALLCAP", "VNALL": "VNALLSHARE",
    "VNALLSHARE": "VNALLSHARE", "VNSI": "VNSI", "VNX50": "VNX50", "VNXALL": "VNXALL",
    "HNX30": "HNX30", "VNDIAMOND": "VNDIAMOND", "VNFINLEAD": "VNFINLEAD",
    "VNFINSELECT": "VNFINSELECT", "VNIT": "VNIT", "VNIND": "VNIND", "VNCONS": "VNCONS",
    "VNCOND": "VNCOND", "VNHEAL": "VNHEAL", "VNENE": "VNENE", "VNUTI": "VNUTI",
    "VNREAL": "VNREAL", "VNFIN": "VNFIN", "VNMAT": "VNMAT",
}
# SSI names the two non-HOSE headline indices differently.
_SSI_INDEX = {"HNXIndex": "HNXINDEX", "HNXUpcomIndex": "UPCOMINDEX"}

_TIMEFRAME = {"1m": "ONE_MINUTE", "1H": "ONE_HOUR", "1D": "ONE_DAY"}
_SSI_RESOLUTION = {"1m": "1", "1H": "60", "1D": "1D"}
# Bars per trading day, to size VCI's count-back from a date range.
_BARS_PER_DAY = {"1m": 255, "1H": 5, "1D": 1}

COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def is_index(symbol: str) -> bool:
    return symbol.upper() in INDEX_CODES


def _day(s: str | date | None, default: date) -> date:
    if s is None:
        return default
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()


def _frame(t, o, h, low, c, v, interval: str, scale: float) -> pd.DataFrame:
    if not t:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame({"time": [int(x) for x in t], "open": o, "high": h,
                       "low": low, "close": c, "volume": v})
    ts = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(_ICT).dt.tz_localize(None)
    df["time"] = ts.dt.normalize() if interval == "1D" else ts
    for col in ("open", "high", "low", "close"):
        df[col] = (pd.to_numeric(df[col], errors="coerce") / scale).round(2)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df


class Quote:
    """vnstock-compatible `Quote(symbol=...).history(start, end, interval)`."""

    def __init__(self, symbol: str, source: str | None = None, **_ignored):
        self.symbol = symbol.upper().strip()
        self.index = is_index(self.symbol)
        self.code = INDEX_CODES.get(self.symbol, self.symbol)

    def history(self, start=None, end=None, interval: str = "1D",
                count_back: int | None = None, **_ignored) -> pd.DataFrame:
        if interval not in _TIMEFRAME:
            raise ValueError(f"interval {interval!r} not supported; use one of {sorted(_TIMEFRAME)}")
        today = datetime.now(_ICT).date()
        end_d = _day(end, today)
        start_d = _day(start, end_d - timedelta(days=30))
        if start_d > end_d:
            raise ValueError("start is after end")

        df = pd.DataFrame(columns=COLUMNS)
        for fetch in (self._vci, self._ssi):
            try:
                df = fetch(start_d, end_d, interval, count_back)
            except (SourceError, KeyError, TypeError, ValueError):
                continue
            if not df.empty:
                break
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        days = df["time"].dt.date
        df = df[(days >= start_d) & (days <= end_d)]
        return (df.drop_duplicates("time", keep="last")
                  .sort_values("time").reset_index(drop=True)[COLUMNS])

    def _vci(self, start_d: date, end_d: date, interval: str, count_back):
        to = datetime(end_d.year, end_d.month, end_d.day, tzinfo=_ICT) + timedelta(days=1)
        if count_back is None:
            bdays = len(pd.bdate_range(start_d, end_d))
            count_back = bdays * _BARS_PER_DAY[interval] + 5
        data = post_json(_VCI_CHART, {"timeFrame": _TIMEFRAME[interval], "symbols": [self.code],
                                      "to": int(to.timestamp()), "countBack": int(count_back)})
        if isinstance(data, dict):
            data = data.get("data")
        if not data:
            return pd.DataFrame(columns=COLUMNS)
        d = data[0]
        return _frame(d["t"], d["o"], d["h"], d["l"], d["c"], d["v"], interval,
                      1 if self.index else 1000)

    def _ssi(self, start_d: date, end_d: date, interval: str, _count_back):
        frm = datetime(start_d.year, start_d.month, start_d.day, tzinfo=_ICT)
        to = datetime(end_d.year, end_d.month, end_d.day, tzinfo=_ICT) + timedelta(days=1)
        j = get_json(_SSI_CHART, {"resolution": _SSI_RESOLUTION[interval],
                                  "symbol": _SSI_INDEX.get(self.code, self.code),
                                  "from": int(frm.timestamp()), "to": int(to.timestamp())})
        d = (j or {}).get("data") or {}
        if d.get("s") != "ok":
            return pd.DataFrame(columns=COLUMNS)
        # SSI already quotes stocks in thousands of VND.
        return _frame(d["t"], d["o"], d["h"], d["l"], d["c"], d["v"], interval, 1)
