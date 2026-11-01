"""USD/VND daily rate, from VNDirect (vnstock read it from MSN).

`Market().forex(symbol="USDVND").ohlcv(start, end)` returns
time/open/high/low/close in VND per USD.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ._http import get_json

_FX = "https://api-finfo.vndirect.com.vn/v4/currencies"
_CODES = {"USDVND": "USD_VND"}


class _Forex:
    def __init__(self, symbol: str):
        if symbol.upper() not in _CODES:
            raise ValueError(f"forex symbol {symbol!r} not supported; use one of {sorted(_CODES)}")
        self.code = _CODES[symbol.upper()]

    def ohlcv(self, start: str | None = None, end: str | None = None, **_ignored) -> pd.DataFrame:
        end = end or date.today().isoformat()
        start = start or (date.fromisoformat(end) - timedelta(days=365)).isoformat()
        rows, page = [], 1
        while True:
            j = get_json(_FX, {"q": f"code:{self.code}~tradingDate:gte:{start}~tradingDate:lte:{end}",
                               "sort": "tradingDate:asc", "size": 500, "page": page}) or {}
            rows += j.get("data") or []
            if page >= int(j.get("totalPages") or 1):
                break
            page += 1
        if not rows:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        out = pd.DataFrame({"time": pd.to_datetime(df["tradingDate"]),
                            "open": df["openPrice"], "high": df["highPrice"],
                            "low": df["lowPrice"], "close": df["closePrice"]})
        return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


class Market:
    """vnstock-compatible `Market()` -- only `.forex(...)` is provided."""

    def __init__(self, **_ignored):
        pass

    def forex(self, symbol: str = "USDVND") -> _Forex:
        return _Forex(symbol)
