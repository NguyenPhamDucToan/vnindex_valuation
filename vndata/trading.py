"""Live price board from VCI, in vnstock's two-level column layout.

`Trading(...).price_board([...])` returns one row per symbol with columns like
("match", "match_price") and ("listing", "reference_price"), prices in raw VND.
"""
from __future__ import annotations

import pandas as pd

from ._http import post_json
from ._names import camel_to_snake

_BOARD = "https://trading.vietcap.com.vn/api/price/symbols/getList"

_DROP = {("listing", "code"), ("listing", "symbol"), ("listing", "received_time"),
         ("listing", "message_type"), ("listing", "time"), ("match", "code"),
         ("match", "symbol"), ("match", "received_time"), ("match", "message_type"),
         ("match", "time"), ("match", "session")}


def _flatten(d: dict, prefix: str) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif not isinstance(v, list):
            out[key] = v
    return out


class Trading:
    """vnstock-compatible `Trading(symbol=...)`."""

    def __init__(self, symbol: str | None = None, source: str | None = None, **_ignored):
        self.symbol = symbol

    def price_board(self, symbols_list: list[str] | None = None, **_ignored) -> pd.DataFrame:
        symbols = [s.upper() for s in (symbols_list or [self.symbol]) if s]
        data = post_json(_BOARD, {"symbols": symbols})
        if not data:
            return pd.DataFrame()
        rows = []
        for item in data:
            row = {**_flatten(item.get("listingInfo") or {}, "listing"),
                   **_flatten(item.get("matchPrice") or {}, "match")}
            ba = item.get("bidAsk") or {}
            for side, key in (("bid", "bidPrices"), ("ask", "askPrices")):
                for i, lvl in enumerate(ba.get(key) or [], start=1):
                    row[f"bidAsk_{side}_{i}_price"] = lvl.get("price")
                    row[f"bidAsk_{side}_{i}_volume"] = lvl.get("volume")
            rows.append(row)
        df = pd.DataFrame(rows)
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(camel_to_snake(p) for p in c.split("_", 1)) for c in df.columns])
        df = df.drop(columns=[c for c in df.columns if c in _DROP])
        return df.rename(columns={"board": "exchange"}, level=1)
