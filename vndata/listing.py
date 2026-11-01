"""Symbol lists and industry classification from VCI."""
from __future__ import annotations

import pandas as pd

from ._http import SourceError, get_json
from ._names import camel_to_snake
from .company import IQ

_SYMBOLS = "https://trading.vietcap.com.vn/api/price/symbols/getAll"

# VCI calls HOSE "HSX"; config.EXCHANGE and every row in `companies` say HOSE.
_BOARD = {"HSX": "HOSE"}

# The sector names the app keys its sector logic on ("Ngân hàng" picks the bank
# chart of accounts, ...). ICB has no single level that spells all of them --
# brokers are "Dịch vụ tài chính" at level 2 -- so the three financial company
# types VCI flags directly are named from that flag, the rest from ICB level 2.
_COM_TYPE_SECTOR = {"NH": "Ngân hàng", "CK": "Chứng khoán", "BH": "Bảo hiểm"}


class Listing:
    """vnstock-compatible `Listing()`."""

    def __init__(self, source: str | None = None, **_ignored):
        pass

    def symbols_by_exchange(self, lang: str = "vi", **_ignored) -> pd.DataFrame:
        data = get_json(_SYMBOLS)
        if not data:
            raise SourceError("VCI returned no symbols")
        df = pd.DataFrame(data)
        df.columns = [camel_to_snake(c) for c in df.columns]
        df = df.rename(columns={"board": "exchange"}).drop(columns=["id"], errors="ignore")
        df["exchange"] = df["exchange"].replace(_BOARD)
        if lang == "vi":
            df = df[[c for c in df.columns if not c.startswith("en_")]]
        else:
            df = df.drop(columns=["organ_name", "organ_short_name"], errors="ignore")
            df.columns = [c.replace("en_", "") for c in df.columns]
        first = ["symbol", "exchange", "type"]
        return df[first + [c for c in df.columns if c not in first]].reset_index(drop=True)

    def all_symbols(self, **_ignored) -> pd.DataFrame:
        df = self.symbols_by_exchange()
        return (df[df["type"] == "STOCK"][["symbol", "organ_name"]]
                .drop_duplicates("symbol").reset_index(drop=True))

    def symbols_by_industries(self, lang: str = "vi", **_ignored) -> pd.DataFrame:
        """One row per symbol: ICB codes/names for levels 1-4, VCI's company
        type (CT/NH/CK/BH) and an `industry_name` for the companies table."""
        data = (get_json(f"{IQ}/v2/company/search-bar", {"language": 1 if lang == "vi" else 2})
                or {}).get("data")
        if not data:
            raise SourceError("VCI returned no industry list")
        rows = []
        for c in data:
            row = {"symbol": c.get("code"), "organ_name": c.get("name"),
                   "com_type_code": c.get("comTypeCode")}
            for lv in range(1, 5):
                icb = c.get(f"icbLv{lv}") or {}
                row[f"icb_code{lv}"] = icb.get("code")
                row[f"icb_name{lv}"] = icb.get("name")
            row["industry_name"] = (_COM_TYPE_SECTOR.get(row["com_type_code"])
                                    or row["icb_name2"])
            rows.append(row)
        df = pd.DataFrame(rows)
        return df[df["symbol"].notna()].drop_duplicates("symbol").reset_index(drop=True)
