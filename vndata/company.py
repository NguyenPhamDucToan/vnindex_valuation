"""Company overview, shareholders, officers, news and events from VCI.

Column names are vnstock's (snake_case of VCI's camelCase, plus its renames:
`issue_share`, `share_holder`, `officer_position`, ...) so callers read them
unchanged.
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timedelta

import pandas as pd

from ._http import get_json
from ._names import camel_to_snake

IQ = "https://iq.vietcap.com.vn/api/iq-insight-service"

_ALL_EVENTS = "DIV,ISS,DDIND,DDINS,DDRP,AGME,AGMR,EGME,AIS,MA,MOVE,NLIS,OTHE,RETU,SUSP"
_EVENT_DATES = ["public_date", "issue_date", "record_date", "exright_date", "display_date"]


def _snake(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [camel_to_snake(str(c)) for c in df.columns]
    return df


def _drop(df: pd.DataFrame, exact=(), prefixes=()) -> pd.DataFrame:
    keep = [c for c in df.columns
            if c not in exact and not any(str(c).startswith(p) for p in prefixes)]
    return df[keep]


def _strip_html(text):
    if not isinstance(text, str):
        return text
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", text)).split())


def _window(years: int = 10) -> tuple[str, str]:
    now = datetime.now()
    return (now - timedelta(days=365 * years)).strftime("%Y%m%d"), now.strftime("%Y%m%d")


class Company:
    """vnstock-compatible `Company(symbol=...)`."""

    def __init__(self, symbol: str, source: str | None = None, **_ignored):
        self.symbol = symbol.upper().strip()
        self._holders: list | None = None

    # -- raw calls ---------------------------------------------------------
    def _details(self) -> dict:
        return (get_json(f"{IQ}/v1/company/details", {"ticker": self.symbol}) or {}).get("data") or {}

    def _holder_list(self) -> list:
        if self._holders is None:
            data = (get_json(f"{IQ}/v1/company/{self.symbol}/shareholder") or {}).get("data")
            self._holders = data if isinstance(data, list) else []
        return self._holders

    def _paged(self, path: str, params: dict) -> list:
        data = (get_json(f"{IQ}/v1/{path}", params) or {}).get("data")
        return data.get("content", []) if isinstance(data, dict) else []

    # -- vnstock surface ---------------------------------------------------
    def overview(self) -> pd.DataFrame:
        data = self._details()
        if not data:
            return pd.DataFrame()
        df = _drop(_snake(pd.DataFrame([data])), prefixes=("en_", "__"))
        if "sector_vn" in df.columns and "sector" in df.columns:
            df = df.drop(columns=["sector_vn"])
        # Both numberOfSharesMktCap and issueShare end up as `issue_share`,
        # exactly as in vnstock; collectors/shares.py takes the first number.
        df = df.rename(columns={"vi_organ_name": "organ_name",
                                "vi_organ_short_name": "organ_short_name",
                                "profile": "company_profile",
                                "number_of_shares_mkt_cap": "issue_share",
                                "ticker": "symbol"})
        if "company_profile" in df.columns:
            df["company_profile"] = df["company_profile"].map(_strip_html)
        # Reorder by position: selecting by label would repeat each duplicate
        # `issue_share` once per occurrence.
        order = sorted(range(len(df.columns)), key=lambda i: df.columns[i] != "symbol")
        return df.iloc[:, order]

    def shareholders(self, mode: str = "detailed") -> pd.DataFrame:
        if mode != "detailed":
            raise ValueError("only mode='detailed' is supported")
        rows = self._holder_list()
        if not rows:
            return pd.DataFrame()
        df = _snake(pd.DataFrame(rows))
        df = df.rename(columns={"owner_name": "share_holder", "percentage": "share_own_percent"})
        df.insert(0, "symbol", self.symbol)
        cols = ["symbol", "share_holder", "quantity", "share_own_percent", "update_date"]
        return df[[c for c in cols if c in df.columns]]

    def officers(self, filter_by: str = "working") -> pd.DataFrame:
        if filter_by not in ("working", "resigned", "all"):
            raise ValueError("filter_by must be 'working', 'resigned' or 'all'")
        rows = self._holder_list()
        if not rows:
            return pd.DataFrame()
        df = _snake(pd.DataFrame(rows))
        if "owner_type" in df.columns:
            df = df[df["owner_type"] == "INDIVIDUAL"]
        if "position_name" in df.columns:
            df = df[df["position_name"].notna()]
        df = df.rename(columns={"owner_name": "officer_name", "position_name": "officer_position",
                                "percentage": "officer_own_percent",
                                "quantity": "officer_own_quantity"})
        df.insert(0, "symbol", self.symbol)
        cols = ["symbol", "officer_name", "officer_position", "officer_own_percent",
                "officer_own_quantity", "update_date"]
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    def news(self, page: int = 0, size: int = 50) -> pd.DataFrame:
        frm, to = _window()
        rows = self._paged("news", {"ticker": self.symbol, "fromDate": frm, "toDate": to,
                                    "languageId": 1, "page": page, "size": size})
        if not rows:
            return pd.DataFrame()
        df = _snake(pd.DataFrame(rows))
        return _drop(df, exact={"organ_code", "symbol", "__typename", "is_event", "event",
                                "create_by", "update_by", "status", "create_date",
                                "update_date", "source_code", "expert_id", "is_tranfer"})

    def events(self, event_codes: str = _ALL_EVENTS, page: int = 0, size: int = 50) -> pd.DataFrame:
        frm, to = _window()
        rows = self._paged("events", {"ticker": self.symbol, "fromDate": frm, "toDate": to,
                                      "eventCode": event_codes, "page": page, "size": size})
        if not rows:
            return pd.DataFrame()
        df = _snake(pd.DataFrame(rows))
        df = _drop(df, exact={"organ_code", "symbol", "__typename", "is_event", "event",
                              "organ_name_en", "organ_name_vi"})
        for col in _EVENT_DATES:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[col] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), df[col])
        return df.reset_index(drop=True)
