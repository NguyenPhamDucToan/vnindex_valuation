"""Financial statements from VCI, in the wide layout vnstock produced.

`Finance(...)._get_report(...)` returns what collectors/financials.py and the
web exporter were written against: one row per line item, an `item_id`
column naming it, and one column per period ("2026-Q1", ..., or "2025" for
years), values in raw VND.

The field codes in VCI's payload ("isa1", "bsa2", ...) mean different things
for different company types -- bsa2 is cash for an industrial, cash and
precious metals for a bank -- so the code -> title dictionary is fetched per
symbol from the same metrics endpoint vnstock used, and `item_id` is derived
from the English title by `_names.item_id`, which reproduces vnstock's
naming exactly.
"""
from __future__ import annotations

import pandas as pd

from ._http import SourceError, get_json
from .company import IQ
from ._names import item_id as _item_id

_SECTIONS = {
    "balance_sheet": "BALANCE_SHEET",
    "income_statement": "INCOME_STATEMENT",
    "cash_flow": "CASH_FLOW",
}

# symbol -> {field code: (vi title, en title)}. One call per symbol per process.
_METRICS: dict[str, dict[str, tuple[str, str]]] = {}


def _metrics(symbol: str) -> dict[str, tuple[str, str]]:
    if symbol not in _METRICS:
        data = (get_json(f"{IQ}/v1/company/{symbol}/financial-statement/metrics") or {}).get("data")
        if not data:
            raise SourceError(f"VCI returned no statement metrics for {symbol}")
        out = {}
        for rows in data.values():
            for r in rows:
                out.setdefault(r["field"], (r.get("titleVi"), r.get("titleEn")))
        _METRICS[symbol] = out
    return _METRICS[symbol]


def _period_label(year, length, yearly: bool) -> str:
    try:
        y, q = int(year), int(length)
    except (TypeError, ValueError):
        return "N/A"
    return f"{y}" if yearly or q >= 5 else f"{y}-Q{q}"


class Finance:
    """vnstock-compatible `Finance(symbol, period=...)` for the VCI source."""

    def __init__(self, symbol: str, period: str = "quarter", get_all: bool = True,
                 show_log: bool = False, source: str | None = None, **_ignored):
        if period not in ("year", "quarter"):
            raise ValueError("period must be 'year' or 'quarter'")
        self.symbol = symbol.upper().strip()
        self.period = period

    def _raw(self, report_type: str, period: str) -> pd.DataFrame:
        if report_type not in _SECTIONS:
            raise ValueError(f"report_type must be one of {sorted(_SECTIONS)}")
        data = (get_json(f"{IQ}/v1/company/{self.symbol}/financial-statement",
                         {"section": _SECTIONS[report_type]}) or {}).get("data")
        if data is None:
            raise SourceError(f"VCI returned no {report_type} for {self.symbol}")
        rows = data.get("years" if period == "year" else "quarters") or []
        return pd.DataFrame(rows)

    def _get_report(self, report_type: str, lang: str = "en", show_log: bool = False,
                    mode: str = "final", period: str | None = None,
                    limit: int | None = None, **_ignored) -> pd.DataFrame:
        """Items as rows, periods as columns. `limit` caps the number of
        periods, taken from the start of VCI's list (oldest first), as in vnstock."""
        period = period or self.period
        if period in ("Q", "Y"):
            period = "quarter" if period == "Q" else "year"
        raw = self._raw(report_type, period)
        if limit is not None and len(raw) > limit:
            raw = raw.head(limit)
        if mode == "raw" or raw.empty:
            return raw

        titles = _metrics(self.symbol)
        # Keep VCI's column order: where two codes share a title, the first
        # one wins every `item_id == ...` lookup, the same as under vnstock.
        item_cols = [c for c in raw.columns if c in titles]
        labels = [_period_label(y, q, period == "year")
                  for y, q in zip(raw.get("yearReport", []), raw.get("lengthReport", []))]
        wide = raw[item_cols].copy()
        wide.index = labels
        wide = wide.T
        wide.columns.name = None
        en = [titles[c][1] for c in item_cols]
        vi = [titles[c][0] for c in item_cols]
        wide.insert(0, "item_id", [_item_id(t) if t else c for t, c in zip(en, item_cols)])
        wide.insert(0, "item_en", en)
        wide.insert(0, "item", vi)
        return wide.reset_index(drop=True)

    def _report(self, report_type: str, period: str | None, dropna: bool) -> pd.DataFrame:
        df = self._get_report(report_type, period=period)
        if dropna and not df.empty:
            df = df.fillna(0).infer_objects()
            df = df.loc[:, (df != 0).any(axis=0)]
        return df

    def balance_sheet(self, period: str | None = None, lang: str = "en",
                      dropna: bool = True, **_ignored) -> pd.DataFrame:
        return self._report("balance_sheet", period, dropna)

    def income_statement(self, period: str | None = None, lang: str = "en",
                         dropna: bool = True, **_ignored) -> pd.DataFrame:
        return self._report("income_statement", period, dropna)

    def cash_flow(self, period: str | None = None, lang: str = "en",
                  dropna: bool = True, **_ignored) -> pd.DataFrame:
        return self._report("cash_flow", period, dropna)
