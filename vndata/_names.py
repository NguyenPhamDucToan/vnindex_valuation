"""Column and line-item naming, kept identical to what vnstock produced.

Everything downstream -- collectors/financials.py, the web exporter, the
dashboard -- looks line items up by the `item_id` vnstock derived from VCI's
English titles ("Net revenue" -> "net_revenue"), and reads API fields by
vnstock's snake_case column names. These two functions reproduce that naming
so the DataFrames this package returns are drop-in replacements.
"""
from __future__ import annotations

import re
import unicodedata

_LEAD_NUMBERING = (
    re.compile(r"^\d+(\.\d+)*\.\s*"),          # 1.  1.1.  1.1.2.
    re.compile(r"^[IVXivx]+(\.\d+)*\.\s*"),    # I.  IV.
    re.compile(r"^[A-Za-z]\.\s*"),             # A.  b.
)


def camel_to_snake(name: str) -> str:
    """`matchPrice` -> `match_price`, `displayDate1` -> `display_date1`."""
    name = re.sub(r"[\s.\-]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"_+", "_", name.lower()).strip("_")


def item_id(title: str) -> str:
    """English line-item title -> snake_case id ("1. Cash & equivalents" -> "cash_and_equivalents")."""
    if not title or not title.strip():
        return ""
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    for pat in _LEAD_NUMBERING:
        text = pat.sub("", text)
    text = text.lower()
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"^&", "and ", text)
    text = re.sub(r"&$", " and", text)
    text = re.sub(r"['\"`]", "", text)
    text = re.sub(r"[^a-z0-9\s_-]", " ", text)
    text = re.sub(r"[\s\-/\\]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text and text[0].isdigit():
        text = f"n_{text}"
    if not text:
        text = re.sub(r"[^a-zA-Z0-9]", "", title).lower()[:20]
    return text
