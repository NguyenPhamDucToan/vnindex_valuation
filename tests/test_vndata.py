"""vndata reshapes VCI/SSI/VNDirect JSON into the frames vnstock used to return.

No network: every source call is patched with a canned payload shaped like the
real one, so these pin the contract the collectors depend on -- item_id naming,
the items-as-rows statement layout, and price units.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from vndata import Company, Finance, Quote
from vndata import finance as vfinance
from vndata._names import camel_to_snake, item_id


# Pairs checked against vnstock 4.0.4's own normalizer on all ~1,500 VCI titles.
@pytest.mark.parametrize("title, expected", [
    ("Net revenue", "net_revenue"),
    ("1. Cash & cash equivalents", "cash_and_cash_equivalents"),
    ("NET ACCOUNTING PROFIT/(LOSS) BEFORE TAX", "net_accounting_profit_loss_before_tax"),
    ("Shareholders' equity", "shareholders_equity"),
    ("A. ASSETS", "assets"),
    ("IV. Inventories", "inventories"),
    ("Financial assets at fair value through profit or loss (FVTPL)",
     "financial_assets_at_fair_value_through_profit_or_loss_fvtpl"),
])
def test_item_id_matches_vnstock_naming(title, expected):
    assert item_id(title) == expected


@pytest.mark.parametrize("name, expected", [
    ("matchPrice", "match_price"),
    ("displayDate1", "display_date1"),
    ("numberOfSharesMktCap", "number_of_shares_mkt_cap"),
    ("projectedTSRPercentage", "projected_tsr_percentage"),
])
def test_camel_to_snake(name, expected):
    assert camel_to_snake(name) == expected


_METRICS = {"data": {
    "INCOME_STATEMENT": [{"field": "isa1", "titleVi": "Doanh thu", "titleEn": "Net revenue"},
                         {"field": "isa2", "titleVi": "LNST", "titleEn": "Net profit after tax"}],
    "BALANCE_SHEET": [{"field": "bsa1", "titleVi": "Tổng tài sản", "titleEn": "TOTAL ASSETS"}],
}}
_STATEMENT = {"data": {
    "quarters": [
        {"ticker": "AAA", "yearReport": 2026, "lengthReport": 1, "isa1": 100.0, "isa2": 10.0, "isx9": 5},
        {"ticker": "AAA", "yearReport": 2026, "lengthReport": 2, "isa1": 200.0, "isa2": None, "isx9": 5},
    ],
    "years": [{"ticker": "AAA", "yearReport": 2025, "lengthReport": 5, "isa1": 900.0, "isa2": 90.0}],
}}


def _fake_get(url, params=None, **_kw):
    return _METRICS if url.endswith("/metrics") else _STATEMENT


@pytest.fixture(autouse=True)
def _clear_metrics_cache():
    vfinance._METRICS.clear()


def test_finance_report_is_items_by_periods():
    with patch("vndata.finance.get_json", side_effect=_fake_get):
        df = Finance("AAA", period="quarter")._get_report("income_statement", limit=50)
    assert list(df.columns) == ["item", "item_en", "item_id", "2026-Q1", "2026-Q2"]
    # Codes without a title for this symbol (isx9) are not line items.
    assert df["item_id"].tolist() == ["net_revenue", "net_profit_after_tax"]
    row = df[df["item_id"] == "net_revenue"].iloc[0]
    assert (row["2026-Q1"], row["2026-Q2"]) == (100.0, 200.0)


def test_finance_year_labels_and_limit():
    with patch("vndata.finance.get_json", side_effect=_fake_get):
        years = Finance("AAA")._get_report("income_statement", period="year")
        first = Finance("AAA")._get_report("income_statement", limit=1)
    assert "2025" in years.columns
    assert [c for c in first.columns if c.startswith("2026")] == ["2026-Q1"]


def test_financials_collector_reads_vndata_frame():
    from collectors.financials import _get
    with patch("vndata.finance.get_json", side_effect=_fake_get):
        df = Finance("AAA")._get_report("income_statement", limit=50)
    assert _get(df, "net_revenue", "2026-Q2") == pytest.approx(200 / 1e9)
    assert _get(df, "net_profit_after_tax", "2026-Q2") is None


_VCI_BARS = [{"symbol": "VCB", "t": ["1793145600", "1793232000"], "o": [56200, 57900],
              "h": [57600, 58400], "l": [56100, 57000], "c": [57500, 57200],
              "v": [6453700, 4049800]}]


def test_quote_scales_stocks_to_thousands():
    with patch("vndata.quote.post_json", return_value=_VCI_BARS):
        df = Quote("VCB").history(start="2026-10-28", end="2026-10-29")
    assert df["close"].tolist() == [57.5, 57.2]
    assert df["time"].dt.date.astype(str).tolist() == ["2026-10-28", "2026-10-29"]
    assert df["volume"].dtype == "int64"


def test_quote_leaves_index_in_points():
    bars = [{**_VCI_BARS[0], "c": [1766.65, 1755.61]}]
    with patch("vndata.quote.post_json", return_value=bars):
        df = Quote("VNINDEX").history(start="2026-10-28", end="2026-10-29")
    assert df["close"].tolist() == [1766.65, 1755.61]


def test_quote_falls_back_to_ssi():
    ssi = {"data": {"s": "ok", "t": [1793232000], "o": [57.9], "h": [58.4],
                    "l": [57.0], "c": [57.2], "v": [4049800]}}
    with patch("vndata.quote.post_json", return_value=[]), \
         patch("vndata.quote.get_json", return_value=ssi):
        df = Quote("VCB").history(start="2026-10-29", end="2026-10-29")
    assert df["close"].tolist() == [57.2]


def test_company_overview_keeps_vnstock_columns():
    details = {"data": {"ticker": "VCB", "viOrganName": "Vietcombank", "enOrganName": "VCB",
                        "numberOfSharesMktCap": 8.0e9, "issueShare": 8.0e9,
                        "targetPrice": 76500.0, "rating": "BUY"}}
    with patch("vndata.company.get_json", return_value=details):
        ov = Company("VCB").overview()
    row = ov.iloc[0]
    assert row["symbol"] == "VCB" and row["organ_name"] == "Vietcombank"
    assert row["target_price"] == 76500.0 and row["rating"] == "BUY"
    assert (ov.columns == "issue_share").sum() == 2
    assert not any(c.startswith("en_") for c in ov.columns)
