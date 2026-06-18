"""Fetch Vietnam macroeconomic indicators from the World Bank Open Data API.

No API key required. Data is annual (World Bank's native granularity), unlike the
monthly/quarterly figures scraped from nso.gov.vn in macro_collector.py — this module
fills in the broader categories (trade, labor, monetary, etc.) that nso.gov.vn doesn't
publish in an easily-scrapable form.
"""
from __future__ import annotations

from datetime import date

import requests
from loguru import logger
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from models.database import IS_SQLITE, get_session
from models.schema import MacroIndicator

_WB_API = "https://api.worldbank.org/v2/country/VNM/indicator/{code}?format=json&per_page=200"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# category (matches the Vĩ mô tab sub-tabs) -> list of (our_key, wb_code, label_vi, unit)
WB_CATALOG: dict[str, list[tuple[str, str, str, str]]] = {
    "GDP": [
        ("wb_gdp_usd",          "NY.GDP.MKTP.CD",     "Quy mô GDP",                    "USD"),
        ("wb_gdp_growth",       "NY.GDP.MKTP.KD.ZG",  "Tăng trưởng GDP",                "%"),
        ("wb_gdp_per_capita",   "NY.GDP.PCAP.CD",     "GDP bình quân đầu người",        "USD"),
        ("wb_gni_per_capita",   "NY.GNP.PCAP.CD",     "GNI bình quân đầu người",        "USD"),
        ("wb_industry_gdp",     "NV.IND.TOTL.ZS",     "Công nghiệp (% GDP)",            "%"),
        ("wb_agri_gdp",         "NV.AGR.TOTL.ZS",     "Nông nghiệp (% GDP)",            "%"),
        ("wb_services_gdp",     "NV.SRV.TOTL.ZS",     "Dịch vụ (% GDP)",                "%"),
    ],
    "Giá cả": [
        ("wb_cpi_inflation",    "FP.CPI.TOTL.ZG",     "Lạm phát (CPI)",                 "%"),
        ("wb_gdp_deflator",     "NY.GDP.DEFL.KD.ZG",  "Giảm phát GDP",                  "%"),
    ],
    "Kinh doanh": [
        ("wb_capital_formation","NE.GDI.TOTL.ZS",     "Tổng vốn đầu tư (% GDP)",        "%"),
        ("wb_industry_value",   "NV.IND.TOTL.ZS",     "Giá trị công nghiệp (% GDP)",    "%"),
        ("wb_gross_savings",    "NY.GNS.ICTR.ZS",     "Tổng tiết kiệm quốc gia (% GDP)","%"),
    ],
    "Thương mại": [
        ("wb_trade_balance_gdp","NE.RSB.GNFS.ZS",     "Cán cân thương mại (% GDP)",     "%"),
        ("wb_exports",          "TX.VAL.MRCH.CD.WT",  "Tổng xuất khẩu",                 "USD"),
        ("wb_imports",          "TM.VAL.MRCH.CD.WT",  "Tổng nhập khẩu",                 "USD"),
        ("wb_current_account",  "BN.CAB.XOKA.CD",     "Tài khoản vãng lai",             "USD"),
        ("wb_current_account_gdp","BN.CAB.XOKA.GD.ZS","Tài khoản vãng lai (% GDP)",     "%"),
        ("wb_fdi",               "BX.KLT.DINV.CD.WD", "Đầu tư nước ngoài (FDI)",        "USD"),
        ("wb_external_debt",    "DT.DOD.DECT.CD",     "Nợ nước ngoài",                  "USD"),
        ("wb_tourist_arrivals", "ST.INT.ARVL",        "Lượng khách du lịch",            "người"),
        ("wb_trade_pct_gdp",    "NE.TRD.GNFS.ZS",     "Thương mại (% GDP)",             "%"),
    ],
    "Lao động": [
        ("wb_unemployment",     "SL.UEM.TOTL.ZS",     "Tỷ lệ thất nghiệp",              "%"),
        ("wb_labor_participation","SL.TLF.CACT.ZS",   "Tỷ lệ tham gia lực lượng lao động","%"),
        ("wb_employment_ratio", "SL.EMP.TOTL.SP.ZS",  "Tỷ lệ có việc làm",              "%"),
    ],
    "Tiền tệ": [
        ("wb_broad_money_gdp",  "FM.LBL.BMNY.GD.ZS",  "Cung tiền M2 (% GDP)",           "%"),
        ("wb_exchange_rate",    "PA.NUS.FCRF",        "Tỷ giá USD/VND",                 "VND"),
        ("wb_broad_money_growth","FM.LBL.BMNY.ZG",    "Tăng trưởng cung tiền M2",       "%"),
    ],
    "Tiêu dùng": [
        ("wb_consumption_gdp",  "NE.CON.PRVT.ZS",     "Tiêu dùng hộ gia đình (% GDP)",  "%"),
        ("wb_consumption_growth","NE.CON.PRVT.KD.ZG", "Tăng trưởng tiêu dùng",          "%"),
    ],
    "Thuế": [
        ("wb_tax_revenue_gdp",  "GC.TAX.TOTL.GD.ZS",  "Thu thuế (% GDP)",               "%"),
    ],
    "Lãi suất": [
        ("wb_lending_rate",     "FR.INR.LEND",        "Lãi suất cho vay",               "%"),
        ("wb_real_interest_rate","FR.INR.RINR",       "Lãi suất thực",                  "%"),
        ("wb_deposit_rate",     "FR.INR.DPST",        "Lãi suất tiền gửi",              "%"),
    ],
}

# Flat lookup used by the dashboard to render labels/units without re-importing the catalog shape.
WB_INDICATOR_META: dict[str, dict[str, str]] = {
    key: {"label": label, "unit": unit, "category": category}
    for category, items in WB_CATALOG.items()
    for key, _code, label, unit in items
}


def fetch_wb_series(wb_code: str) -> list[tuple[int, float]]:
    """Return (year, value) pairs for a World Bank indicator for Vietnam, non-null only, ascending."""
    url = _WB_API.format(code=wb_code)
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return []
    out = []
    for row in payload[1]:
        if row.get("value") is not None:
            out.append((int(row["date"]), float(row["value"])))
    return sorted(out)


def _upsert(rows: list[dict]) -> None:
    if not rows:
        return
    insert_fn = sqlite_insert if IS_SQLITE else pg_insert
    with get_session() as session:
        for row in rows:
            stmt = insert_fn(MacroIndicator).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["indicator", "period"],
                set_={"value": row["value"], "unit": row["unit"], "source_url": row["source_url"]},
            )
            session.execute(stmt)


def collect_worldbank_macro() -> int:
    """Fetch every indicator in WB_CATALOG and upsert into macro_indicators. Returns row count written."""
    total = 0
    for category, items in WB_CATALOG.items():
        for our_key, wb_code, label, unit in items:
            try:
                series = fetch_wb_series(wb_code)
            except Exception as e:
                logger.warning(f"World Bank fetch failed for {wb_code} ({label}): {e}")
                continue
            rows = [
                {
                    "indicator": our_key,
                    "period": date(year, 1, 1),
                    "value": round(value, 4),
                    "unit": unit,
                    "source_url": f"https://api.worldbank.org/v2/country/VNM/indicator/{wb_code}",
                }
                for year, value in series
            ]
            _upsert(rows)
            total += len(rows)
            logger.info(f"[{category}] {label} ({wb_code}): {len(rows)} years")
    logger.info(f"World Bank macro collect: {total} rows written")
    return total


if __name__ == "__main__":
    collect_worldbank_macro()
