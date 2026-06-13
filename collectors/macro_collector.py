"""Scrape Vietnam macroeconomic indicators (CPI, GDP growth, trade balance) from nso.gov.vn."""
from __future__ import annotations

import io
import re
import unicodedata
from datetime import date

import pandas as pd
import requests
import truststore
from loguru import logger

truststore.inject_into_ssl()  # use the OS certificate store (nso.gov.vn's CA isn't in certifi)
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models.database import get_session, IS_SQLITE
from models.schema import MacroIndicator


_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_CPI_LIST_URL = "https://www.nso.gov.vn/cpi-vi/"
_GDP_LIST_URL = "https://www.nso.gov.vn/tai-khoan-quoc-gia/"
_TRADE_LIST_URL = "https://www.nso.gov.vn/xuat-nhap-khau/"

_MONTH_SLUGS = {
    "mot": 1, "hai": 2, "ba": 3, "tu": 4, "nam": 5, "sau": 6,
    "bay": 7, "tam": 8, "chin": 9, "muoi-mot": 11, "muoi-hai": 12, "muoi": 10,
}

_QUARTER_END_MONTH = {"I": 3, "II": 6, "III": 9, "IV": 12}

_MOM_RE = re.compile(r"(tăng|giảm)(?:\s+\S+)?\s+([\d,]+)%\s+so với tháng trước")
_YOY_RE = re.compile(r"(tăng|giảm)(?:\s+\S+)?\s+([\d,]+)%\s*(?:\[\d+\]\s*)?so với cùng kỳ năm trước")
_YOY_RE_OLD = re.compile(r"[Ss]o với cùng kỳ năm trước,.{0,80}?(tăng|giảm)\s+([\d,]+)%")


_CPI_BODY_RE = re.compile(r"tiêu dùng(?:\s*\(CPI\))?\s+tháng\s+\S+\s+(?:tăng|giảm)")


def _cpi_window(text: str) -> str:
    """Isolate the CPI body paragraph from the title and the gold/USD-price paragraphs that follow it."""
    m = _CPI_BODY_RE.search(text)
    if not m:
        return text
    start = m.start()
    end = len(text)
    for marker in ("Giá vàng", "giá vàng", "Giá đô la", "giá đô la", "Giá đôla", "giá đôla"):
        j = text.find(marker, start + 10)
        if j != -1:
            end = min(end, j)
    return text[start:end]
_SLUG_RE_WORD = re.compile(r"thang-(muoi-hai|muoi-mot|mot|hai|ba|tu|nam|sau|bay|tam|chin|muoi)-.*?nam-(\d{4})")
_SLUG_RE_NUM = re.compile(r"thang-(\d{1,2})-nam-(\d{4})")
_SLUG_RE_NUM_FALLBACK = re.compile(r"-(\d{1,2})-nam-(\d{4})")
_GDP_RE = re.compile(
    r"Tổng sản phẩm trong nước \(GDP\) quý (I{1,3}|IV)/(\d{4}).{0,150}?([\d,]+)%\s*(?:\[\d+\]\s*)?so với cùng kỳ năm trước"
)
_TRADE_YEAR_RE = re.compile(r"so-lieu-xuat-nhap-khau-cac-thang-nam-(\d{4})")
_TRADE_XLS_RE = re.compile(r'href="([^"]*V0([12])-\d{4}-\d+\.xls)"')
_FDI_RE = re.compile(
    r"Tổng vốn đầu tư nước ngoài đăng ký vào Việt Nam.{0,250}?đạt\s*(?:gần\s*)?([\d,]+)\s*tỷ USD"
)

# Retail sales (Tổng mức bán lẻ hàng hóa và doanh thu dịch vụ tiêu dùng) YoY growth.
# Wording varies by press release; tried in order, normal quarter mentions first,
# then cumulative 9-month / 6-month wording used in some Q3 / Q2 releases.
_RETAIL_RE_QUARTER_FIRST = re.compile(
    r"Tổng mức bán lẻ hàng hóa và doanh thu dịch vụ tiêu dùng(?: theo giá hiện hành)? quý (I{1,3}|IV)/(\d{4})"
    r".{0,100}?(tăng|giảm)\s+([\d,]+)%\s*so với cùng kỳ năm trước"
)
_RETAIL_RE_QUARTER_AFTER = re.compile(
    r"[Qq]uý (I{1,3}|IV)/(\d{4})\s*,?\s*[Tt]ổng mức bán lẻ hàng hóa và doanh thu dịch vụ tiêu dùng"
    r".{0,100}?(tăng|giảm)\s+([\d,]+)%\s*so với cùng kỳ năm trước"
)
_RETAIL_RE_9M = re.compile(
    r"[Tt]ính chung chín tháng năm (\d{4}).{0,30}?tổng mức bán lẻ hàng hóa và doanh thu dịch vụ tiêu dùng"
    r".{0,100}?(tăng|giảm)\s+([\d,]+)%\s*so với cùng kỳ năm trước"
)
_RETAIL_RE_6M = re.compile(
    r"[Tt]ính chung (?:sáu|6) tháng đầu năm (\d{4}).{0,30}?tổng mức bán lẻ hàng hóa và doanh thu dịch vụ tiêu dùng"
    r".{0,100}?(tăng|giảm)\s+([\d,]+)%\s*so với cùng kỳ năm trước"
)


def _to_float(sign: str, num: str) -> float:
    value = float(num.replace(",", "."))
    return -value if sign == "giảm" else value


def discover_cpi_article_urls(max_pages: int = 10) -> list[str]:
    """Return CPI article URLs from the first `max_pages` listing pages (newest first)."""
    urls: list[str] = []
    for page in range(1, max_pages + 1):
        list_url = _CPI_LIST_URL if page == 1 else f"{_CPI_LIST_URL}?paged={page}"
        try:
            resp = requests.get(list_url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {list_url}: {e}")
            break
        found = re.findall(
            r'href="(https://www\.nso\.gov\.vn/du-lieu-va-so-lieu-thong-ke/[^"]*chi-so-gia-tieu-dung[^"]*)"',
            resp.text,
        )
        if not found:
            break
        urls.extend(found)
    return list(dict.fromkeys(urls))  # de-dup, preserve order


def _parse_period(url: str) -> date | None:
    m = _SLUG_RE_WORD.search(url)
    if m:
        return date(int(m.group(2)), _MONTH_SLUGS[m.group(1)], 1)
    m = _SLUG_RE_NUM.search(url)
    if m:
        return date(int(m.group(2)), int(m.group(1)), 1)
    m = _SLUG_RE_NUM_FALLBACK.search(url)
    if m:
        return date(int(m.group(2)), int(m.group(1)), 1)
    return None


def parse_cpi_article(url: str) -> list[dict]:
    """Return [{indicator, period, value, unit, source_url}, ...] for one CPI article."""
    period = _parse_period(url)
    if period is None:
        logger.warning(f"Could not parse month/year from URL: {url}")
        return []

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return []

    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = text.replace("&#8211;", "-").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    cpi_text = _cpi_window(text)

    rows = []
    mom = _MOM_RE.search(cpi_text)
    if mom:
        rows.append({
            "indicator": "cpi_mom", "period": period,
            "value": _to_float(mom.group(1), mom.group(2)),
            "unit": "%", "source_url": url,
        })
    yoy = _YOY_RE.search(cpi_text) or _YOY_RE_OLD.search(cpi_text)
    if yoy:
        rows.append({
            "indicator": "cpi_yoy", "period": period,
            "value": _to_float(yoy.group(1), yoy.group(2)),
            "unit": "%", "source_url": url,
        })
    return rows


def discover_gdp_article_urls(max_pages: int = 3) -> list[str]:
    """Return quarterly socio-economic press release URLs (newest first)."""
    urls: list[str] = []
    for page in range(1, max_pages + 1):
        list_url = _GDP_LIST_URL if page == 1 else f"{_GDP_LIST_URL}?paged={page}"
        try:
            resp = requests.get(list_url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {list_url}: {e}")
            break
        found = re.findall(
            r'href="(https://www\.nso\.gov\.vn/(?:du-lieu-va-so-lieu-thong-ke|tin-tuc-thong-ke)/[^"]*thong-cao-bao-chi[^"]*)"',
            resp.text,
        )
        if not found:
            break
        urls.extend(found)
    return list(dict.fromkeys(urls))


def parse_gdp_article(url: str) -> list[dict]:
    """Return [{indicator: 'gdp_growth', period, value, ...}] for one quarterly press release."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return []

    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = text.replace("&#8211;", "-").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFC", text)

    m = _GDP_RE.search(text)
    if not m:
        logger.warning(f"Could not parse GDP growth from {url}")
        return []
    quarter, year, value = m.group(1), int(m.group(2)), float(m.group(3).replace(",", "."))
    period = date(year, _QUARTER_END_MONTH[quarter], 1)
    rows = [{"indicator": "gdp_growth", "period": period, "value": value, "unit": "%", "source_url": url}]

    fdi = _FDI_RE.search(text)
    if fdi:
        # Year-to-date cumulative registered FDI; converted to per-quarter deltas in _derive_fdi_quarterly().
        rows.append({
            "indicator": "fdi_cumulative", "period": period,
            "value": float(fdi.group(1).replace(",", ".")), "unit": "tỷ USD", "source_url": url,
        })

    retail = (
        _RETAIL_RE_QUARTER_FIRST.search(text)
        or _RETAIL_RE_QUARTER_AFTER.search(text)
    )
    if retail:
        retail_quarter, retail_year, sign, val = retail.groups()
        retail_period = date(int(retail_year), _QUARTER_END_MONTH[retail_quarter], 1)
        rows.append({
            "indicator": "retail_sales_growth", "period": retail_period,
            "value": _to_float(sign, val), "unit": "%", "source_url": url,
        })
    else:
        retail = _RETAIL_RE_9M.search(text)
        retail_month = 9
        if not retail:
            retail = _RETAIL_RE_6M.search(text)
            retail_month = 6
        if retail:
            retail_year, sign, val = retail.groups()
            retail_period = date(int(retail_year), retail_month, 1)
            rows.append({
                "indicator": "retail_sales_growth", "period": retail_period,
                "value": _to_float(sign, val), "unit": "%", "source_url": url,
            })
    return rows


def discover_trade_article_urls(max_pages: int = 8) -> list[str]:
    """Return yearly export/import data-table article URLs (newest first)."""
    urls: list[str] = []
    for page in range(1, max_pages + 1):
        list_url = _TRADE_LIST_URL if page == 1 else f"{_TRADE_LIST_URL}?paged={page}"
        try:
            resp = requests.get(list_url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {list_url}: {e}")
            break
        found = re.findall(
            r'href="(https://www\.nso\.gov\.vn/[^"]*so-lieu-xuat-nhap-khau-cac-thang-nam-\d{4}[^"]*)"',
            resp.text,
        )
        if not found:
            break
        urls.extend(found)
    return list(dict.fromkeys(urls))


def _trade_monthly_totals(xls_url: str) -> "pd.Series | None":
    resp = requests.get(xls_url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.ExcelFile(io.BytesIO(resp.content)).parse(0, header=None)
    for _, row in df.iterrows():
        if isinstance(row[0], str) and "Tổng số" in row[0]:
            return row
    return None


def parse_trade_article(url: str) -> list[dict]:
    """Return monthly [{indicator: 'trade_balance', period, value (tỷ USD), ...}] for one year."""
    m = _TRADE_YEAR_RE.search(url)
    if not m:
        return []
    year = int(m.group(1))

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return []

    xls_urls = dict((v, u) for u, v in _TRADE_XLS_RE.findall(resp.text))
    if "1" not in xls_urls or "2" not in xls_urls:
        logger.warning(f"Could not find export/import data tables on {url}")
        return []

    try:
        export_row = _trade_monthly_totals(xls_urls["1"])
        import_row = _trade_monthly_totals(xls_urls["2"])
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch trade data tables for {url}: {e}")
        return []
    if export_row is None or import_row is None:
        logger.warning(f"Could not find 'Tổng số' row in trade data tables for {url}")
        return []

    # Layout: col 0 = label, then (Lượng, Trị giá) pairs per month, last pair = year-to-date total.
    num_months = (len(export_row) - 1) // 2 - 1
    rows = []
    for month in range(1, num_months + 1):
        col = 2 * month
        export_val, import_val = export_row[col], import_row[col]
        if pd.isna(export_val) or pd.isna(import_val):
            continue
        balance = (float(export_val) - float(import_val)) / 1e6  # thousand USD -> tỷ (billion) USD
        rows.append({
            "indicator": "trade_balance", "period": date(year, month, 1),
            "value": round(balance, 3), "unit": "tỷ USD", "source_url": url,
        })
    return rows


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


def collect_cpi(max_pages: int = 10) -> int:
    """Scrape CPI YoY/MoM from nso.gov.vn and upsert into macro_indicators. Returns row count."""
    urls = discover_cpi_article_urls(max_pages=max_pages)
    logger.info(f"Found {len(urls)} CPI articles")
    total = 0
    for url in urls:
        rows = parse_cpi_article(url)
        _upsert(rows)
        total += len(rows)
    logger.info(f"Upserted {total} CPI data points")
    return total


def collect_gdp(max_pages: int = 3) -> int:
    """Scrape quarterly GDP growth and FDI from nso.gov.vn and upsert into macro_indicators."""
    urls = discover_gdp_article_urls(max_pages=max_pages)
    logger.info(f"Found {len(urls)} GDP press releases")
    total = 0
    for url in urls:
        rows = parse_gdp_article(url)
        _upsert(rows)
        total += len(rows)
    logger.info(f"Upserted {total} GDP growth / FDI data points")
    total += _derive_fdi_quarterly()
    return total


def _derive_fdi_quarterly() -> int:
    """Convert year-to-date cumulative registered FDI into per-quarter net inflows."""
    with get_session() as session:
        cumulative = session.execute(
            select(MacroIndicator.period, MacroIndicator.value, MacroIndicator.source_url)
            .where(MacroIndicator.indicator == "fdi_cumulative")
            .order_by(MacroIndicator.period.asc())
        ).all()

    rows = []
    prev_by_year: dict[int, float] = {}
    for period, value, source_url in cumulative:
        prev = prev_by_year.get(period.year, 0.0)
        rows.append({
            "indicator": "fdi", "period": period,
            "value": round(value - prev, 3), "unit": "tỷ USD", "source_url": source_url,
        })
        prev_by_year[period.year] = value

    _upsert(rows)
    logger.info(f"Derived {len(rows)} quarterly FDI data points")
    return len(rows)


def collect_trade(max_pages: int = 8) -> int:
    """Scrape monthly trade balance from nso.gov.vn and upsert into macro_indicators."""
    urls = discover_trade_article_urls(max_pages=max_pages)
    logger.info(f"Found {len(urls)} trade data articles")
    total = 0
    for url in urls:
        rows = parse_trade_article(url)
        _upsert(rows)
        total += len(rows)
    logger.info(f"Upserted {total} trade balance data points")
    return total


if __name__ == "__main__":
    collect_cpi(max_pages=30)
    collect_gdp(max_pages=5)
    collect_trade(max_pages=8)
