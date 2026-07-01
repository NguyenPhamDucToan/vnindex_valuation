"""
Annual report downloader — fetches individual PDFs from Zenodo period ZIPs
using HTTP Range requests (remotezip) without downloading full archives.

Flow:
  1. get_index_df()          → download & cache file_index_full.csv once (2.5 MB)
  2. get_available_years()   → filter index by ticker, return list of years
  3. fetch_annual_report_pdf() → RemoteZip extraction, cache to data/annual_reports/
  4. extract_report_sections() → pdfplumber text + regex section split, cache as JSON
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    from remotezip import RemoteZip
    _HAS_REMOTEZIP = True
except ImportError:
    _HAS_REMOTEZIP = False

# ── Zenodo config ──────────────────────────────────────────────────────────────
ZENODO_RECORD = "20949551"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

_PERIOD_ZIPS: dict[tuple[int, int], str] = {
    (2000, 2005): "vn_bctn_2000_2005.zip",
    (2006, 2010): "vn_bctn_2006_2010.zip",
    (2011, 2015): "vn_bctn_2011_2015.zip",
    (2016, 2020): "vn_bctn_2016_2020.zip",
    (2021, 2025): "vn_bctn_2021_2025.zip",
}

# ── Local cache paths ──────────────────────────────────────────────────────────
CACHE_DIR = Path("data/annual_reports")
INDEX_PATH = CACHE_DIR / "file_index.csv"

# ── Section definitions (Vietnamese headers) ───────────────────────────────────
SECTIONS: dict[str, dict] = {
    "gioi_thieu": {
        "label": "Giới thiệu công ty",
        "patterns": [
            r"GIỚI\s+THIỆU\s+(?:CHUNG|CÔNG\s+TY|VỀ\s+CÔNG\s+TY)",
            r"TỔNG\s+QUAN\s+(?:VỀ\s+)?CÔNG\s+TY",
            r"THÔNG\s+TIN\s+CÔNG\s+TY",
            r"VỀ\s+CHÚNG\s+TÔI",
            r"TẦM\s+NHÌN.{0,20}SỨ\s+MỆNH",
        ],
    },
    "hoat_dong": {
        "label": "Hoạt động kinh doanh",
        "patterns": [
            r"LĨNH\s+VỰC\s+HOẠT\s+ĐỘNG",
            r"HOẠT\s+ĐỘNG\s+KINH\s+DOANH",
            r"SẢN\s+PHẨM\s+(?:VÀ\s+)?DỊCH\s+VỤ",
            r"NGÀNH\s+NGHỀ\s+KINH\s+DOANH",
        ],
    },
    "ket_qua": {
        "label": "Kết quả kinh doanh",
        "patterns": [
            r"KẾT\s+QUẢ\s+KINH\s+DOANH",
            r"BÁO\s+CÁO\s+CỦA\s+BAN\s+(?:ĐIỀU\s+HÀNH|GIÁM\s+ĐỐC|TỔNG\s+GIÁM\s+ĐỐC)",
            r"ĐÁNH\s+GIÁ\s+CỦA\s+BAN\s+(?:ĐIỀU\s+HÀNH|GIÁM\s+ĐỐC)",
            r"TÌNH\s+HÌNH\s+HOẠT\s+ĐỘNG",
        ],
    },
    "rui_ro": {
        "label": "Quản trị rủi ro",
        "patterns": [
            r"(?:CÁC\s+)?YẾU\s+TỐ\s+RỦI\s+RO",
            r"QUẢN\s+TRỊ\s+RỦI\s+RO",
            r"RỦI\s+RO\s+(?:VÀ\s+)?(?:THÁCH\s+THỨC|CƠ\s+HỘI)",
        ],
    },
}

_MAX_SECTION_CHARS = 4000  # cap per section to keep UI readable


# ── Helpers ────────────────────────────────────────────────────────────────────

def _zip_url(zip_name: str) -> str:
    return f"{ZENODO_BASE}/{zip_name}?download=1"


def _year_to_zip(year: int) -> Optional[str]:
    for (lo, hi), name in _PERIOD_ZIPS.items():
        if lo <= year <= hi:
            return name
    return None


def _pdf_cache_path(ticker: str, year: int) -> Path:
    year_short = str(year)[2:]
    fname = f"{ticker.upper()}_{year_short}CN_BCTN.pdf"
    return CACHE_DIR / ticker.upper() / fname


def _sections_cache_path(ticker: str, year: int) -> Path:
    return _pdf_cache_path(ticker, year).with_suffix(".sections.json")


# ── Public API ─────────────────────────────────────────────────────────────────

def get_index_df() -> pd.DataFrame:
    """Return the full file index, downloading from Zenodo once if not cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        logger.info("Downloading file index from Zenodo (one-time, 2.5 MB)...")
        url = f"{ZENODO_BASE}/file_index_full.csv?download=1"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        INDEX_PATH.write_bytes(resp.content)
        logger.info(f"Index cached at {INDEX_PATH}")
    return pd.read_csv(INDEX_PATH)


def get_available_years(ticker: str) -> list[int]:
    """Sorted (desc) list of years with annual reports for this ticker."""
    try:
        df = get_index_df()
        mask = df["ticker_file"].str.upper() == ticker.upper()
        return sorted(
            df.loc[mask, "year_full"].dropna().astype(int).unique().tolist(),
            reverse=True,
        )
    except Exception:
        logger.exception(f"Cannot load available years for {ticker}")
        return []


def is_pdf_cached(ticker: str, year: int) -> bool:
    return _pdf_cache_path(ticker, year).exists()


def fetch_annual_report_pdf(ticker: str, year: int) -> Optional[Path]:
    """
    Download a single annual report PDF from the Zenodo ZIP archive using
    HTTP Range requests (remotezip).  The PDF is cached locally; subsequent
    calls return the cached path immediately.
    """
    if not _HAS_REMOTEZIP:
        logger.error("remotezip not installed — run: pip install remotezip")
        return None

    cache_path = _pdf_cache_path(ticker, year)
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    year_short = str(year)[2:]
    # Internal ZIP path uses forward slashes
    zip_member = f"{ticker.upper()}/{ticker.upper()}_{year_short}CN_BCTN.pdf"

    zip_name = _year_to_zip(year)
    if zip_name is None:
        logger.warning(f"No Zenodo ZIP covers year {year}")
        return None

    zip_url = _zip_url(zip_name)
    logger.info(f"Fetching {zip_member} from {zip_name} via Range requests...")

    try:
        with RemoteZip(zip_url) as zf:
            # Files are stored as full_data/<ticker>/<file>.pdf inside the ZIP
            full_member = f"full_data/{zip_member}"
            available = zf.namelist()
            if full_member not in available:
                logger.warning(f"{full_member} not found in {zip_name}")
                return None
            data = zf.read(full_member)

        cache_path.write_bytes(data)
        mb = len(data) / 1e6
        logger.info(f"PDF cached: {cache_path} ({mb:.1f} MB)")
        return cache_path

    except Exception:
        logger.exception(f"Failed to fetch {zip_member} from {zip_name}")
        return None


def extract_report_sections(ticker: str, year: int) -> dict[str, str]:
    """
    Extract key text sections from the cached PDF.
    Returns section_key → text.  Caches result as JSON next to PDF.
    Returns {} if PDF not downloaded yet, {"_scanned": "1"} if scan-only.
    """
    sections_path = _sections_cache_path(ticker, year)
    if sections_path.exists():
        return json.loads(sections_path.read_text(encoding="utf-8"))

    pdf_path = _pdf_cache_path(ticker, year)
    if not pdf_path.exists():
        return {}

    if not _HAS_PDFPLUMBER:
        logger.error("pdfplumber not installed")
        return {}

    logger.info(f"Extracting sections from {pdf_path.name}...")

    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    full_text += t + "\n"
    except Exception:
        logger.exception(f"pdfplumber failed on {pdf_path}")
        return {}

    if not full_text.strip():
        result = {"_scanned": "1"}
        sections_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    upper_text = full_text.upper()

    # Find the earliest match position for each section
    hits: list[tuple[int, str]] = []
    for key, meta in SECTIONS.items():
        for pat in meta["patterns"]:
            m = re.search(pat, upper_text, re.MULTILINE)
            if m:
                hits.append((m.start(), key))
                break  # one match per section

    hits.sort()

    result: dict[str, str] = {}
    for i, (pos, key) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else pos + 5000
        snippet = _clean_text(full_text[pos:end])
        if len(snippet) > _MAX_SECTION_CHARS:
            snippet = snippet[:_MAX_SECTION_CHARS] + "\n\n[... xem toàn bộ trong tài liệu gốc]"
        result[key] = snippet

    if not result:
        # Fallback: return first ~1500 chars (Giới thiệu usually at front)
        result["raw_intro"] = _clean_text(full_text[: min(1500, len(full_text))])

    sections_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Sections cached ({len(result)} sections): {sections_path}")
    return result


def _clean_text(text: str) -> str:
    """Clean raw pdfplumber output into readable paragraphs."""
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        line = line.strip()
        # Drop lone page numbers, short noise lines, and separator lines
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if re.fullmatch(r"[_\-=\.]{3,}", line):
            continue
        if len(line) < 3:
            continue
        cleaned.append(line)

    # Re-join broken lines: if a line doesn't end with sentence-ending
    # punctuation, merge it with the next line (PDF column wrapping artifact)
    _SENT_END = re.compile(r"[.!?:;)\]»]$")
    _BULLET = re.compile(r"^[•\-–—\d]+[\.\)]?\s")
    merged: list[str] = []
    buf = ""
    for line in cleaned:
        if not buf:
            buf = line
        elif _BULLET.match(line) or _BULLET.match(buf):
            # Bullet items → keep separate
            merged.append(buf)
            buf = line
        elif _SENT_END.search(buf):
            # Previous line ended a sentence → new paragraph
            merged.append(buf)
            buf = line
        else:
            # Continuation → join with space
            buf = buf + " " + line
    if buf:
        merged.append(buf)

    # Group into paragraphs: two+ consecutive sentence-ending lines → split
    paragraphs: list[str] = []
    para: list[str] = []
    for line in merged:
        para.append(line)
        if _SENT_END.search(line) and len(para) >= 1:
            paragraphs.append(" ".join(para))
            para = []
    if para:
        paragraphs.append(" ".join(para))

    return "\n\n".join(p for p in paragraphs if p.strip())
