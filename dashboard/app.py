"""Streamlit dashboard — VNIndex Valuation Tool.

Run with:  streamlit run dashboard/app.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# vnstock prints emoji/Vietnamese banners on import; on Windows the default
# console encoding (cp1252) can't render them, which raises UnicodeEncodeError.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Signal labels with invisible sort prefix (U+2060 Word Joiner)
_WJ_GLOBAL = "⁠"
_SIG_LABELS = {
    "Strong Buy":  _WJ_GLOBAL * 1 + "Strong Buy",
    "Buy":         _WJ_GLOBAL * 2 + "Buy",
    "Watch":       _WJ_GLOBAL * 3 + "Watch",
    "Neutral":     _WJ_GLOBAL * 4 + "Neutral",
    "Reduce":      _WJ_GLOBAL * 5 + "Reduce",
    "Sell":        _WJ_GLOBAL * 6 + "Sell",
    "Strong Sell": _WJ_GLOBAL * 7 + "Strong Sell",
}

# Sector → relevant commodity symbols (Yahoo Finance) for input/output price chart
_SECTOR_COMMODITIES = {
    "Thực phẩm - Đồ uống": {
        "title": "Food & Beverage Input Prices",
        "input":  [("ZW=F","Wheat","#3b82f6"),("SB=F","Sugar","#8b5cf6"),("ZC=F","Corn","#f59e0b")],
        "output": [("KC=F","Coffee","#92400e"),("ZS=F","Soybean","#22c55e")],
    },
    "Nông - Lâm - Ngư": {
        "title": "Agricultural Commodity Prices",
        "input":  [("ZC=F","Corn","#f59e0b"),("ZW=F","Wheat","#3b82f6")],
        "output": [("ZS=F","Soybean","#22c55e"),("SB=F","Sugar","#8b5cf6")],
    },
    "Chế biến Thủy sản": {
        "title": "Seafood Input Prices",
        "input":  [("ZC=F","Corn/Feed","#f59e0b"),("ZS=F","Soybean/Feed","#22c55e")],
        "output": [],
    },
    "Khai khoáng": {
        "title": "Mining Commodity Prices",
        "input":  [("CL=F","Crude Oil","#ef4444")],
        "output": [("GC=F","Gold","#eab308"),("SI=F","Silver","#9ca3af")],
    },
    "Tiện ích": {
        "title": "Energy Input Prices",
        "input":  [("CL=F","Crude Oil","#ef4444"),("NG=F","Natural Gas","#3b82f6")],
        "output": [],
    },
    "Vận tải - kho bãi": {
        "title": "Transport Fuel Prices",
        "input":  [("BZ=F","Brent Oil","#ef4444"),("NG=F","Natural Gas","#3b82f6")],
        "output": [],
    },
    "Vật liệu xây dựng": {
        "title": "Construction Material Prices",
        "input":  [("TIO=F","Iron Ore (in)","#ef4444"),("CL=F","Energy (in)","#6b7280")],
        "output": [("SLX","Steel ETF (out)","#3b82f6")],
    },
    "SX Phụ trợ": {
        "title": "Manufacturing Input/Output Prices",
        "input":  [("TIO=F","Iron Ore (in)","#ef4444"),("HG=F","Copper (in)","#f59e0b")],
        "output": [("SLX","Steel ETF (out)","#3b82f6")],
    },
    "SX Thiết bị, máy móc": {
        "title": "Machinery Input Prices",
        "input":  [("TIO=F","Iron Ore (in)","#ef4444"),("ALI=F","Aluminum (in)","#9ca3af")],
        "output": [("HG=F","Copper (out)","#f59e0b")],
    },
    "SX Nhựa - Hóa chất": {
        "title": "Chemical/Plastic Input Prices",
        "input":  [("CL=F","Crude Oil (in)","#ef4444"),("NG=F","Natural Gas (in)","#3b82f6")],
        "output": [],
    },
    "Thiết bị điện": {
        "title": "Electrical Equipment Input Prices",
        "input":  [("HG=F","Copper (in)","#f59e0b"),("ALI=F","Aluminum (in)","#9ca3af")],
        "output": [],
    },
    "Sản phẩm cao su": {
        "title": "Rubber/Petroleum Input Prices",
        "input":  [("CL=F","Crude Oil (in)","#ef4444")],
        "output": [],
    },
    "Xây dựng": {
        "title": "Steel & Construction Material Prices",
        "input":  [("TIO=F","Iron Ore (in)","#ef4444"),("CL=F","Energy (in)","#6b7280")],
        "output": [("SLX","Steel (out)","#3b82f6")],
    },
}

import io
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import select, func as sqlfunc

from models.database import get_session
from models.schema import Financial, Price, Company, Valuation, PinnedTicker, MacroIndicator
from collectors.worldbank_collector import WB_INDICATOR_META
from valuation.inputs import compute_ttm, compute_fcff_ttm, prepare_dcf_inputs, build_quarter_history
from valuation.dcf import dcf_valuation
from valuation.graham import graham_number, bvps_from_financials
from valuation.wacc import cost_of_equity, DEFAULT_BETA, DEFAULT_COD
from valuation.multiples import (
    pb_implied, ev_ebitda_implied, epv_implied,
    ps_implied, residual_income_implied, pocf_implied,
)
from config import MARKET_PE
from valuation.ratios import (
    gross_margin, net_margin, operating_margin, roe, roa,
    current_ratio, debt_to_equity, profit_quality, fcf_margin,
    quick_ratio, absolute_liquidity, debt_to_assets, ocf_to_current_liabilities,
)
from valuation.signals import compute_quality_score, classify_signal

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Page config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="VNIndex Valuation",
    page_icon="📈",
    layout="wide",
)

# Kill all animations/transitions globally — prevents white flash and dialog delay
st.markdown("""<style>
html { background:#0e1117!important; color-scheme:dark; }
body,[data-testid="stApp"],.main,.block-container { background-color:#0e1117!important; }
[data-testid="stApp"]*{ animation-duration:0.001s!important; transition-duration:0.001s!important; }
[data-testid="stStatusWidget"]{ visibility:hidden!important; }
[data-testid="stDecoration"]{ display:none!important; }
[data-testid="stModal"],[data-testid="stModalContent"],[data-testid="stModalOverlay"]{
    animation:none!important; transition:none!important;
}
</style>""", unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Utility functions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fmt_bn(v) -> str:
    if v is None: return "—"
    return f"{v:,.0f} bn"


def fmt_pct(v) -> str:
    if v is None: return "—"
    return f"{v*100:.1f}%"


def fmt_vnd(v) -> str:
    if v is None: return "—"
    return f"{v:,.0f}"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Data helpers (cached so they don't re-query on every rerun)
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_prices(ticker: str) -> pd.DataFrame:
    from datetime import date as _date, timedelta
    from collectors.prices import fetch_prices, upsert_prices, incremental_start_date

    def _read_db():
        with get_session() as s:
            rows = s.execute(
                select(Price).where(Price.ticker == ticker).order_by(Price.date.asc())
            ).scalars().all()
            return pd.DataFrame([{
                "date": r.date, "open": r.open, "high": r.high,
                "low": r.low, "close": r.close, "volume": r.volume,
            } for r in rows])

    df_db = _read_db()

    # Auto-fetch if DB is sparse (< 200 rows in the last 2 years) or stale (latest
    # row more than 4 calendar days old, e.g. the daily refresh hasn't run).
    cutoff = _date.today() - timedelta(days=730)
    recent = len(df_db[pd.to_datetime(df_db["date"]).dt.date >= cutoff]) if not df_db.empty else 0
    stale = (df_db.empty or
             pd.to_datetime(df_db["date"]).max().date() < _date.today() - timedelta(days=4))
    if recent < 200 or stale:
        try:
            start = incremental_start_date(ticker)
            df_fresh = fetch_prices(ticker, start, _date.today())
            if not df_fresh.empty:
                upsert_prices(ticker, df_fresh)
                df_db = _read_db()
        except Exception:
            pass

    return df_db


@st.cache_data(ttl=30)
def load_live_quote(ticker: str) -> dict | None:
    """Return a live price snapshot during trading hours, else None.

    Values are in thousands VND (same convention as Price.close).
    """
    from collectors.live_quote import is_market_hours_ict, fetch_live_quote
    if not is_market_hours_ict():
        return None
    return fetch_live_quote(ticker)


@st.cache_data(ttl=3600)
def load_macro_indicator(indicator: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(
            select(MacroIndicator)
            .where(MacroIndicator.indicator == indicator)
            .order_by(MacroIndicator.period.asc())
        ).scalars().all()
        return pd.DataFrame([{"period": r.period, "value": r.value, "unit": r.unit} for r in rows])


_MACRO_INDICATOR_LABELS = {
    "gdp_growth": "Tăng trưởng GDP",
    "cpi_yoy": "Lạm phát (so với cùng kỳ năm trước)",
    "cpi_mom": "Lạm phát (so với tháng trước)",
    "trade_balance": "Cán cân thương mại",
    "retail_sales_growth": "Tăng trưởng bán lẻ",
    "fdi": "FDI",
    "gdp_sector_agri": "GDP - Nông, lâm nghiệp và thủy sản",
    "gdp_sector_industry": "GDP - Công nghiệp và xây dựng",
    "gdp_sector_services": "GDP - Dịch vụ",
    "gdp_nominal_usd": "Quy mô GDP (USD)",
    "gdp_per_capita_usd": "GDP bình quân đầu người (USD)",
    "investment_growth": "Tăng trưởng vốn đầu tư toàn xã hội",
    "core_inflation_yoy": "Lạm phát cơ bản (so với cùng kỳ năm trước)",
    "cpi_food": "Lạm phát lương thực (so với tháng trước)",
    "cpi_transport": "CPI nhóm giao thông (so với tháng trước)",
    "ppi_yoy": "Chỉ số giá sản xuất công nghiệp (so với cùng kỳ năm trước)",
    "unemployment_rate": "Tỷ lệ thất nghiệp",
    "underemployment_rate": "Tỷ lệ thiếu việc làm",
    "labor_force": "Lực lượng lao động",
    "avg_income": "Thu nhập bình quân người lao động",
}
_MACRO_SEEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "macro_seen.json")


@st.cache_data(ttl=300)
def load_macro_latest_periods() -> dict[str, str]:
    """Most recent data period per macro indicator, as ISO date strings."""
    with get_session() as s:
        rows = s.execute(
            select(MacroIndicator.indicator, sqlfunc.max(MacroIndicator.period))
            .group_by(MacroIndicator.indicator)
        ).all()
        return {indicator: period.isoformat() for indicator, period in rows if period}


def check_macro_updates() -> list[str]:
    """Compare latest macro data periods to the last-seen snapshot.

    Returns the indicators (keys of _MACRO_INDICATOR_LABELS) that have newer data
    than last time this was checked, and records the new snapshot.
    """
    latest = {k: v for k, v in load_macro_latest_periods().items() if k in _MACRO_INDICATOR_LABELS}
    try:
        with open(_MACRO_SEEN_PATH, encoding="utf-8") as f:
            seen = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        seen = {}

    updated = [k for k, v in latest.items() if v != seen.get(k)]
    if updated:
        os.makedirs(os.path.dirname(_MACRO_SEEN_PATH), exist_ok=True)
        with open(_MACRO_SEEN_PATH, "w", encoding="utf-8") as f:
            json.dump(latest, f)
    return updated


@st.cache_data(ttl=3600)
def load_financials_q(ticker: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Q")
            .order_by(Financial.period.asc())
        ).scalars().all()
        cols = [c.key for c in Financial.__table__.columns]
        return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])


@st.cache_data(ttl=3600)
def load_financials_y(ticker: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Y")
            .order_by(Financial.period.asc())
        ).scalars().all()
        cols = [c.key for c in Financial.__table__.columns]
        return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])


@st.cache_data(ttl=3600)
def get_dcf(ticker: str) -> dict | None:
    inputs = prepare_dcf_inputs(ticker)
    if inputs is None or inputs["fcff_base"] <= 0:
        return None
    result = dcf_valuation(
        fcff_base        = inputs["fcff_base"],
        net_debt_bn      = inputs["net_debt_bn"],
        shares_millions  = inputs["shares_millions"],
        fcff_growth_rate = inputs["fcff_growth_rate"],
        beta             = inputs["beta"],
        cost_of_debt     = inputs["cost_of_debt"],
        debt_bn          = inputs["debt_bn"],
        equity_bn        = inputs["equity_bn"],
    )
    result["inputs"] = inputs
    return result


@st.cache_data(ttl=3600)
def get_all_valuations(ticker: str) -> dict:
    """Compute all 10 intrinsic price estimates for a ticker."""
    ttm   = compute_ttm(ticker)
    dcf_r = get_dcf(ticker)
    dcf_price = dcf_r["price_per_share"] if dcf_r else None

    empty = {k: None for k in ("dcf","fcfe","graham","pe","pb","ev_ebitda","epv","ps","ri","pocf")}
    if ttm is None:
        empty["dcf"] = dcf_price
        return empty

    shares     = ttm.get("shares_outstanding") or 0
    equity     = ttm.get("equity")
    net_income = ttm.get("net_income")
    debt       = ttm.get("debt") or 0.0
    cash       = ttm.get("cash") or 0.0
    net_debt   = debt - cash
    g          = dcf_r["inputs"]["fcff_growth_rate"] if dcf_r else 0.12
    wacc_val   = dcf_r["wacc"] if dcf_r else None

    # 2. FCFE — operating CF − capex discounted at cost of equity
    fcfe_price = None
    if (fcfe_val := ttm.get("fcf")) and fcfe_val > 0 and shares > 0:
        res = dcf_valuation(fcff_base=fcfe_val, net_debt_bn=0,
                            shares_millions=shares, fcff_growth_rate=g,
                            wacc_override=cost_of_equity(DEFAULT_BETA))
        fcfe_price = res.get("price_per_share")

    # 3. Graham Number
    graham_price = None
    if equity and shares > 0:
        bvps = bvps_from_financials(equity, shares)
        eps  = (net_income * 1_000 / shares) if net_income and net_income > 0 else None
        graham_price = graham_number(eps, bvps) if eps else None

    # 4. P/E implied
    pe_price = None
    if net_income and net_income > 0 and shares > 0:
        pe_price = (net_income * 1_000 / shares) * MARKET_PE

    # 5. P/B implied (target 1.5×)
    pb_price = pb_implied(equity, shares) if equity and shares > 0 else None

    # 6. EV/EBITDA implied (target 8×)
    ebitda = ttm.get("ebitda") or (
        ((ttm.get("ebit") or 0) + (ttm.get("depreciation") or 0))
        if ttm.get("ebit") else None
    )
    ev_price = ev_ebitda_implied(ebitda, net_debt, shares) if ebitda and shares > 0 else None

    # 7. Earnings Power Value (zero-growth floor)
    epv_price = epv_implied(ttm.get("ebit"), net_debt, shares, wacc=wacc_val) if shares > 0 else None

    # 8. P/Sales implied (target 1.2×)
    ps_price = ps_implied(ttm.get("revenue"), shares) if shares > 0 else None

    # 9. Residual Income
    ri_price = residual_income_implied(equity, shares, net_income, g=g) if equity and shares > 0 else None

    # 10. Price/OCF (target 10×)
    pocf_price = pocf_implied(ttm.get("operating_cf"), shares) if shares > 0 else None

    return {
        "dcf": dcf_price, "fcfe": fcfe_price, "graham": graham_price, "pe": pe_price,
        "pb": pb_price, "ev_ebitda": ev_price, "epv": epv_price, "ps": ps_price,
        "ri": ri_price, "pocf": pocf_price,
    }


@st.cache_data(ttl=3600)
def valuation_history(ticker: str) -> pd.DataFrame:
    """Compute intrinsic-value estimates at each quarterly TTM snapshot."""
    import calendar
    from valuation.inputs import compute_fcff_ttm, _FLOW, _STOCK

    _cols = [c.key for c in Financial.__table__.columns]
    with get_session() as s:
        orm_rows = s.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Q")
            .order_by(Financial.period.asc())
        ).scalars().all()
        rows = [{c: getattr(r, c) for c in _cols} for r in orm_rows]

    if len(rows) < 4:
        return pd.DataFrame()

    inputs = prepare_dcf_inputs(ticker)
    g = inputs["fcff_growth_rate"] if inputs else 0.12
    coe = cost_of_equity(DEFAULT_BETA)

    records = []
    for i in range(3, len(rows)):
        window = rows[i - 3: i + 1]

        ttm: dict = {}
        for col in _FLOW:
            vals = [r[col] for r in window if r.get(col) is not None]
            ttm[col] = sum(vals) if vals else None
        for col in _STOCK:
            ttm[col] = window[-1].get(col)
        ca0 = window[-1].get("current_assets"); cl0 = window[-1].get("current_liabilities")
        ca1 = window[0].get("current_assets");  cl1 = window[0].get("current_liabilities")
        nwc0 = (ca0 - cl0) if ca0 and cl0 else None
        nwc1 = (ca1 - cl1) if ca1 and cl1 else None
        ttm["delta_nwc"] = (nwc0 - nwc1) if nwc0 is not None and nwc1 is not None else 0.0

        period = window[-1]["period"]
        yr, qn = int(period[:4]), int(period[6])
        mo = qn * 3
        dt = pd.Timestamp(f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}")

        shares = ttm.get("shares_outstanding") or 0
        equity = ttm.get("equity")
        net_income = ttm.get("net_income")

        dcf_price = None
        fcff = compute_fcff_ttm(ttm)
        if fcff and fcff > 0 and shares > 0:
            debt_bn   = ttm.get("debt") or 0
            equity_bn = ttm.get("equity") or 1
            cash_bn   = ttm.get("cash") or 0
            res = dcf_valuation(fcff_base=fcff, net_debt_bn=debt_bn - cash_bn,
                                shares_millions=shares, fcff_growth_rate=g,
                                beta=DEFAULT_BETA, cost_of_debt=DEFAULT_COD,
                                debt_bn=debt_bn, equity_bn=equity_bn)
            dcf_price = res.get("price_per_share")

        fcfe_price = None
        fcfe_val = ttm.get("fcf")
        if fcfe_val and fcfe_val > 0 and shares > 0:
            res2 = dcf_valuation(fcff_base=fcfe_val, net_debt_bn=0,
                                 shares_millions=shares, fcff_growth_rate=g,
                                 wacc_override=coe)
            fcfe_price = res2.get("price_per_share")

        graham_price = None
        if equity and shares > 0:
            bvps = bvps_from_financials(equity, shares)
            eps  = (net_income * 1_000 / shares) if net_income and net_income > 0 else None
            graham_price = graham_number(eps, bvps) if eps else None

        pe_price = None
        if net_income and net_income > 0 and shares > 0:
            pe_price = (net_income * 1_000 / shares) * MARKET_PE

        # 5. P/B
        pb_price = pb_implied(ttm.get("equity"), shares)

        # 6. EV/EBITDA
        ebitda_h = ttm.get("ebitda") or (
            ((ttm.get("ebit") or 0) + (ttm.get("depreciation") or 0))
            if ttm.get("ebit") else None
        )
        ev_price = ev_ebitda_implied(ebitda_h, (ttm.get("debt") or 0) - (ttm.get("cash") or 0), shares)

        # 7. EPV
        epv_price = epv_implied(ttm.get("ebit"), (ttm.get("debt") or 0) - (ttm.get("cash") or 0), shares)

        # 8. P/Sales
        ps_price = ps_implied(ttm.get("revenue"), shares)

        # 9. Residual Income
        ri_price = residual_income_implied(ttm.get("equity"), shares, net_income, g=g)

        # 10. P/OCF
        pocf_price = pocf_implied(ttm.get("operating_cf"), shares)

        valid = [p for p in [dcf_price, fcfe_price, graham_price, pe_price,
                              pb_price, ev_price, epv_price, ps_price, ri_price, pocf_price]
                 if p and p > 0]
        records.append({
            "date": dt,
            "dcf": dcf_price, "fcfe": fcfe_price,
            "graham": graham_price, "pe": pe_price,
            "avg": sum(valid) / len(valid) if valid else None,
        })

    return pd.DataFrame(records)


@st.cache_data(ttl=3600)
def load_available_tickers() -> list[str]:
    """Return sorted list of tickers that have at least 4 quarterly rows in DB."""
    with get_session() as s:
        rows = s.execute(
            select(Financial.ticker, sqlfunc.count(Financial.id))
            .where(Financial.period_type == "Q")
            .group_by(Financial.ticker)
            .having(sqlfunc.count(Financial.id) >= 4)
            .order_by(Financial.ticker)
        ).all()
    return [r[0] for r in rows] or ["VNM", "FPT", "VIC", "HPG"]


@st.cache_data(ttl=1800)
def load_valuation_screen_data() -> pd.DataFrame:
    """Return pre-computed valuation rows joined with latest prices + sector.

    Includes raw numeric columns for filtering (_roe_raw, _de_raw, etc.)
    and a composite quality_score (0-100).
    """
    with get_session() as s:
        subq = (
            select(Valuation.ticker, sqlfunc.max(Valuation.calc_date).label("max_date"))
            .group_by(Valuation.ticker)
            .subquery()
        )
        val_rows = s.execute(
            select(Valuation)
            .join(subq, (Valuation.ticker == subq.c.ticker) &
                         (Valuation.calc_date == subq.c.max_date))
            .order_by(Valuation.ticker)
        ).scalars().all()

        price_subq = (
            select(Price.ticker, sqlfunc.max(Price.date).label("max_date"))
            .group_by(Price.ticker)
            .subquery()
        )
        price_rows = s.execute(
            select(Price.ticker, Price.close, Price.date)
            .join(price_subq, (Price.ticker == price_subq.c.ticker) &
                               (Price.date == price_subq.c.max_date))
        ).all()
        price_map = {r[0]: r[1] * 1000 for r in price_rows if r[1]}

        # Sector lookup
        company_rows = s.execute(select(Company.ticker, Company.sector)).all()
        sector_map = {r[0]: (r[1] or "Unknown") for r in company_rows}

        records = []
        for v in val_rows:
            cur_price = price_map.get(v.ticker)
            upside = None
            if v.dcf_estimate and cur_price and cur_price > 0:
                upside = (v.dcf_estimate - cur_price) / cur_price

            # Avg Estimate = pre-computed average of all 10 valuation methods
            avg_est = v.avg_intrinsic_value if hasattr(v, "avg_intrinsic_value") else None
            avg_upside = ((avg_est - cur_price) / cur_price) if (avg_est and cur_price and cur_price > 0) else None

            qs = compute_quality_score(
                v.roe, v.net_margin, v.profit_quality,
                v.fcf_margin, v.current_ratio, v.debt_to_equity
            )

            records.append({
                "Ticker":        v.ticker,
                "Sector":        sector_map.get(v.ticker, "Unknown"),
                "Price (VND)":   f"{cur_price:,.0f}" if cur_price else "—",
                "Avg Estimate":  f"{avg_est:,.0f}" if avg_est else "—",
                "Avg Upside":    f"{avg_upside*100:+.1f}%" if avg_upside is not None else "—",
                "DCF Estimate":  f"{v.dcf_estimate:,.0f}" if v.dcf_estimate else "—",
                "FCFE Estimate": f"{v.fcfe_estimate:,.0f}" if (hasattr(v,'fcfe_estimate') and v.fcfe_estimate) else "—",
                "Upside":        f"{upside*100:+.1f}%" if upside is not None else "—",
                "Quality":       f"{qs:.0f}",
                "Graham Number": f"{v.graham_number:,.0f}" if v.graham_number else "—",
                "P/E":           f"{v.pe:.1f}x" if v.pe else "—",
                "P/B":           f"{v.pb:.1f}x" if v.pb else "—",
                "Net Margin":    fmt_pct(v.net_margin),
                "ROE":           fmt_pct(v.roe),
                "FCF Margin":    fmt_pct(v.fcf_margin),
                "D/E":           f"{v.debt_to_equity:.2f}x" if v.debt_to_equity else "—",
                "Current Ratio": f"{v.current_ratio:.2f}x" if v.current_ratio else "—",
                # Raw values for filtering and sorting
                "_upside_raw":   upside if upside is not None else -999,
                "_roe_raw":      v.roe if v.roe is not None else -999,
                "_nm_raw":        v.net_margin if v.net_margin is not None else -999,
                "_de_raw":        v.debt_to_equity if v.debt_to_equity is not None else 999,
                "_fcfm_raw":      v.fcf_margin if v.fcf_margin is not None else -999,
                "_cr_raw":        v.current_ratio if v.current_ratio is not None else 0,
                "_qs_raw":        qs,
                "_avg_upside_raw": avg_upside if avg_upside is not None else -9999,
            })

    return pd.DataFrame(records)


@st.cache_data(ttl=1800)
def load_sector_ticker_data() -> pd.DataFrame:
    """Return ticker-level valuation data with sector — for heatmap and top-N per sector."""
    with get_session() as s:
        subq = (
            select(Valuation.ticker, sqlfunc.max(Valuation.calc_date).label("max_date"))
            .group_by(Valuation.ticker).subquery()
        )
        val_rows = s.execute(
            select(Valuation)
            .join(subq, (Valuation.ticker == subq.c.ticker) &
                         (Valuation.calc_date == subq.c.max_date))
        ).scalars().all()
        price_subq = (
            select(Price.ticker, sqlfunc.max(Price.date).label("max_date"))
            .group_by(Price.ticker).subquery()
        )
        price_rows = s.execute(
            select(Price.ticker, Price.close)
            .join(price_subq, (Price.ticker == price_subq.c.ticker) &
                               (Price.date == price_subq.c.max_date))
        ).all()
        price_map = {r[0]: r[1] * 1000 for r in price_rows if r[1]}
        comp_rows = s.execute(
            select(Company.ticker, Company.sector, Company.name)
        ).all()
        sector_map = {r[0]: (r[1] or "Unknown") for r in comp_rows}
        name_map   = {r[0]: (r[2] or r[0])       for r in comp_rows}

        # Get latest shares_outstanding per ticker for market cap
        fin_subq = (
            select(Financial.ticker, sqlfunc.max(Financial.period).label("max_period"))
            .where(Financial.period_type == "Q")
            .group_by(Financial.ticker).subquery()
        )
        shares_rows = s.execute(
            select(Financial.ticker, Financial.shares_outstanding)
            .join(fin_subq, (Financial.ticker == fin_subq.c.ticker) &
                             (Financial.period == fin_subq.c.max_period))
            .where(Financial.period_type == "Q")
        ).all()
        shares_map = {r[0]: r[1] for r in shares_rows if r[1]}

        rows = []
        for v in val_rows:
            sec = sector_map.get(v.ticker, "Unknown")
            if sec == "Unknown":
                continue
            cur = price_map.get(v.ticker)
            upside = ((v.dcf_estimate - cur) / cur * 100) if (v.dcf_estimate and cur and cur > 0) else None
            shares_m = shares_map.get(v.ticker)
            # market cap in VND billions (price_vnd × shares_M × 1e6 / 1e9)
            mcap = (cur * shares_m * 1e6 / 1e9) if (cur and shares_m) else 1.0
            # Avg estimate = mean of available intrinsic value methods
            _estimates = [e for e in [v.dcf_estimate, v.graham_number] if e and e > 0]
            avg_est   = sum(_estimates) / len(_estimates) if _estimates else None
            avg_upside = ((avg_est - cur) / cur * 100) if (avg_est and cur and cur > 0) else None

            rows.append({
                "ticker":     v.ticker,
                "name":       name_map.get(v.ticker, v.ticker),
                "sector":     sec,
                "price":      cur,
                "dcf":        v.dcf_estimate,
                "graham":     v.graham_number,
                "avg_est":    avg_est,
                "upside_pct": upside,
                "avg_upside": avg_upside,
                "pe":         v.pe,
                "pb":         v.pb,
                "roe":        v.roe * 100 if v.roe else None,
                "net_margin": v.net_margin * 100 if v.net_margin else None,
                "mcap":       max(mcap, 1.0),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def load_sector_data() -> pd.DataFrame:
    """Return sector-level summary: median metrics per sector from valuations table."""
    with get_session() as s:
        subq = (
            select(Valuation.ticker, sqlfunc.max(Valuation.calc_date).label("max_date"))
            .group_by(Valuation.ticker)
            .subquery()
        )
        val_rows = s.execute(
            select(Valuation)
            .join(subq, (Valuation.ticker == subq.c.ticker) &
                         (Valuation.calc_date == subq.c.max_date))
        ).scalars().all()

        price_subq = (
            select(Price.ticker, sqlfunc.max(Price.date).label("max_date"))
            .group_by(Price.ticker)
            .subquery()
        )
        price_rows = s.execute(
            select(Price.ticker, Price.close)
            .join(price_subq, (Price.ticker == price_subq.c.ticker) &
                               (Price.date == price_subq.c.max_date))
        ).all()
        price_map = {r[0]: r[1] * 1000 for r in price_rows if r[1]}

        company_rows = s.execute(select(Company.ticker, Company.sector)).all()
        sector_map = {r[0]: (r[1] or "Unknown") for r in company_rows}

        rows = []
        for v in val_rows:
            cur_price = price_map.get(v.ticker)
            upside = None
            if v.dcf_estimate and cur_price and cur_price > 0:
                upside = (v.dcf_estimate - cur_price) / cur_price

            qs = compute_quality_score(
                v.roe, v.net_margin, v.profit_quality,
                v.fcf_margin, v.current_ratio, v.debt_to_equity
            )

            rows.append({
                "sector":       sector_map.get(v.ticker, "Unknown"),
                "ticker":       v.ticker,
                "upside":       upside,
                "roe":          v.roe,
                "net_margin":   v.net_margin,
                "fcf_margin":   v.fcf_margin,
                "de":           v.debt_to_equity,
                "current_ratio": v.current_ratio,
                "pe":           v.pe,
                "pb":           v.pb,
                "quality":      qs,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["sector"] != "Unknown"]

    def _med(s):
        return s.dropna().median()

    grouped = (
        df.groupby("sector")
        .agg(
            tickers=("ticker", "count"),
            dcf_upside_pct=("upside", lambda s: _med(s) * 100 if not s.dropna().empty else None),
            roe_pct=("roe", lambda s: _med(s) * 100),
            net_margin_pct=("net_margin", lambda s: _med(s) * 100),
            fcf_margin_pct=("fcf_margin", lambda s: _med(s) * 100),
            de=("de", _med),
            current_ratio=("current_ratio", _med),
            pe=("pe", _med),
            pb=("pb", lambda s: _med(s)),
            quality=("quality", _med),
        )
        .reset_index()
        .sort_values("dcf_upside_pct", ascending=False)
    )
    return grouped


@st.cache_data(ttl=1800)
def load_watchlist_data(min_upside: float = 0.20) -> pd.DataFrame:
    """Return watchlist rows: DCF upside > min_upside AND positive FCFF."""
    with get_session() as s:
        subq = (
            select(Valuation.ticker, sqlfunc.max(Valuation.calc_date).label("max_date"))
            .group_by(Valuation.ticker)
            .subquery()
        )
        val_rows = s.execute(
            select(Valuation)
            .join(subq, (Valuation.ticker == subq.c.ticker) &
                         (Valuation.calc_date == subq.c.max_date))
            .where(Valuation.fcff > 0)
            .order_by(Valuation.ticker)
        ).scalars().all()

        price_subq = (
            select(Price.ticker, sqlfunc.max(Price.date).label("max_date"))
            .group_by(Price.ticker)
            .subquery()
        )
        price_rows = s.execute(
            select(Price.ticker, Price.close)
            .join(price_subq, (Price.ticker == price_subq.c.ticker) &
                               (Price.date == price_subq.c.max_date))
        ).all()
        price_map = {r[0]: r[1] * 1000 for r in price_rows if r[1]}

        records = []
        for v in val_rows:
            if not v.dcf_estimate:
                continue
            cur_price = price_map.get(v.ticker)
            if not cur_price or cur_price <= 0:
                continue
            upside = (v.dcf_estimate - cur_price) / cur_price
            if upside < min_upside:
                continue
            records.append({
                "Ticker":       v.ticker,
                "Price (VND)":  f"{cur_price:,.0f}",
                "DCF Estimate": f"{v.dcf_estimate:,.0f}",
                "Upside":       f"{upside*100:+.1f}%",
                "WACC":         f"{v.wacc*100:.2f}%" if v.wacc else "—",
                "TTM FCFF":     fmt_bn(v.fcff),
                "Net Margin":   fmt_pct(v.net_margin),
                "ROE":          fmt_pct(v.roe),
                "_upside_raw":  upside,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("_upside_raw", ascending=False)
    return df


def get_pinned_tickers() -> list[str]:
    """Return list of pinned ticker symbols from DB."""
    from models.database import engine
    from models.schema import Base
    Base.metadata.create_all(engine, checkfirst=True)  # ensure table exists
    with get_session() as s:
        rows = s.execute(select(PinnedTicker.ticker)).scalars().all()
        return list(rows)


def pin_ticker(ticker: str) -> None:
    from datetime import date as _date
    from models.database import engine
    from models.schema import Base
    Base.metadata.create_all(engine, checkfirst=True)
    with get_session() as s:
        existing = s.execute(
            select(PinnedTicker).where(PinnedTicker.ticker == ticker)
        ).scalar_one_or_none()
        if not existing:
            s.add(PinnedTicker(ticker=ticker, added_date=_date.today(), note=""))


def unpin_ticker(ticker: str) -> None:
    with get_session() as s:
        s.execute(
            PinnedTicker.__table__.delete().where(PinnedTicker.ticker == ticker)
        )


@st.cache_data(ttl=300)
def load_market_snapshot() -> "pd.DataFrame":
    """Return today's price snapshot: close, prev_close, change%, volume for all tickers."""
    with get_session() as s:
        price_subq = (
            select(Price.ticker, sqlfunc.max(Price.date).label("max_date"))
            .group_by(Price.ticker).subquery()
        )
        # Latest price
        latest = s.execute(
            select(Price.ticker, Price.close, Price.volume, Price.date)
            .join(price_subq, (Price.ticker == price_subq.c.ticker) &
                               (Price.date == price_subq.c.max_date))
        ).all()
        latest_map = {r[0]: {"close": r[1], "volume": r[2], "date": r[3]} for r in latest}

        # Second latest price (for change calc)
        all_prices = s.execute(
            select(Price.ticker, Price.close, Price.date)
            .order_by(Price.ticker, Price.date.desc())
        ).all()
        comp_rows = s.execute(select(Company.ticker, Company.sector, Company.name)).all()
        sector_map = {r[0]: r[1] for r in comp_rows}
        name_map   = {r[0]: r[2] for r in comp_rows}

    # Build prev-close map (second most recent)
    from collections import defaultdict
    ticker_dates: dict = defaultdict(list)
    for r in all_prices:
        ticker_dates[r[0]].append((r[2], r[1]))
    prev_map = {}
    for tkr, dates in ticker_dates.items():
        dates.sort(key=lambda x: x[0], reverse=True)
        if len(dates) >= 2:
            prev_map[tkr] = dates[1][1]

    rows = []
    for tkr, data in latest_map.items():
        cur = data["close"] * 1000 if data["close"] else None
        prev = prev_map.get(tkr)
        prev_vnd = prev * 1000 if prev else None
        chg_pct = ((cur - prev_vnd) / prev_vnd * 100) if (cur and prev_vnd and prev_vnd > 0) else None
        rows.append({
            "ticker":   tkr,
            "name":     name_map.get(tkr, tkr),
            "sector":   sector_map.get(tkr, "Unknown"),
            "price":    cur,
            "prev":     prev_vnd,
            "chg_pct":  chg_pct,
            "volume":   data["volume"],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_vnindex_prices(days: int = 504) -> "pd.DataFrame":
    """Fetch VNINDEX daily close prices for benchmark comparison."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol="VNINDEX", source="VCI")
            from datetime import date, timedelta
            end = date.today().strftime("%Y-%m-%d")
            start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = stock.quote.history(start=start, end=end, interval="1D")
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"time": "date"})
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "close"]].sort_values("date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_foreign_flow(code: str = "VNINDEX", sessions: int = 15) -> "pd.DataFrame":
    """Foreign investors' net trading value over the last N sessions.

    Source: VNDirect finfo. `code="VNINDEX"` returns whole-market (HOSE) net
    flow; a ticker symbol returns that stock's foreign net. `netVal` is in VND.
    Returns columns: date, buy_val, sell_val, net_val (all VND), sorted ascending.
    """
    import requests
    from datetime import date, timedelta
    try:
        # pull a generous window (calendar days) then keep the last N sessions
        start = (date.today() - timedelta(days=sessions * 3 + 20)).strftime("%Y-%m-%d")
        url = (
            "https://api-finfo.vndirect.com.vn/v4/foreigns"
            f"?q=code:{code}~tradingDate:gte:{start}"
            f"&size=200&sort=tradingDate:asc"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
            "Referer": "https://dstock.vndirect.com.vn/",
        }
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["tradingDate"])
        for col in ("buyVal", "sellVal", "netVal", "buyVol", "sellVol", "netVol"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df = (df.rename(columns={"buyVal": "buy_val", "sellVal": "sell_val", "netVal": "net_val",
                                  "buyVol": "buy_vol", "sellVol": "sell_vol", "netVol": "net_vol"})
                [["date", "buy_val", "sell_val", "net_val", "buy_vol", "sell_vol", "net_vol"]]
                .dropna(subset=["net_val"])
                .sort_values("date")
                .tail(sessions)
                .reset_index(drop=True))
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_latest_prices() -> dict[str, float]:
    """Return {ticker: latest_close_vnd} for all tickers with price data."""
    with get_session() as s:
        price_subq = (
            select(Price.ticker, sqlfunc.max(Price.date).label("max_date"))
            .group_by(Price.ticker)
            .subquery()
        )
        rows = s.execute(
            select(Price.ticker, Price.close)
            .join(price_subq, (Price.ticker == price_subq.c.ticker) &
                               (Price.date == price_subq.c.max_date))
        ).all()
    return {r[0]: r[1] * 1000 for r in rows if r[1]}


@st.cache_data(ttl=3600)
def load_detailed_financials(ticker: str):
    """Fetch raw VCI income statement + balance sheet for detailed sub-item charts.

    Returns (income_df, balance_df) in wide format (item_id × periods).
    Falls back to (empty, empty) on any error.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from vnstock.explorer.vci.financial import Finance
            fin = Finance(ticker, period="quarter", show_log=False)
            inc = fin._get_report("income_statement", period="quarter",
                                  lang="en", show_log=False, limit=50)
            bal = fin._get_report("balance_sheet", period="quarter",
                                  lang="en", show_log=False, limit=50)
        return inc, bal
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def load_annual_cf(ticker: str):
    """Fetch annual cash flow + income statement from VCI for dividends chart."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from vnstock.explorer.vci.financial import Finance
            fin = Finance(ticker, period="year", show_log=False)
            cf  = fin._get_report("cash_flow",        period="year", lang="en", show_log=False, limit=20)
            inc = fin._get_report("income_statement", period="year", lang="en", show_log=False, limit=20)
        return cf, inc
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def load_analyst_recommendation(ticker: str) -> dict:
    """Fetch current analyst recommendation from VCI via vnstock.

    Returns dict with keys: rating, target_price, upside, analyst, rating_as_of, current_price
    Returns empty dict on failure.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=ticker, source="VCI")
            ov = stock.company.overview()
            if ov is None or ov.empty:
                return {}
            row = ov.iloc[0]
            return {
                "rating":       str(row.get("rating") or ""),
                "target_price": float(row.get("target_price") or 0),
                "upside":       float(row.get("upside_to_target_percent") or 0),
                "analyst":      str(row.get("analyst") or ""),
                "rating_as_of": str(row.get("rating_as_of") or ""),
                "current_price": float(row.get("current_price") or 0),
            }
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_commodity_prices(symbols: tuple, period: str = "1y") -> "pd.DataFrame":
    """Fetch Yahoo Finance commodity prices, return normalized (base=100) DataFrame."""
    import yfinance as yf
    frames = {}
    for sym, label, _ in symbols:
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                close = df["Close"].squeeze()
                base = close.iloc[0]
                if base and base != 0:
                    frames[label] = (close / base * 100).round(2)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


@st.cache_data(ttl=600)
def load_bank_kbs_data(ticker: str) -> "pd.DataFrame":
    """Fetch KBS balance sheet + income for banking stocks.

    Returns a DataFrame indexed by period label (e.g. 'Q1/26') with columns:
      interest_income, interest_expense, nii, net_profit, operating_expenses,
      provision, deposits, sbv_borrowings, interbank_borrowings, total_assets
    All values in VND billions.
    """
    import warnings, re
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from vnstock import Finance
            fin = Finance(symbol=ticker, source="KBS")
            inc = fin.income_statement(period="quarter")
            bal = fin.balance_sheet(period="quarter")
    except Exception:
        return pd.DataFrame()

    if inc is None or inc.empty or bal is None or bal.empty:
        return pd.DataFrame()

    # Period columns: "2026-Q1", "2025-Q4" (skip "_1" restated duplicates)
    def _period_cols(df):
        return sorted(
            [c for c in df.columns if re.match(r"^\d{4}-Q[1-4]$", str(c))],
            reverse=True,
        )

    def _get(df, item_id, col, scale=1e9):
        mask = df["item_id"] == item_id
        if not mask.any() or col not in df.columns:
            return None
        try:
            v = float(df.loc[mask, col].values[0])
            return None if pd.isna(v) else v / scale
        except Exception:
            return None

    inc_cols = _period_cols(inc)
    bal_cols = _period_cols(bal)
    # align to income statement periods (shorter set)
    periods = inc_cols

    def _fmt(p):
        yr, q = p.split("-Q")
        return f"Q{q}/{yr[2:]}"

    rows = []
    for p in periods:
        bp = p if p in bal_cols else None
        rows.append({
            "period": _fmt(p),
            "interest_income":      _get(inc, "interest_income_and_similar_income", p),
            "interest_expense":     _get(inc, "interest_expense_and_similar_expenses", p),
            "nii":                  _get(inc, "net_interest_income", p),
            "net_profit":           _get(inc, "net_profit", p),
            "operating_expenses":   _get(inc, "operating_expenses", p),
            "provision":            _get(inc, "provision_for_credit_losses", p),
            "deposits":             _get(bal, "deposits_from_customers", bp) if bp else None,
            "sbv_borrowings":       _get(bal, "due_to_government_and_borrowings_from_the_state_bank_of_vietnam", bp) if bp else None,
            "interbank_borrowings": _get(bal, "placements_and_borrowings_from_other_credit_institutions", bp) if bp else None,
            "total_assets":         _get(bal, "total_assets", bp) if bp else None,
        })

    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)  # oldest first


@st.cache_data(ttl=3600)
def load_annual_financials(ticker: str) -> "pd.DataFrame":
    """Load annual revenue + net_income from DB (period_type='Y')."""
    with get_session() as s:
        rows = s.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Y")
            .order_by(Financial.period.asc())
        ).scalars().all()
        return pd.DataFrame([{
            "period":     r.period,
            "revenue":    r.revenue,
            "net_income": r.net_income,
        } for r in rows])


@st.cache_data(ttl=3600)
def load_valuation_multiples(ticker: str) -> "pd.DataFrame":
    """Compute quarterly P/E and P/B from DB price history + financial data."""
    import datetime
    from sqlalchemy import select
    from models.schema import Price, Financial
    _fin_cols = ["period", "net_income", "equity", "shares_outstanding"]
    with get_session() as session:
        q_list = [
            {c: getattr(r, c) for c in _fin_cols}
            for r in session.execute(
                select(Financial)
                .where(Financial.ticker == ticker, Financial.period_type == "Q")
                .order_by(Financial.period.asc())
            ).scalars().all()
        ]
        prices_raw = session.execute(
            select(Price.date, Price.close)
            .where(Price.ticker == ticker)
            .order_by(Price.date.asc())
        ).all()
    if not q_list or not prices_raw:
        return pd.DataFrame()
    price_s = pd.Series(
        {pd.Timestamp(r[0]): float(r[1]) * 1000 for r in prices_raw}
    ).sort_index()
    _qend = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    results = []
    for i, row in enumerate(q_list):
        year, qnum = int(row["period"][:4]), int(row["period"][6])
        m, d = _qend[qnum]
        qend = pd.Timestamp(datetime.date(year, m, d))
        past = price_s[price_s.index <= qend]
        if past.empty:
            continue
        price = float(past.iloc[-1])
        window = q_list[max(0, i - 3): i + 1]
        if len(window) < 4:
            continue
        ni_sum = sum(r["net_income"] for r in window if r["net_income"] is not None)
        sh = row["shares_outstanding"]
        ttm_eps = ni_sum * 1000 / sh if sh and sh > 0 and ni_sum != 0 else None
        bvps = row["equity"] * 1000 / sh if row["equity"] and sh and sh > 0 else None
        _pe = round(price / ttm_eps, 2) if ttm_eps else None
        _pb = round(price / bvps,    2) if bvps and bvps > 0 else None
        results.append({
            "period": row["period"],
            "pe": _pe,
            "pb": _pb,
        })
    return pd.DataFrame(results)


@st.cache_data(ttl=3600)
def load_market_valuation_history() -> "pd.DataFrame":
    """Median market-wide P/E and P/B per quarter, across all tickers.

    For each ticker/quarter: TTM EPS = trailing-4Q net income / shares,
    BVPS = equity / shares, price = last close on/before quarter-end.
    P/E and P/B per ticker/quarter are then medianed across tickers.
    """
    import datetime
    _qend = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

    with get_session() as s:
        fin_rows = s.execute(
            select(Financial.ticker, Financial.period, Financial.net_income,
                   Financial.equity, Financial.shares_outstanding)
            .where(Financial.period_type == "Q")
            .order_by(Financial.ticker, Financial.period)
        ).all()
        price_rows = s.execute(
            select(Price.ticker, Price.date, Price.close)
            .order_by(Price.ticker, Price.date)
        ).all()

    if not fin_rows or not price_rows:
        return pd.DataFrame()

    fin_df = pd.DataFrame(fin_rows, columns=["ticker", "period", "net_income", "equity", "shares"])
    fin_df["year"]  = fin_df["period"].str[:4].astype(int)
    fin_df["qnum"]  = fin_df["period"].str[6].astype(int)
    fin_df["qend"]  = fin_df.apply(lambda r: pd.Timestamp(datetime.date(r["year"], *_qend[r["qnum"]])), axis=1)
    fin_df = fin_df.sort_values(["ticker", "qend"])
    # TTM net income = trailing 4 quarters (per ticker)
    fin_df["ttm_ni"] = fin_df.groupby("ticker")["net_income"].transform(
        lambda s: s.rolling(4, min_periods=4).sum())
    fin_df["ttm_eps"] = np.where(
        (fin_df["shares"] > 0) & fin_df["ttm_ni"].notna() & (fin_df["ttm_ni"] != 0),
        fin_df["ttm_ni"] * 1000 / fin_df["shares"], np.nan)
    fin_df["bvps"] = np.where(
        (fin_df["shares"] > 0) & (fin_df["equity"] > 0),
        fin_df["equity"] * 1000 / fin_df["shares"], np.nan)

    price_df = pd.DataFrame(price_rows, columns=["ticker", "date", "close"])
    price_df["date"]  = pd.to_datetime(price_df["date"])
    price_df["close"] = price_df["close"] * 1000

    out = []
    for tkr, fin_g in fin_df.groupby("ticker"):
        px = price_df[price_df["ticker"] == tkr]
        if px.empty:
            continue
        merged = pd.merge_asof(
            fin_g.sort_values("qend"), px.sort_values("date")[["date", "close"]],
            left_on="qend", right_on="date", direction="backward")
        merged["pe"] = np.where(merged["ttm_eps"].notna() & (merged["ttm_eps"] > 0),
                                 merged["close"] / merged["ttm_eps"], np.nan)
        merged["pb"] = np.where(merged["bvps"].notna() & (merged["bvps"] > 0),
                                 merged["close"] / merged["bvps"], np.nan)
        out.append(merged[["period", "qend", "pe", "pb"]])

    if not out:
        return pd.DataFrame()

    all_df = pd.concat(out, ignore_index=True)
    # Drop unreasonable outliers before taking the median
    all_df.loc[(all_df["pe"] <= 0) | (all_df["pe"] > 100), "pe"] = np.nan
    all_df.loc[(all_df["pb"] <= 0) | (all_df["pb"] > 20),  "pb"] = np.nan

    agg = (all_df.groupby(["period", "qend"])
                  .agg(median_pe=("pe", "median"), median_pb=("pb", "median"),
                       n_pe=("pe", "count"), n_pb=("pb", "count"))
                  .reset_index()
                  .sort_values("qend"))
    agg = agg[(agg["n_pe"] >= 10) | (agg["n_pb"] >= 10)]
    return agg


_INDEX_SYMBOLS = ["VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30"]


def _fetch_one_index_intraday(sym: str, today, week_ago) -> tuple[str, dict | None]:
    from vnstock import Quote
    try:
        q = Quote(symbol=sym, source="VCI")
        intraday = q.history(start=today.isoformat(), end=today.isoformat(), interval="1m")
        if intraday.empty:
            return sym, None
        intraday_date = pd.to_datetime(intraday["time"]).dt.date.iloc[-1]
        if intraday_date != today:
            # Weekend/holiday: vnstock returns just the last 1-minute bar of
            # the most recent session — re-fetch and keep only that session's rows
            # (vnstock's 1m interval ignores the single-day range and may
            # return a wider recent window).
            intraday = q.history(start=intraday_date.isoformat(), end=intraday_date.isoformat(), interval="1m")
            intraday = intraday[pd.to_datetime(intraday["time"]).dt.date == intraday_date]
            if intraday.empty:
                return sym, None
        daily = q.history(start=week_ago.isoformat(), end=today.isoformat(), interval="1D")
        current  = float(intraday["close"].iloc[-1])
        day_open = float(intraday["open"].iloc[0])
        prev_close = day_open
        if not daily.empty:
            # Use the actual session date returned by the intraday call (not
            # `today`) — on weekends/holidays vnstock returns the most recent
            # trading day's intraday data, so prev_close must come from the
            # day BEFORE that session, not the day before today.
            intraday_date = pd.to_datetime(intraday["time"]).dt.date.iloc[-1]
            daily_dates = pd.to_datetime(daily["time"]).dt.date
            prior = daily[daily_dates < intraday_date]
            if not prior.empty:
                prev_close = float(prior["close"].iloc[-1])
        chg = current - prev_close
        return sym, {
            "intraday": intraday,
            "current": current,
            "open": day_open,
            "prev_close": prev_close,
            "chg": chg,
            "chg_pct": (chg / prev_close * 100) if prev_close else 0.0,
            "volume": float(intraday["volume"].sum()),
        }
    except Exception:
        return sym, None


@st.cache_data(ttl=60)
def load_index_intraday() -> dict:
    """Today's 1-minute OHLCV for the main indices, via vnstock VCI source.

    Returns {symbol: {"intraday": df, "current", "open", "prev_close",
    "chg", "chg_pct", "volume"}}. Symbols with no data are omitted.
    Fetched in parallel since each symbol needs 2 sequential API calls.
    """
    import datetime
    from concurrent.futures import ThreadPoolExecutor

    today    = datetime.date.today()
    week_ago = today - datetime.timedelta(days=10)
    out = {}
    with ThreadPoolExecutor(max_workers=len(_INDEX_SYMBOLS)) as ex:
        for sym, data in ex.map(lambda s: _fetch_one_index_intraday(s, today, week_ago), _INDEX_SYMBOLS):
            if data is not None:
                out[sym] = data
    # Preserve the canonical display order
    return {sym: out[sym] for sym in _INDEX_SYMBOLS if sym in out}


_IDX_LABELS = {
    "VNINDEX": "VNINDEX", "HNXINDEX": "HNXINDEX", "UPCOMINDEX": "UPINDEX",
    "VN30": "VN30", "HNX30": "HNX30",
}


@st.fragment(run_every=30)
def _render_index_ticker_bar():
    """Live intraday mini-charts for the main indices — re-fetches every 30s during trading hours."""
    _idx_data = load_index_intraday()
    if not _idx_data:
        return
    _idx_cols = st.columns(len(_idx_data))
    for _col, (_sym, _d) in zip(_idx_cols, _idx_data.items()):
        _up   = _d["chg"] >= 0
        _clr  = "#22c55e" if _up else "#ef4444"
        _ia  = (_d["intraday"].copy()
                .drop_duplicates(subset=["time"])
                .sort_values("time")
                .reset_index(drop=True))
        _ia["tlabel"] = pd.to_datetime(_ia["time"]).dt.strftime("%H:%M")
        _y_vals = _ia["close"].dropna()
        _y_min  = float(_y_vals.min()) if not _y_vals.empty else _d["open"]
        _y_max  = float(_y_vals.max()) if not _y_vals.empty else _d["open"]
        _y_pad  = max((_y_max - _y_min) * 0.25, 1.0)
        _y_rng  = (_y_max + _y_pad) - (_y_min - _y_pad)
        _xs     = list(range(len(_ia)))
        _cdata  = list(zip(_ia["tlabel"], _ia["volume"].fillna(0).astype(int)))

        # Scale volume bars into bottom 25% of the price y-range (no subplot needed)
        _v_max  = max(float(_ia["volume"].max()), 1)
        _v_base = _y_min - _y_pad
        _v_h    = [(_v / _v_max) * 0.25 * _y_rng
                   for _v in _ia["volume"].fillna(0)]

        _fig_i = go.Figure()
        _fig_i.add_trace(go.Bar(
            x=_xs, y=_v_h, base=_v_base,
            marker_color=_clr, opacity=0.35, hoverinfo="skip"))
        _fig_i.add_trace(go.Scatter(
            x=_xs, y=_ia["close"].tolist(), mode="lines",
            line=dict(color=_clr, width=1.5),
            customdata=_cdata,
            hovertemplate="<b>%{customdata[0]}</b>  %{y:,.2f}<br>KL: %{customdata[1]:,}<extra></extra>"))

        # Header annotations in top margin
        _fig_i.add_annotation(x=0.5, y=1.58, xref="paper", yref="paper",
            text=f"<b>{_IDX_LABELS.get(_sym, _sym)}</b>",
            showarrow=False, font=dict(size=11, color="#9ca3af"), xanchor="center")
        _fig_i.add_annotation(x=0.5, y=1.36, xref="paper", yref="paper",
            text=f"<b>{_d['current']:,.2f}</b>",
            showarrow=False, font=dict(size=17, color=_clr), xanchor="center")
        _fig_i.add_annotation(x=0.5, y=1.16, xref="paper", yref="paper",
            text=f"{_d['chg']:+.2f} ({_d['chg_pct']:+.2f}%)",
            showarrow=False, font=dict(size=11, color=_clr), xanchor="center")

        # Card border
        _fig_i.add_shape(type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1.7,
            line=dict(color="rgba(148,163,184,0.2)", width=1),
            fillcolor="rgba(0,0,0,0)")

        _fig_i.update_layout(
            height=175, margin=dict(l=8, r=8, t=70, b=8), dragmode=False,
            showlegend=False, hovermode="x", bargap=0,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, showticklabels=False, ticks="",
                       showspikes=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showticklabels=False, ticks="",
                       showgrid=False, zeroline=False,
                       range=[_y_min - _y_pad, _y_max + _y_pad]),
            hoverlabel=dict(bgcolor="#1e293b", font_size=11, font_color="#f9fafb",
                            bordercolor="rgba(255,255,255,0.1)"))
        with _col:
            st.plotly_chart(_fig_i, width="stretch", config={"displayModeBar": False})
    st.caption("Xanh = đang tăng so với hôm trước · Đỏ = đang giảm")
    st.divider()


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────


@st.cache_data(ttl=600)
def check_data_completeness() -> "pd.DataFrame":
    """Return data quality per ticker: financials, prices, valuations count."""
    with get_session() as s:
        tickers_fin   = {r[0]: r[1] for r in s.execute(
            select(Financial.ticker, sqlfunc.count(Financial.id).label("n"))
            .where(Financial.period_type == "Q")
            .group_by(Financial.ticker)).all()}
        tickers_price = {r[0]: r[1] for r in s.execute(
            select(Price.ticker, sqlfunc.count(Price.id).label("n"))
            .group_by(Price.ticker)).all()}
        tickers_val   = {r[0]: r[1] for r in s.execute(
            select(Valuation.ticker, sqlfunc.count(Valuation.id).label("n"))
            .group_by(Valuation.ticker)).all()}
        comp_rows = s.execute(select(Company.ticker, Company.sector)).all()
    rows = []
    for tkr, sec in comp_rows:
        fin_n  = tickers_fin.get(tkr, 0)
        pri_n  = tickers_price.get(tkr, 0)
        val_n  = tickers_val.get(tkr, 0)
        ok     = fin_n >= 4 and pri_n >= 10 and val_n >= 1
        rows.append({"Ticker": tkr, "Sector": sec or "Unknown",
                     "Quarterly Fins": fin_n, "Price Days": pri_n, "Valuations": val_n,
                     "Status": "OK" if ok else "Needs Update"})
    return pd.DataFrame(rows)


def build_excel_export(df: "pd.DataFrame", screen_name: str = "Valuation Screen") -> bytes:
    """Build multi-sheet Excel file."""
    import io
    buf = io.BytesIO()
    raw_cols = [c for c in df.columns if c.startswith("_")]
    exp = df.drop(columns=raw_cols, errors="ignore")
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            exp.to_excel(writer, sheet_name=screen_name[:31], index=False)
            if "Sector" in exp.columns:
                sec_s = exp.groupby("Sector").size().reset_index(name="Count")
                sec_s.to_excel(writer, sheet_name="By Sector", index=False)
    except Exception:
        exp.to_csv(buf, index=False)
    return buf.getvalue()


def _hv(v, sfx=""):
    return f"{v:,.0f}{sfx}" if v is not None else "—"


def _hf(v, d=1, sfx=""):
    return f"{v:.{d}f}{sfx}" if v is not None else "—"


def _mc(label, value, accent=False):
    vc = "#60a5fa" if accent else "#f9fafb"
    return (
        f'<div style="min-width:0;">'
        f'<div style="font-size:16px;color:#9ca3af;white-space:nowrap;">{label}</div>'
        f'<div style="font-size:20px;font-weight:600;color:{vc};white-space:nowrap;">{value}</div>'
        f'</div>'
    )


def _company_header_html(ticker, prices_df, co_name, co_exch, co_sect, sh, eq, ni, ebit_, dep_, debt_, cash_):
    """Build the styled stock-info header HTML, overlaying a live quote during trading hours."""
    from collectors.live_quote import apply_live_overlay

    current_price = float(prices_df["close"].iloc[-1]) * 1000
    _last   = prices_df.iloc[-1]
    _prev_c = float(prices_df["close"].iloc[-2]) * 1000 if len(prices_df) > 1 else current_price
    _high_d = float(_last["high"]) * 1000
    _low_d  = float(_last["low"])  * 1000

    _overlay = apply_live_overlay(current_price, _prev_c, _high_d, _low_d, load_live_quote(ticker))
    current_price, _prev_c, _high_d, _low_d, _live_as_of = (
        _overlay["current_price"], _overlay["prev_close"],
        _overlay["high"], _overlay["low"], _overlay["as_of"],
    )

    _chg      = current_price - _prev_c
    _chg_pct  = _chg / _prev_c * 100 if _prev_c else 0
    _chg_disp = f"+{_chg:,.0f}" if _chg >= 0 else f"{_chg:,.0f}"

    # Price band limit by exchange (HOSE ±7%, HNX ±10%, UPCOM ±15%)
    _exch_up  = co_exch.upper() if co_exch else "HOSE"
    _band_pct = 0.10 if "HNX" in _exch_up else 0.15 if "UPCOM" in _exch_up or "UPC" in _exch_up else 0.07
    _ceil_p   = _prev_c * (1 + _band_pct)
    _floor_p  = _prev_c * (1 - _band_pct)

    if _prev_c and current_price >= _ceil_p * 0.9995:   # at ceiling (within 0.05% rounding)
        _cc, _cbg, _arrow = "#a855f7", "#4a1d96", "▲"
    elif _chg > 0:
        _cc, _cbg, _arrow = "#22c55e", "#166534", "▲"
    elif _chg == 0:
        _cc, _cbg, _arrow = "#eab308", "#713f12", "—"
    elif _prev_c and current_price <= _floor_p * 1.0005:  # at floor
        _cc, _cbg, _arrow = "#22d3ee", "#164e63", "▼"
    else:
        _cc, _cbg, _arrow = "#ef4444", "#7f1d1d", "▼"

    _mcap     = round(current_price * sh / 1e3) if sh else None
    _bvps     = (eq * 1e9 / (sh * 1e6))        if sh else None
    _eps_h    = (ni * 1e9 / (sh * 1e6))        if sh else None
    _pe_h     = round(current_price / _eps_h, 1) if _eps_h and _eps_h > 0 else None
    _pb_h     = round(current_price / _bvps,   1) if _bvps  and _bvps  > 0 else None
    _ebitda_h = ebit_ + dep_
    _ev_h     = (_mcap or 0) + debt_ - cash_
    _evebitda = round(_ev_h / _ebitda_h, 1)      if _ebitda_h > 0 else None
    _avgvol   = prices_df["volume"].tail(15).mean() / 1000 if len(prices_df) >= 15 else None
    _rng_pct  = round((current_price - _low_d) / (_high_d - _low_d) * 100) if _high_d > _low_d else 50
    _rng_pct  = max(2, min(98, _rng_pct))  # keep dot inside bar

    _live_badge = (
        f'<span style="font-size:13px;background:#7f1d1d;color:#fca5a5;padding:3px 10px;'
        f'border-radius:8px;font-weight:600;">🔴 LIVE · {_live_as_of}</span>'
        if _live_as_of else
        f'<span style="font-size:13px;background:#374151;color:#9ca3af;padding:3px 10px;'
        f'border-radius:8px;font-weight:600;">EOD</span>'
    )

    return (
        f'<div style="padding:22px 0px;margin-bottom:18px;display:flex;gap:28px;align-items:center;">'

        # LEFT — ticker + name
        f'<div style="min-width:230px;padding-right:28px;">'
        f'<div style="font-size:33px;font-weight:800;color:#f9fafb;line-height:1.2;">'
        f'{ticker}'
        f'<span style="font-size:16px;background:#1e3a5f;color:#60a5fa;padding:3px 10px;'
        f'border-radius:6px;margin-left:9px;vertical-align:middle;">{co_exch}</span>'
        f'</div>'
        f'<div style="font-size:18px;color:#9ca3af;margin-top:8px;">{co_name}</div>'
        f'</div>'

        # CENTER — price + change + day range
        f'<div style="min-width:300px;padding-right:28px;">'
        f'<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-size:45px;font-weight:800;color:#f9fafb;">{current_price:,.0f}</span>'
        f'<span style="font-size:22px;color:{_cc};font-weight:600;">{_chg_disp}</span>'
        f'<span style="font-size:18px;background:{_cbg};color:{_cc};padding:3px 12px;'
        f'border-radius:8px;font-weight:600;">{_arrow}{abs(_chg_pct):.2f}%</span>'
        f'{_live_badge}'
        f'</div>'
        f'<div style="margin-top:10px;width:100%;">'
        f'<div style="display:flex;justify-content:space-between;font-size:16px;color:#9ca3af;margin-bottom:5px;">'
        f'<span>Low &nbsp;<b style="color:#f9fafb;">{_low_d:,.0f}</b></span>'
        f'<span>High <b style="color:#f9fafb;">{_high_d:,.0f}</b></span>'
        f'</div>'
        f'<div style="position:relative;height:4px;background:#374151;border-radius:2px;margin-bottom:10px;">'
        f'<div style="position:absolute;left:0;top:0;height:100%;width:{_rng_pct}%;'
        f'background:{_cc};border-radius:2px;opacity:0.7;"></div>'
        f'<div style="position:absolute;left:{_rng_pct}%;top:50%;transform:translate(-50%,-50%);'
        f'width:11px;height:11px;background:{_cc};border-radius:50%;box-shadow:0 0 0 2px #1f2937;"></div>'
        f'<div style="position:absolute;left:{_rng_pct}%;top:10px;transform:translateX(-50%);'
        f'width:0;height:0;'
        f'border-left:6px solid transparent;'
        f'border-right:6px solid transparent;'
        f'border-bottom:8px solid {_cc};"></div>'
        f'</div>'
        f'</div>'
        f'</div>'

        # RIGHT — 3×3 metrics grid
        f'<div style="flex:1;display:grid;grid-template-columns:repeat(3,1fr);gap:16px 18px;">'
        + _mc("Market Cap (bn)", _hv(_mcap))
        + _mc("Book Value (bn)", _hv(eq))
        + _mc("P/E", _hf(_pe_h, 1, "x"), accent=True)
        + _mc("Avg Vol 15D (K)", _hv(_avgvol))
        + _mc("EPS (VND)", _hv(_eps_h))
        + _mc("P/B", _hf(_pb_h, 1, "x"), accent=True)
        + _mc("Shares (M)", _hv(sh))
        + _mc("EV/EBITDA", _hf(_evebitda, 1, "x"))
        + _mc("Sector", co_sect)
        + f'</div>'

        f'</div>'
    )


@st.fragment(run_every=30)
def _render_company_header(ticker, prices_df, co_name, co_exch, co_sect, sh, eq, ni, ebit_, dep_, debt_, cash_):
    """Live-updating stock header — re-fetches the live quote every 30s during trading hours."""
    st.markdown(
        _company_header_html(ticker, prices_df, co_name, co_exch, co_sect, sh, eq, ni, ebit_, dep_, debt_, cash_),
        unsafe_allow_html=True,
    )


VIEWS = [
    "Company Analysis",
    "Stock Screener",
    "Sector Analysis",
    "Portfolio Tracker",
    "Market Overview",
]
# Handle navigation from popup buttons (must be before radio renders)
if st.session_state.get("hm_navigate_to"):
    _pre_nav = st.session_state.pop("hm_navigate_to")
    st.session_state["view_selector"] = "Company Analysis"
    st.session_state["ticker_input"] = _pre_nav

def _on_view_change():
    # Clear any open popup the instant the tab changes (before heavy view renders)
    st.session_state["hm_popup_ticker"] = None
    st.session_state["hm_popup_sector"] = None

view = st.sidebar.radio("View", VIEWS, key="view_selector", on_change=_on_view_change)

# ── Global ticker quick-view popup (works from any tab) ──────────────────────
if st.session_state.get("hm_popup_ticker"):
    _gpt = st.session_state["hm_popup_ticker"]

    @st.dialog(f"{_gpt}", width="large")
    def _global_ticker_popup():
        from sqlalchemy import select as _sel2
        _gprices = load_prices(_gpt)
        _gtd_rows = load_sector_ticker_data()
        _gtd = _gtd_rows[_gtd_rows["ticker"] == _gpt].iloc[0] if not _gtd_rows[_gtd_rows["ticker"] == _gpt].empty else None

        with get_session() as _gds:
            _gco = _gds.execute(_sel2(Company.name, Company.sector).where(Company.ticker == _gpt)).first()
        _gco_name = _gco[0] if _gco else _gpt
        _gco_sect = _gco[1] if _gco else ""
        st.markdown(f"**{_gco_name}** · *{_gco_sect}*")

        if _gprices.empty:
            st.warning("No price data.")
        else:
            _gl = _gprices.iloc[-1]; _gpv = _gprices.iloc[-2] if len(_gprices) > 1 else _gl
            _gcur = float(_gl["close"])*1000; _gref = float(_gpv["close"])*1000
            _gopen = float(_gl["open"])*1000; _ghigh = float(_gl["high"])*1000
            _glow = float(_gl["low"])*1000; _gvol = float(_gl["volume"])
            _gchg = _gcur - _gref; _gchgp = _gchg/_gref*100 if _gref else 0
            _gcc = "#22c55e" if _gchg >= 0 else "#ef4444"
            _gmini = _gprices.tail(113).copy()
            _gmini["ma10"] = _gmini["close"].rolling(10).mean()*1000
            _gmini["ma50"] = _gmini["close"].rolling(50).mean()*1000
            _gmini = _gmini.tail(63)
            _gmini["dlabel"] = pd.to_datetime(_gmini["date"]).dt.strftime("%Y-%m-%d")
            _gmini["vc"] = _gmini.apply(lambda r: "#22c55e" if r["close"]>=r["open"] else "#ef4444", axis=1)
            _gavol10 = int(_gprices.tail(10)["volume"].mean()) if len(_gprices)>=10 else 0
            _gsh = _gtd["shares_outstanding"] if _gtd is not None and pd.notna(_gtd.get("shares_outstanding")) else None
            _left, _right = st.columns([4, 1.5])
            with _left:
                st.markdown(
                    f"<div style='font-size:11px;color:#9ca3af;margin-bottom:2px;'>"
                    f"O&nbsp;<b style='color:#f9fafb'>{_gopen:,.0f}</b>&nbsp;"
                    f"H&nbsp;<b style='color:#22c55e'>{_ghigh:,.0f}</b>&nbsp;"
                    f"L&nbsp;<b style='color:#ef4444'>{_glow:,.0f}</b>&nbsp;"
                    f"C&nbsp;<b style='color:{_gcc}'>{_gcur:,.0f}</b>&nbsp;"
                    f"Vol&nbsp;<b style='color:#f9fafb'>{_gvol/1e6:.2f}M</b><br>"
                    f"MA10&nbsp;<b style='color:#60a5fa'>{_gmini['ma10'].dropna().iloc[-1]:,.0f}</b>&nbsp;"
                    f"MA50&nbsp;<b style='color:#fb923c'>{_gmini['ma50'].dropna().iloc[-1]:,.0f}</b></div>",
                    unsafe_allow_html=True)
                _gfig = go.Figure()
                _gfig.add_trace(go.Candlestick(x=_gmini["dlabel"],open=_gmini["open"]*1000,high=_gmini["high"]*1000,low=_gmini["low"]*1000,close=_gmini["close"]*1000,increasing_line_color="#22c55e",decreasing_line_color="#ef4444",showlegend=False,hoverinfo="skip",yaxis="y"))
                _gfig.add_trace(go.Scatter(x=_gmini["dlabel"],y=_gmini["ma10"],yaxis="y",mode="lines",line=dict(color="#60a5fa",width=1.2),showlegend=False,name="MA10",hovertemplate="MA10 %{y:,.0f}<extra></extra>"))
                _gfig.add_trace(go.Scatter(x=_gmini["dlabel"],y=_gmini["ma50"],yaxis="y",mode="lines",line=dict(color="#fb923c",width=1.2),showlegend=False,name="MA50",hovertemplate="MA50 %{y:,.0f}<extra></extra>"))
                _gmini["vol_m"] = _gmini["volume"]/1e6
                _gfig.add_trace(go.Bar(x=_gmini["dlabel"],y=_gmini["volume"],marker_color=_gmini["vc"].tolist(),yaxis="y2",showlegend=False,hoverinfo="skip"))
                _gfig.add_trace(go.Scatter(x=_gmini["dlabel"],y=_gmini["close"]*1000,yaxis="y",mode="markers",marker=dict(color="rgba(0,0,0,0)",size=1),showlegend=False,name="Vol",customdata=_gmini["vol_m"],hovertemplate="Vol %{customdata:.2f}M<extra></extra>"))
                _gfig.update_layout(height=360,margin=dict(l=0,r=0,t=0,b=0),dragmode=False,hovermode="x unified",xaxis=dict(type="category",rangeslider=dict(visible=False),nticks=6,showgrid=False),yaxis=dict(domain=[0.25,1.0],showgrid=True,gridcolor="rgba(255,255,255,0.06)"),yaxis2=dict(domain=[0.0,0.22],showgrid=False),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(_gfig, width="stretch")
            with _right:
                def _gs(label,value,color="#f9fafb"):
                    return (f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1f2937;font-size:13px;'><span style='color:#9ca3af'>{label}</span><span style='color:{color};font-weight:600'>{value}</span></div>")
                _gmcap = f"{_gcur*_gsh/1e12:,.1f} tn" if _gsh else "—"
                _gpe = f"{_gtd['pe']:.1f}×" if _gtd is not None and pd.notna(_gtd.get("pe")) else "—"
                _gpb = f"{_gtd['pb']:.2f}×" if _gtd is not None and pd.notna(_gtd.get("pb")) else "—"
                _groe = f"{_gtd['roe']:.1f}%" if _gtd is not None and pd.notna(_gtd.get("roe")) else "—"
                _gup = _gtd.get("avg_upside") if _gtd is not None else None
                _gup_str = f"{_gup:+.1f}%" if _gup is not None and pd.notna(_gup) else "—"
                _gup_col = "#22c55e" if (_gup or 0) > 0 else "#ef4444"
                st.markdown(f"<div style='font-size:28px;font-weight:800;color:{_gcc}'>{_gcur:,.0f}</div><div style='font-size:13px;color:{_gcc};margin-bottom:8px;'>{_gchg:+,.0f} / {_gchgp:+.2f}%</div>"+_gs("Tham chiếu",f"{_gref:,.0f}","#eab308")+_gs("Mở cửa",f"{_gopen:,.0f}")+_gs("Thấp – Cao",f"{_glow:,.0f} – {_ghigh:,.0f}")+_gs("Khối lượng",f"{_gvol:,.0f}")+_gs("KLTB 10D",f"{_gavol10:,.0f}")+_gs("Thị giá vốn",_gmcap)+_gs("P/E",_gpe)+_gs("P/B",_gpb)+_gs("ROE",_groe)+_gs("Avg Upside",_gup_str,_gup_col),unsafe_allow_html=True)

        st.divider()
        if st.button("Open Full Analysis →", type="primary", use_container_width=True):
            st.session_state["hm_popup_ticker"] = None
            st.session_state["view_selector"] = "Company Analysis"
            st.session_state["ticker_input"] = _gpt
            st.rerun()

    _global_ticker_popup()
    st.session_state["hm_popup_ticker"] = None


# ═══════════════════════════════════════════════════════════════
# VIEW 1 — COMPANY ANALYSIS
# ═══════════════════════════════════════════════════════════════
if view == "Company Analysis":

    _available = load_available_tickers()
    # Allow navigation from heatmap click
    _nav_ticker = st.session_state.pop("ticker_input", None)
    if _nav_ticker and _nav_ticker in _available:
        _default_idx = _available.index(_nav_ticker)
    elif "VNM" in _available:
        _default_idx = _available.index("VNM")
    else:
        _default_idx = 0
    ticker = st.sidebar.selectbox("Ticker", _available, index=_default_idx, key="ticker_selector")

    prices_df   = load_prices(ticker)
    fin_q       = load_financials_q(ticker)
    ttm         = compute_ttm(ticker)
    valuations  = get_all_valuations(ticker)

    if prices_df.empty:
        st.warning("No price data in DB. Run: python collectors/prices.py")
        st.stop()

    current_price = float(prices_df["close"].iloc[-1]) * 1000

    # ── Styled stock header ────────────────────────────────────
    with get_session() as _hs:
        _co_row = _hs.execute(select(Company).where(Company.ticker == ticker)).scalar_one_or_none()
        _co_name = (_co_row.name     or ticker) if _co_row else ticker
        _co_exch = (_co_row.exchange or "HOSE") if _co_row else "HOSE"
        _co_sect = (_co_row.sector   or "—")    if _co_row else "—"

    _sh    = (ttm.get("shares_outstanding") or 0) if ttm else 0
    _eq    = (ttm.get("equity")             or 0) if ttm else 0
    _ni    = (ttm.get("net_income")         or 0) if ttm else 0
    _ebit_ = (ttm.get("ebit")               or 0) if ttm else 0
    _dep_  = (ttm.get("depreciation")       or 0) if ttm else 0
    _debt_ = (ttm.get("debt")               or 0) if ttm else 0
    _cash_ = (ttm.get("cash")               or 0) if ttm else 0

    # Live-updating header (re-fetches the live quote every 30s during trading hours)
    _render_company_header(ticker, prices_df, _co_name, _co_exch, _co_sect, _sh, _eq, _ni, _ebit_, _dep_, _debt_, _cash_)
    st.markdown('<hr style="border:none;border-top:1px solid #2d3748;margin:0 0 8px 0;">', unsafe_allow_html=True)

    # During trading hours, overlay a live quote on top of the last stored
    # (previous-day) EOD bar — DB is only refreshed after market close.
    # Used by the valuation comparisons below.
    from collectors.live_quote import apply_live_overlay
    _last   = prices_df.iloc[-1]
    _prev_c = float(prices_df["close"].iloc[-2]) * 1000 if len(prices_df) > 1 else current_price
    _high_d = float(_last["high"]) * 1000
    _low_d  = float(_last["low"])  * 1000
    _overlay = apply_live_overlay(current_price, _prev_c, _high_d, _low_d, load_live_quote(ticker))
    current_price = _overlay["current_price"]

    # ── Price chart ────────────────────────────────────────────
    # Marker + CSS so the last bordered card in each of the two columns
    # below (Foreign & Proprietary Trading / Phân tích kỹ thuật) stretches
    # to fill the column, keeping both columns' bottom edges aligned.
    st.markdown('''
        <div id="ta-align-row"></div>
        <style>
        div[data-testid="element-container"]:has(#ta-align-row)
            + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"]
            > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        div[data-testid="element-container"]:has(#ta-align-row)
            + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"]
            > div[data-testid="column"] > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"]:last-child {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        div[data-testid="element-container"]:has(#ta-align-row)
            + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"]
            > div[data-testid="column"] > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"]:last-child
            > div[data-testid="stVerticalBlockBorderWrapper"] {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        div[data-testid="element-container"]:has(#ta-align-row)
            + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"]
            > div[data-testid="column"] > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"]:last-child
            > div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
            flex: 1;
        }
        </style>
    ''', unsafe_allow_html=True)
    col_chart, col_dcf = st.columns([5, 2], gap="large")

    with col_chart:
        # Time range selector
        _range_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756}
        _sel = st.radio("", list(_range_map.keys()), index=3, horizontal=True,
                        key=f"price_range_{ticker}", label_visibility="collapsed")
        _n_days = _range_map[_sel]

        # Compute MAs on extra lookback so they start from day 1 of the display range
        _lookback = 60  # enough for MA50
        df_ext = prices_df.tail(_n_days + _lookback).copy()
        df_ext["close_vnd"] = df_ext["close"] * 1000
        df_ext["ma10"] = df_ext["close_vnd"].rolling(10).mean()
        df_ext["ma50"] = df_ext["close_vnd"].rolling(50).mean()
        # Now trim to the display range
        df1y = df_ext.tail(_n_days).copy()
        df1y["open_vnd"]  = df1y["open"]  * 1000
        df1y["high_vnd"]  = df1y["high"]  * 1000
        df1y["low_vnd"]   = df1y["low"]   * 1000
        df1y["vol_color"] = df1y.apply(
            lambda r: "#22c55e" if r["close"] >= r["open"] else "#ef4444", axis=1
        )
        # String labels for categorical x-axis → no gaps for weekends/holidays
        df1y["dlabel"] = pd.to_datetime(df1y["date"]).dt.strftime("%Y-%m-%d")

        # Single figure, two y-axes with domains — one shared x-axis so spike spans full height
        fig_price = go.Figure()
        df1y["vol_m"] = df1y["volume"] / 1e6

        # Candlestick (yaxis=y, top 75%)
        fig_price.add_trace(go.Candlestick(
            x=df1y["dlabel"],
            open=df1y["open_vnd"], high=df1y["high_vnd"],
            low=df1y["low_vnd"],  close=df1y["close_vnd"],
            name="Price", yaxis="y",
            increasing_line_color="#22c55e", increasing_fillcolor="#22c55e",
            decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444",
            showlegend=False,
            hoverinfo="none",
        ))
        # MA10
        fig_price.add_trace(go.Scatter(
            x=df1y["dlabel"], y=df1y["ma10"], yaxis="y",
            mode="lines", name="MA10",
            line=dict(color="#60a5fa", width=1.5),
            hoverinfo="none",
        ))
        # MA50
        fig_price.add_trace(go.Scatter(
            x=df1y["dlabel"], y=df1y["ma50"], yaxis="y",
            mode="lines", name="MA50",
            line=dict(color="#fb923c", width=1.5),
            hoverinfo="none",
        ))
        # Valuation lines (hidden by default) — computed first so proxy can include values
        vh = valuation_history(ticker)
        _val_series = [
            ("dcf",    "#4ade80", "DCF Intrinsic Value",         "dash",  1),
            ("fcfe",   "#00bcd4", "FCFE / Cash Flow to Equity",  "dash",  1),
            ("graham", "#fbbf24", "Graham Number",               "dash",  1),
            ("pe",     "#c084fc", f"P/E Implied (×{MARKET_PE})", "dash",  1),
            ("avg",    "#f87171", "Avg of Estimates",            "solid", 2),
        ]
        vh_daily = pd.DataFrame()
        if not vh.empty:
            daily_idx = pd.to_datetime(df1y["date"])
            vh_daily = (
                vh.set_index(pd.to_datetime(vh["date"]))
                  .drop(columns=["date"])
                  .reindex(daily_idx)
                  .interpolate(method="time")  # smooth daily transition between quarters
                  .ffill().bfill()             # fill edges outside data range
            )
            vh_daily.index = df1y["dlabel"].values
            for _vc, color, label, dash, width in _val_series:
                series = vh_daily[_vc].dropna() if _vc in vh_daily.columns else pd.Series(dtype=float)
                if series.empty:
                    continue
                fig_price.add_trace(go.Scatter(
                    x=series.index, y=series.values, yaxis="y",
                    mode="lines", name=label,
                    line=dict(color=color, dash=dash, width=width),
                    visible="legendonly",
                    hovertemplate=f"{label} %{{y:,.0f}}<extra></extra>",
                ))

        # Data-window proxy: OHLCV + MA only (valuation lines add themselves when enabled)
        _top_y = float(df1y["high_vnd"].max())
        _ma10_f = df1y["ma10"].fillna(0)
        _ma50_f = df1y["ma50"].fillna(0)
        _proxy_cd = list(zip(
            df1y["open_vnd"], df1y["high_vnd"], df1y["low_vnd"], df1y["close_vnd"],
            df1y["vol_m"], _ma10_f, _ma50_f,
        ))
        fig_price.add_trace(go.Scatter(
            x=df1y["dlabel"],
            y=[_top_y] * len(df1y),
            yaxis="y",
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=1, opacity=0),
            showlegend=False,
            customdata=_proxy_cd,
            hovertemplate=(
                "O %{customdata[0]:,.0f}  "
                "H %{customdata[1]:,.0f}  "
                "L %{customdata[2]:,.0f}  "
                "C %{customdata[3]:,.0f}  "
                "Vol %{customdata[4]:.2f}M<br>"
                "MA10 %{customdata[5]:,.0f}  "
                "MA50 %{customdata[6]:,.0f}"
                "<extra></extra>"
            ),
            hoverlabel=dict(
                bgcolor="#1e2533", bordercolor="#374151",
                font=dict(color="#f9fafb", size=12, family="monospace"),
                align="left",
            ),
        ))

        # Volume bars (yaxis=y2, bottom 22%)
        fig_price.add_trace(go.Bar(
            x=df1y["dlabel"], y=df1y["volume"], yaxis="y2",
            name="Volume",
            marker_color=df1y["vol_color"].tolist(),
            showlegend=False,
            hoverinfo="none",
        ))

        # Reference line (add as shape so it doesn't affect autorange)
        fig_price.add_shape(
            type="line", xref="paper", x0=0, x1=1,
            yref="y", y0=_prev_c, y1=_prev_c,
            line=dict(color="#ef4444", width=1, dash="dash"),
            opacity=0.6,
        )

        fig_price.update_layout(
            height=480, margin=dict(l=0, r=10, t=4, b=0),
            hovermode="x unified", dragmode=False,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="top", y=-0.06,
                xanchor="left", x=0, font=dict(size=11),
                itemclick="toggle", itemdoubleclick="toggleothers",
            ),
            # Single x-axis — spike spans full figure height automatically
            xaxis=dict(
                type="category", showgrid=False,
                rangeslider=dict(visible=False), nticks=8,
                # remove blank padding on left/right edges
                range=[-0.5, len(df1y) - 0.5],
                showspikes=True, spikemode="across",
                spikesnap="cursor", spikecolor="rgba(255,255,255,0.3)",
                spikethickness=1, spikedash="dot",
            ),
            # Price y-axis: top 75%
            yaxis=dict(
                domain=[0.25, 1.0], title="VND",
                showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                showspikes=True, spikemode="across+toaxis",
                spikesnap="cursor", spikecolor="rgba(255,255,255,0.25)",
                spikethickness=1, spikedash="dot",
            ),
            # Volume y-axis: bottom 22%
            yaxis2=dict(
                domain=[0.0, 0.22], title="Vol",
                showgrid=False, anchor="x",
            ),
        )
        st.plotly_chart(fig_price, width="stretch")

        # ── Peer Comparison vs Sector (fills space below chart) ──
        if ttm and _co_sect and _co_sect != "—":
            _peers = load_sector_ticker_data()
            _peers = _peers[_peers["sector"] == _co_sect]
            if len(_peers) >= 3:
              st.write("")
              with st.container(border=True):
                st.markdown(
                    f'<div style="font-size:17px;font-weight:700;color:#f9fafb;margin-bottom:8px;">'
                    f'Peer Comparison <span style="color:#9ca3af;font-size:13px;font-weight:400;">'
                    f'· {_co_sect}</span></div>', unsafe_allow_html=True)

                def _pmed(col):
                    s = _peers[col].dropna()
                    return s.median() if not s.empty else None

                def _pself(col):
                    r = _peers[_peers["ticker"] == ticker]
                    return r.iloc[0][col] if not r.empty and pd.notna(r.iloc[0][col]) else None

                _pmetrics = [("pe", "P/E", False), ("pb", "P/B", False),
                             ("roe", "ROE %", True), ("avg_upside", "Avg Upside %", True)]
                _pcc = st.columns(4)
                for _i, (_col, _lbl, _hb) in enumerate(_pmetrics):
                    _sv = _pself(_col); _mv = _pmed(_col)
                    if _sv is None or _mv is None:
                        _pcc[_i].metric(_lbl, "—")
                        continue
                    _sfx = "%" if "%" in _lbl else "×"
                    _pcc[_i].metric(_lbl, f"{_sv:.1f}{_sfx}",
                        delta=f"{_sv-_mv:+.1f} vs median",
                        delta_color="normal" if _hb else "inverse")

                _pp = _peers.dropna(subset=["pe", "roe"]).copy()
                if len(_pp) >= 3:
                    _pp["is_self"] = _pp["ticker"] == ticker
                    fig_pc = go.Figure()
                    _oth = _pp[~_pp["is_self"]]; _slf = _pp[_pp["is_self"]]
                    fig_pc.add_trace(go.Scatter(
                        x=_oth["pe"], y=_oth["roe"], mode="markers+text",
                        marker=dict(size=15, color="#5b9bd5", opacity=0.7,
                                    line=dict(width=0.5, color="#1f2937")),
                        text=_oth["ticker"], textposition="top center",
                        textfont=dict(size=10, color="#9ca3af"),
                        hovertemplate="%{text}<br>P/E %{x:.1f}× · ROE %{y:.1f}%<extra></extra>",
                        name="Peers"))
                    if not _slf.empty:
                        fig_pc.add_trace(go.Scatter(
                            x=_slf["pe"], y=_slf["roe"], mode="markers+text",
                            marker=dict(size=28, color="#f59e0b", symbol="star",
                                        line=dict(width=1.5, color="#fff")),
                            text=_slf["ticker"], textposition="top center",
                            textfont=dict(size=14, color="#f59e0b"),
                            hovertemplate="<b>%{text}</b><br>P/E %{x:.1f}× · ROE %{y:.1f}%<extra></extra>",
                            name=ticker))
                    fig_pc.update_layout(
                        height=340, margin=dict(l=0, r=0, t=6, b=0), dragmode=False,
                        xaxis_title="P/E (×)", yaxis_title="ROE (%)",
                        showlegend=False, hovermode="closest")
                    st.plotly_chart(fig_pc, width="stretch")
                    st.caption(f"★ {ticker} · Lower-right = cheap & profitable (low P/E, high ROE)")

        # ── Foreign & Proprietary Trading — last 20 sessions (this ticker) ──
        st.write("")
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:17px;font-weight:700;color:#f9fafb;margin-bottom:8px;">'
                'Foreign & Proprietary Trading <span style="color:#9ca3af;font-size:13px;'
                'font-weight:400;">· last 20 sessions</span></div>', unsafe_allow_html=True)

            tab_nn, tab_td = st.tabs(["Nước ngoài", "Tự doanh"])

            with tab_nn:
                _ff20 = load_foreign_flow(ticker, sessions=20)
                if not _ff20.empty:
                    _last = _ff20.iloc[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("KL Mua", f"{_last['buy_vol']:,.0f}")
                    c2.metric("KL Bán", f"{_last['sell_vol']:,.0f}")
                    c3.metric("KL Mua-Bán", f"{_last['net_vol']:+,.0f}")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("GT Mua (tỷ)", f"{_last['buy_val']/1e9:,.2f}")
                    c5.metric("GT Bán (tỷ)", f"{_last['sell_val']/1e9:,.2f}")
                    c6.metric("GT Mua-Bán (tỷ)", f"{_last['net_val']/1e9:+,.2f}")

                    _ff20 = _ff20.copy()
                    _ff20["net_bn"] = _ff20["net_val"] / 1e9
                    _ff20["dlabel"] = _ff20["date"].dt.strftime("%d/%m")
                    _ff_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in _ff20["net_bn"]]

                    # Align price line to the same x categories as the foreign-flow
                    # bars — mismatched date sets on a categorical axis break the chart.
                    _px = load_prices(ticker)
                    if not _px.empty:
                        _px2 = _px[["date", "close"]].copy()
                        _px2["date"] = pd.to_datetime(_px2["date"])
                        _ff20 = _ff20.merge(_px2, on="date", how="left")

                    fig_nn = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_nn.add_trace(go.Bar(
                        x=_ff20["dlabel"], y=_ff20["net_bn"], marker_color=_ff_colors,
                        name="GTNN mua ròng (tỷ)",
                        customdata=list(zip(_ff20["buy_val"] / 1e9, _ff20["sell_val"] / 1e9)),
                        hovertemplate=("Net: %{y:,.2f} tỷ · Buy: %{customdata[0]:,.2f} tỷ · "
                                       "Sell: %{customdata[1]:,.2f} tỷ<extra>GTNN ròng</extra>")),
                        secondary_y=False)
                    if "close" in _ff20.columns and _ff20["close"].notna().any():
                        fig_nn.add_trace(go.Scatter(
                            x=_ff20["dlabel"], y=_ff20["close"], mode="lines",
                            line=dict(color="#60a5fa", width=2), name="Giá đóng cửa",
                            connectgaps=True,
                            hovertemplate="%{y:,.1f}<extra>Giá đóng cửa</extra>"),
                            secondary_y=True)
                    fig_nn.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
                    fig_nn.update_layout(
                        height=280, margin=dict(l=0, r=0, t=10, b=0), dragmode=False,
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#f9fafb"),
                        xaxis=dict(type="category", showgrid=False, tickfont=dict(size=10)),
                        legend=dict(orientation="h", y=1.1, x=0))
                    fig_nn.update_yaxes(title_text="GTNN ròng (tỷ)", showgrid=True,
                                         gridcolor="rgba(255,255,255,0.06)", zeroline=False, secondary_y=False)
                    fig_nn.update_yaxes(title_text="Giá (nghìn VND)", showgrid=False, secondary_y=True)
                    st.plotly_chart(fig_nn, width="stretch")
                    st.caption("GTNN = giá trị giao dịch ròng của nhà đầu tư nước ngoài.")
                else:
                    st.info("Foreign trading data unavailable right now.")

            with tab_td:
                st.info("Dữ liệu giao dịch tự doanh chưa có sẵn từ nguồn dữ liệu hiện tại.")

    # ── Valuation panel ────────────────────────────────────────
    with col_dcf:
        st.subheader("Valuation Estimates")

        v = dict(valuations)  # copy so sector overrides don't mutate cache

        # ── Sector-specific valuation overrides ────────────────────
        if ttm:
            _sh_s = ttm.get("shares_outstanding") or 0
            def _ps_val(val_bn, mult):
                return round(val_bn * 1e9 / (_sh_s * 1e6) * mult) if (val_bn and _sh_s > 0) else None

            if _co_sect == "Ngân hàng":
                v["ev_ebitda"] = _ps_val(ttm.get("gross_profit"), 8)   # P/NII ×8
                v["epv"]       = _ps_val(ttm.get("ebit"),          6)   # P/PPOP ×6
                v["ps"]        = _ps_val(ttm.get("revenue"),        5)   # P/TOI ×5

            elif _co_sect == "Bất động sản":
                # RE: EV/EBITDA ×15, NAV proxy (Book × 1.5 premium), P/Revenue ×3.5
                _ebitda_re = (ttm.get("ebit") or 0) + (ttm.get("depreciation") or 0)
                _nd_re     = (ttm.get("debt") or 0) - (ttm.get("cash") or 0)
                if _ebitda_re > 0 and _sh_s > 0:
                    _ev_re = _ebitda_re * 15 - _nd_re
                    v["ev_ebitda"] = round(_ev_re * 1e9 / (_sh_s * 1e6)) if _ev_re > 0 else None
                v["ps"] = _ps_val(ttm.get("revenue"), 3.5)
                # EPV → NAV proxy: Book value × 1.8 (RE trades at premium to book for land bank)
                _eq_re = ttm.get("equity")
                if _eq_re and _sh_s > 0:
                    v["epv"] = round(_eq_re * 1.8 * 1e9 / (_sh_s * 1e6))

            elif _co_sect == "Chứng khoán":
                # Securities: P/Revenue ×3, EPV = BVPS × 1.2 (book value target)
                v["ps"]        = _ps_val(ttm.get("revenue"), 3)
                v["ev_ebitda"] = None
                _eq_sec = ttm.get("equity")
                if _eq_sec and _sh_s > 0:
                    v["epv"] = round(_eq_sec * 1.2 * 1e9 / (_sh_s * 1e6))

            elif _co_sect == "Bảo hiểm":
                # Insurance: P/Revenue ×2, Embedded Value proxy = Book × 2.0
                # (insurers trade at premium to book for VIF embedded value)
                v["ps"]        = _ps_val(ttm.get("revenue"), 2)
                v["ev_ebitda"] = None
                _eq_ins = ttm.get("equity")
                if _eq_ins and _sh_s > 0:
                    v["epv"] = round(_eq_ins * 2.0 * 1e9 / (_sh_s * 1e6))

        # ── Build method label list (sector-aware) ─────────────────
        _IS_BANK   = (_co_sect == "Ngân hàng")
        _IS_RE     = (_co_sect == "Bất động sản")
        _IS_SEC    = (_co_sect == "Chứng khoán")
        _IS_INS    = (_co_sect == "Bảo hiểm")

        _ALL_METHODS = [
            ("dcf",       "DCF / FCFF",              "NOPAT-based DCF at WACC"),
            ("fcfe",      "FCFE / Cash Flow to Equity", "OCF−CapEx discounted at cost of equity"),
            ("graham",    "Graham Number",            "√(22.5 × EPS × BVPS)"),
            ("pe",        f"P/E Implied (×{MARKET_PE})", f"TTM EPS × {MARKET_PE}×"),
            ("pb",        "P/B Implied (×1.5)",       "BVPS × 1.5× VN market"),
            ("ev_ebitda",
                "P/NII (×8)"      if _IS_BANK else
                "EV/EBITDA (×15)" if _IS_RE   else
                "EV/EBITDA (×8)",
                "NII/share × 8×"         if _IS_BANK else
                "EBITDA × 15× − net debt (RE)" if _IS_RE else
                "EBITDA × 8× − net debt"),
            ("epv",
                "P/PPOP (×6)"          if _IS_BANK else
                "NAV Proxy (Book×1.8)" if _IS_RE   else
                "Book Value (×1.2)"    if _IS_SEC  else
                "Embedded Value (×2.0)"if _IS_INS  else
                "Earnings Power Value",
                "PPOP/share × 6×"          if _IS_BANK else
                "Book × 1.8 (land bank premium)" if _IS_RE else
                "Book × 1.2 (securities target)" if _IS_SEC else
                "Book × 2.0 (VIF embedded value)" if _IS_INS else
                "NOPAT ÷ WACC, zero growth"),
            ("ps",
                "P/TOI (×5)"      if _IS_BANK else
                "P/Revenue (×3.5)"if _IS_RE   else
                "P/Revenue (×3)"  if _IS_SEC  else
                "P/Revenue (×2)"  if _IS_INS  else
                "P/Sales (×1.2)",
                "TOI/share × 5×"        if _IS_BANK else
                "Revenue/share × 3.5× (RE)" if _IS_RE else
                "Revenue/share × 1.2×"),
            ("ri",        "Residual Income",          "BVPS + excess ROE perpetuity"),
            ("pocf",      "Price/OCF (×10)",          "Operating CF/share × 10×"),
        ]

        # Only show methods with valid values (skip — entries)
        _valid_methods = [(k, lbl, hint) for k, lbl, hint in _ALL_METHODS if v.get(k) and v[k] > 0]

        if current_price:
            st.markdown(
                f"<div style='font-size:12.5px;color:#9ca3af;margin:-6px 0 10px;'>"
                f"Giá thị trường hiện tại: <b style='color:#f9fafb;'>{current_price:,.0f} ₫</b></div>",
                unsafe_allow_html=True)

        def _val_card(label, price_val, hint):
            u = (price_val - current_price) / current_price * 100 if current_price else 0
            clr = "#4ade80" if u >= 0 else "#f87171"
            arrow = "▲" if u >= 0 else "▼"
            return (
                f'<div title="{hint}" style="background:#1e293b;border:1px solid #334155;'
                f'border-radius:8px;padding:12px 14px;">'
                f'<div style="font-size:12px;color:#9ca3af;margin-bottom:6px;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>'
                f'<div style="font-size:21px;font-weight:800;color:#f9fafb;letter-spacing:.3px;">'
                f'{price_val:,.0f} <span style="font-size:13px;font-weight:600;color:#9ca3af;">₫</span></div>'
                f'<div style="font-size:12px;font-weight:700;color:{clr};margin-top:4px;">'
                f'{arrow} {u:+.1f}% <span style="color:#6b7280;font-weight:400;">vs market</span></div>'
                f'</div>')

        _cards_html = "".join(_val_card(label, v[key], hint) for key, label, hint in _valid_methods)
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">{_cards_html}</div>',
            unsafe_allow_html=True)

        valid_prices = [v[k] for k, *_ in _valid_methods if v.get(k) and v[k] > 0]
        if valid_prices and current_price:
            avg_val = sum(valid_prices) / len(valid_prices)
            u = (avg_val - current_price) / current_price * 100
            clr = "#4ade80" if u >= 0 else "#f87171"
            arrow = "▲" if u >= 0 else "▼"
            st.markdown(
                f'<div style="margin-top:10px;background:linear-gradient(135deg,#1e3a5f,#1e293b);'
                f'border:1px solid #3b82f6;border-radius:8px;padding:14px 16px;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<div>'
                f'<div style="font-size:11.5px;color:#93c5fd;font-weight:700;letter-spacing:.5px;">'
                f'TRUNG BÌNH {len(valid_prices)} PHƯƠNG PHÁP</div>'
                f'<div style="font-size:27px;font-weight:800;color:#f9fafb;margin-top:2px;">'
                f'{avg_val:,.0f} <span style="font-size:15px;font-weight:600;color:#9ca3af;">₫</span></div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:17px;font-weight:800;color:{clr};">{arrow} {u:+.1f}%</div>'
                f'<div style="font-size:11px;color:#9ca3af;">so với giá thị trường</div>'
                f'</div></div>',
                unsafe_allow_html=True)

        # ── Technical analysis — daily ───────────────────────────
        st.write("")
        with st.container(border=True, key="ta_card"):
            st.markdown('''
                <style>
                .st-key-ta_card [data-testid="stDataFrame"] * { font-size: 14px !important; }
                .st-key-ta_card table { font-size: 14px !important; }
                .st-key-ta_card p, .st-key-ta_card td, .st-key-ta_card th { font-size: 14px !important; }
                .st-key-ta_card { border: none !important; padding: 0 !important; }
                </style>
            ''', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:17px;font-weight:700;color:#f9fafb;margin-bottom:8px;">'
                'Phân tích kỹ thuật <span style="color:#9ca3af;font-size:13px;'
                'font-weight:400;">· 1 ngày</span></div>', unsafe_allow_html=True)

            _ta = prices_df.copy().reset_index(drop=True)
            if len(_ta) >= 100:
                close, high, low, open_ = _ta["close"], _ta["high"], _ta["low"], _ta["open"]
                cur_price = close.iloc[-1]

                # RSI(14)
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                rsi = 100 - 100 / (1 + avg_gain / avg_loss)
                rsi_val = rsi.iloc[-1]

                # Stochastic(14,3)
                low14, high14 = low.rolling(14).min(), high.rolling(14).max()
                stoch_k = (close - low14) / (high14 - low14) * 100
                stoch_val = stoch_k.iloc[-1]

                # Stochastic RSI(14)
                rsi_min, rsi_max = rsi.rolling(14).min(), rsi.rolling(14).max()
                stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
                stoch_rsi_val = stoch_rsi.iloc[-1]

                # MACD(12,26)
                macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
                macd_val = macd.iloc[-1]

                # Williams %R(14)
                willr = (high14 - close) / (high14 - low14) * -100
                willr_val = willr.iloc[-1]

                # CCI(14)
                tp = (high + low + close) / 3
                sma_tp = tp.rolling(14).mean()
                mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
                cci = (tp - sma_tp) / (0.015 * mad)
                cci_val = cci.iloc[-1]

                # Awesome Oscillator(5,34)
                median_price = (high + low) / 2
                ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
                ao_val = ao.iloc[-1]

                def _sig_bounded(v, lo=30, hi=70):
                    if v < lo: return "Quá bán"
                    if v > hi: return "Quá mua"
                    return "Mua" if v > 50 else "Bán"

                def _sig_zero(v):
                    return "Mua" if v > 0 else "Bán"

                ind_rows = [
                    ("RSI(14)",                  rsi_val,       _sig_bounded(rsi_val)),
                    ("Stochastic(14,3)",         stoch_val,     _sig_bounded(stoch_val, 20, 80)),
                    ("Stochastic RSI(14)",       stoch_rsi_val, _sig_bounded(stoch_rsi_val, 20, 80)),
                    ("MACD(12,26)",              macd_val,      _sig_zero(macd_val)),
                    ("WilliamR(14)",             willr_val,     _sig_bounded(willr_val, -80, -20)),
                    ("CCI(14)",                  cci_val,       _sig_zero(cci_val)),
                    ("Awesome Oscillator(5,34)", ao_val,        _sig_zero(ao_val)),
                ]

                # Moving averages
                ma_periods = [p for p in [5, 10, 20, 50, 100, 150, 200] if len(_ta) > p]
                ma_rows = []
                for p in ma_periods:
                    sma_v = close.rolling(p).mean().iloc[-1]
                    ema_v = close.ewm(span=p, adjust=False).mean().iloc[-1]
                    ma_rows.append((
                        f"MA{p}",
                        sma_v, "Mua" if cur_price > sma_v else "Bán",
                        ema_v, "Mua" if cur_price > ema_v else "Bán",
                    ))

                # ── Overall summary ──────────────────────────────
                all_sigs = [r[2] for r in ind_rows] + [r[2] for r in ma_rows] + [r[4] for r in ma_rows]
                n_buy  = sum(1 for s in all_sigs if "mua" in s.lower())
                n_sell = sum(1 for s in all_sigs if "bán" in s.lower())
                total = n_buy + n_sell
                score = (n_buy - n_sell) / total if total else 0

                if score <= -0.6: verdict, vcolor = "BÁN MẠNH", "#ef4444"
                elif score <= -0.2: verdict, vcolor = "BÁN", "#f97316"
                elif score < 0.2:   verdict, vcolor = "TRUNG LẬP", "#eab308"
                elif score < 0.6:   verdict, vcolor = "MUA", "#84cc16"
                else:                verdict, vcolor = "MUA MẠNH", "#22c55e"

                def _verdict(buy, sell):
                    t = buy + sell
                    s = (buy - sell) / t if t else 0
                    if s <= -0.6: return "BÁN MẠNH"
                    if s <= -0.2: return "BÁN"
                    if s < 0.2:   return "TRUNG LẬP"
                    if s < 0.6:   return "MUA"
                    return "MUA MẠNH"

                ind_buy  = sum(1 for r in ind_rows if "mua" in r[2].lower())
                ind_sell = sum(1 for r in ind_rows if "bán" in r[2].lower())
                ma_sigs  = [r[2] for r in ma_rows] + [r[4] for r in ma_rows]
                ma_buy   = sum(1 for s in ma_sigs if "mua" in s.lower())
                ma_sell  = sum(1 for s in ma_sigs if "bán" in s.lower())

                gc1, gc2 = st.columns([3, 2])
                with gc1:
                    st.markdown(
                        f'<div style="margin-top:12px;">TỔNG HỢP: '
                        f'<span style="background:{vcolor};color:white;font-weight:700;'
                        f'padding:2px 10px;border-radius:4px;">{verdict}</span></div>',
                        unsafe_allow_html=True)
                    st.markdown(f"""
| | | Mua | Bán |
|---|---|---|---|
| Đường trung bình | **{_verdict(ma_buy, ma_sell)}** | {ma_buy} | {ma_sell} |
| Chỉ số kỹ thuật | **{_verdict(ind_buy, ind_sell)}** | {ind_buy} | {ind_sell} |
""")
                with gc2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number={"valueformat": ".2f", "font": {"size": 20}},
                        gauge={
                            "axis": {"range": [-1, 1], "visible": False},
                            "bar": {"color": "rgba(0,0,0,0)"},
                            "bgcolor": "rgba(0,0,0,0)",
                            "steps": [
                                {"range": [-1, -0.6], "color": "#dc2626"},
                                {"range": [-0.6, -0.2], "color": "#f97316"},
                                {"range": [-0.2, 0.2], "color": "#eab308"},
                                {"range": [0.2, 0.6], "color": "#84cc16"},
                                {"range": [0.6, 1], "color": "#22c55e"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 4},
                                "thickness": 0.85,
                                "value": score,
                            },
                        }))
                    fig_gauge.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=0),
                                             font=dict(color="#f9fafb"))
                    st.plotly_chart(fig_gauge, width="stretch")

                st.markdown(
                    '<div style="background:#1e293b;border-left:4px solid #3b82f6;'
                    'border-radius:6px;padding:10px 14px;margin-top:8px;font-size:12.5px;'
                    'color:#94a3b8;line-height:1.7;">'
                    '💡 <b style="color:#cbd5e1;">TỔNG HỢP</b> = tổng hợp tất cả tín hiệu '
                    'Mua/Bán từ 7 chỉ số kỹ thuật và các đường trung bình '
                    '(Simple &amp; Exponential) bên dưới.<br>'
                    '📊 <b style="color:#cbd5e1;">Điểm gauge</b> chạy từ -1 (Bán mạnh) đến '
                    '+1 (Mua mạnh) = (Số tín hiệu Mua − Số tín hiệu Bán) / Tổng số tín hiệu. '
                    '0 = cân bằng giữa Mua và Bán.'
                    '</div>',
                    unsafe_allow_html=True)
                st.caption("* Dữ liệu được tính toán tự động từ giá đóng cửa lịch sử")

                # ── Pivot points ──────────────────────────────────
                st.markdown("**Pivot Points**")
                last = _ta.iloc[-1]
                H, L, C, O = last["high"], last["low"], last["close"], last["open"]

                P_c = (H + L + C) / 3
                R1c, S1c = 2*P_c - L, 2*P_c - H
                R2c, S2c = P_c + (H-L), P_c - (H-L)
                R3c, S3c = H + 2*(P_c-L), L - 2*(H-P_c)

                P_f = P_c
                R1f, S1f = P_f + 0.382*(H-L), P_f - 0.382*(H-L)
                R2f, S2f = P_f + 0.618*(H-L), P_f - 0.618*(H-L)
                R3f, S3f = P_f + 1.0*(H-L),   P_f - 1.0*(H-L)

                P_ca = (H + L + C) / 3
                R1ca, S1ca = C + (H-L)*1.1/12, C - (H-L)*1.1/12
                R2ca, S2ca = C + (H-L)*1.1/6,  C - (H-L)*1.1/6
                R3ca, S3ca = C + (H-L)*1.1/4,  C - (H-L)*1.1/4

                P_w = (H + L + 2*C) / 4
                R1w, S1w = 2*P_w - L, 2*P_w - H
                R2w, S2w = P_w + (H-L), P_w - (H-L)
                R3w, S3w = H + 2*(P_w-L), L - 2*(H-P_w)

                if C < O:   X = H + 2*L + C
                elif C > O: X = 2*H + L + C
                else:       X = H + L + 2*C
                P_d = X / 4
                R1d, S1d = X/2 - L, X/2 - H
                R2d, S2d = P_d + (R1d-S1d), P_d - (R1d-S1d)
                R3d, S3d = R1d + (H-L), S1d - (H-L)

                pivot_df = pd.DataFrame({
                    "S3":     [S3c, S3f, S3ca, S3w, S3d],
                    "S2":     [S2c, S2f, S2ca, S2w, S2d],
                    "S1":     [S1c, S1f, S1ca, S1w, S1d],
                    "Points": [P_c, P_f, P_ca, P_w, P_d],
                    "R1":     [R1c, R1f, R1ca, R1w, R1d],
                    "R2":     [R2c, R2f, R2ca, R2w, R2d],
                    "R3":     [R3c, R3f, R3ca, R3w, R3d],
                }, index=["Classic", "Fibonacci", "Camarilla", "Woodie", "DeMark"]).round(2)
                st.dataframe(
                    pivot_df.style
                    .format("{:.2f}")
                    .set_properties(subset=["S1", "S2", "S3"], **{"color": "#4ade80"})
                    .set_properties(subset=["R1", "R2", "R3"], **{"color": "#f87171"})
                    .set_properties(subset=["Points"], **{"color": "#facc15", "font-weight": "700"}),
                    width="stretch")
                st.caption("S = Hỗ trợ (xanh) · R = Kháng cự (đỏ) · Points = Điểm xoay (vàng)")

                # ── Indicators & Moving averages ──────────────────
                ic1, ic2 = st.columns(2)

                def _color_action(val):
                    if "mua" in str(val).lower():
                        return "color: #22c55e; font-weight: 600"
                    if "bán" in str(val).lower():
                        return "color: #ef4444; font-weight: 600"
                    return ""

                with ic1:
                    st.markdown("**Chỉ số kỹ thuật**")
                    df_ind = pd.DataFrame(ind_rows, columns=["Tên", "Giá trị", "Tín hiệu"])
                    st.table(
                        df_ind.style
                        .format({"Giá trị": "{:.2f}"})
                        .map(_color_action, subset=["Tín hiệu"])
                        .set_properties(subset=["Tín hiệu"], **{"white-space": "nowrap"})
                        .hide(axis="index"))

                with ic2:
                    st.markdown("**Đường trung bình**")
                    df_ma = pd.DataFrame(
                        [(r[0], f"{r[1]:.2f} ({r[2]})", f"{r[3]:.2f} ({r[4]})") for r in ma_rows],
                        columns=["Tên", "Simple", "Exponential"])
                    st.table(
                        df_ma.style
                        .map(_color_action, subset=["Simple", "Exponential"])
                        .hide(axis="index"))
            else:
                st.info("Not enough price history to compute technical indicators.")


    st.divider()

    # ── Quarterly financial charts ─────────────────────────────
    if not fin_q.empty and len(fin_q) >= 4:
        df_q = fin_q.sort_values("period").copy().reset_index(drop=True)

        # Format period labels: "2024-Q3" → "Q3/24"
        def _fmt_period(p: str) -> str:
            try:
                yr, q = p.split("-Q")
                return f"Q{q}/{yr[2:]}"
            except Exception:
                return p

        df_q["label"] = df_q["period"].apply(_fmt_period)

        # YoY % change (vs same quarter 1 year ago)
        def _yoy(series, lag=4):
            out = [None] * len(series)
            for i in range(lag, len(series)):
                curr, prev = series.iloc[i], series.iloc[i - lag]
                if prev and prev != 0 and curr is not None:
                    out[i] = (curr - prev) / abs(prev) * 100
            return out

        rev_yoy  = _yoy(df_q["revenue"])
        ni_yoy   = _yoy(df_q["net_income"])
        ebit_yoy = _yoy(df_q["ebit"])

        # Trim everything to start where YoY data is first available
        # (first 4 quarters have no prior-year comparison)
        first_yoy = next((i for i, v in enumerate(rev_yoy) if v is not None), 0)
        df_q_full = df_q.copy()   # keep pre-trim version for YoY lookback
        df_q     = df_q.iloc[first_yoy:].reset_index(drop=True)
        rev_yoy  = rev_yoy[first_yoy:]
        ni_yoy   = ni_yoy[first_yoy:]
        ebit_yoy = ebit_yoy[first_yoy:]

        def _yoy_full(col):
            """YoY computed on pre-trim data so all displayed quarters have values."""
            vals = _yoy(df_q_full[col])
            return vals[first_yoy:]

        labels  = df_q["label"].tolist()

        # Margins (%)
        def _pct(a, b):
            try:
                return a / b * 100 if b and b != 0 else None
            except Exception:
                return None

        gm_pct  = [_pct(r.gross_profit, r.revenue) for _, r in df_q.iterrows()]
        ebitda_v = [
            r.ebitda if r.ebitda else
            ((r.ebit or 0) + (r.depreciation or 0)) if r.ebit else None
            for _, r in df_q.iterrows()
        ]
        em_pct = [_pct(e, r.revenue) for e, (_, r) in zip(ebitda_v, df_q.iterrows())]
        nm_pct = [_pct(r.net_income, r.revenue) for _, r in df_q.iterrows()]

        _CHART_H    = 300
        _CHART_M    = dict(l=0, r=0, t=36, b=0)
        _LEG_LAYOUT = dict(orientation="h", y=-0.25, x=0, font=dict(size=11))


        def _dual_bar(vals, yoy, bar_color, neg_color, title, bar_name, yoy_color):
            """Bar chart (left axis) + YoY% line (right axis)."""
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            colors = [neg_color if (v or 0) < 0 else bar_color for v in vals]
            fig.add_trace(
                go.Bar(x=labels, y=vals, name=bar_name, marker_color=colors,
                       hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=labels, y=yoy, name="Tăng trưởng YoY %", mode="lines+markers",
                           line=dict(color=yoy_color, width=2),
                           marker=dict(size=5),
                           hovertemplate="%{y:.1f}%<extra></extra>"),
                secondary_y=True,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray",
                          opacity=0.4, secondary_y=True)
            fig.update_layout(
                title=title, height=_CHART_H, margin=_CHART_M,
                legend=_LEG_LAYOUT, barmode="relative",
                hovermode="x unified", dragmode=False,
            )
            fig.update_yaxes(title_text="tỷ VND", secondary_y=False)
            fig.update_yaxes(title_text="Tăng trưởng %", secondary_y=True, showgrid=False)
            return fig

        # Chart 1 — Doanh Thu
        fig_rev = _dual_bar(
            df_q["revenue"].tolist(), rev_yoy,
            bar_color="#5b9bd5", neg_color="#d62728",
            title="Doanh thu", bar_name="Doanh thu", yoy_color="#e07b39",
        )

        # Chart 2 — Lợi Nhuận Sau Thuế
        fig_ni = _dual_bar(
            df_q["net_income"].tolist(), ni_yoy,
            bar_color="#2ca02c", neg_color="#d62728",
            title="Lợi nhuận sau thuế", bar_name="Lợi nhuận", yoy_color="#f5c518",
        )

        # Chart 3 — Biên Lợi Nhuận
        fig_mg = go.Figure()
        for series, name, color in [
            (gm_pct,  "Biên lợi nhuận gộp",  "#f5c518"),
            (em_pct,  "Biên EBITDA",         "#e07b39"),
            (nm_pct,  "Biên lợi nhuận thuần","#2ca02c"),
        ]:
            if any(v is not None for v in series):
                fig_mg.add_trace(go.Scatter(
                    x=labels, y=series, name=name, mode="lines",
                    line=dict(color=color, width=2),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ))
        fig_mg.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
        fig_mg.update_layout(
            title="Biên lợi nhuận (%)", height=_CHART_H, margin=_CHART_M,
            yaxis_title="%", legend=_LEG_LAYOUT, hovermode="x unified",
            dragmode=False,
        )

        c1, c2, c3 = st.columns(3)

        if _co_sect == "Ngân hàng":
            # ── Bank Row 1: Interest / Deposits / Deposit Structure ──

            # Chart B1 — Interest Income & Expense + NIM / Cost of Funds
            ii_v  = [((r.gross_profit or 0) + (r.interest_expense or 0)) for _, r in df_q.iterrows()]
            ie_v  = [-(r.interest_expense or 0) for _, r in df_q.iterrows()]
            nim_v = [_pct((r.gross_profit or 0) * 4, r.total_assets) for _, r in df_q.iterrows()]
            cof_v = [_pct((r.interest_expense or 0) * 4, r.payables) for _, r in df_q.iterrows()]

            fig_b1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_b1.add_trace(go.Bar(x=labels, y=ii_v, name="Thu nhập lãi",
                marker_color="#5b9bd5", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                secondary_y=False)
            fig_b1.add_trace(go.Bar(x=labels, y=ie_v, name="Chi phí lãi vay",
                marker_color="#ef4444", opacity=0.85,
                hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                secondary_y=False)
            fig_b1.add_trace(go.Scatter(x=labels, y=nim_v, name="NIM % (năm hóa)",
                mode="lines+markers", line=dict(color="#f5c518", width=2),
                marker=dict(size=5), hovertemplate="NIM %{y:.2f}%<extra></extra>"),
                secondary_y=True)
            fig_b1.add_trace(go.Scatter(x=labels, y=cof_v, name="Chi phí vốn %",
                mode="lines+markers", line=dict(color="#fb923c", width=2, dash="dash"),
                marker=dict(size=5), hovertemplate="CoF %{y:.2f}%<extra></extra>"),
                secondary_y=True)
            fig_b1.update_layout(title="Thu nhập & chi phí lãi", height=_CHART_H,
                margin=_CHART_M, barmode="relative", legend=_LEG_LAYOUT,
                hovermode="x unified", dragmode=False)
            fig_b1.update_yaxes(title_text="tỷ VND", secondary_y=False)
            fig_b1.update_yaxes(title_text="%", secondary_y=True, showgrid=False)

            # Chart B2 — Customer Deposits + YoY Growth
            dep_v   = df_q["payables"].tolist()
            dep_yoy = _yoy_full("payables")

            fig_b2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_b2.add_trace(go.Bar(x=labels, y=dep_v, name="Tiền gửi khách hàng",
                marker_color="#60a5fa", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                secondary_y=False)
            fig_b2.add_trace(go.Scatter(x=labels, y=dep_yoy, name="Tăng trưởng YoY %",
                mode="lines+markers", line=dict(color="#f5c518", width=2),
                marker=dict(size=5), hovertemplate="%{y:.1f}%<extra></extra>"),
                secondary_y=True)
            fig_b2.add_hline(y=0, line_dash="dot", line_color="gray",
                             opacity=0.4, secondary_y=True)
            fig_b2.update_layout(title="Tiền gửi khách hàng", height=_CHART_H,
                margin=_CHART_M, legend=_LEG_LAYOUT,
                hovermode="x unified", dragmode=False)
            fig_b2.update_yaxes(title_text="tỷ VND", secondary_y=False)
            fig_b2.update_yaxes(title_text="Tăng trưởng %", secondary_y=True, showgrid=False)

            # Chart B3 — Deposit Structure: Customer vs Interbank+SBV (% of total)
            cust_dep_v  = df_q["payables"].tolist()
            interbank_v = df_q["debt"].tolist()
            total_fund_v = [(c or 0) + (i or 0) for c, i in zip(cust_dep_v, interbank_v)]
            cust_pct_v      = [_pct(c, t) for c, t in zip(cust_dep_v, total_fund_v)]
            interbank_pct_v = [_pct(i, t) for i, t in zip(interbank_v, total_fund_v)]

            fig_b3 = go.Figure()
            fig_b3.add_trace(go.Bar(x=labels, y=cust_pct_v, name="Tiền gửi khách hàng",
                marker_color="#60a5fa", hovertemplate="%{y:.1f}%<extra></extra>"))
            fig_b3.add_trace(go.Bar(x=labels, y=interbank_pct_v, name="Liên ngân hàng & NHNN",
                marker_color="#1e3a5f", hovertemplate="%{y:.1f}%<extra></extra>"))
            fig_b3.update_layout(title="Cấu trúc nguồn huy động", height=_CHART_H,
                margin=_CHART_M, barmode="stack", legend=_LEG_LAYOUT,
                yaxis_title="%", hovermode="x unified", dragmode=False)

            with c1: st.plotly_chart(fig_b1, width="stretch")
            with c2: st.plotly_chart(fig_b2, width="stretch")
            with c3: st.plotly_chart(fig_b3, width="stretch")

        else:
            with c1:
                st.plotly_chart(fig_rev, width="stretch")
            with c2:
                st.plotly_chart(fig_ni,  width="stretch")
            with c3:
                st.plotly_chart(fig_mg,  width="stretch")

        st.divider()

        # ── Row 2: Profit breakdown · SG&A · CAPEX & Depreciation ──
        r2c1, r2c2, r2c3 = st.columns(3)
        _is_bank = (_co_sect == "Ngân hàng")

        # Chart 4 — Profit Structure (EBIT + financial drag + YoY)
        with r2c1:
            ebit_vals  = df_q["ebit"].tolist()
            # interest stored as positive expense → show as negative (drag on profit)
            int_vals   = [-(v or 0) for v in df_q["interest_expense"].tolist()]

            fig4 = make_subplots(specs=[[{"secondary_y": True}]])
            fig4.add_trace(
                go.Bar(x=labels, y=ebit_vals, name="Lợi nhuận hoạt động (EBIT)",
                       marker_color="#4472c4",
                       hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                secondary_y=False,
            )
            fig4.add_trace(
                go.Bar(x=labels, y=int_vals, name="Chi phí lãi vay (−)",
                       marker_color="#ffc000",
                       hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                secondary_y=False,
            )
            fig4.add_trace(
                go.Scatter(x=labels, y=ebit_yoy, name="Tăng trưởng EBIT %",
                           mode="lines", line=dict(color="#c00000", width=2),
                           hovertemplate="%{y:.1f}%<extra></extra>"),
                secondary_y=True,
            )
            fig4.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=True)
            fig4.update_layout(
                title="Cấu trúc lợi nhuận trước thuế", height=_CHART_H, margin=_CHART_M,
                barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified",
                dragmode=False,
            )
            fig4.update_yaxes(title_text="tỷ VND", secondary_y=False)
            fig4.update_yaxes(title_text="Tăng trưởng %", secondary_y=True, showgrid=False)

            st.plotly_chart(fig4, width="stretch")

        # Chart 5 — SG&A Expenses (stacked: selling + G&A)
        with r2c2:
            # Stored as negatives → abs for display
            sell_vals = [abs(v) if v else 0 for v in df_q["selling_expense"].tolist()]
            ga_vals   = [abs(v) if v else 0 for v in df_q["ga_expense"].tolist()]

            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=labels, y=ga_vals, name="Chi phí QLDN",
                marker_color="#7030a0",
                hovertemplate="%{y:,.0f} tỷ<extra></extra>",
            ))
            fig5.add_trace(go.Bar(
                x=labels, y=sell_vals, name="Chi phí bán hàng",
                marker_color="#b4a0e0",
                hovertemplate="%{y:,.0f} tỷ<extra></extra>",
            ))
            fig5.update_layout(
                title="Chi phí bán hàng & QLDN", height=_CHART_H, margin=_CHART_M,
                barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified",
                yaxis_title="tỷ VND", dragmode=False,
            )

            st.plotly_chart(fig5, width="stretch")

        # Chart 6 — CAPEX & Depreciation / [Bank] Credit Cost
        with r2c3:
            if _is_bank:
                prov_v    = df_q["cogs"].tolist()   # provisions for credit losses
                loans_v6  = df_q["receivables"].tolist()
                cc_pct    = [_pct((p or 0) * 4, l) for p, l in zip(prov_v, loans_v6)]
                prov_yoy  = _yoy(df_q["cogs"])
                fig6 = make_subplots(specs=[[{"secondary_y": True}]])
                fig6.add_trace(go.Bar(x=labels, y=prov_v, name="Trích lập dự phòng",
                    marker_color="#ef4444", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig6.add_trace(go.Scatter(x=labels, y=cc_pct, name="Chi phí tín dụng % (năm hóa)",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="%{y:.2f}%<extra></extra>"),
                    secondary_y=True)
                fig6.update_layout(title="Chi phí tín dụng & dự phòng", height=_CHART_H,
                    margin=_CHART_M, legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig6.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig6.update_yaxes(title_text="Chi phí tín dụng %", secondary_y=True, showgrid=False)
            else:
                capex_vals = df_q["capex"].tolist()
                dep_vals   = df_q["depreciation"].tolist()
                dep_ratio  = [(d / a * 100) if d and a and a > 0 else None
                              for d, a in zip(dep_vals, df_q["total_assets"].tolist())]
                fig6 = make_subplots(specs=[[{"secondary_y": True}]])
                fig6.add_trace(go.Bar(x=labels, y=capex_vals, name="CAPEX",
                    marker_color="#70ad47", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig6.add_trace(go.Bar(x=labels, y=dep_vals, name="Khấu hao",
                    marker_color="#ffc000", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig6.add_trace(go.Scatter(x=labels, y=dep_ratio, name="Khấu hao/Tài sản %",
                    mode="lines", fill="tozeroy", fillcolor="rgba(255,100,100,0.15)",
                    line=dict(color="rgba(255,100,100,0.6)", width=1.5),
                    hovertemplate="%{y:.2f}%<extra></extra>"), secondary_y=True)
                fig6.update_layout(title="CAPEX & Khấu hao", height=_CHART_H,
                    margin=_CHART_M, barmode="group", legend=_LEG_LAYOUT,
                    hovermode="x unified", dragmode=False)
                fig6.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig6.update_yaxes(title_text="Khấu hao/Tài sản %", secondary_y=True, showgrid=False)
            st.plotly_chart(fig6, width="stretch")

        # ── Row 3: Provisions · Financial Revenue · Financial Costs ──
        inc_raw, bal_raw = load_detailed_financials(ticker)
        periods_r3 = df_q["period"].tolist()

        def _rv(df, iid, p, scale=1e9):
            if df is None or df.empty or "item_id" not in df.columns or p not in df.columns:
                return None
            mask = df["item_id"] == iid
            if not mask.any():
                return None
            try:
                v = float(df.loc[mask, p].values[0])
                return None if pd.isna(v) else v / scale
            except Exception:
                return None

        def _ser(df, ids, periods, scale=1e9):
            out = []
            for p in periods:
                v = None
                for iid in ids:
                    v = _rv(df, iid, p, scale)
                    if v is not None:
                        break
                out.append(v)
            return out

        r3c1, r3c2, r3c3 = st.columns(3)

        if _is_bank:
            # ── Bank Row 3: Income Mix / OPEX+PPOP / Profitability KPIs ──
            with r3c1:
                nii_b   = df_q["gross_profit"].tolist()
                non_ii  = [(rv - n) if rv and n else None
                           for rv, n in zip(df_q["revenue"].tolist(), nii_b)]
                nii_pct = [_pct(n, rv) for n, rv in zip(nii_b, df_q["revenue"].tolist())]
                fig7b = make_subplots(specs=[[{"secondary_y": True}]])
                fig7b.add_trace(go.Bar(x=labels, y=nii_b, name="Thu nhập lãi thuần",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig7b.add_trace(go.Bar(x=labels, y=non_ii, name="Thu nhập ngoài lãi",
                    marker_color="#70ad47", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig7b.add_trace(go.Scatter(x=labels, y=nii_pct, name="Tỷ trọng NII %",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="%{y:.1f}%<extra></extra>"),
                    secondary_y=True)
                fig7b.update_layout(title="Cơ cấu thu nhập", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig7b.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig7b.update_yaxes(title_text="NII %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig7b, width="stretch")

            with r3c2:
                opex_v = df_q["ga_expense"].tolist()
                ppop_v = df_q["ebit"].tolist()
                cir_b  = [_pct(o, rv) for o, rv in zip(opex_v, df_q["revenue"].tolist())]
                fig8b = make_subplots(specs=[[{"secondary_y": True}]])
                fig8b.add_trace(go.Bar(x=labels, y=opex_v, name="Chi phí hoạt động",
                    marker_color="#ef4444", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig8b.add_trace(go.Bar(x=labels, y=ppop_v, name="Lợi nhuận trước dự phòng",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig8b.add_trace(go.Scatter(x=labels, y=cir_b, name="CIR %",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="CIR %{y:.1f}%<extra></extra>"),
                    secondary_y=True)
                fig8b.update_layout(title="Chi phí hoạt động & PPOP", height=_CHART_H, margin=_CHART_M,
                    barmode="group", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig8b.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig8b.update_yaxes(title_text="CIR %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig8b, width="stretch")

            with r3c3:
                roa_v  = [_pct((r.net_income or 0)*4, r.total_assets) for _, r in df_q.iterrows()]
                roe_v  = [_pct((r.net_income or 0)*4, r.equity)       for _, r in df_q.iterrows()]
                nim_v3 = [_pct((r.gross_profit or 0)*4, r.total_assets) for _, r in df_q.iterrows()]
                nm_v3  = [_pct(r.net_income, r.revenue)                  for _, r in df_q.iterrows()]
                fig9b  = go.Figure()
                for vals, name, color in [
                    (roa_v, "ROA % (năm hóa)", "#60a5fa"),
                    (roe_v, "ROE % (năm hóa)", "#f59e0b"),
                    (nim_v3,"NIM % (năm hóa)", "#22c55e"),
                    (nm_v3, "Biên lợi nhuận thuần %", "#c084fc"),
                ]:
                    if any(v is not None for v in vals):
                        fig9b.add_trace(go.Scatter(x=labels, y=vals, name=name,
                            mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                            hovertemplate="%{y:.2f}%<extra></extra>"))
                fig9b.update_layout(title="Các chỉ số sinh lời", height=_CHART_H,
                    margin=_CHART_M, yaxis_title="%", legend=_LEG_LAYOUT,
                    hovermode="x unified", dragmode=False)
                st.plotly_chart(fig9b, width="stretch")

        else:
            with r3c1:
                prov_st_rec = _ser(bal_raw, ["provision_for_doubtful_debts",
                    "provision_for_short_term_receivables",
                    "provision_for_diminution"], periods_r3)
                prov_inv    = _ser(bal_raw, ["provision_for_decline_in_inventories",
                    "provision_for_diminution_in_value_of_trading_securities"], periods_r3)
                prov_lt_rec = _ser(bal_raw, ["provision_for_doubtful_lt_receivable",
                    "provision_for_long_term_receivables"], periods_r3)
                prov_lt_inv = _ser(bal_raw, ["provision_for_long_term_investments",
                    "provision_for_diminution_in_value_of_long_term_investments"], periods_r3)
                fig7 = go.Figure()
                for vals, name, color in [
                    (prov_lt_rec, "DP phải thu dài hạn",  "#f4a460"),
                    (prov_lt_inv, "DP đầu tư dài hạn",    "#ff8c00"),
                    (prov_inv,    "DP hàng tồn kho",      "#ffd700"),
                    (prov_st_rec, "DP phải thu ngắn hạn", "#70ad47"),
                ]:
                    if any(v is not None for v in vals):
                        fig7.add_trace(go.Bar(x=labels, y=vals, name=name,
                            marker_color=color, hovertemplate="%{y:,.1f} tỷ<extra></extra>"))
                fig7.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig7.update_layout(title="Trích lập dự phòng", height=_CHART_H, margin=_CHART_M,
                    barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig7.update_yaxes(title_text="tỷ VND")
                st.plotly_chart(fig7, width="stretch")

            with r3c2:
                fin_income = _ser(inc_raw, ["financial_income",
                    "revenue_from_financial_activities"], periods_r3)
                import re as _re2
                _qpat = _re2.compile(r'^\d{4}-Q[1-4]$')
                _all_p = sorted([c for c in (inc_raw.columns if not inc_raw.empty else []) if _qpat.match(str(c))])
                _fi_full = [_rv(inc_raw, "financial_income", p) for p in _all_p]
                _fi_yoy_full = _yoy(pd.Series(_fi_full))
                _p2yoy = dict(zip(_all_p, _fi_yoy_full))
                fi_yoy = [_p2yoy.get(p) for p in periods_r3]
                fig8 = make_subplots(specs=[[{"secondary_y": True}]])
                fig8.add_trace(go.Bar(x=labels, y=fin_income, name="Doanh thu tài chính",
                    marker_color="#0d6efd", hovertemplate="%{y:,.1f} tỷ<extra></extra>"),
                    secondary_y=False)
                if any(v is not None for v in fi_yoy):
                    fig8.add_trace(go.Scatter(x=labels, y=fi_yoy, name="Tăng trưởng %",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:.1f}%<extra></extra>"), secondary_y=True)
                fig8.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
                fig8.update_layout(title="Doanh thu tài chính", height=_CHART_H, margin=_CHART_M,
                    legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig8.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig8.update_yaxes(title_text="Tăng trưởng %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig8, width="stretch")

            with r3c3:
                fin_exp_raw = _ser(inc_raw, ["financial_expenses",
                    "expense_from_financial_activities"], periods_r3)
                int_exp_raw = _ser(inc_raw, ["interest_expenses", "interests_expenses"], periods_r3)
                fin_exp_abs = [abs(v) if v is not None else None for v in fin_exp_raw]
                int_exp_abs = [abs(v) if v is not None else None for v in int_exp_raw]
                other_exp   = [round(fe - ie, 3) if (fe is not None and ie is not None) else fe
                               for fe, ie in zip(fin_exp_abs, int_exp_abs)]
                fig9 = make_subplots(specs=[[{"secondary_y": True}]])
                fig9.add_trace(go.Bar(x=labels, y=int_exp_abs, name="Chi phí lãi vay",
                    marker_color="#c00000", hovertemplate="%{y:,.1f} tỷ<extra></extra>"),
                    secondary_y=False)
                if any(v is not None and v > 0 for v in other_exp):
                    fig9.add_trace(go.Bar(x=labels, y=other_exp, name="Chi phí tài chính khác",
                        marker_color="#4472c4", hovertemplate="%{y:,.1f} tỷ<extra></extra>"),
                        secondary_y=False)
                fig9.add_trace(go.Scatter(x=labels, y=fin_exp_abs, name="Tổng chi phí tài chính",
                    mode="lines", line=dict(color="#ffb3b3", width=2),
                    hovertemplate="%{y:,.1f} tỷ<extra></extra>"), secondary_y=True)
                fig9.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
                fig9.update_layout(title="Chi phí tài chính", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig9.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig9.update_yaxes(title_text="tỷ VND (tổng)", secondary_y=True, showgrid=False)
                st.plotly_chart(fig9, width="stretch")

        st.divider()

        # ── Row 4: Balance Sheet Structure ──────────────────────────
        r4c1, r4c2, r4c3 = st.columns(3)

        if _is_bank:
            # ── Bank Row 4: Asset Mix / Funding Structure / Liquidity ──
            loans_r4  = df_q["receivables"].tolist()
            dep_r4    = df_q["payables"].tolist()
            ib_r4     = df_q["debt"].tolist()
            eq_r4     = df_q["equity"].tolist()
            cash_r4   = df_q["cash"].tolist()
            ta_r4     = df_q["total_assets"].tolist()

            with r4c1:
                # Bank Asset Mix: Cash, Loans, Other
                other_r4 = [max(0, (ta or 0) - (c or 0) - (l or 0))
                            for ta, c, l in zip(ta_r4, cash_r4, loans_r4)]
                fig10b = go.Figure()
                fig10b.add_trace(go.Bar(x=labels, y=cash_r4, name="Tiền mặt & NHNN",
                    marker_color="#5bc0de", hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig10b.add_trace(go.Bar(x=labels, y=loans_r4, name="Dư nợ cho vay",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig10b.add_trace(go.Bar(x=labels, y=other_r4, name="Chứng khoán & Khác",
                    marker_color="#9467bd", hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig10b.update_layout(title="Cơ cấu tài sản", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig10b.update_yaxes(title_text="tỷ VND")
                st.plotly_chart(fig10b, width="stretch")

            with r4c2:
                # Funding Structure: Deposits, Interbank, Equity, Other
                other_fund = [max(0, (ta or 0) - (d or 0) - (ib or 0) - (e or 0))
                              for ta, d, ib, e in zip(ta_r4, dep_r4, ib_r4, eq_r4)]
                fig11b = go.Figure()
                fig11b.add_trace(go.Bar(x=labels, y=dep_r4, name="Tiền gửi khách hàng",
                    marker_color="#60a5fa", hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig11b.add_trace(go.Bar(x=labels, y=ib_r4, name="Liên ngân hàng & NHNN",
                    marker_color="#1e40af", hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig11b.add_trace(go.Bar(x=labels, y=eq_r4, name="Vốn chủ sở hữu",
                    marker_color="#22c55e", hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig11b.add_trace(go.Bar(x=labels, y=other_fund, name="Nợ phải trả khác",
                    marker_color="#6b7280", hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig11b.update_layout(title="Cơ cấu nguồn vốn", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig11b.update_yaxes(title_text="tỷ VND")
                st.plotly_chart(fig11b, width="stretch")

            with r4c3:
                # Liquidity: LDR, Loan/Asset, Cash/Deposit
                ldr_r4  = [_pct(l, d) for l, d in zip(loans_r4, dep_r4)]
                la_r4   = [_pct(l, ta) for l, ta in zip(loans_r4, ta_r4)]
                cd_r4   = [_pct(c, d)  for c, d  in zip(cash_r4, dep_r4)]
                fig12b = go.Figure()
                for vals, name, color in [
                    (ldr_r4, "LDR % (Vay/Huy động)",     "#f59e0b"),
                    (la_r4,  "Dư nợ/Tổng tài sản %",      "#60a5fa"),
                    (cd_r4,  "Tiền mặt/Huy động %",       "#22c55e"),
                ]:
                    if any(v is not None for v in vals):
                        fig12b.add_trace(go.Scatter(x=labels, y=vals, name=name,
                            mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                            hovertemplate="%{y:.1f}%<extra></extra>"))
                fig12b.update_layout(title="Chỉ số thanh khoản", height=_CHART_H, margin=_CHART_M,
                    yaxis_title="%", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                st.plotly_chart(fig12b, width="stretch")

        else:
            # Reuse bal_raw and periods_r3 already loaded for Row 3
            cash_v   = _ser(bal_raw, ["cash_and_cash_equivalents"], periods_r3)
            recv_v   = _ser(bal_raw, ["trade_accounts_receivable"],  periods_r3)
            inv_v    = _ser(bal_raw, ["inventories_net"],             periods_r3)
            curr_v   = _ser(bal_raw, ["current_assets"],              periods_r3)
            total_v  = _ser(bal_raw, ["total_assets"],                periods_r3)
            curr_liab_v = _ser(bal_raw, ["current_liabilities"],      periods_r3)
            st_borrow   = _ser(bal_raw, ["short_term_borrowings"],    periods_r3)
            lt_borrow   = _ser(bal_raw, ["long_term_borrowings"],     periods_r3)
            payable_v   = _ser(bal_raw, ["trade_accounts_payable"],   periods_r3)
            equity_v    = _ser(bal_raw, ["owners_equity"],            periods_r3)
            other_curr = [max(0.0, (c or 0) - (k or 0) - (r or 0) - (i or 0))
                          for c, k, r, i in zip(curr_v, cash_v, recv_v, inv_v)]
            non_curr = [max(0.0, (t or 0) - (c or 0)) for t, c in zip(total_v, curr_v)]
            other_liab = [max(0.0, (t or 0) - (s or 0) - (l or 0) - (p or 0) - (e or 0))
                          for t, s, l, p, e in zip(total_v, st_borrow, lt_borrow, payable_v, equity_v)]
            cash_pct = [round(k / t * 100, 2) if k and t and t > 0 else None
                        for k, t in zip(cash_v, total_v)]
            recv_pct = [round(r / t * 100, 2) if r and t and t > 0 else None
                        for r, t in zip(recv_v, total_v)]

            with r4c1:
                fig10 = go.Figure()
                for vals, name, color in [
                    (cash_v,     "Tiền & tương đương",   "#5bc0de"),
                    (recv_v,     "Phải thu",             "#f0ad4e"),
                    (inv_v,      "Hàng tồn kho",         "#5cb85c"),
                    (other_curr, "TS ngắn hạn khác",     "#9b59b6"),
                    (non_curr,   "TS dài hạn",           "#e74c3c"),
                ]:
                    if any(v is not None and v > 0 for v in vals):
                        fig10.add_trace(go.Bar(x=labels, y=vals, name=name,
                            marker_color=color, hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig10.update_layout(title="Cấu trúc tài sản", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig10.update_yaxes(title_text="tỷ VND")
                st.plotly_chart(fig10, width="stretch")

            with r4c2:
                fig11 = go.Figure()
                for vals, name, color in [
                    (equity_v,   "Vốn chủ sở hữu",      "#2ca02c"),
                    (lt_borrow,  "Vay dài hạn",          "#d62728"),
                    (st_borrow,  "Vay ngắn hạn",         "#ff7f0e"),
                    (payable_v,  "Phải trả người bán",   "#1f77b4"),
                    (other_liab, "Nợ phải trả khác",     "#9467bd"),
                ]:
                    if any(v is not None and v > 0 for v in vals):
                        fig11.add_trace(go.Bar(x=labels, y=vals, name=name,
                            marker_color=color, hovertemplate="%{y:,.0f} tỷ<extra></extra>"))
                fig11.update_layout(title="Cấu trúc nguồn vốn", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig11.update_yaxes(title_text="tỷ VND")
                st.plotly_chart(fig11, width="stretch")

            with r4c3:
                cr_hist  = [current_ratio(ca, cl)         for ca, cl in zip(curr_v, curr_liab_v)]
                qr_hist  = [quick_ratio(ca, iv, cl)       for ca, iv, cl in zip(curr_v, inv_v, curr_liab_v)]
                cashr_hist = [absolute_liquidity(ck, cl)  for ck, cl in zip(cash_v, curr_liab_v)]
                fig12 = go.Figure()
                for vals, name, color in [
                    (cr_hist,    "Current ratio",  "#22c55e"),
                    (qr_hist,    "Quick ratio",    "#f59e0b"),
                    (cashr_hist, "Cash ratio",     "#60a5fa"),
                ]:
                    if any(v is not None for v in vals):
                        fig12.add_trace(go.Scatter(x=labels, y=vals, name=name,
                            mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                            hovertemplate="%{y:.2f}x<extra></extra>"))
                fig12.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.5)
                fig12.update_layout(title="Hệ số thanh khoản", height=_CHART_H, margin=_CHART_M,
                    yaxis_title="lần (x)", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                st.plotly_chart(fig12, width="stretch")

        st.divider()

        # ── Row 5: Receivables · Inventory · Debt Breakdown ─────────
        r5c1, r5c2, r5c3 = st.columns(3)

        if _is_bank:
            # ── Bank Row 5: Loan Book / Asset Quality / Capital Ratios ──
            loans_r5 = df_q["receivables"].tolist()
            prov_r5  = df_q["cogs"].tolist()   # provisions
            eq_r5    = df_q["equity"].tolist()
            ta_r5    = df_q["total_assets"].tolist()

            with r5c1:
                # Loan Book: net loans + YoY growth
                loans_yoy = _yoy_full("receivables")
                fig13b = make_subplots(specs=[[{"secondary_y": True}]])
                fig13b.add_trace(go.Bar(x=labels, y=loans_r5, name="Dư nợ cho vay (ròng)",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} tỷ<extra></extra>"),
                    secondary_y=False)
                fig13b.add_trace(go.Scatter(x=labels, y=loans_yoy, name="Tăng trưởng YoY %",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="%{y:.1f}%<extra></extra>"),
                    secondary_y=True)
                fig13b.add_hline(y=0, line_dash="dot", line_color="gray",
                                 opacity=0.4, secondary_y=True)
                fig13b.update_layout(title="Dư nợ cho vay", height=_CHART_H, margin=_CHART_M,
                    legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig13b.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig13b.update_yaxes(title_text="Tăng trưởng %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig13b, width="stretch")

            with r5c2:
                # Provisions & Reversals — green for reversals (hoàn nhập), red for new provisions
                _prov_colors = [
                    "#22c55e" if (p or 0) < 0 else "#ef4444"
                    for p in prov_r5
                ]
                _prov_labels = [
                    f"{'Hoàn nhập' if (p or 0) < 0 else 'Trích lập'}: {abs(p or 0):,.0f} tỷ"
                    for p in prov_r5
                ]
                cc_r5 = [_pct((p or 0) * 4, l) for p, l in zip(prov_r5, loans_r5)]

                fig14b = make_subplots(specs=[[{"secondary_y": True}]])
                fig14b.add_trace(go.Bar(
                    x=labels, y=prov_r5, name="Trích lập / Hoàn nhập",
                    marker_color=_prov_colors,
                    text=["★ HOÀN NHẬP" if (p or 0) < 0 else "" for p in prov_r5],
                    textposition="outside",
                    textfont=dict(color="#22c55e", size=10),
                    hovertemplate="%{x}: %{y:,.0f} tỷ<extra></extra>",
                ), secondary_y=False)
                fig14b.add_hline(y=0, line_dash="dot", line_color="gray",
                                 opacity=0.5, secondary_y=False)
                fig14b.add_trace(go.Scatter(
                    x=labels, y=cc_r5, name="Chi phí tín dụng % (năm hóa)",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="CoC %{y:.2f}%<extra></extra>",
                ), secondary_y=True)
                fig14b.update_layout(
                    title="Trích lập & hoàn nhập dự phòng", height=_CHART_H, margin=_CHART_M,
                    legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False,
                    annotations=[dict(
                        text="Xanh = Hoàn nhập | Đỏ = Trích lập mới",
                        x=0, xref="paper", y=-0.28, yref="paper",
                        xanchor="left", font=dict(size=10, color="#6b7280"),
                        showarrow=False,
                    )]
                )
                fig14b.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig14b.update_yaxes(title_text="Chi phí tín dụng %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig14b, width="stretch")

            with r5c3:
                # Capital Ratios: Equity/Assets, Equity/Loans, ROE
                ea_r5  = [_pct(e, ta) for e, ta in zip(eq_r5, ta_r5)]
                el_r5  = [_pct(e, l)  for e, l  in zip(eq_r5, loans_r5)]
                roe_r5 = [_pct((r.net_income or 0) * 4, r.equity) for _, r in df_q.iterrows()]
                fig15b = go.Figure()
                for vals, name, color in [
                    (ea_r5,  "Vốn CSH/Tổng TS %", "#60a5fa"),
                    (el_r5,  "Vốn CSH/Dư nợ %",   "#f59e0b"),
                    (roe_r5, "ROE % (năm hóa)",   "#22c55e"),
                ]:
                    if any(v is not None for v in vals):
                        fig15b.add_trace(go.Scatter(x=labels, y=vals, name=name,
                            mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                            hovertemplate="%{y:.1f}%<extra></extra>"))
                fig15b.update_layout(title="Chỉ số vốn", height=_CHART_H, margin=_CHART_M,
                    yaxis_title="%", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                st.plotly_chart(fig15b, width="stretch")

        else:
            # Chart 13 — Receivables Structure
            with r5c1:
                r_trade  = _ser(bal_raw, ["trade_accounts_receivable"],        periods_r3)
                r_other  = _ser(bal_raw, ["other_receivables"],                periods_r3)
                r_lt     = _ser(bal_raw, ["long_term_loans_receivables",
                                          "long_term_trade_receivables"],      periods_r3)
                r_prov_st = _ser(bal_raw, ["provision_for_doubtful_debts"],    periods_r3)
                r_prov_lt = _ser(bal_raw, ["provision_for_doubtful_lt_receivable"], periods_r3)
                # provisions come as negative from VCI — keep sign for stacked bar
                r_prov_st_neg = [-(abs(v)) if v is not None and v != 0 else v for v in r_prov_st]
                r_prov_lt_neg = [-(abs(v)) if v is not None and v != 0 else v for v in r_prov_lt]
                # net total line
                r_net = [
                    sum(x for x in [t, o, l, ps, pl] if x is not None)
                    for t, o, l, ps, pl in zip(r_trade, r_other, r_lt, r_prov_st_neg, r_prov_lt_neg)
                ]

                fig13 = go.Figure()
                for vals, name, color in [
                    (r_trade,      "Phải thu khách hàng",       "#5bc0de"),
                    (r_other,      "Phải thu ngắn hạn khác",    "#f0ad4e"),
                    (r_lt,         "Phải thu dài hạn",          "#555555"),
                    (r_prov_st_neg,"DP phải thu ngắn hạn",      "#d9534f"),
                    (r_prov_lt_neg,"DP phải thu dài hạn",       "#e87c6e"),
                ]:
                    if any(v is not None and v != 0 for v in vals):
                        fig13.add_trace(go.Bar(
                            x=labels, y=vals, name=name, marker_color=color,
                            hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                        ))
                if any(v != 0 for v in r_net):
                    fig13.add_trace(go.Scatter(
                        x=labels, y=r_net, name="Tổng ròng",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                    ))
                fig13.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig13.update_layout(
                    title="Các khoản phải thu", height=_CHART_H, margin=_CHART_M,
                    barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified",
                    dragmode=False,
                )
                fig13.update_yaxes(title_text="tỷ VND")
                st.plotly_chart(fig13, width="stretch")

            # Chart 14 — Inventory / Trading Securities (for securities firms)
            with r5c2:
                if _co_sect == "Chứng khoán":
                    _ts = _ser(bal_raw, [
                        "financial_assets_at_fair_value_through_profit_or_loss_fvtpl",
                        "trading_securities", "trading_securities_2",
                        "available_for_sale_financial_assets_afs",
                    ], periods_r3)
                    _ts = _ts or [None] * len(labels)
                    _ts_qoq = _yoy(pd.Series(_ts), lag=1)  # QoQ: starts from 2nd bar
                    fig14 = make_subplots(specs=[[{"secondary_y": True}]])
                    if any(v is not None and v != 0 for v in _ts):
                        fig14.add_trace(go.Bar(x=labels, y=_ts, name="Tài sản tài chính (FVTPL/AFS)",
                            marker_color="#5b9bd5",
                            hovertemplate="%{y:,.0f} tỷ<extra></extra>"), secondary_y=False)
                    if any(v is not None for v in _ts_qoq):
                        fig14.add_trace(go.Scatter(x=labels, y=_ts_qoq, name="Tăng trưởng QoQ %",
                            mode="lines+markers", line=dict(color="#f5c518", width=2),
                            marker=dict(size=5),
                            hovertemplate="%{y:.1f}%<extra></extra>"), secondary_y=True)
                    fig14.update_layout(title="Danh mục tài sản tài chính", height=_CHART_H,
                        margin=_CHART_M, legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                    fig14.update_yaxes(title_text="tỷ VND", secondary_y=False)
                    fig14.update_yaxes(title_text="Tăng trưởng %", secondary_y=True, showgrid=False)
                    st.plotly_chart(fig14, width="stretch")
                else:
                    inv_gross = _ser(bal_raw, ["inventories"],                        periods_r3)
                    inv_net   = _ser(bal_raw, ["inventories_net"],                    periods_r3)
                    inv_prov  = _ser(bal_raw, ["provision_for_decline_in_inventories"],periods_r3)
                    inv_prov_neg = [-(abs(v)) if v is not None and v != 0 else v for v in inv_prov]
                    inv_pct = [round(n / t * 100, 2) if n and t and t > 0 else None
                               for n, t in zip(inv_net, total_v)]
                    fig14 = make_subplots(specs=[[{"secondary_y": True}]])
                    if any(v is not None and v > 0 for v in inv_gross):
                        fig14.add_trace(go.Bar(x=labels, y=inv_gross, name="Hàng tồn kho (gộp)",
                            marker_color="#f0ad4e",
                            hovertemplate="%{y:,.0f} tỷ<extra></extra>"), secondary_y=False)
                    elif any(v is not None and v > 0 for v in inv_net):
                        fig14.add_trace(go.Bar(x=labels, y=inv_net, name="Hàng tồn kho (ròng)",
                            marker_color="#f0ad4e",
                            hovertemplate="%{y:,.0f} tỷ<extra></extra>"), secondary_y=False)
                    if any(v is not None and v != 0 for v in inv_prov_neg):
                        fig14.add_trace(go.Bar(x=labels, y=inv_prov_neg, name="DP giảm giá hàng tồn kho",
                            marker_color="#d9534f",
                            hovertemplate="%{y:,.0f} tỷ<extra></extra>"), secondary_y=False)
                    if any(v is not None for v in inv_pct):
                        fig14.add_trace(go.Scatter(x=labels, y=inv_pct, name="Tồn kho/Tổng TS %",
                            mode="lines", line=dict(color="#c00000", width=2),
                            hovertemplate="%{y:.1f}%<extra></extra>"), secondary_y=True)
                    fig14.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
                    fig14.update_layout(title="Hàng tồn kho", height=_CHART_H, margin=_CHART_M,
                        barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                    fig14.update_yaxes(title_text="tỷ VND", secondary_y=False)
                    fig14.update_yaxes(title_text="%", secondary_y=True, showgrid=False)
                    st.plotly_chart(fig14, width="stretch")

            # Chart 15 — Financial Leverage
            with r5c3:
                d_st    = _ser(bal_raw, ["short_term_borrowings"],  periods_r3)
                d_lt    = _ser(bal_raw, ["long_term_borrowings"],   periods_r3)
                d_bonds = _ser(bal_raw, ["convertible_bonds"],      periods_r3)
                d_other = _ser(bal_raw, ["other_long_term_payables"], periods_r3)
                eq_v    = equity_v  # already computed above
                total_debt = [
                    (s or 0) + (l or 0) + (b or 0) + (o or 0)
                    for s, l, b, o in zip(d_st, d_lt, d_bonds, d_other)
                ]
                de_ratio = [
                    round(td / e, 2) if e and e > 0 and td > 0 else None
                    for td, e in zip(total_debt, eq_v)
                ]

                fig15 = make_subplots(specs=[[{"secondary_y": True}]])
                for vals, name, color in [
                    (d_st,    "Vay ngắn hạn",         "#ff7f0e"),
                    (d_lt,    "Vay dài hạn",          "#1f77b4"),
                    (d_bonds, "Trái phiếu chuyển đổi", "#e74c3c"),
                    (d_other, "Phải trả dài hạn khác", "#9467bd"),
                ]:
                    if any(v is not None and v > 0 for v in vals):
                        fig15.add_trace(go.Bar(
                            x=labels, y=vals, name=name, marker_color=color,
                            hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                        ), secondary_y=False)
                if any(v is not None for v in de_ratio):
                    fig15.add_trace(go.Scatter(
                        x=labels, y=de_ratio, name="Tỷ số Nợ/VCSH (D/E)",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:.2f}x<extra></extra>",
                    ), secondary_y=True)
                fig15.update_layout(
                    title="Đòn bẩy tài chính", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified",
                    dragmode=False,
                )
                fig15.update_yaxes(title_text="tỷ VND", secondary_y=False)
                fig15.update_yaxes(title_text="D/E (x)", secondary_y=True, showgrid=False)
                st.plotly_chart(fig15, width="stretch")


        st.divider()

        # ── Row 6: Cash Flow · Dividends · Valuation Multiples ──────
        r6c1, r6c2, r6c3 = st.columns(3)

        # Chart 16 — Cash Flow Waterfall
        with r6c1:
            ocf_v = df_q["operating_cf"].tolist()
            icf_v = df_q["investing_cf"].tolist()
            fcf_v = df_q["financing_cf"].tolist()
            # cash_v: use DB field (works for both regular and banking stocks)
            cash_v = df_q["cash"].tolist()

            fig16 = make_subplots(specs=[[{"secondary_y": True}]])
            for vals, name, color in [
                (ocf_v, "Dòng tiền HĐKD",  "#2ca02c"),
                (icf_v, "Dòng tiền đầu tư", "#1f77b4"),
                (fcf_v, "Dòng tiền tài chính", "#ffc000"),
            ]:
                if any(v is not None for v in vals):
                    fig16.add_trace(go.Bar(
                        x=labels, y=vals, name=name, marker_color=color,
                        hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                    ), secondary_y=False)
            if any(v is not None for v in cash_v):
                fig16.add_trace(go.Scatter(
                    x=labels, y=cash_v, name="Tiền cuối kỳ",
                    mode="lines", line=dict(color="#d62728", width=2),
                    hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                ), secondary_y=True)
            fig16.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
            fig16.update_layout(
                title="Dòng tiền", height=_CHART_H, margin=_CHART_M,
                barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified",
                dragmode=False,
            )
            fig16.update_yaxes(title_text="tỷ VND", secondary_y=False)
            fig16.update_yaxes(title_text="Tiền (tỷ)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig16, width="stretch")

        # Chart 17 — Dividends (annual)
        with r6c2:
            cf_ann, inc_ann = load_annual_cf(ticker)
            import re as _re3
            _ypat = _re3.compile(r'^\d{4}$')

            def _yr_ser(df, iid):
                if df is None or df.empty or "item_id" not in df.columns:
                    return {}, []
                ycols = sorted([c for c in df.columns if _ypat.match(str(c))])
                mask = df["item_id"] == iid
                if not mask.any():
                    return {}, ycols
                row_vals = df.loc[mask].iloc[0]
                return {c: row_vals[c] for c in ycols}, ycols

            div_map, ycols = _yr_ser(cf_ann,  "dividends_paid")
            ni_map,  _     = _yr_ser(inc_ann, "net_profit_loss_after_tax")

            ylabels   = ycols
            div_vals  = [abs(div_map.get(y, 0) or 0) / 1e9 for y in ycols]  # raw VND → bn
            payout    = []
            for y in ycols:
                d  = abs(div_map.get(y) or 0)
                ni = abs(ni_map.get(y)  or 0)
                payout.append(round(d / ni * 100, 2) if ni > 0 else None)

            fig17 = make_subplots(specs=[[{"secondary_y": True}]])
            if any(v > 0 for v in div_vals):
                fig17.add_trace(go.Bar(
                    x=ylabels, y=div_vals, name="Cổ tức tiền mặt",
                    marker_color="#5bc0de",
                    hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                ), secondary_y=False)
            if any(v is not None for v in payout):
                fig17.add_trace(go.Scatter(
                    x=ylabels, y=payout, name="Tỷ lệ chi trả %",
                    mode="lines", line=dict(color="#c00000", width=2),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ), secondary_y=True)
            fig17.update_layout(
                title="Cổ tức (hàng năm)", height=_CHART_H, margin=_CHART_M,
                legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False,
            )
            fig17.update_yaxes(title_text="tỷ VND", secondary_y=False)
            fig17.update_yaxes(title_text="Tỷ lệ chi trả %", secondary_y=True, showgrid=False)
            st.plotly_chart(fig17, width="stretch")

        # Chart 18 — Valuation Multiples (P/E primary, P/B secondary axis)
        with r6c3:
            mult_df = load_valuation_multiples(ticker)
            fig18 = make_subplots(specs=[[{"secondary_y": True}]])
            if not mult_df.empty:
                mult_labels = [_fmt_period(p) for p in mult_df["period"]]
                if mult_df["pe"].notna().any():
                    fig18.add_trace(go.Scatter(
                        x=mult_labels, y=mult_df["pe"].tolist(),
                        mode="lines", line=dict(color="#2ca02c", width=1.5, dash="dash"),
                        connectgaps=True, showlegend=False, opacity=0.4,
                        hoverinfo="skip",
                    ), secondary_y=False)
                    fig18.add_trace(go.Scatter(
                        x=mult_labels, y=mult_df["pe"].tolist(),
                        name="P/E", mode="lines",
                        line=dict(color="#2ca02c", width=2),
                        connectgaps=False,
                        hovertemplate="P/E %{y:.1f}x<extra></extra>",
                    ), secondary_y=False)
                if mult_df["pb"].notna().any():
                    fig18.add_trace(go.Scatter(
                        x=mult_labels, y=mult_df["pb"].tolist(),
                        mode="lines", line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                        connectgaps=True, showlegend=False, opacity=0.4,
                        hoverinfo="skip",
                    ), secondary_y=True)
                    fig18.add_trace(go.Scatter(
                        x=mult_labels, y=mult_df["pb"].tolist(),
                        name="P/B", mode="lines",
                        line=dict(color="#ff7f0e", width=2),
                        connectgaps=False,
                        hovertemplate="P/B %{y:.2f}x<extra></extra>",
                    ), secondary_y=True)
            # dynamic P/E y-range clipped at ±3× p90 so one spike doesn't flatten everything
            _pe_vals = mult_df["pe"].dropna().tolist() if not mult_df.empty else []
            if _pe_vals:
                _p90 = float(np.percentile([abs(v) for v in _pe_vals], 90))
                _ymax = max(_p90 * 3, 20)
                _ymin = -_ymax * 0.5
            else:
                _ymax, _ymin = 50, -25
            fig18.update_layout(
                title="Định giá (P/E & P/B)", height=_CHART_H, margin=_CHART_M,
                legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False,
            )
            fig18.update_yaxes(title_text="P/E (x)", secondary_y=False,
                               range=[_ymin, _ymax], zeroline=True,
                               zerolinecolor="gray", zerolinewidth=1)
            fig18.update_yaxes(title_text="P/B (x)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig18, width="stretch")

        st.divider()

        # ── Row 7: Business Projection · Price vs Value · Volume ────
        r7c1, r7c2, r7c3 = st.columns(3)

        # Chart 19 — Business Projection (bars primary, net margin % line secondary)
        with r7c1:
            ann_df = load_annual_financials(ticker)
            fig19 = make_subplots(specs=[[{"secondary_y": True}]])
            if not ann_df.empty:
                tail6   = ann_df.tail(6)
                act_yrs = tail6["period"].tolist()
                rev_act = tail6["revenue"].tolist()
                ni_act  = tail6["net_income"].tolist()
                mgn_act = [round(ni / rv * 100, 1) if rv and rv != 0 else None
                           for ni, rv in zip(ni_act, rev_act)]

                fig19.add_trace(go.Bar(
                    x=act_yrs, y=rev_act, name="Doanh thu",
                    marker_color="#1f77b4",
                    hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                ), secondary_y=False)
                fig19.add_trace(go.Bar(
                    x=act_yrs, y=ni_act, name="Lợi nhuận sau thuế",
                    marker_color="#aec7e8",
                    hovertemplate="%{y:,.0f} tỷ<extra></extra>",
                ), secondary_y=False)
                fig19.add_trace(go.Scatter(
                    x=act_yrs, y=mgn_act, name="Biên lợi nhuận thuần %",
                    mode="lines+markers",
                    line=dict(color="#f59e0b", width=2),
                    marker=dict(size=5),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ), secondary_y=True)

                if len(tail6) >= 3:
                    r3 = rev_act[-3:]
                    n3 = ni_act[-3:]
                    cagr_r = (r3[-1] / r3[0]) ** (1/2) - 1 if r3[0] and r3[0] > 0 else 0
                    cagr_n = (n3[-1] / n3[0]) ** (1/2) - 1 if n3[0] and n3[0] > 0 else 0
                    last_yr = int(act_yrs[-1])
                    _nf = 3
                    proj_yrs = [f"{last_yr + k}F" for k in range(1, _nf + 1)]
                    proj_rev = [round(rev_act[-1] * (1 + cagr_r) ** k, 1) for k in range(1, _nf + 1)]
                    proj_ni  = [round(ni_act[-1]  * (1 + cagr_n) ** k, 1) for k in range(1, _nf + 1)]
                    proj_mgn = [round(pni / prv * 100, 1) if prv and prv != 0 else None
                                for pni, prv in zip(proj_ni, proj_rev)]
                    _cr_str = f"{cagr_r * 100:.1f}"
                    _cn_str = f"{cagr_n * 100:.1f}"
                    fig19.add_trace(go.Bar(
                        x=proj_yrs, y=proj_rev, name="Doanh thu (DB)",
                        marker=dict(color="#1f77b4", opacity=0.7,
                                    pattern=dict(shape="/", size=6, solidity=0.4)),
                        showlegend=False,
                        hovertemplate=f"%{{x}}: %{{y:,.0f}} tỷ (CAGR {_cr_str}%)<extra></extra>",
                    ), secondary_y=False)
                    fig19.add_trace(go.Bar(
                        x=proj_yrs, y=proj_ni, name="LNST (DB)",
                        marker=dict(color="#aec7e8", opacity=0.7,
                                    pattern=dict(shape="/", size=6, solidity=0.4)),
                        showlegend=False,
                        hovertemplate=f"%{{x}}: %{{y:,.0f}} tỷ (CAGR {_cn_str}%)<extra></extra>",
                    ), secondary_y=False)
                    fig19.add_trace(go.Scatter(
                        x=proj_yrs, y=proj_mgn, name="Biên LN thuần % (DB)",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                        marker=dict(size=5),
                        showlegend=False,
                        hovertemplate="%{y:.1f}%<extra></extra>",
                    ), secondary_y=True)

            fig19.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4,
                            secondary_y=False)
            fig19.update_layout(
                title="Dự báo kinh doanh", height=_CHART_H, margin=_CHART_M,
                barmode="group", legend=_LEG_LAYOUT, hovermode="x unified",
                dragmode=False,
            )
            fig19.update_yaxes(title_text="tỷ VND", secondary_y=False)
            fig19.update_yaxes(title_text="Biên LN thuần %", secondary_y=True, showgrid=False)
            st.plotly_chart(fig19, width="stretch")

        # Chart 20 — Price vs Avg Intrinsic Value
        with r7c2:
            price_df20 = load_prices(ticker)
            vh_df20    = valuation_history(ticker)
            fig20 = go.Figure()
            if not price_df20.empty:
                _p20 = price_df20.copy()
                _p20["date"] = pd.to_datetime(_p20["date"])
                _p20 = _p20.sort_values("date")
                _p20["price_vnd"] = _p20["close"] * 1000  # thousands VND → raw VND
                fig20.add_trace(go.Scatter(
                    x=_p20["date"], y=_p20["price_vnd"],
                    name="Giá thị trường", mode="lines",
                    line=dict(color="#1f77b4", width=1.5),
                    fill="tozeroy", fillcolor="rgba(31,119,180,0.12)",
                    hovertemplate="%{y:,.0f} VND<extra></extra>",
                ))
            if not vh_df20.empty and "avg" in vh_df20.columns:
                _v20 = vh_df20[vh_df20["avg"].notna()].copy()
                _v20["date"] = pd.to_datetime(_v20["date"])
                if not _v20.empty and not price_df20.empty:
                    # extend to match full price date range
                    p_dates = pd.to_datetime(_p20["date"])
                    _ext = pd.DataFrame({"date": [p_dates.iloc[0], p_dates.iloc[-1]], "avg": [None, None]})
                    _v20 = pd.concat([_ext, _v20], ignore_index=True).sort_values("date")
                    _v20["avg"] = _v20["avg"].interpolate(method="linear").ffill().bfill()
                    fig20.add_trace(go.Scatter(
                        x=_v20["date"], y=_v20["avg"],
                        name="Giá trị nội tại TB", mode="lines",
                        line=dict(color="#c00000", width=2, dash="dash"),
                        hovertemplate="%{y:,.0f} VND<extra></extra>",
                    ))
            fig20.update_layout(
                title="Giá so với giá trị nội tại", height=_CHART_H, margin=_CHART_M,
                legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False,
            )
            fig20.update_yaxes(title_text="Giá (VND)")
            st.plotly_chart(fig20, width="stretch")

        # Chart 21 — Commodity Input/Output Prices (sector-based) or Analyst Rec fallback
        with r7c3:
            _comm_def = _SECTOR_COMMODITIES.get(_co_sect)
            _all_comm = tuple((_comm_def or {}).get("input", []) + (_comm_def or {}).get("output", []))

            if _comm_def and _all_comm:
                # Fetch normalized commodity prices
                comm_df = load_commodity_prices(_all_comm, period="1y")
                fig21 = go.Figure()
                if not comm_df.empty:
                    _in_labels  = {lbl for _, lbl, _ in _comm_def.get("input", [])}
                    _out_labels = {lbl for _, lbl, _ in _comm_def.get("output", [])}
                    _color_map  = {lbl: clr for _, lbl, clr in _all_comm}
                    for col in comm_df.columns:
                        is_input = col in _in_labels
                        fig21.add_trace(go.Scatter(
                            x=comm_df.index.strftime("%Y-%m-%d"),
                            y=comm_df[col],
                            name=col,
                            mode="lines",
                            line=dict(
                                color=_color_map.get(col, "#9ca3af"),
                                width=2,
                                dash="dash" if is_input else "solid",
                            ),
                            hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
                        ))
                fig21.update_layout(
                    title=_comm_def["title"], height=_CHART_H, margin=_CHART_M,
                    legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False,
                    yaxis_title="Chỉ số (gốc=100)",
                    xaxis=dict(type="category", tickangle=-45, nticks=8),
                )
            else:
                # Fallback: Valuation Estimates bar chart (all methods vs market price)
                _val_methods = [
                    ("dcf",       f"DCF / FCFF"),
                    ("fcfe",      "Cash Flow to Equity"),
                    ("graham",    "Graham Number"),
                    ("pe",        f"P/E Implied (×{MARKET_PE})"),
                    ("pb",        "P/B Implied (×1.5)"),
                    ("ev_ebitda", "EV/EBITDA (×8)"),
                    ("epv",       "Earnings Power Value"),
                    ("ps",        "P/Sales (×1.2)"),
                    ("ri",        "Residual Income"),
                    ("pocf",      "Price/OCF (×10)"),
                ]
                _v21 = valuations or {}
                _vm_names = [label for _, label in _val_methods if _v21.get(_) and _v21[_] > 0]
                _vm_vals  = [_v21[k] for k, _ in _val_methods if _v21.get(k) and _v21[k] > 0]
                _vm_colors = [
                    "#22c55e" if v > current_price else "#ef4444"
                    for v in _vm_vals
                ]
                _vm_pct = [
                    f"{(v - current_price) / current_price * 100:+.1f}%"
                    for v in _vm_vals
                ]

                fig21 = go.Figure()
                fig21.add_trace(go.Bar(
                    x=_vm_names, y=_vm_vals,
                    marker_color=_vm_colors,
                    text=_vm_pct,
                    textposition="outside",
                    textfont=dict(size=11),
                    hovertemplate="%{x}: %{y:,.0f} VND<extra></extra>",
                    name="Valuation",
                ))
                # Market price & Avg Est: shape spans full width, invisible scatter for hover
                if current_price:
                    fig21.add_shape(type="line", xref="paper", x0=0, x1=1,
                        yref="y", y0=current_price, y1=current_price,
                        line=dict(color="#f59e0b", width=2, dash="dash"))
                    fig21.add_trace(go.Scatter(
                        x=_vm_names, y=[current_price] * len(_vm_names),
                        mode="lines",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                        name=f"Thị giá {current_price:,.0f}",
                        hovertemplate=f"Thị giá: {current_price:,.0f} VND<extra></extra>",
                    ))
                if _vm_vals:
                    _avg21 = sum(_vm_vals) / len(_vm_vals)
                    _avg_pct = f"{(_avg21 - current_price) / current_price * 100:+.1f}%" if current_price else ""
                    fig21.add_shape(type="line", xref="paper", x0=0, x1=1,
                        yref="y", y0=_avg21, y1=_avg21,
                        line=dict(color="#f87171", width=2, dash="dot"))
                    fig21.add_trace(go.Scatter(
                        x=_vm_names, y=[_avg21] * len(_vm_names),
                        mode="lines",
                        line=dict(color="#f87171", width=2, dash="dot"),
                        name=f"TB ước tính {_avg21:,.0f}",
                        hovertemplate=f"TB ước tính: {_avg21:,.0f} VND ({_avg_pct})<extra></extra>",
                    ))
                fig21.update_layout(
                    title="Các phương pháp định giá so với thị giá",
                    height=_CHART_H, margin=_CHART_M,
                    showlegend=False, dragmode=False,
                    hovermode="x unified",
                    xaxis=dict(tickangle=-30),
                )
                fig21.update_yaxes(title_text="VND")
            st.plotly_chart(fig21, width="stretch")

        st.divider()

    # ── Key ratios ─────────────────────────────────────────────
    if ttm:
        st.subheader("Chỉ số tài chính chủ chốt (TTM)")

        gm  = gross_margin(ttm.get("gross_profit"), ttm.get("revenue"))
        nm  = net_margin(ttm.get("net_income"),     ttm.get("revenue"))
        om  = operating_margin(ttm.get("ebit"),     ttm.get("revenue"))
        roe_val = roe(ttm.get("net_income"),         ttm.get("equity"))
        roa_val = roa(ttm.get("net_income"),         ttm.get("total_assets"))
        cr  = current_ratio(ttm.get("current_assets"), ttm.get("current_liabilities"))
        de  = debt_to_equity(ttm.get("debt"),        ttm.get("equity"))
        pq  = profit_quality(ttm.get("operating_cf"), ttm.get("net_income"))
        fcfm = fcf_margin(ttm.get("fcf"),             ttm.get("revenue"))
        qr  = quick_ratio(ttm.get("current_assets"), ttm.get("inventory"), ttm.get("current_liabilities"))
        cashr = absolute_liquidity(ttm.get("cash"), ttm.get("current_liabilities"))
        da  = debt_to_assets(ttm.get("debt"),        ttm.get("total_assets"))
        ocf_cl = ocf_to_current_liabilities(ttm.get("operating_cf"), ttm.get("current_liabilities"))

        def _x(value):
            return f"{value:.2f}x" if value is not None else "—"

        def _color(val, good, ok, higher_better=True):
            if val is None:
                return "#94a3b8"
            above_good = val >= good if higher_better else val <= good
            above_ok   = val >= ok   if higher_better else val <= ok
            return "#16a34a" if above_good else ("#d97706" if above_ok else "#dc2626")

        def _scorecard(title: str, metrics: list) -> str:
            def _esc(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            def _tip_html(t: dict) -> str:
                return (
                    '<div class="ttm-tooltip">'
                    f'<div style="font-size:10px;color:#64748b;margin-bottom:4px;font-style:italic;">{_esc(t["f"])}</div>'
                    f'<div style="font-size:12px;color:#e2e8f0;margin-bottom:10px;line-height:1.45;">{_esc(t["d"])}</div>'
                    '<div style="display:flex;flex-direction:column;gap:4px;font-size:11px;color:#cbd5e1;">'
                    f'<span><span style="color:#16a34a;margin-right:6px;font-size:9px;">&#9679;</span>Tốt: {_esc(t["g"])}</span>'
                    f'<span><span style="color:#d97706;margin-right:6px;font-size:9px;">&#9679;</span>Cảnh báo: {_esc(t["w"])}</span>'
                    f'<span><span style="color:#dc2626;margin-right:6px;font-size:9px;">&#9679;</span>Nguy hiểm: {_esc(t["b"])}</span>'
                    '</div>'
                    '<div class="ttm-tt-arrow"></div>'
                    '</div>'
                )

            cells = ""
            for i, metric in enumerate(metrics):
                label, value, color = metric[0], metric[1], metric[2]
                tip_dict = metric[3] if len(metric) > 3 else {}
                sep = "border-right:1px solid rgba(148,163,184,0.2);" if i < len(metrics) - 1 else ""
                tooltip = _tip_html(tip_dict) if tip_dict else ""
                cells += (
                    f'<div class="ttm-cell" style="flex:1;text-align:center;padding:16px 10px;{sep}position:relative;">'
                    f'{tooltip}'
                    f'<div style="font-size:11px;color:#94a3b8;margin-bottom:6px;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>'
                    f'<div style="font-size:22px;font-weight:700;color:{color};letter-spacing:-0.5px;">{value}</div>'
                    f'</div>'
                )
            return (
                f'<div style="border:1px solid rgba(148,163,184,0.2);border-radius:10px;'
                f'margin-bottom:10px;">'
                f'<div style="background:rgba(148,163,184,0.08);padding:6px 14px;'
                f'font-size:10px;font-weight:700;letter-spacing:1.4px;color:#94a3b8;'
                f'border-radius:9px 9px 0 0;">'
                f'{title}</div>'
                f'<div style="display:flex;overflow:visible;">{cells}</div>'
                f'</div>'
            )

        _TOOLTIP_CSS = (
            "<style>"
            ".ttm-cell{position:relative;cursor:default;}"
            ".ttm-tooltip{"
            "display:none;"
            "position:absolute;"
            "bottom:calc(100% + 10px);"
            "left:50%;"
            "transform:translateX(-50%);"
            "background:#1e293b;"
            "border:1px solid rgba(148,163,184,0.15);"
            "border-radius:8px;"
            "padding:12px 14px;"
            "width:230px;"
            "z-index:9999;"
            "box-shadow:0 6px 24px rgba(0,0,0,0.5);"
            "pointer-events:none;"
            "text-align:left;"
            "}"
            ".ttm-tt-arrow{"
            "position:absolute;"
            "top:100%;"
            "left:50%;"
            "transform:translateX(-50%);"
            "border:7px solid transparent;"
            "border-top-color:#1e293b;"
            "}"
            ".ttm-cell:hover .ttm-tooltip{display:block;}"
            "</style>"
        )

        scorecard_html = _TOOLTIP_CSS + (
            _scorecard("SINH LỜI", [
                ("Biên LN gộp",    fmt_pct(gm),      _color(gm,      25, 15), {
                    "f": "Lợi nhuận gộp / Doanh thu",
                    "d": "Đo hiệu quả sản xuất cốt lõi trước chi phí vận hành",
                    "g": "≥ 25%", "w": "15 – 25%", "b": "< 15%",
                }),
                ("Biên hoạt động", fmt_pct(om),      _color(om,      15,  5), {
                    "f": "EBIT / Doanh thu",
                    "d": "Lợi nhuận sau chi phí bán hàng & quản lý, trước lãi vay và thuế",
                    "g": "≥ 15%", "w": "5 – 15%", "b": "< 5%",
                }),
                ("Biên LN ròng",   fmt_pct(nm),      _color(nm,      10,  5), {
                    "f": "Lợi nhuận sau thuế / Doanh thu",
                    "d": "Tỷ suất sinh lời thực tế cuối cùng giữ lại cho cổ đông",
                    "g": "≥ 10%", "w": "5 – 10%", "b": "< 5%",
                }),
                ("ROE",            fmt_pct(roe_val),  _color(roe_val, 15, 10), {
                    "f": "Lợi nhuận ròng / Vốn chủ sở hữu",
                    "d": "Đo mức sinh lời trên đồng vốn cổ đông bỏ ra",
                    "g": "≥ 15%", "w": "10 – 15%", "b": "< 10%",
                }),
                ("ROA",            fmt_pct(roa_val),  _color(roa_val,  8,  5), {
                    "f": "Lợi nhuận ròng / Tổng tài sản",
                    "d": "Đo hiệu quả sử dụng toàn bộ tài sản",
                    "g": "≥ 8%", "w": "5 – 8%", "b": "< 5%",
                }),
            ]) +
            _scorecard("THANH KHOẢN", [
                ("Current ratio",      _x(cr),     _color(cr,     2,   1), {
                    "f": "Tài sản ngắn hạn / Nợ ngắn hạn",
                    "d": "Khả năng trả nợ ngắn hạn bằng tài sản lưu động",
                    "g": "≥ 2x", "w": "1 – 2x", "b": "< 1x",
                }),
                ("Quick ratio",        _x(qr),     _color(qr,     1, 0.5), {
                    "f": "(Tài sản ngắn hạn − Hàng tồn kho) / Nợ ngắn hạn",
                    "d": "Loại trừ hàng tồn kho để đo thanh khoản thực tế hơn",
                    "g": "≥ 1x", "w": "0.5 – 1x", "b": "< 0.5x",
                }),
                ("Cash ratio",         _x(cashr),  _color(cashr, 0.5, 0.2), {
                    "f": "Tiền & tương đương tiền / Nợ ngắn hạn",
                    "d": "Khả năng thanh toán ngay lập tức bằng tiền mặt",
                    "g": "≥ 0.5x", "w": "0.2 – 0.5x", "b": "< 0.2x",
                }),
                ("OCF / Nợ ngắn hạn", _x(ocf_cl), _color(ocf_cl, 0.4, 0.2), {
                    "f": "Dòng tiền hoạt động / Nợ ngắn hạn",
                    "d": "Khả năng trả nợ từ tiền kinh doanh tạo ra",
                    "g": "≥ 0.4x", "w": "0.2 – 0.4x", "b": "< 0.2x",
                }),
            ]) +
            _scorecard("ĐÒN BẨY & DÒNG TIỀN", [
                ("Nợ / Vốn chủ (D/E)", _x(de),       _color(de,   1,  2, higher_better=False), {
                    "f": "Tổng nợ vay / Vốn chủ sở hữu",
                    "d": "Mức độ đòn bẩy tài chính",
                    "g": "≤ 1x", "w": "1 – 2x", "b": "> 2x",
                }),
                ("Nợ / Tổng tài sản",  fmt_pct(da),  _color(da,  30, 60, higher_better=False), {
                    "f": "Tổng nợ vay / Tổng tài sản",
                    "d": "Tỷ trọng nợ trong cơ cấu vốn",
                    "g": "≤ 30%", "w": "30 – 60%", "b": "> 60%",
                }),
                ("Biên FCF",           fmt_pct(fcfm), _color(fcfm, 10,  0), {
                    "f": "Dòng tiền tự do (FCF) / Doanh thu",
                    "d": "Khả năng tạo tiền thực sau đầu tư CAPEX",
                    "g": "≥ 10%", "w": "0 – 10%", "b": "< 0%",
                }),
                ("Chất lượng LN",      fmt_pct(pq),   _color(pq,   1, 0.8), {
                    "f": "Dòng tiền hoạt động / Lợi nhuận ròng",
                    "d": "> 1x: lợi nhuận được bảo chứng bằng tiền mặt thực",
                    "g": "≥ 1x", "w": "0.8 – 1x", "b": "< 0.8x",
                }),
            ])
        )
        _LEGEND = (
            '<div style="display:flex;gap:18px;justify-content:flex-end;'
            'padding:2px 4px 8px;font-size:11px;color:#94a3b8;">'
            '<span style="display:flex;align-items:center;gap:5px;">'
            '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            'background:#16a34a;flex-shrink:0;"></span>Tốt</span>'
            '<span style="display:flex;align-items:center;gap:5px;">'
            '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            'background:#d97706;flex-shrink:0;"></span>Cảnh báo</span>'
            '<span style="display:flex;align-items:center;gap:5px;">'
            '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            'background:#dc2626;flex-shrink:0;"></span>Nguy hiểm</span>'
            '</div>'
        )
        st.markdown(scorecard_html + _LEGEND, unsafe_allow_html=True)


    # ── Valuation Football Field ───────────────────────────────
    if valuations and current_price:
        st.divider()
        st.subheader("Valuation Football Field")
        _ff_methods = [
            ("dcf", "DCF/FCFF"), ("fcfe", "FCFE"), ("graham", "Graham"),
            ("pe", f"P/E ×{MARKET_PE}"), ("pb", "P/B"), ("ev_ebitda", "EV/EBITDA"),
            ("epv", "EPV"), ("ps", "P/Sales"), ("ri", "Residual Income"),
            ("pocf", "P/OCF"),
        ]
        _ff = [(lbl, valuations.get(k)) for k, lbl in _ff_methods
               if valuations.get(k) and valuations[k] > 0]
        if _ff:
            _ff_vals = [v for _, v in _ff]
            _ff_avg  = sum(_ff_vals) / len(_ff_vals)
            # sort by value for visual ladder
            _ff.sort(key=lambda x: x[1])
            _labels  = [x[0] for x in _ff]
            _vals    = [x[1] for x in _ff]
            _colors  = ["#22c55e" if v >= current_price else "#ef4444" for v in _vals]
            fig_ff = go.Figure(go.Bar(
                x=_vals, y=_labels, orientation="h",
                marker_color=_colors,
                text=[f"{v:,.0f}" for v in _vals], textposition="outside",
                hovertemplate="%{y}: %{x:,.0f} VND<extra></extra>",
            ))
            # Current price line
            fig_ff.add_vline(x=current_price, line_dash="dash", line_color="#f59e0b",
                             line_width=2, annotation_text=f"Market {current_price:,.0f}",
                             annotation_position="top", annotation_font_color="#f59e0b")
            # Average line
            fig_ff.add_vline(x=_ff_avg, line_dash="dot", line_color="#60a5fa",
                             line_width=2, annotation_text=f"Avg {_ff_avg:,.0f}",
                             annotation_position="bottom", annotation_font_color="#60a5fa")
            fig_ff.update_layout(
                height=max(300, len(_ff) * 38), margin=dict(l=0, r=60, t=20, b=0),
                dragmode=False, xaxis_title="Intrinsic Value (VND)",
                showlegend=False)
            st.plotly_chart(fig_ff, width="stretch")
            _ff_up = (_ff_avg - current_price) / current_price * 100
            _ff_cc = "#22c55e" if _ff_up >= 0 else "#ef4444"
            st.markdown(
                f"<div style='font-size:13px;color:#9ca3af;'>"
                f"{len(_ff)} methods · Average intrinsic value "
                f"<b style='color:{_ff_cc}'>{_ff_avg:,.0f} VND ({_ff_up:+.1f}% vs market)</b> · "
                f"Green = above market price (undervalued signal)</div>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# VIEW 2 — VALUATION SCREEN
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# VIEW 2 — STOCK SCREENER
# ═══════════════════════════════════════════════════════════════
elif view == "Stock Screener":
    _ss_tab1, _ss_tab2 = st.tabs(["📋 Lọc cổ phiếu", "⭐ Watchlist"])
    with _ss_tab1:
        st.title("📋 Valuation Screen")

        screen_df = load_valuation_screen_data()

        if screen_df.empty:
            st.warning("No pre-computed valuation data. Run: `python -m collectors.compute_valuations`")
            st.info("If you haven't loaded tickers yet, run `python -m collectors.bulk_load` first.")
        else:

            # ── Sidebar filters ────────────────────────────────────
            st.sidebar.markdown("### Filters")

            all_sectors = sorted(s for s in screen_df["Sector"].unique() if s != "Unknown")
            sel_sectors = st.sidebar.multiselect(
                "Sector", all_sectors,
                default=[],
                placeholder="All sectors",
            )

            min_roe = st.sidebar.slider("Min ROE (%)", -50, 50, -50, step=5)
            max_de  = st.sidebar.slider("Max D/E (x)", 0.0, 30.0, 30.0, step=0.5)
            min_upside_pct = st.sidebar.slider("Min Avg upside (%)", -1000, 200, -1000, step=50)
            min_quality = st.sidebar.slider("Min Quality score", 0, 100, 0, step=5)

            sort_col = st.sidebar.selectbox(
                "Sort by",
                ["Signal (Strong Buy first)", "Signal (Strong Sell first)",
                 "Upside (best first)", "Quality (best first)",
                 "ROE (best first)", "Net Margin (best first)", "Ticker (A-Z)"],
                index=0,
            )
            _WJ2 = "⁠"
            _ALL_SIGNALS_V2 = [_WJ2*1+"Strong Buy", _WJ2*2+"Buy", _WJ2*3+"Watch",
                               _WJ2*4+"Neutral", _WJ2*5+"Reduce", _WJ2*6+"Sell", _WJ2*7+"Strong Sell"]
            f_signals = st.sidebar.multiselect(
                "Filter Signal", _ALL_SIGNALS_V2, default=[],
                placeholder="All signals",
            )

            # ── Apply filters ──────────────────────────────────────
            filtered = screen_df.copy()
            if sel_sectors:
                filtered = filtered[filtered["Sector"].isin(sel_sectors)]
            filtered = filtered[filtered["_roe_raw"] >= min_roe / 100]
            filtered = filtered[filtered["_de_raw"] <= max_de]
            filtered = filtered[filtered["_avg_upside_raw"] >= min_upside_pct / 100]
            filtered = filtered[filtered["_qs_raw"] >= min_quality]

            # ── Sort ───────────────────────────────────────────────
            if sort_col == "Ticker (A-Z)":
                filtered = filtered.sort_values("Ticker")
            elif sort_col == "ROE (best first)":
                filtered = filtered.sort_values("_roe_raw", ascending=False)
            elif sort_col == "Net Margin (best first)":
                filtered = filtered.sort_values("_nm_raw", ascending=False)
            elif sort_col == "Quality (best first)":
                filtered = filtered.sort_values("_qs_raw", ascending=False)
            elif sort_col in ("Signal (Strong Buy first)", "Signal (Strong Sell first)"):
                pass  # applied after signal column is built (needs _sig_rank)
            else:
                filtered = filtered.sort_values("_upside_raw", ascending=False)

            n_total = len(screen_df)
            n_filtered = len(filtered)
            st.caption(
                f"Showing {n_filtered} of {n_total} tickers"
                + (" (filters applied)" if n_filtered < n_total else "")
            )

            # ── Ticker multiselect — just above table ───────────────
            # Use ALL tickers from the full dataset (not filtered) so nothing is hidden
            st.markdown("""
    <style>
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #1e40af !important;
        color: #bfdbfe !important;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {
        color: #93c5fd !important;
    }
    </style>""", unsafe_allow_html=True)
            # ── Build display table ────────────────────────────────
            raw_cols = [c for c in filtered.columns if c.startswith("_")]
            display = filtered.drop(columns=raw_cols).copy()

            # Sector: hide Unknown → "-"
            display["Sector"] = filtered["Sector"].apply(
                lambda s: "-" if s == "Unknown" else s
            ).values

            # 7-level signal — prefixed with number so column-header click sorts correctly
            # "1-Strong Buy" < "2-Buy" < ... alphabetically = our intended order
            # _SIG_LABELS defined at module level above
            signals = [_SIG_LABELS[classify_signal(u, q)]
                       for u, q in zip(filtered["_avg_upside_raw"], filtered["_qs_raw"])]
            display["Signal"] = signals
            _signal_rank = {v: 6 - i for i, v in enumerate(_SIG_LABELS.values())}
            display["_sig_rank"] = [_signal_rank.get(s, 3) for s in signals]

            # Quality as integer for color bar
            display["Quality"] = [int(round(q)) for q in filtered["_qs_raw"]]

            # Column order — Avg Estimate first, then DCF
            ordered = ["Signal", "Sector", "Price (VND)",
                       "Avg Estimate", "Avg Upside",
                       "DCF Estimate", "FCFE Estimate", "Upside",
                       "Quality", "Graham Number", "P/E", "P/B",
                       "Net Margin", "ROE", "FCF Margin", "D/E", "Current Ratio"]
            ordered = [c for c in ordered if c in display.columns]

            # Apply signal filter
            if f_signals:
                display = display[display["Signal"].isin(f_signals)]

            # Apply signal sort after display is built (needs _sig_rank)
            if sort_col == "Signal (Strong Buy first)":
                display = display.sort_values("_sig_rank", ascending=False)
            elif sort_col == "Signal (Strong Sell first)":
                display = display.sort_values("_sig_rank", ascending=True)

            # ── CSV export ─────────────────────────────────────────
            csv_bytes = display[["Ticker"] + ordered].to_csv(index=False).encode("utf-8-sig")
            xlsx_bytes = build_excel_export(filtered)
            _dl1, _dl2, _ = st.columns([2, 2, 6])
            _dl1.download_button("Download CSV", data=csv_bytes,
                file_name="valuation_screen.csv", mime="text/csv", use_container_width=True)
            _dl2.download_button("Download Excel", data=xlsx_bytes,
                file_name="valuation_screen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
            if False:
                st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="valuation_screen.csv",
                mime="text/csv",
            )

            # ── Table ──────────────────────────────────────────────
            def _signal_color(val):
                # strip invisible prefix for lookup
                _v = val.lstrip("⁠") if val else val
                return {
                    "Strong Buy":  "background-color:#14532d; color:#86efac; font-weight:700",
                    "Buy":         "background-color:#166534; color:#bbf7d0; font-weight:600",
                    "Watch":       "background-color:#713f12; color:#fde68a; font-weight:600",
                    "Neutral":     "background-color:#1e293b; color:#94a3b8",
                    "Reduce":      "background-color:#7c2d12; color:#fdba74",
                    "Sell":        "background-color:#7f1d1d; color:#fca5a5; font-weight:600",
                    "Strong Sell": "background-color:#450a0a; color:#f87171; font-weight:700",
                }.get(_v, "")

            def _quality_color(val):
                try:
                    q = int(val)
                    if q >= 70:   return "background-color:#166534; color:#86efac"
                    elif q >= 50: return "background-color:#713f12; color:#fde047"
                    elif q >= 30: return "background-color:#7c2d12; color:#fdba74"
                    else:         return "background-color:#7f1d1d; color:#fca5a5"
                except Exception:
                    return ""

            styled = (
                display.set_index("Ticker")[ordered]
                .style
                .map(_signal_color, subset=["Signal"])
                .map(_quality_color, subset=["Quality"])
            )
            st.dataframe(styled, width="stretch")

            # ── Colored legend below table ─────────────────────────
            st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:8px;font-size:12px;padding:8px 0 4px 0;justify-content:flex-end;">
      <span style="color:#6b7280;font-size:11px;align-self:center;">Avg of 10 valuation methods vs market price · Quality Score 0–100 (ROE, margin, FCF, liquidity) · click Signal to sort ·</span>
      <span style="background:#14532d;color:#86efac;padding:2px 8px;border-radius:4px;font-weight:700;">Strong Buy: upside ≥+20% &amp; quality ≥60</span>
      <span style="background:#166534;color:#bbf7d0;padding:2px 8px;border-radius:4px;font-weight:600;">Buy: upside ≥+10% &amp; quality ≥45</span>
      <span style="background:#713f12;color:#fde68a;padding:2px 8px;border-radius:4px;">Watch: upside ≥0%</span>
      <span style="background:#1e293b;color:#94a3b8;padding:2px 8px;border-radius:4px;">Neutral: -10% to 0%</span>
      <span style="background:#7c2d12;color:#fdba74;padding:2px 8px;border-radius:4px;">Reduce: -30% to -10%</span>
      <span style="background:#7f1d1d;color:#fca5a5;padding:2px 8px;border-radius:4px;font-weight:600;">Sell: -50% to -30%</span>
      <span style="background:#450a0a;color:#f87171;padding:2px 8px;border-radius:4px;font-weight:700;">Strong Sell: &lt;-50%</span>
    </div>
    """, unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════
            # ANALYTICS — charts below the table
            # ════════════════════════════════════════════════════════
            st.divider()

            # Build a clean analysis frame from filtered data
            _an = filtered.copy()
            _an["_signal_clean"] = [
                ("Strong Buy" if u >= 0.20 and q >= 60 else
                 "Buy"        if u >= 0.10 and q >= 45 else
                 "Watch"      if u >= 0.00 else
                 "Neutral"    if u >= -0.10 else
                 "Reduce"     if u >= -0.30 else
                 "Sell"       if u >= -0.50 else
                 "Strong Sell")
                for u, q in zip(_an["_avg_upside_raw"], _an["_qs_raw"])
            ]

            # ── Signal count KPI cards ──────────────────────────────
            _sig_order = ["Strong Buy", "Buy", "Watch", "Neutral", "Reduce", "Sell", "Strong Sell"]
            _sig_colors_map = {
                "Strong Buy": "#22c55e", "Buy": "#4ade80", "Watch": "#eab308",
                "Neutral": "#94a3b8", "Reduce": "#fb923c", "Sell": "#ef4444",
                "Strong Sell": "#b91c1c",
            }
            _counts = _an["_signal_clean"].value_counts().to_dict()
            _kpi_cols = st.columns(7)
            for _i, _sig in enumerate(_sig_order):
                _cnt = _counts.get(_sig, 0)
                _clr = _sig_colors_map[_sig]
                _kpi_cols[_i].markdown(
                    f"<div style='text-align:center;padding:6px;border-radius:6px;"
                    f"background:rgba(255,255,255,0.03);border-top:3px solid {_clr};'>"
                    f"<div style='font-size:24px;font-weight:800;color:{_clr};'>{_cnt}</div>"
                    f"<div style='font-size:11px;color:#9ca3af;'>{_sig}</div></div>",
                    unsafe_allow_html=True)

            # ── Top Picks — best Strong Buy / Buy by Quality ────────
            st.write("")
            _picks = (_an[_an["_signal_clean"].isin(["Strong Buy", "Buy"])]
                      .sort_values(["_qs_raw", "_avg_upside_raw"], ascending=False).head(5))
            if not _picks.empty:
                st.markdown("**Top Picks** · highest Quality among Buy / Strong Buy")
                _pk_cols = st.columns(len(_picks))
                for _col, (_, _r) in zip(_pk_cols, _picks.iterrows()):
                    _clr = _sig_colors_map[_r["_signal_clean"]]
                    _col.markdown(
                        f"<div style='text-align:center;padding:10px 6px;border-radius:8px;"
                        f"background:rgba(255,255,255,0.03);border:1px solid {_clr}44;'>"
                        f"<div style='font-size:16px;font-weight:800;color:#f9fafb;'>{_r['Ticker']}</div>"
                        f"<div style='font-size:11px;color:#9ca3af;margin-bottom:4px;'>{_r['Sector']}</div>"
                        f"<div style='font-size:13px;color:{_clr};font-weight:700;'>+{_r['_avg_upside_raw']*100:.0f}%</div>"
                        f"<div style='font-size:11px;color:#9ca3af;'>Upside</div>"
                        f"<div style='font-size:13px;color:#f9fafb;margin-top:4px;'>Q{int(round(_r['_qs_raw']))}</div>"
                        f"<div style='font-size:11px;color:#9ca3af;'>Quality</div>"
                        f"</div>", unsafe_allow_html=True)

            st.write("")
            _c1, _c2 = st.columns(2)

            # ── Chart 1: Signal distribution donut ──────────────────
            with _c1:
                st.subheader("Signal Distribution")
                _dist = [(s, _counts.get(s, 0)) for s in _sig_order if _counts.get(s, 0) > 0]
                if _dist:
                    fig_sig = go.Figure(go.Pie(
                        labels=[d[0] for d in _dist],
                        values=[d[1] for d in _dist],
                        marker=dict(colors=[_sig_colors_map[d[0]] for d in _dist]),
                        hole=0.5, textinfo="label+value",
                        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                    ))
                    fig_sig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                                          showlegend=False, dragmode=False)
                    st.plotly_chart(fig_sig, width="stretch")

            # ── Chart 2: Buy opportunities by sector ────────────────
            with _c2:
                st.subheader("Buy Opportunities by Sector")
                _buys = _an[_an["_signal_clean"].isin(["Strong Buy", "Buy"])].copy()
                if not _buys.empty:
                    _by_sec = (_buys.groupby("Sector")["_signal_clean"]
                               .value_counts().unstack(fill_value=0))
                    for _col in ["Strong Buy", "Buy"]:
                        if _col not in _by_sec.columns:
                            _by_sec[_col] = 0
                    _by_sec["_tot"] = _by_sec["Strong Buy"] + _by_sec["Buy"]
                    _by_sec = _by_sec.sort_values("_tot", ascending=True).tail(12)
                    fig_sec_buy = go.Figure()
                    fig_sec_buy.add_trace(go.Bar(
                        y=_by_sec.index, x=_by_sec["Strong Buy"], orientation="h",
                        name="Strong Buy", marker_color="#22c55e",
                        hovertemplate="%{y}: %{x} Strong Buy<extra></extra>"))
                    fig_sec_buy.add_trace(go.Bar(
                        y=_by_sec.index, x=_by_sec["Buy"], orientation="h",
                        name="Buy", marker_color="#4ade80",
                        hovertemplate="%{y}: %{x} Buy<extra></extra>"))
                    fig_sec_buy.update_layout(
                        height=340, margin=dict(l=0, r=10, t=10, b=0),
                        barmode="stack", dragmode=False,
                        legend=dict(orientation="h", y=-0.12),
                        xaxis_title="# Tickers")
                    st.plotly_chart(fig_sec_buy, width="stretch")
                else:
                    st.info("No Buy/Strong Buy tickers in current filter.")

            # ── Chart 2b: Avg Upside by Sector ───────────────────────
            st.subheader("Avg Upside by Sector")
            _sec_up = (_an.dropna(subset=["_avg_upside_raw"])
                          .groupby("Sector")["_avg_upside_raw"].median()
                          .sort_values())
            if not _sec_up.empty:
                _sec_up_pct = (_sec_up * 100)
                fig_sec_up = go.Figure(go.Bar(
                    y=_sec_up_pct.index, x=_sec_up_pct.values, orientation="h",
                    marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in _sec_up_pct.values],
                    hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
                fig_sec_up.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_sec_up.update_layout(
                    height=max(340, 24 * len(_sec_up_pct)), margin=dict(l=0, r=10, t=10, b=0),
                    dragmode=False, xaxis_title="Median Avg Upside %")
                st.plotly_chart(fig_sec_up, width="stretch")
                st.caption("Median of 'Avg Estimate' upside across tickers in each sector — "
                           "negative = sector trading above estimated fair value.")

            # ── Chart 3: Quality vs Upside scatter ──────────────────
            st.subheader("Quality vs Avg Upside (all filtered tickers)")
            _sc = _an.dropna(subset=["_qs_raw", "_avg_upside_raw"]).copy()
            _sc["_upside_pct"] = _sc["_avg_upside_raw"] * 100
            _CLIP_LO, _CLIP_HI = -100, 150
            _n_clipped = int(((_sc["_upside_pct"] < _CLIP_LO) | (_sc["_upside_pct"] > _CLIP_HI)).sum())
            _sc["_upside_clip"] = _sc["_upside_pct"].clip(_CLIP_LO, _CLIP_HI)
            # Jitter points sitting exactly on the clip edge so they don't form a solid wall
            _is_clipped = (_sc["_upside_pct"] < _CLIP_LO) | (_sc["_upside_pct"] > _CLIP_HI)
            if _is_clipped.any():
                _rng = np.random.default_rng(42)
                _sc.loc[_is_clipped, "_upside_clip"] += _rng.uniform(-4, 4, size=int(_is_clipped.sum()))
            fig_qs = go.Figure(go.Scatter(
                x=_sc["_upside_clip"], y=_sc["_qs_raw"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=[_sig_colors_map[s] for s in _sc["_signal_clean"]],
                    line=dict(width=0.5, color="#1f2937"), opacity=0.75),
                text=_sc["Ticker"],
                customdata=_sc[["Sector", "_upside_pct"]].values,
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>"
                              "Upside %{customdata[1]:.1f}%<br>Quality %{y:.0f}<extra></extra>",
            ))
            # Quadrant reference lines
            fig_qs.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
            fig_qs.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.4)
            fig_qs.add_annotation(x=75, y=85, text="★ Cheap & Quality", showarrow=False,
                                  font=dict(color="#22c55e", size=12))
            fig_qs.add_annotation(x=-50, y=15, text="Avoid", showarrow=False,
                                  font=dict(color="#ef4444", size=12))
            fig_qs.update_layout(
                height=420, margin=dict(l=0, r=0, t=10, b=0), dragmode=False,
                xaxis_title=f"Avg Upside % (clipped to {_CLIP_LO}…{_CLIP_HI}%)", yaxis_title="Quality Score",
                hovermode="closest")
            st.plotly_chart(fig_qs, width="stretch")
            _clip_note = (f" · {_n_clipped} tickers off-chart (upside outside "
                           f"{_CLIP_LO}%…{_CLIP_HI}%, jittered at the edge)" if _n_clipped else "")
            st.caption("Top-right quadrant = undervalued + high quality (best opportunities). "
                       f"Bubble color = signal.{_clip_note}")

            # ── Chart 4: Upside distribution histogram ──────────────
            st.subheader("Avg Upside Distribution")
            _n_offrange = int(((_sc["_upside_pct"] < _CLIP_LO) | (_sc["_upside_pct"] > _CLIP_HI)).sum())
            _bin_size = 10
            _bin_edges = np.arange(_CLIP_LO, _CLIP_HI + _bin_size, _bin_size)
            _counts, _ = np.histogram(_sc["_upside_pct"], bins=_bin_edges)
            _bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
            # Red = expensive (negative upside) -> Green = cheap (positive upside)
            _bin_colors = ["#ef4444" if c < 0 else "#22c55e" for c in _bin_centers]
            fig_hist = go.Figure(go.Bar(
                x=_bin_centers, y=_counts, marker_color=_bin_colors, width=_bin_size * 0.9,
                customdata=np.stack([_bin_edges[:-1], _bin_edges[1:]], axis=-1),
                hovertemplate="%{customdata[0]:.0f}% to %{customdata[1]:.0f}%: %{y} tickers<extra></extra>",
            ))
            fig_hist.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.6)
            fig_hist.update_layout(
                height=260, margin=dict(l=0, r=0, t=10, b=0), dragmode=False,
                xaxis=dict(title="Avg Upside %", range=[_CLIP_LO, _CLIP_HI]),
                yaxis_title="# Tickers", bargap=0.05)
            st.plotly_chart(fig_hist, width="stretch")
            _offrange_note = (f" - {_n_offrange} tickers with upside outside "
                              f"{_CLIP_LO}%...{_CLIP_HI}% not shown" if _n_offrange else "")
            st.caption(
                "Mỗi cột = số mã có mức upside trong khoảng đó. "
                "🟢 Xanh (bên phải vạch 0) = đang bị định giá thấp, có thể tăng giá. "
                "🔴 Đỏ (bên trái vạch 0) = đang đắt hơn giá trị ước tính, có thể giảm giá."
                f"{_offrange_note}")


    with _ss_tab2:
        st.title("Screening & Watchlist")

        screen_df = load_valuation_screen_data()
        pinned    = get_pinned_tickers()

        # ── Data Health expander in sidebar ────────────────────────
        with st.sidebar.expander("Data Health Check"):
            if st.button("Check completeness", key="btn_health"):
                _comp = check_data_completeness()
                _missing = _comp[_comp["Status"] == "Needs Update"]
                st.caption(f"{len(_missing)} / {len(_comp)} tickers need update")
                if not _missing.empty:
                    st.dataframe(_missing.head(20), width="stretch")
                else:
                    st.success("All tickers have full data!")


        if screen_df.empty:
            st.warning("No pre-computed valuation data. Run: `python -m collectors.compute_valuations`")
            st.stop()

        # ── Sidebar filters ─────────────────────────────────────────
        st.sidebar.subheader("Filters")

        # Preset buttons
        preset_col1, preset_col2, preset_col3 = st.sidebar.columns(3)
        _apply_bank   = preset_col1.button("Bank",    use_container_width=True)
        _apply_value  = preset_col2.button("Value",   use_container_width=True)
        _apply_growth = preset_col3.button("Growth",  use_container_width=True)

        # Preset defaults
        if _apply_bank:
            st.session_state.update({
                "f_sectors": ["Ngân hàng"], "f_min_upside": -50,
                "f_max_pe": 20, "f_max_pb": 1.5, "f_min_roe": 12, "f_min_nm": 10,
            })
        elif _apply_value:
            st.session_state.update({
                "f_sectors": [], "f_min_upside": 20,
                "f_max_pe": 15, "f_max_pb": 2.0, "f_min_roe": 10, "f_min_nm": 5,
            })
        elif _apply_growth:
            st.session_state.update({
                "f_sectors": [], "f_min_upside": 10,
                "f_max_pe": 40, "f_max_pb": 5.0, "f_min_roe": 15, "f_min_nm": 8,
            })

        all_sectors = sorted(screen_df["Sector"].dropna().unique().tolist()) if "Sector" in screen_df.columns else []
        _saved_sectors = [s for s in st.session_state.get("f_sectors", []) if s in all_sectors]
        f_sectors  = st.sidebar.multiselect("Sectors", all_sectors, default=_saved_sectors)
        f_min_upside = st.sidebar.slider("Min Avg Upside (%)", -1000, 200,
                         st.session_state.get("f_min_upside", -1000), step=50)
        f_max_pe  = st.sidebar.slider("Max P/E (×)",   0, 100,
                         st.session_state.get("f_max_pe", 100))
        f_max_pb  = st.sidebar.slider("Max P/B (×)",   0.0, 10.0,
                         float(st.session_state.get("f_max_pb", 10.0)), step=0.1)
        f_min_roe = st.sidebar.slider("Min ROE (%)",  -50, 50,
                         st.session_state.get("f_min_roe", -50))
        f_min_nm  = st.sidebar.slider("Min Net Margin (%)", -50, 50,
                         st.session_state.get("f_min_nm", -50))
        f_min_qs  = st.sidebar.slider("Min Quality Score", 0, 100,
                         st.session_state.get("f_min_qs", 0), step=5)
        f_pinned_only = st.sidebar.checkbox("Saved watchlist only", value=False)

        # Reset stale session state defaults to new permissive values
        for _k, _v in [("f_min_upside", -1000), ("f_min_roe", -50),
                       ("f_min_nm", -50), ("f_max_pe", 100), ("f_max_pb", 10.0)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        # Save current filter state
        st.session_state.update({
            "f_sectors": f_sectors, "f_min_upside": f_min_upside,
            "f_max_pe": f_max_pe, "f_max_pb": f_max_pb,
            "f_min_roe": f_min_roe, "f_min_nm": f_min_nm, "f_min_qs": f_min_qs,
        })

        # ── Apply filters ───────────────────────────────────────────
        res = screen_df.copy()
        if f_sectors:
            res = res[res["Sector"].isin(f_sectors)]
        res = res[res["_avg_upside_raw"] >= f_min_upside / 100]
        res = res[res["_roe_raw"]       >= f_min_roe / 100]
        res = res[res["_nm_raw"]        >= f_min_nm  / 100]
        if f_max_pe < 100:
            res = res[(res["_upside_raw"] > -999)]  # ensure column exists
            # parse P/E from display col
            def _pe_raw(s):
                try: return float(str(s).replace("x","").replace("—","999"))
                except: return 999
            res = res[res["P/E"].apply(_pe_raw) <= f_max_pe]
        if f_max_pb < 10:
            def _pb_raw(s):
                try: return float(str(s).replace("x","").replace("—","999"))
                except: return 999
            res = res[res["P/B"].apply(_pb_raw) <= f_max_pb]
        if f_min_qs > 0:
            res = res[res["_qs_raw"] >= f_min_qs]
        if f_pinned_only:
            res = res[res["Ticker"].isin(pinned)]

        res = res.sort_values("_upside_raw", ascending=False)

        _all_tickers = sorted(screen_df["Ticker"].tolist())

        # ── Results ─────────────────────────────────────────────────
        tab_screen, tab_saved = st.tabs([
            f"Screen Results ({len(res)})",
            f"Saved Watchlist ({len(pinned)})",
        ])

        with tab_screen:
            # Search filter only (no buttons — click row to save/remove)
            _sel_tickers = st.multiselect(
                "Ticker", _all_tickers, default=[],
                placeholder="Search to filter tickers...",
                key="screen_ticker_ms", label_visibility="collapsed",
            )

            # Apply ticker filter to results
            _res_display = res[res["Ticker"].isin(_sel_tickers)] if _sel_tickers else res
            if _res_display.empty:
                st.info("No tickers match. Adjust filters in the sidebar.")
            else:
                st.caption(f"{len(_res_display)} tickers shown")

                res_disp = _res_display.copy()
                # Star in same column as Ticker — ★ lit (saved) or ✩ dim (not saved)
                res_disp["Ticker"] = res_disp["Ticker"].apply(
                    lambda t: f"★ {t}" if t in pinned else f"✩ {t}"
                )

                # Signal column with colors
                _sc_sigs = [_SIG_LABELS[classify_signal(u, q)]
                            for u, q in zip(res_disp["_avg_upside_raw"], res_disp["_qs_raw"])]
                res_disp["Signal"] = _sc_sigs
                res_disp["Quality"] = [int(round(q)) for q in res_disp["_qs_raw"]]

                display_cols = ["Signal","Sector","Price (VND)",  # "☆" added below
                                "Avg Estimate","Avg Upside",
                                "DCF Estimate","Upside","Quality",
                                "P/E","P/B","ROE","Net Margin"]
                display_cols = [c for c in display_cols if c in res_disp.columns]

                def _sc_sig_col(val):
                    _v = val.lstrip("⁠") if val else val
                    return {"Strong Buy":"background-color:#14532d;color:#86efac;font-weight:700",
                            "Buy":"background-color:#166534;color:#bbf7d0;font-weight:600",
                            "Watch":"background-color:#713f12;color:#fde68a",
                            "Neutral":"background-color:#1e293b;color:#94a3b8",
                            "Reduce":"background-color:#7c2d12;color:#fdba74",
                            "Sell":"background-color:#7f1d1d;color:#fca5a5;font-weight:600",
                            "Strong Sell":"background-color:#450a0a;color:#f87171;font-weight:700",
                            }.get(_v, "")

                def _sc_q_col(val):
                    try:
                        q = int(val)
                        if q >= 70: return "background-color:#166534;color:#86efac"
                        elif q >= 50: return "background-color:#713f12;color:#fde047"
                        elif q >= 30: return "background-color:#7c2d12;color:#fdba74"
                        else: return "background-color:#7f1d1d;color:#fca5a5"
                    except: return ""

                _final_cols = ["Ticker"] + display_cols
                _final_cols = [c for c in _final_cols if c in res_disp.columns]
                _plain_df = res_disp[_final_cols].reset_index(drop=True)
                _styled_sc = (
                    _plain_df.style
                    .map(_sc_sig_col, subset=["Signal"])
                    .map(_sc_q_col,   subset=["Quality"] if "Quality" in display_cols else [])
                )
                _selection = st.dataframe(
                    _styled_sc, width="stretch", hide_index=True,
                    on_select="rerun", selection_mode="single-row",
                )

                # Handle row click → save or remove
                _sel_rows = _selection.selection.get("rows", []) if _selection else []
                if _sel_rows:
                    _row_idx = _sel_rows[0]
                    # Strip star prefix to get raw ticker
                    _clicked_raw = _plain_df.iloc[_row_idx]["Ticker"]
                    _clicked_ticker = _clicked_raw.lstrip("★✩ ")
                    _is_pinned = _clicked_ticker in pinned

                    _act_col, _info_col = st.columns([3, 7])
                    _info_col.markdown(
                        f"**{_clicked_ticker}** — "
                        f"{'already in watchlist ★' if _is_pinned else 'not in watchlist ✩'}"
                    )
                    if _is_pinned:
                        if _act_col.button(f"✕ Remove {_clicked_ticker} from watchlist",
                                           key="wl_rm_click", use_container_width=True):
                            if st.session_state.get("wl_rm_confirm") == _clicked_ticker:
                                unpin_ticker(_clicked_ticker)
                                st.session_state.pop("wl_rm_confirm", None)
                                st.cache_data.clear(); st.rerun()
                            else:
                                st.session_state["wl_rm_confirm"] = _clicked_ticker
                                st.warning(f"Click Remove again to confirm removing **{_clicked_ticker}**")
                    else:
                        if _act_col.button(f"★ Save {_clicked_ticker} to watchlist",
                                           key="wl_save_click", use_container_width=True):
                            pin_ticker(_clicked_ticker)
                            st.success(f"★ {_clicked_ticker} saved!")
                            st.cache_data.clear(); st.rerun()

        with tab_saved:
            if not pinned:
                st.info("No tickers saved yet. Use the Screen tab to find and save tickers.")
            else:
                saved_df = screen_df[screen_df["Ticker"].isin(pinned)].copy()

                # Compute signals for saved tickers
                _saved_sigs = [_SIG_LABELS[classify_signal(u, q)]
                               for u, q in zip(saved_df["_avg_upside_raw"], saved_df["_qs_raw"])]
                saved_df["Signal"] = _saved_sigs

                # Search + remove row
                sw_col, rem_col = st.columns([4, 2])
                _saved_search = sw_col.multiselect(
                    "Filter", sorted(saved_df["Ticker"].tolist()),
                    default=[], placeholder="Search saved tickers...",
                    key="wl_saved_search", label_visibility="collapsed",
                )
                if _saved_search:
                    saved_df = saved_df[saved_df["Ticker"].isin(_saved_search)]

                saved_df = saved_df.sort_values("_sig_rank" if "_sig_rank" in saved_df.columns
                                                 else "_avg_upside_raw", ascending=False)

                display_cols2 = ["Signal","Sector","Price (VND)","Avg Estimate","Avg Upside",
                                 "DCF Estimate","Upside","Quality","P/E","P/B","ROE","Net Margin"]
                display_cols2 = [c for c in display_cols2 if c in saved_df.columns]

                # Color Signal + Quality
                def _saved_sig_color(val):
                    _v = val.lstrip("⁠") if val else val
                    return {"Strong Buy":"background-color:#14532d;color:#86efac;font-weight:700",
                            "Buy":"background-color:#166534;color:#bbf7d0;font-weight:600",
                            "Watch":"background-color:#713f12;color:#fde68a",
                            "Neutral":"background-color:#1e293b;color:#94a3b8",
                            "Reduce":"background-color:#7c2d12;color:#fdba74",
                            "Sell":"background-color:#7f1d1d;color:#fca5a5;font-weight:600",
                            "Strong Sell":"background-color:#450a0a;color:#f87171;font-weight:700",
                            }.get(_v, "")

                def _saved_q_color(val):
                    try:
                        q = int(val)
                        if q >= 70: return "background-color:#166534;color:#86efac"
                        elif q >= 50: return "background-color:#713f12;color:#fde047"
                        elif q >= 30: return "background-color:#7c2d12;color:#fdba74"
                        else: return "background-color:#7f1d1d;color:#fca5a5"
                    except: return ""

                saved_disp = saved_df[["Ticker"] + display_cols2].copy()
                saved_disp["Quality"] = [int(round(q)) for q in saved_df["_qs_raw"]]
                styled2 = (
                    saved_disp.set_index("Ticker")[display_cols2]
                    .style
                    .map(_saved_sig_color, subset=["Signal"])
                    .map(_saved_q_color,   subset=["Quality"] if "Quality" in display_cols2 else [])
                )
                st.dataframe(styled2, width="stretch")
                st.caption(f"{len(pinned)} saved · showing {len(saved_df)}")

                # Remove button
                _rem2 = rem_col.text_input("Remove ticker", placeholder="e.g. VNM",
                                           key="wl_rem2_input", label_visibility="collapsed").upper().strip()
                if rem_col.button("✕ Remove from watchlist", key="wl_rem2_btn", use_container_width=True):
                    if _rem2:
                        unpin_ticker(_rem2)
                        st.cache_data.clear()
                        st.rerun()




# ═══════════════════════════════════════════════════════════════
# VIEW 3 — SECTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif view == "Sector Analysis":
    st.title("Sector Analysis")

    sector_df  = load_sector_data()
    ticker_df  = load_sector_ticker_data()

    if sector_df.empty:
        st.warning("No valuation data. Run: `python -m collectors.compute_valuations`")
    else:
        n_sectors = len(sector_df)
        st.caption(f"{n_sectors} sectors · Metrics are medians across tickers · "
                   f"{len(ticker_df)} tickers with valuation data")

        # ── Global metric selector — used by heatmap AND top-5 ────
        _METRIC_OPTIONS = {
            "Avg Estimates vs Market":  ("avg_upside",  -80, 150,  "Avg Est Upside %",   "upside", False),
            "DCF vs Market":            ("upside_pct",  -80, 150,  "DCF Upside %",        "upside", False),
            "P/E (lower = cheaper)":    ("pe",           0,   50,  "P/E",                 "pe",     True),
            "P/B (lower = cheaper)":    ("pb",           0,    5,  "P/B",                 "pb",     True),
            "ROE (higher = better)":    ("roe",        -20,   40,  "ROE %",               "roe",    False),
        }
        _metric_sel = st.radio(
            "Color / Sort by:", list(_METRIC_OPTIONS.keys()),
            horizontal=True, index=0, key="heatmap_metric",
            label_visibility="collapsed",
        )
        _metric_col, _cmin_m, _cmax_m, _metric_label, _sort_col, _sort_asc = _METRIC_OPTIONS[_metric_sel]
        _invert = _metric_sel.startswith("P/")

        # ── 1. SECTOR HEATMAP ─────────────────────────────────────
        st.subheader("Sector Heatmap")
        # Hide iframe white flash
        st.markdown("""<style>
iframe[title="heatmap_click.heatmap_click"] {
    background: #0e1117 !important;
    border: none !important;
}
</style>""", unsafe_allow_html=True)

        if not ticker_df.empty and _metric_col in ticker_df.columns:
            hm_df = ticker_df.dropna(subset=[_metric_col, "sector"]).copy()
            hm_df["_val"] = hm_df[_metric_col].clip(_cmin_m, _cmax_m)

            # Normalize each metric to the -80..+150 range used by _upside_to_hex
            if _metric_sel.endswith("Market"):      # upside % already in range
                hm_df["color_norm"] = hm_df["_val"]
            elif _metric_sel.startswith("P/E"):     # 0→+150 (green), 50→-80 (red)
                hm_df["color_norm"] = 150 - (hm_df["_val"] / 50) * 230
            elif _metric_sel.startswith("P/B"):     # 0→+150 (green), 5→-80 (red)
                hm_df["color_norm"] = 150 - (hm_df["_val"] / 5) * 230
            elif _metric_sel.startswith("ROE"):     # -20→-80 (red), 40→+150 (green)
                hm_df["color_norm"] = (hm_df["_val"] + 20) / 60 * 230 - 80
            else:
                hm_df["color_norm"] = hm_df["_val"]
            hm_df["upside_clipped"] = hm_df["color_norm"]

            # Size: log-compress + add floor so small companies get minimum visible space
            hm_df["mcap_log"] = np.log1p(hm_df["mcap"])
            # floor = 40% of the sector median → no cell is too tiny
            _sec_med = hm_df.groupby("sector")["mcap_log"].transform("median")
            hm_df["mcap_sized"] = hm_df["mcap_log"] + _sec_med * 0.4
            _sec_total_sized = hm_df.groupby("sector")["mcap_sized"].transform("sum")
            hm_df["mcap_norm"] = hm_df["mcap_sized"] / _sec_total_sized

            # Pre-compute hex colors manually so sector headers can be fixed dark gray
            def _upside_to_hex(val):
                # Red (#dc2626) → Amber (#d97706) → Green (#16a34a)
                # No white — neutral is amber/orange
                t = np.clip((val + 80) / 230, 0, 1)  # -80..+150 → 0..1
                if t <= 0.5:
                    # dark red → amber
                    r = int(127 + (217 - 127) * t * 2)
                    g = int(29  + (119 - 29)  * t * 2)
                    b = int(29  + (6   - 29)  * t * 2)
                else:
                    # amber → dark green
                    r = int(217 + (20  - 217) * (t - 0.5) * 2)
                    g = int(119 + (83  - 119) * (t - 0.5) * 2)
                    b = int(6   + (45  - 6)   * (t - 0.5) * 2)
                return f"#{r:02x}{g:02x}{b:02x}"

            _SECTOR_HDR = "#1e293b"   # fixed dark slate for sector header bars

            # Build ids / labels / parents / values / hex_colors / text
            ids, labels, parents, values, hex_colors, texts, customs = [], [], [], [], [], [], []

            # Sector nodes — fixed dark color, equal size (1.0 = sum of children)
            for sec in hm_df["sector"].unique():
                ids.append(sec)
                labels.append(sec)
                parents.append("")
                values.append(1.0)
                hex_colors.append(_SECTOR_HDR)
                texts.append(sec)
                sec_up = hm_df.loc[hm_df["sector"] == sec, "upside_pct"].median()
                customs.append([sec, round(sec_up, 1) if pd.notna(sec_up) else 0, 0, 0, 0])

            # Ticker nodes — sized by relative market cap, colored by upside
            for _, row in hm_df.iterrows():
                ids.append(f"{row['sector']}_{row['ticker']}")
                labels.append(row["ticker"])
                parents.append(row["sector"])
                values.append(row["mcap_norm"])
                hex_colors.append(_upside_to_hex(row["upside_clipped"]))
                _mv = row[_metric_col]
                _mv_str = f"{_mv:+.1f}%" if _metric_sel.endswith("Market") else f"{_mv:.1f}"
                texts.append(f"{row['ticker']}<br>{_mv_str}" if pd.notna(_mv) else row["ticker"])
                customs.append([
                    row["ticker"],
                    round(row.get(_metric_col) or 0, 2),  # [1] pre-rounded
                    row["pe"] or 0, row["pb"] or 0, row["roe"] or 0,
                ])

            fig_hm = go.Figure(go.Treemap(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                customdata=customs,
                text=texts,
                texttemplate="%{text}",
                textposition="middle center",
                textfont=dict(size=13, color="white"),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + (_metric_label + ": %{customdata[1]:.2f}%<br>"
                       if _metric_sel.endswith("Market") else
                       _metric_label + ": %{customdata[1]:.2f}<br>")
                    + "P/E: %{customdata[2]:.1f}×<br>"
                      "P/B: %{customdata[3]:.2f}×<br>"
                      "ROE: %{customdata[4]:.1f}%"
                      "<extra></extra>"
                ),
                marker=dict(
                    colors=hex_colors,
                    showscale=False,
                    pad=dict(t=22, l=2, r=2, b=2),
                    line=dict(width=1, color="#0f172a"),
                ),
                branchvalues="total",
                tiling=dict(squarifyratio=1),
            ))
            fig_hm.update_layout(
                height=1050,
                margin=dict(l=0, r=0, t=0, b=0),
                dragmode=False,
            )
            # Heatmap — proper component (JS blocks drill-down, returns ticker in one rerun)
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from heatmap_component import heatmap_click as _heatmap_click
            _hm_clicked = _heatmap_click(fig_hm, height=1050, key="hm_comp_fixed")
            if _hm_clicked and isinstance(_hm_clicked, str) and _hm_clicked.strip():
                _val = _hm_clicked.strip()
                if _val.startswith("TICKER:"):
                    st.session_state["hm_popup_ticker"] = _val[7:].upper()
                    st.session_state.pop("hm_popup_sector", None)
                elif _val.startswith("SECTOR:"):
                    st.session_state["hm_popup_sector"] = _val[7:]
                    st.session_state.pop("hm_popup_ticker", None)
                else:
                    # Legacy: plain ticker
                    st.session_state["hm_popup_ticker"] = _val.upper()

            # Ticker popup handled by global popup block above (before radio)

            # Dynamic color legend
            _legend_configs = {
                "Avg Estimates vs Market": [
                    ("#7f1d1d","-80%+"), ("#dc2626","-40%"), ("#b45309","-10%"),
                    ("#d97706","0% (fair)"), ("#4ade80","+40%"), ("#16a34a","+100%"), ("#14532d","+150%+"),
                ],
                "DCF vs Market": [
                    ("#7f1d1d","-80%+"), ("#dc2626","-40%"), ("#b45309","-10%"),
                    ("#d97706","0% (fair)"), ("#4ade80","+40%"), ("#16a34a","+100%"), ("#14532d","+150%+"),
                ],
                "P/E (lower = cheaper)": [
                    ("#14532d","<5×"), ("#16a34a","10×"), ("#4ade80","15×"),
                    ("#d97706","20×"), ("#b45309","30×"), ("#dc2626","40×"), ("#7f1d1d","50×+"),
                ],
                "P/B (lower = cheaper)": [
                    ("#14532d","<0.5×"), ("#16a34a","1×"), ("#4ade80","1.5×"),
                    ("#d97706","2.5×"), ("#b45309","3.5×"), ("#dc2626","4.5×"), ("#7f1d1d","5×+"),
                ],
                "ROE (higher = better)": [
                    ("#7f1d1d","<-20%"), ("#dc2626","-10%"), ("#b45309","0%"),
                    ("#d97706","5%"), ("#4ade80","15%"), ("#16a34a","25%"), ("#14532d","40%+"),
                ],
            }
            _swatches = _legend_configs.get(_metric_sel, _legend_configs["DCF vs Market"])
            _swatch_html = "".join(
                f'<div style="display:flex;align-items:center;gap:4px;">'
                f'<div style="width:18px;height:14px;background:{c};border-radius:2px;"></div>'
                f'<span>{lbl}</span></div>'
                for c, lbl in _swatches
            )
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;'
                f'color:#9ca3af;padding:4px 0 12px 0;">'
                f'<span>{_metric_label}:</span>{_swatch_html}'
                f'<span style="margin-left:8px;">· Cell size = relative market cap within sector</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No ticker-level upside data available.")

        st.divider()

        # ── 2. P/E · P/B · ROE comparison across sectors ──
        st.subheader("Valuation Multiples by Sector")
        mult_col1, mult_col2 = st.columns(2)

        with mult_col1:
            _pe_df = sector_df.dropna(subset=["pe"]).sort_values("pe", ascending=True)
            _pb_df = sector_df.dropna(subset=["pb"]).sort_values("pb", ascending=True)

            fig_pe = go.Figure()
            fig_pe.add_trace(go.Bar(
                x=_pe_df["pe"], y=_pe_df["sector"], orientation="h",
                name="P/E (×)", marker_color="#5b9bd5",
                text=[f"{x:.1f}×" for x in _pe_df["pe"]], textposition="outside",
                hovertemplate="%{y}: %{x:.1f}×<extra></extra>",
            ))
            fig_pe.update_layout(
                title="Median P/E by Sector",
                height=max(300, n_sectors * 26), margin=dict(l=0, r=50, t=36, b=0),
                showlegend=False, dragmode=False,
            )
            st.plotly_chart(fig_pe, width="stretch")

        with mult_col2:
            fig_pb = go.Figure()
            fig_pb.add_trace(go.Bar(
                x=_pb_df["pb"], y=_pb_df["sector"], orientation="h",
                name="P/B (×)", marker_color="#70ad47",
                text=[f"{x:.2f}×" for x in _pb_df["pb"]], textposition="outside",
                hovertemplate="%{y}: %{x:.2f}×<extra></extra>",
            ))
            fig_pb.update_layout(
                title="Median P/B by Sector",
                height=max(300, n_sectors * 26), margin=dict(l=0, r=50, t=36, b=0),
                showlegend=False, dragmode=False,
            )
            st.plotly_chart(fig_pb, width="stretch")

        # ROE + Net Margin side by side
        roe_col, nm_col = st.columns(2)
        with roe_col:
            _roe_df = sector_df.dropna(subset=["roe_pct"]).sort_values("roe_pct", ascending=True)
            fig_roe = go.Figure(go.Bar(
                x=_roe_df["roe_pct"], y=_roe_df["sector"], orientation="h",
                marker_color="#f59e0b",
                text=[f"{x:.1f}%" for x in _roe_df["roe_pct"]], textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_roe.update_layout(
                title="Median ROE by Sector",
                height=max(300, n_sectors * 26), margin=dict(l=0, r=60, t=36, b=0),
                showlegend=False, dragmode=False,
            )
            st.plotly_chart(fig_roe, width="stretch")

        with nm_col:
            _nm_df = sector_df.dropna(subset=["net_margin_pct"]).sort_values("net_margin_pct", ascending=True)
            fig_nm = go.Figure(go.Bar(
                x=_nm_df["net_margin_pct"], y=_nm_df["sector"], orientation="h",
                marker_color=["#2ca02c" if x >= 0 else "#d62728" for x in _nm_df["net_margin_pct"]],
                text=[f"{x:.1f}%" for x in _nm_df["net_margin_pct"]], textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_nm.update_layout(
                title="Median Net Margin by Sector",
                height=max(300, n_sectors * 26), margin=dict(l=0, r=60, t=36, b=0),
                showlegend=False, dragmode=False,
            )
            st.plotly_chart(fig_nm, width="stretch")

        st.divider()

        # ── 3. SUMMARY TABLE ────────────────────────────────────
        st.subheader("Sector Summary Table")
        disp = sector_df.copy()
        for col, fmt in [
            ("dcf_upside_pct", lambda x: f"{x:+.1f}%"),
            ("roe_pct",        lambda x: f"{x:.1f}%"),
            ("net_margin_pct", lambda x: f"{x:.1f}%"),
            ("fcf_margin_pct", lambda x: f"{x:.1f}%"),
            ("de",             lambda x: f"{x:.2f}×"),
            ("current_ratio",  lambda x: f"{x:.2f}×"),
            ("pe",             lambda x: f"{x:.1f}×"),
            ("pb",             lambda x: f"{x:.2f}×"),
            ("quality",        lambda x: f"{x:.0f}"),
        ]:
            disp[col] = disp[col].apply(lambda x, f=fmt: f(x) if pd.notna(x) else "—")
        disp = disp.rename(columns={
            "sector":"Sector","tickers":"# Tickers",
            "dcf_upside_pct":"DCF Upside","roe_pct":"ROE","net_margin_pct":"Net Margin",
            "fcf_margin_pct":"FCF Margin","de":"D/E","current_ratio":"Curr Ratio",
            "pe":"P/E","pb":"P/B","quality":"Quality",
        })
        st.dataframe(
            disp[["Sector","# Tickers","DCF Upside","P/E","P/B","ROE",
                  "Net Margin","FCF Margin","D/E","Curr Ratio","Quality"]]
            .set_index("Sector"),
            width="stretch",
        )

        st.divider()

        # ── 4. TOP 5 PER SECTOR (sorted by selected metric) ────────
        _top5_label = {
            "Avg Estimates vs Market": "Avg Est Upside",
            "DCF vs Market":           "DCF Upside",
            "P/E (lower = cheaper)":   "Lowest P/E",
            "P/B (lower = cheaper)":   "Lowest P/B",
            "ROE (higher = better)":   "Highest ROE",
        }.get(_metric_sel, "DCF Upside")
        st.subheader(f"Top 5 per Sector — by {_top5_label}")
        if not ticker_df.empty:
            _top_sort_col = _metric_col if _metric_col in ticker_df.columns else "upside_pct"
            _top_asc      = _sort_asc  # True for P/E, P/B (lower = better); False otherwise
            _top_df = (
                ticker_df.dropna(subset=[_top_sort_col])
                .sort_values(_top_sort_col, ascending=_top_asc)
            )
            _sectors_sorted = (
                sector_df.dropna(subset=["dcf_upside_pct"])
                .sort_values("dcf_upside_pct", ascending=False)["sector"]
                .tolist()
            )
            for _sec in _sectors_sorted:
                _sec_top = _top_df[_top_df["sector"] == _sec].head(5)
                if _sec_top.empty:
                    continue
                _exp_label = f"**{_sec}** — top {len(_sec_top)} by {_top5_label}"
                with st.expander(_exp_label, expanded=False):
                    # Always show price + selected metric first, then the rest
                    _base_cols = ["ticker","name","price"]
                    _metric_display = {
                        "avg_upside": ("avg_upside", "Avg Est Upside", lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"),
                        "upside_pct": ("upside_pct", "DCF Upside",     lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"),
                        "pe":         ("pe",          "P/E",            lambda x: f"{x:.1f}×"  if pd.notna(x) else "—"),
                        "pb":         ("pb",          "P/B",            lambda x: f"{x:.2f}×"  if pd.notna(x) else "—"),
                        "roe":        ("roe",         "ROE",            lambda x: f"{x:.1f}%"  if pd.notna(x) else "—"),
                    }
                    _mcol, _mlabel, _mfmt = _metric_display.get(_metric_col, _metric_display["upside_pct"])
                    # Extra cols (exclude selected metric to avoid duplicate)
                    _extra = [c for c in ["avg_upside","upside_pct","pe","pb","roe"] if c != _mcol]

                    _all_cols = _base_cols + [_mcol] + _extra
                    _avail = [c for c in _all_cols if c in _sec_top.columns]
                    _t = _sec_top[_avail].copy()

                    _t["price"] = _t["price"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
                    _t[_mcol]   = _t[_mcol].apply(_mfmt)
                    _fmt_map = {
                        "avg_upside": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                        "upside_pct": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                        "pe":         lambda x: f"{x:.1f}×"  if pd.notna(x) else "—",
                        "pb":         lambda x: f"{x:.2f}×"  if pd.notna(x) else "—",
                        "roe":        lambda x: f"{x:.1f}%"  if pd.notna(x) else "—",
                    }
                    for ec in _extra:
                        if ec in _t.columns:
                            _t[ec] = _t[ec].apply(_fmt_map[ec])

                    _rename = {
                        "ticker":"Ticker","name":"Company","price":"Price (VND)",
                        _mcol: _mlabel,
                        "avg_upside":"Avg Est Upside","upside_pct":"DCF Upside",
                        "pe":"P/E","pb":"P/B","roe":"ROE",
                    }
                    _t = _t.rename(columns=_rename)
                    st.dataframe(_t.set_index("Ticker"), width="stretch")

        st.divider()

        # ── 5. QUALITY vs X SCATTER ────────────────────────────
        _SC_METRICS = {
            "DCF Upside %":  ("dcf_upside_pct", "Median DCF Upside (%)"),
            "P/E (median)":  ("pe",              "Median P/E (×)"),
            "P/B (median)":  ("pb",              "Median P/B (×)"),
            "ROE %":         ("roe_pct",         "Median ROE (%)"),
            "Net Margin %":  ("net_margin_pct",  "Median Net Margin (%)"),
        }
        _sc_sel = st.radio(
            "X axis:", list(_SC_METRICS.keys()),
            horizontal=True, index=0, key="scatter_x",
            label_visibility="collapsed",
        )
        _sc_col, _sc_label = _SC_METRICS[_sc_sel]
        st.subheader(f"Quality Score vs {_sc_sel}")

        scatter_df = sector_df.dropna(subset=[_sc_col, "quality"])
        fig_sc = px.scatter(
            scatter_df, x=_sc_col, y="quality",
            size="tickers", text="sector", color="roe_pct",
            color_continuous_scale="RdYlGn",
            labels={_sc_col: _sc_label, "quality": "Median Quality Score",
                    "tickers": "# Tickers", "roe_pct": "ROE (%)"},
        )
        fig_sc.update_traces(textposition="top center")
        fig_sc.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sc.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), dragmode=False)
        st.plotly_chart(fig_sc, width="stretch")


# ═══════════════════════════════════════════════════════════════
# VIEW 5 — PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════
elif view == "Portfolio Tracker":
    st.title("Portfolio Tracker")
    st.caption("Format: `TICKER  SHARES  ENTRY_PRICE`  (entry price optional — needed for P&L)")

    # ── Holdings input — editable table ────────────────────────
    # Initialize ONCE — never overwrite between reruns (prevents lost edits)
    if "portfolio_df" not in st.session_state:
        st.session_state["portfolio_df"] = pd.DataFrame([
            {"Ticker": "VNM",  "Shares": 1000, "Entry Price (VND)": 58000},
            {"Ticker": "FPT",  "Shares": 500,  "Entry Price (VND)": 120000},
            {"Ticker": "VIC",  "Shares": 200,  "Entry Price (VND)": 45000},
            {"Ticker": "HPG",  "Shares": 2000, "Entry Price (VND)": 0},
        ])

    edited = st.data_editor(
        st.session_state["portfolio_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small",
                help="Stock ticker e.g. VNM"),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0,
                format="%d", width="small"),
            "Entry Price (VND)": st.column_config.NumberColumn("Entry Price (VND)",
                min_value=0, format="%d", width="medium",
                help="Avg buy price. Leave 0 to skip P&L."),
        },
        key="portfolio_editor",
    )
    # Persist edits so they survive page navigation
    st.session_state["portfolio_df"] = edited

    # Parse from data editor
    holdings: dict[str, tuple] = {}
    for _, row in edited.iterrows():
        tkr = str(row.get("Ticker", "") or "").upper().strip()
        try:
            shares = float(row.get("Shares") or 0)
            entry  = float(row.get("Entry Price (VND)") or 0)
            if tkr and shares > 0:
                holdings[tkr] = (shares, entry)
        except (ValueError, TypeError):
            pass

    if not holdings:
        st.info("Enter your holdings above.")
        st.stop()

    # ── Load data ───────────────────────────────────────────────
    price_map  = load_latest_prices()
    screen_df  = load_valuation_screen_data()
    vnidx_df   = load_vnindex_prices(days=730)

    val_lookup: dict[str, dict] = {}
    sector_lookup: dict[str, str] = {}
    if not screen_df.empty:
        for _, row in screen_df.iterrows():
            val_lookup[row["Ticker"]] = row.to_dict()
            sector_lookup[row["Ticker"]] = row.get("Sector", "Unknown")

    # ── Build portfolio rows ────────────────────────────────────
    rows = []
    for tkr, (shares, entry_price) in holdings.items():
        cur     = price_map.get(tkr, 0)
        vd      = val_lookup.get(tkr, {})
        sector  = sector_lookup.get(tkr, "Unknown")

        mkt_val  = cur * shares
        cost     = entry_price * shares if entry_price > 0 else None
        pnl      = (mkt_val - cost) if cost else None
        pnl_pct  = (pnl / cost * 100) if cost else None

        avg_est_raw = None
        ae_str = vd.get("Avg Estimate", "—")
        if ae_str and ae_str not in ("—", "-"):
            try: avg_est_raw = float(str(ae_str).replace(",",""))
            except: pass
        avg_upside = ((avg_est_raw - cur) / cur * 100) if avg_est_raw and cur else None

        rows.append({
            "Ticker":       tkr,
            "Sector":       sector,
            "Shares":       f"{shares:,.0f}",
            "Entry (VND)":  f"{entry_price:,.0f}" if entry_price > 0 else "—",
            "Price (VND)":  f"{cur:,.0f}" if cur else "—",
            "Mkt Value":    f"{mkt_val:,.0f}" if cur else "—",
            "P&L (VND)":    f"{pnl:+,.0f}" if pnl is not None else "—",
            "P&L %":        f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—",
            "Avg Est":      vd.get("Avg Estimate", "—"),
            "Avg Upside":   f"{avg_upside:+.1f}%" if avg_upside is not None else "—",
            "Signal":       vd.get("Signal", "—"),
            "Quality":      vd.get("Quality", "—"),
            "_mkt":         mkt_val,
            "_cost":        cost or 0,
            "_pnl":         pnl or 0,
            "_pnl_pct":     pnl_pct,
            "_shares":      shares,
            "_entry":       entry_price,
            "_sector":      sector,
            "_avg_upside":  avg_upside,
        })

    port_df = pd.DataFrame(rows)
    total_mkt  = port_df["_mkt"].sum()
    total_cost = port_df["_cost"].sum()
    total_pnl  = port_df["_pnl"].sum()
    has_cost   = total_cost > 0

    # ── Summary KPIs ────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Market Value",  f"{total_mkt/1e9:,.2f} bn" if total_mkt else "—")
    k2.metric("Cost Basis",    f"{total_cost/1e9:,.2f} bn" if has_cost else "—")
    k3.metric("P&L",
              f"{total_pnl/1e9:+,.2f} bn" if has_cost else "—",
              delta=f"{total_pnl/total_cost*100:+.1f}%" if has_cost and total_cost else None,
              delta_color="normal" if total_pnl >= 0 else "inverse")
    # Weighted avg upside
    _wu = port_df.dropna(subset=["_avg_upside"])
    wavg = (_wu["_avg_upside"] * _wu["_mkt"]).sum() / _wu["_mkt"].sum() if not _wu.empty and _wu["_mkt"].sum() > 0 else None
    k4.metric("Wtd Avg Upside", f"{wavg:+.1f}%" if wavg is not None else "—")
    k5.metric("Holdings", f"{len(port_df)} tickers")

    st.divider()

    # ── Holdings table ──────────────────────────────────────────
    display_cols = ["Sector","Shares","Entry (VND)","Price (VND)","Mkt Value",
                    "P&L (VND)","P&L %","Avg Est","Avg Upside","Signal","Quality"]
    raw_cols = [c for c in port_df.columns if c.startswith("_")]

    def _pnl_color(val):
        try:
            v = float(str(val).replace(",","").replace("%","").replace("+",""))
            if v > 0: return "color:#22c55e"
            elif v < 0: return "color:#ef4444"
        except: pass
        return ""

    styled_port = (
        port_df[["Ticker"] + display_cols].set_index("Ticker")
        .style.map(_pnl_color, subset=["P&L (VND)", "P&L %", "Avg Upside"])
    )
    st.dataframe(styled_port, width="stretch")

    st.divider()

    # ── Charts ──────────────────────────────────────────────────
    chart1, chart2 = st.columns(2)

    # Sector allocation pie
    with chart1:
        st.subheader("Sector Allocation")
        sec_alloc = port_df.groupby("_sector")["_mkt"].sum().reset_index()
        sec_alloc.columns = ["Sector", "Value"]
        sec_alloc = sec_alloc[sec_alloc["Value"] > 0].sort_values("Value", ascending=False)
        fig_sec = px.pie(sec_alloc, names="Sector", values="Value", hole=0.4)
        fig_sec.update_traces(textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} VND<extra></extra>")
        fig_sec.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False, dragmode=False)
        st.plotly_chart(fig_sec, width="stretch")

    # Ticker allocation pie
    with chart2:
        st.subheader("Position Allocation")
        tkr_alloc = port_df[port_df["_mkt"] > 0].copy()
        fig_tkr = px.pie(tkr_alloc, names="Ticker", values="_mkt", hole=0.4)
        fig_tkr.update_traces(textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} VND<extra></extra>")
        fig_tkr.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False, dragmode=False)
        st.plotly_chart(fig_tkr, width="stretch")

    # ── P&L chart (if entry prices provided) ───────────────────
    if has_cost:
        st.subheader("P&L by Position")
        pnl_df = port_df[port_df["_entry"] > 0].copy()
        pnl_df = pnl_df.sort_values("_pnl_pct", ascending=True)
        fig_pnl = go.Figure(go.Bar(
            x=pnl_df["_pnl_pct"], y=pnl_df["Ticker"], orientation="h",
            marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in pnl_df["_pnl_pct"]],
            text=[f"{v:+.1f}%" for v in pnl_df["_pnl_pct"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
        ))
        fig_pnl.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig_pnl.update_layout(height=max(250, len(pnl_df)*32),
            margin=dict(l=0,r=60,t=10,b=0), xaxis_title="Return %",
            dragmode=False)
        st.plotly_chart(fig_pnl, width="stretch")

    # ── Portfolio vs VN-Index performance ──────────────────────
    if has_cost and not vnidx_df.empty:
        st.subheader("Portfolio Return vs VN-Index")
        # Use the earliest entry date to anchor performance comparison
        # Show cumulative return from cost basis date (use today as anchor)
        fig_bench = go.Figure()

        # VN-Index normalized to 100 (last 1Y)
        _vi = vnidx_df.tail(252).copy()
        if not _vi.empty:
            _base = float(_vi["close"].iloc[0])
            _vi["return_pct"] = (_vi["close"] / _base - 1) * 100
            fig_bench.add_trace(go.Scatter(
                x=_vi["date"].dt.strftime("%Y-%m-%d"), y=_vi["return_pct"],
                name="VN-Index", mode="lines",
                line=dict(color="#9ca3af", width=2),
                hovertemplate="VN-Index: %{y:+.1f}%<extra></extra>",
            ))

        # Portfolio current return (single point — total P&L %)
        if total_cost > 0:
            port_return = total_pnl / total_cost * 100
            fig_bench.add_hline(y=port_return, line_dash="dash",
                line_color="#22c55e" if port_return >= 0 else "#ef4444",
                line_width=2,
                annotation_text=f"Portfolio {port_return:+.1f}%",
                annotation_position="right",
                annotation_font_color="#22c55e" if port_return >= 0 else "#ef4444")

        fig_bench.update_layout(
            height=320, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Return %", dragmode=False,
            hovermode="x unified",
        )
        fig_bench.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
        st.plotly_chart(fig_bench, width="stretch")
        st.caption("VN-Index return shown over last 1Y. Portfolio return = total P&L / total cost basis.")

    # ── CSV export ──────────────────────────────────────────────
    _exp = port_df[["Ticker"] + display_cols].copy()
    st.download_button("Download Portfolio CSV",
        data=_exp.to_csv(index=False).encode("utf-8-sig"),
        file_name="portfolio.csv", mime="text/csv")



# ═══════════════════════════════════════════════════════════════
# VIEW 6 — MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════
elif view == "Market Overview":
    _mo_tab1, _mo_tab2 = st.tabs(["📊 Thị trường", "📈 Vĩ mô"])
    with _mo_tab1:
        st.title("Market Overview")

        # ── Intraday index ticker bar (auto-refreshes every 30s during trading hours)
        _render_index_ticker_bar()

        with st.spinner("Loading market data..."):
            snap_df = load_market_snapshot()
            vnidx   = load_vnindex_prices(days=252)

        if snap_df.empty:
            st.warning("No price data.")
            st.stop()

        snap_df = snap_df[snap_df["chg_pct"].notna()].copy()
        n_up   = (snap_df["chg_pct"] > 0).sum()
        n_dn   = (snap_df["chg_pct"] < 0).sum()
        n_flat = (snap_df["chg_pct"] == 0).sum()
        total  = len(snap_df)

        # ── VN-Index chart + A/D ratio
        col_idx, col_ad = st.columns([3, 1])
        with col_idx:
            st.subheader("VN-Index (1 Year)")
            if not vnidx.empty:
                _vi = vnidx.copy()
                _vi["ma20"]   = _vi["close"].rolling(20).mean()
                _vi["dlabel"] = _vi["date"].dt.strftime("%Y-%m-%d")
                _last_vi = float(_vi.iloc[-1]["close"])
                _prev_vi = float(_vi.iloc[-2]["close"]) if len(_vi) > 1 else _last_vi
                _chg_vi  = (_last_vi - _prev_vi) / _prev_vi * 100 if _prev_vi else 0
                _cc_vi   = "#22c55e" if _chg_vi >= 0 else "#ef4444"
                _vi_ymin = float(_vi["close"].min()) * 0.975
                _vi_ymax = float(_vi["close"].max()) * 1.015
                _fig_vi  = go.Figure()
                _fig_vi.add_trace(go.Scatter(x=_vi["dlabel"], y=_vi["close"], mode="lines", name="VN-Index",
                    line=dict(color="#5b9bd5", width=2), fill="tozeroy", fillcolor="rgba(91,155,213,0.08)",
                    hovertemplate="%{y:,.2f}<extra></extra>"))
                _fig_vi.add_trace(go.Scatter(x=_vi["dlabel"], y=_vi["ma20"], mode="lines", name="MA20",
                    line=dict(color="#a78bfa", width=1.5), hovertemplate="MA20 %{y:,.2f}<extra></extra>"))
                _fig_vi.update_layout(height=320, margin=dict(l=0,r=10,t=40,b=0), dragmode=False,
                    hovermode="x unified", showlegend=True, legend=dict(orientation="h", y=-0.1),
                    title=dict(text=f"<span style='color:{_cc_vi}'>{_last_vi:,.2f}  ({_chg_vi:+.2f}%)</span>",
                               font=dict(size=16)),
                    xaxis=dict(type="category", nticks=8, showgrid=False, rangeslider=dict(visible=False)),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               range=[_vi_ymin, _vi_ymax]))
                st.plotly_chart(_fig_vi, width="stretch")
            else:
                st.info("No VN-Index data available.")

        with col_ad:
            st.subheader("Advance / Decline")
            _fig_ad = go.Figure(go.Bar(
                x=["Up", "Flat", "Down"], y=[n_up, n_flat, n_dn],
                marker_color=["#22c55e", "#eab308", "#ef4444"],
                text=[n_up, n_flat, n_dn], textposition="outside",
                hovertemplate="%{x}: %{y}<extra></extra>"))
            _fig_ad.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0),
                dragmode=False, showlegend=False, yaxis_visible=False)
            st.plotly_chart(_fig_ad, width="stretch")
            _ratio     = n_up / max(n_dn, 1)
            _rc        = "#22c55e" if _ratio >= 1.5 else "#eab308" if _ratio >= 0.8 else "#ef4444"
            st.markdown(
                f"<div style='text-align:center;font-size:26px;font-weight:700;color:{_rc}'>{_ratio:.2f}"
                f"<span style='font-size:13px;color:#9ca3af'> A/D ratio</span></div>"
                f"<div style='text-align:center;font-size:12px;color:#9ca3af'>"
                f"<span style='color:#22c55e'>▲{n_up}</span>  "
                f"<span style='color:#eab308'>—{n_flat}</span>  "
                f"<span style='color:#ef4444'>▼{n_dn}</span>  "
                f"of {total} tickers</div>", unsafe_allow_html=True)

        st.divider()

        # ── Market-wide P/E & P/B vs history ────────────────────────
        st.subheader("Market Valuation (P/E & P/B) vs History")
        _mv = load_market_valuation_history()
        if not _mv.empty:
            def _fmt_q(p: str) -> str:
                try:
                    yr, q = p.split("-Q")
                    return f"Q{q}/{yr[2:]}"
                except Exception:
                    return p
            _mv_lbl = [_fmt_q(p) for p in _mv["period"]]
            _cur_pe, _cur_pb = _mv["median_pe"].iloc[-1], _mv["median_pb"].iloc[-1]
            _avg_pe, _avg_pb = _mv["median_pe"].mean(), _mv["median_pb"].mean()

            _vc1, _vc2 = st.columns(2)
            with _vc1:
                _pe_diff = (_cur_pe / _avg_pe - 1) * 100 if _avg_pe else 0
                _pe_clr  = "#ef4444" if _pe_diff > 5 else "#22c55e" if _pe_diff < -5 else "#eab308"
                _pe_ymin = max(0.0, float(_mv["median_pe"].min()) * 0.85)
                _pe_ymax = float(_mv["median_pe"].max()) * 1.06
                fig_mpe = go.Figure(go.Scatter(
                    x=_mv_lbl, y=_mv["median_pe"], mode="lines", name="Median P/E",
                    line=dict(color="#60a5fa", width=2), fill="tozeroy",
                    fillcolor="rgba(96,165,250,0.07)",
                    hovertemplate="P/E: %{y:.1f}x<extra></extra>"))
                fig_mpe.add_trace(go.Scatter(
                    x=_mv_lbl, y=[_avg_pe] * len(_mv_lbl), mode="lines", name="Trung bình",
                    line=dict(color="#94a3b8", width=1, dash="dot"),
                    hovertemplate=f"Trung bình: {_avg_pe:.1f}x<extra></extra>"))
                fig_mpe.add_trace(go.Scatter(
                    x=_mv_lbl, y=[_cur_pe] * len(_mv_lbl), mode="lines", name="Hiện tại",
                    line=dict(color=_pe_clr, width=1.2, dash="dash"),
                    hovertemplate=f"Hiện tại: {_cur_pe:.1f}x<extra></extra>"))
                fig_mpe.update_layout(
                    height=300, margin=dict(l=0, r=0, t=30, b=0), dragmode=False,
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#f9fafb"),
                    title=dict(text=f"P/E hiện tại: <span style='color:{_pe_clr}'>{_cur_pe:.1f}x "
                                    f"({_pe_diff:+.0f}% so TB)</span>", font=dict(size=14)),
                    legend=dict(orientation="h", y=-0.15),
                    xaxis=dict(type="category", nticks=8, showgrid=False),
                    yaxis=dict(title="P/E (x)", showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               range=[_pe_ymin, _pe_ymax]))
                st.plotly_chart(fig_mpe, width="stretch")

            with _vc2:
                _pb_diff = (_cur_pb / _avg_pb - 1) * 100 if _avg_pb else 0
                _pb_clr  = "#ef4444" if _pb_diff > 5 else "#22c55e" if _pb_diff < -5 else "#eab308"
                _pb_ymin = max(0.0, float(_mv["median_pb"].min()) * 0.85)
                _pb_ymax = float(_mv["median_pb"].max()) * 1.06
                fig_mpb = go.Figure(go.Scatter(
                    x=_mv_lbl, y=_mv["median_pb"], mode="lines", name="Median P/B",
                    line=dict(color="#34d399", width=2), fill="tozeroy",
                    fillcolor="rgba(52,211,153,0.07)",
                    hovertemplate="P/B: %{y:.2f}x<extra></extra>"))
                fig_mpb.add_trace(go.Scatter(
                    x=_mv_lbl, y=[_avg_pb] * len(_mv_lbl), mode="lines", name="Trung bình",
                    line=dict(color="#94a3b8", width=1, dash="dot"),
                    hovertemplate=f"Trung bình: {_avg_pb:.2f}x<extra></extra>"))
                fig_mpb.add_trace(go.Scatter(
                    x=_mv_lbl, y=[_cur_pb] * len(_mv_lbl), mode="lines", name="Hiện tại",
                    line=dict(color=_pb_clr, width=1.2, dash="dash"),
                    hovertemplate=f"Hiện tại: {_cur_pb:.2f}x<extra></extra>"))
                fig_mpb.update_layout(
                    height=300, margin=dict(l=0, r=0, t=30, b=0), dragmode=False,
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#f9fafb"),
                    title=dict(text=f"P/B hiện tại: <span style='color:{_pb_clr}'>{_cur_pb:.2f}x "
                                    f"({_pb_diff:+.0f}% so TB)</span>", font=dict(size=14)),
                    legend=dict(orientation="h", y=-0.15),
                    xaxis=dict(type="category", nticks=8, showgrid=False),
                    yaxis=dict(title="P/B (x)", showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               range=[_pb_ymin, _pb_ymax]))
                st.plotly_chart(fig_mpb, width="stretch")

            st.caption(
                "P/E, P/B = trung vị (median) của toàn bộ mã có dữ liệu mỗi quý. "
                "Đường chấm = trung bình toàn bộ giai đoạn. Giá trị hiện tại cao hơn đường trung bình "
                "→ thị trường đang đắt hơn so với quá khứ; thấp hơn → đang rẻ hơn.")
        else:
            st.info("Not enough data to compute market-wide P/E and P/B history.")

        st.divider()

        # ── Foreign net trading value — last 15 sessions (whole HOSE market) ──
        st.subheader("Foreign Net Trading Value")
        _ff = load_foreign_flow("VNINDEX", sessions=15)
        if not _ff.empty:
            _ff = _ff.copy()
            _ff["net_bn"]  = _ff["net_val"] / 1e9          # VND → tỷ (billion)
            _ff["dlabel"]  = _ff["date"].dt.strftime("%d/%m")
            _ff_colors     = ["#22c55e" if v >= 0 else "#ef4444" for v in _ff["net_bn"]]
            _last_net      = float(_ff.iloc[-1]["net_bn"])
            _net_15        = float(_ff["net_bn"].sum())
            _cc_net        = "#22c55e" if _last_net >= 0 else "#ef4444"
            _cc_sum        = "#22c55e" if _net_15 >= 0 else "#ef4444"
            _fig_ff = go.Figure(go.Bar(
                x=_ff["dlabel"], y=_ff["net_bn"], marker_color=_ff_colors,
                customdata=list(zip(_ff["buy_val"] / 1e9, _ff["sell_val"] / 1e9)),
                hovertemplate=("<b>%{x}</b><br>Net: %{y:,.1f} tỷ<br>"
                               "Buy: %{customdata[0]:,.1f} tỷ<br>"
                               "Sell: %{customdata[1]:,.1f} tỷ<extra></extra>")))
            _fig_ff.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
            _fig_ff.update_layout(
                height=300, margin=dict(l=0, r=10, t=40, b=0), dragmode=False, showlegend=False,
                title=dict(text=("Foreign net buy/sell · last 15 sessions  "
                                 f"<span style='color:{_cc_net}'>(latest {_last_net:+,.0f} tỷ)</span>"),
                           font=dict(size=14)),
                xaxis=dict(type="category", showgrid=False),
                yaxis=dict(title="Net value (tỷ VND)", showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           zeroline=False))
            st.plotly_chart(_fig_ff, width="stretch")
            st.caption(
                f"Net buy = green, net sell = red. 15-session cumulative: "
                f"<span style='color:{_cc_sum};font-weight:600'>{_net_15:+,.0f} tỷ VND</span>. "
                f"Source: VNDirect (NN = nhà đầu tư nước ngoài).", unsafe_allow_html=True)
        else:
            st.info("Foreign flow data unavailable right now.")

        st.divider()

        # ── Top gainers / losers
        gain_col, lose_col = st.columns(2)
        def _chg_color(val):
            try:
                v = float(str(val).replace("%","").replace("+",""))
                if v > 0: return "color:#22c55e;font-weight:600"
                if v < 0: return "color:#ef4444;font-weight:600"
            except: pass
            return ""

        # Exclude covered warrants (chứng quyền): tickers with digits, e.g. CMBB2601
        _stocks_only = snap_df[snap_df["ticker"].str.match(r'^[A-Z]{3,5}$')]

        with gain_col:
            st.subheader("Top 10 Gainers")
            _g = _stocks_only.nlargest(10, "chg_pct").copy()
            _g["Price"]  = _g["price"].apply(lambda x: f"{x:,.0f}" if x else "—")
            _g["Change"] = _g["chg_pct"].apply(lambda x: f"{x:+.2f}%")
            _g["Volume"] = _g["volume"].apply(lambda x: f"{x/1e6:.2f}M" if x else "—")
            _gd = _g[["ticker","name","Price","Change","Volume"]].rename(columns={"ticker":"Ticker","name":"Company"})
            st.dataframe(_gd.set_index("Ticker").style.map(_chg_color, subset=["Change"]), width="stretch")

        with lose_col:
            st.subheader("Top 10 Losers")
            _l = _stocks_only.nsmallest(10, "chg_pct").copy()
            _l["Price"]  = _l["price"].apply(lambda x: f"{x:,.0f}" if x else "—")
            _l["Change"] = _l["chg_pct"].apply(lambda x: f"{x:+.2f}%")
            _l["Volume"] = _l["volume"].apply(lambda x: f"{x/1e6:.2f}M" if x else "—")
            _ld = _l[["ticker","name","Price","Change","Volume"]].rename(columns={"ticker":"Ticker","name":"Company"})
            st.dataframe(_ld.set_index("Ticker").style.map(_chg_color, subset=["Change"]), width="stretch")

        st.divider()

        # ── Top volume + Sector performance
        vol_col, sec_col = st.columns(2)
        with vol_col:
            st.subheader("Top 10 by Volume")
            _tv = snap_df.nlargest(10, "volume").copy()
            _tv["Price"]  = _tv["price"].apply(lambda x: f"{x:,.0f}" if x else "—")
            _tv["Change"] = _tv["chg_pct"].apply(lambda x: f"{x:+.2f}%")
            _tv["Volume"] = _tv["volume"].apply(lambda x: f"{x/1e6:.2f}M" if x else "—")
            _tvd = _tv[["ticker","name","Price","Change","Volume"]].rename(columns={"ticker":"Ticker","name":"Company"})
            st.dataframe(_tvd.set_index("Ticker").style.map(_chg_color, subset=["Change"]), width="stretch")

        with sec_col:
            st.subheader("Sector Performance")
            _sp = snap_df[snap_df["sector"] != "Unknown"].groupby("sector")["chg_pct"].mean().reset_index()
            _sp = _sp.sort_values("chg_pct", ascending=True)
            _sp_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in _sp["chg_pct"]]
            _fig_sp = go.Figure(go.Bar(
                x=_sp["chg_pct"], y=_sp["sector"], orientation="h",
                marker_color=_sp_colors,
                text=[f"{v:+.2f}%" for v in _sp["chg_pct"]], textposition="outside",
                hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
            _fig_sp.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
            _fig_sp.update_layout(height=max(300, len(_sp)*24), margin=dict(l=0,r=60,t=10,b=0),
                dragmode=False, showlegend=False, xaxis_title="Avg Change %")
            st.plotly_chart(_fig_sp, width="stretch")

        st.divider()

        # ── Market Heatmap
        st.subheader("Market Heatmap Today")
        _hm_df = snap_df[snap_df["ticker"].str.match(r'^[A-Z]{3,5}$')].dropna(subset=["chg_pct"]).copy()
        _hm_df["_vol"]    = _hm_df["volume"].fillna(0).clip(lower=1)
        _hm_df["_chgs"]   = _hm_df["chg_pct"].apply(lambda x: f"{x:+.1f}%")
        _hm_df["_prices"] = _hm_df["price"].fillna(0).apply(lambda x: f"{x:,.0f}")

        _fig_hm = go.Figure(go.Treemap(
            labels=_hm_df["ticker"].tolist(),
            parents=[""] * len(_hm_df),
            values=_hm_df["_vol"].tolist(),
            text=_hm_df["_chgs"].tolist(),
            texttemplate="<b>%{label}</b><br>%{text}",
            textfont=dict(size=11, color="#ffffff"),
            customdata=list(zip(_hm_df["name"].fillna(""), _hm_df["_prices"], _hm_df["_chgs"])),
            hovertemplate="<b>%{label}</b>  %{customdata[0]}<br>Giá: %{customdata[1]} VND<br>Thay đổi: %{customdata[2]}<extra></extra>",
            marker=dict(
                colors=_hm_df["chg_pct"].tolist(),
                colorscale=[
                    [0.0,  "#7f1d1d"],
                    [0.3,  "#dc2626"],
                    [0.5,  "#334155"],
                    [0.7,  "#16a34a"],
                    [1.0,  "#14532d"],
                ],
                cmin=-10, cmax=10,
                showscale=False,
                line=dict(width=1, color="#0f172a"),
            ),
            pathbar=dict(visible=False),
        ))
        _fig_hm.update_layout(
            height=430, margin=dict(l=0, r=0, t=0, b=0), dragmode=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_fig_hm, width="stretch")
        _hm_up   = int((_hm_df["chg_pct"] > 0).sum())
        _hm_dn   = int((_hm_df["chg_pct"] < 0).sum())
        _hm_flat = int((_hm_df["chg_pct"] == 0).sum())
        st.caption(f"▲ Tăng: {_hm_up}  ·  ▼ Giảm: {_hm_dn}  ·  — Đứng: {_hm_flat}  ·  {len(_hm_df)} mã cổ phiếu")



    with _mo_tab2:
        st.title("Kinh tế vĩ mô")
        st.caption("Nguồn: Tổng cục Thống kê (nso.gov.vn)")

        _updated = check_macro_updates()
        if _updated:
            _names = ", ".join(_MACRO_INDICATOR_LABELS.get(k, k) for k in _updated)
            st.toast(f"Có dữ liệu vĩ mô mới: {_names}", icon="📊")
            st.success(f"Có dữ liệu mới cho: {_names}")

        def _macro_bar_chart(df: "pd.DataFrame", title: str, unit: str = "%"):
            if df.empty:
                st.info(f"{title}: chưa có dữ liệu")
                return
            colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["value"]]
            # Annual-only series (all periods in December) are sparse and far apart on a
            # date axis, which makes Plotly auto-size the bars to span huge ranges.
            # Render those as categorical (one bar per year) instead.
            is_annual = all(p.month == 12 for p in df["period"])
            if is_annual:
                x = df["period"].apply(lambda p: str(p.year))
                hover = "%{x}: %{y:+.2f}" + unit + "<extra></extra>"
            else:
                x = df["period"]
                hover = "%{x|%m/%Y}: %{y:+.2f}" + unit + "<extra></extra>"
            fig = go.Figure(go.Bar(x=x, y=df["value"], marker_color=colors, hovertemplate=hover))
            fig.add_hline(y=0, line_color="gray", opacity=0.5)
            fig.update_layout(
                title=title, height=280, margin=dict(l=0, r=0, t=40, b=0),
                dragmode=False, showlegend=False,
                yaxis_title=unit)
            if is_annual:
                fig.update_xaxes(type="category")
            st.plotly_chart(fig, width="stretch")
            latest = df["period"].max()
            st.caption(f"Nguồn: Tổng cục Thống kê (nso.gov.vn) · Cập nhật đến {latest:%m/%Y}")

        def _macro_line_chart(series: dict[str, "pd.DataFrame"], title: str, unit: str = "%"):
            series = {label: df for label, df in series.items() if not df.empty}
            if not series:
                st.info(f"{title}: chưa có dữ liệu")
                return
            _colors = ["#22c55e", "#f59e0b", "#60a5fa"]
            fig = go.Figure()
            for i, (label, df) in enumerate(series.items()):
                fig.add_trace(go.Scatter(
                    x=df["period"], y=df["value"], name=label, mode="lines+markers",
                    line=dict(color=_colors[i % len(_colors)], width=2),
                    hovertemplate="%{x|%m/%Y} · " + label + ": %{y:+.2f}" + unit + "<extra></extra>"))
            fig.add_hline(y=0, line_color="gray", opacity=0.5)
            fig.update_layout(
                title=title, height=340, margin=dict(l=0, r=0, t=40, b=40),
                dragmode=False, showlegend=len(series) > 1,
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
                yaxis_title=unit)
            st.plotly_chart(fig, width="stretch")
            latest = max(df["period"].max() for df in series.values())
            st.caption(f"Nguồn: Tổng cục Thống kê (nso.gov.vn) · Cập nhật đến {latest:%m/%Y}")

        def _macro_summary_row(indicator: str, label: str, unit: str = "%", freq: str = "monthly") -> dict:
            df = load_macro_indicator(indicator)
            if df.empty:
                return {
                    "Chỉ số": label, "Kỳ gần nhất": "—", "Kỳ trước": "—",
                    "Đơn vị": unit, "Kỳ báo cáo": "—", "Khoảng dữ liệu": "Chưa có dữ liệu",
                }
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            if freq == "quarterly":
                period_str = f"Quý {(last['period'].month - 1) // 3 + 1}/{last['period'].year}"
            else:
                period_str = f"Tháng {last['period'].month}/{last['period'].year}"
            return {
                "Chỉ số": label,
                "Kỳ gần nhất": f"{last['value']:+.2f}{unit}",
                "Kỳ trước": f"{prev['value']:+.2f}{unit}" if prev is not None else "—",
                "Đơn vị": unit,
                "Kỳ báo cáo": period_str,
                "Khoảng dữ liệu": f"{df['period'].iloc[0]:%m/%Y} - {df['period'].iloc[-1]:%m/%Y}",
            }

        def _wb_fmt_value(value: float, unit: str) -> str:
            if unit == "USD":
                if abs(value) >= 1e9:
                    return f"{value/1e9:,.2f} tỷ"
                return f"{value/1e6:,.1f} triệu"
            if unit == "người":
                return f"{value:,.0f}"
            if unit == "VND":
                return f"{value:,.0f}"
            return f"{value:+.2f}"

        def _wb_summary_row(indicator: str, label: str, unit: str) -> dict:
            df = load_macro_indicator(indicator)
            if df.empty:
                return {
                    "Chỉ số": label, "Kỳ gần nhất": "—", "Kỳ trước": "—",
                    "Đơn vị": unit, "Kỳ báo cáo": "Hàng năm", "Khoảng dữ liệu": "Chưa có dữ liệu",
                }
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            return {
                "Chỉ số": label,
                "Kỳ gần nhất": _wb_fmt_value(last["value"], unit),
                "Kỳ trước": _wb_fmt_value(prev["value"], unit) if prev is not None else "—",
                "Đơn vị": unit,
                "Kỳ báo cáo": "Hàng năm",
                "Khoảng dữ liệu": f"{df['period'].iloc[0].year} - {df['period'].iloc[-1].year}",
            }

        def _wb_category_tab(category: str):
            rows = [_wb_summary_row(k, m["label"], m["unit"])
                    for k, m in WB_INDICATOR_META.items() if m["category"] == category]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Nguồn: World Bank Open Data (api.worldbank.org) · Dữ liệu theo năm, cập nhật hàng năm")

        tab_overview, tab_gdp, tab_prices, tab_biz, tab_trade, tab_labor, tab_money, tab_consumer, tab_tax, tab_rates = st.tabs(
            ["Tổng quan", "GDP", "Giá cả", "Kinh doanh", "Thương mại", "Lao động", "Tiền tệ", "Tiêu dùng", "Thuế", "Lãi suất"])

        with tab_overview:
            row1 = st.columns(3)
            row2 = st.columns(3)

            with row1[0]:
                _macro_bar_chart(load_macro_indicator("gdp_growth"), "Tăng trưởng GDP (so với cùng kỳ năm trước)")
            with row1[1]:
                _macro_bar_chart(load_macro_indicator("cpi_yoy"), "Lạm phát (so với cùng kỳ năm trước)")
            with row1[2]:
                _macro_bar_chart(load_macro_indicator("cpi_mom"), "Lạm phát (so với tháng trước)")
            with row2[0]:
                _macro_bar_chart(load_macro_indicator("trade_balance"), "Cán cân thương mại", unit=" tỷ USD")
            with row2[1]:
                _macro_bar_chart(load_macro_indicator("retail_sales_growth"), "Tăng trưởng bán lẻ (so với cùng kỳ năm trước)")
            with row2[2]:
                _macro_bar_chart(load_macro_indicator("fdi"), "Vốn đầu tư nước ngoài (FDI đăng ký, theo quý)", unit=" tỷ USD")

        with tab_gdp:
            _macro_line_chart(
                {
                    "Nông, lâm nghiệp và thủy sản": load_macro_indicator("gdp_sector_agri"),
                    "Công nghiệp và xây dựng": load_macro_indicator("gdp_sector_industry"),
                    "Dịch vụ": load_macro_indicator("gdp_sector_services"),
                },
                "Tăng trưởng GDP theo khu vực kinh tế (so với cùng kỳ năm trước)",
            )
            row3 = st.columns(3)
            with row3[0]:
                _macro_line_chart({"Quy mô GDP": load_macro_indicator("gdp_nominal_usd")}, "Quy mô GDP (theo năm)", unit=" tỷ USD")
            with row3[1]:
                _macro_line_chart({"GDP bình quân đầu người": load_macro_indicator("gdp_per_capita_usd")}, "GDP bình quân đầu người (theo năm)", unit=" USD")
            with row3[2]:
                _macro_line_chart({"Tăng trưởng vốn đầu tư": load_macro_indicator("investment_growth")}, "Tăng trưởng vốn đầu tư toàn xã hội (so với cùng kỳ)")

        with tab_prices:
            summary_rows = [
                _macro_summary_row("cpi_yoy", "Tỷ lệ lạm phát (so với cùng kỳ năm trước)"),
                _macro_summary_row("cpi_mom", "Tỷ lệ lạm phát (so với tháng trước)"),
                _macro_summary_row("core_inflation_yoy", "Tỷ lệ lạm phát cơ bản (so với cùng kỳ năm trước)"),
                _macro_summary_row("cpi_food", "Lạm phát lương thực (so với tháng trước)"),
                _macro_summary_row("cpi_transport", "CPI nhóm giao thông (so với tháng trước)"),
                _macro_summary_row("ppi_yoy", "Chỉ số giá sản xuất công nghiệp (so với cùng kỳ năm trước)", freq="quarterly"),
                {
                    "Chỉ số": "Chỉ số giá tiêu dùng so với kỳ gốc 2019",
                    "Kỳ gần nhất": "—", "Kỳ trước": "—", "Đơn vị": "điểm",
                    "Kỳ báo cáo": "—", "Khoảng dữ liệu": "Chưa có dữ liệu",
                },
            ]
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

            row1 = st.columns(2)
            with row1[0]:
                _macro_line_chart(
                    {
                        "Lạm phát (YoY)": load_macro_indicator("cpi_yoy"),
                        "Lạm phát cơ bản (YoY)": load_macro_indicator("core_inflation_yoy"),
                    },
                    "Lạm phát toàn phần và lạm phát cơ bản (so với cùng kỳ năm trước)",
                )
            with row1[1]:
                _macro_line_chart({"Lạm phát (MoM)": load_macro_indicator("cpi_mom")}, "Tỷ lệ lạm phát (so với tháng trước)")

            row2 = st.columns(2)
            with row2[0]:
                _macro_line_chart({"Lương thực": load_macro_indicator("cpi_food")}, "Lạm phát lương thực (so với tháng trước)")
            with row2[1]:
                _macro_line_chart({"Giao thông": load_macro_indicator("cpi_transport")}, "CPI nhóm giao thông (so với tháng trước)")

            _macro_line_chart({"PPI": load_macro_indicator("ppi_yoy")}, "Chỉ số giá sản xuất công nghiệp (so với cùng kỳ năm trước)")

        with tab_biz:
            _wb_category_tab("Kinh doanh")

        with tab_trade:
            _wb_category_tab("Thương mại")

        with tab_labor:
            labor_rows = [
                _macro_summary_row("unemployment_rate", "Tỷ lệ thất nghiệp", freq="quarterly"),
                _macro_summary_row("underemployment_rate", "Tỷ lệ thiếu việc làm", freq="quarterly"),
                _macro_summary_row("labor_force", "Lực lượng lao động", unit=" triệu người", freq="quarterly"),
                _macro_summary_row("avg_income", "Thu nhập bình quân người lao động", unit=" triệu đồng/tháng", freq="quarterly"),
            ]
            st.dataframe(pd.DataFrame(labor_rows), hide_index=True, width="stretch")
            st.caption("Nguồn: Tổng cục Thống kê (nso.gov.vn) · Dữ liệu theo quý")

            row_labor = st.columns(2)
            with row_labor[0]:
                _macro_line_chart({"Thất nghiệp": load_macro_indicator("unemployment_rate"),
                                    "Thiếu việc làm": load_macro_indicator("underemployment_rate")},
                                   "Tỷ lệ thất nghiệp & thiếu việc làm (theo quý)")
            with row_labor[1]:
                _macro_line_chart({"Thu nhập bình quân": load_macro_indicator("avg_income")},
                                   "Thu nhập bình quân người lao động (theo quý)", unit=" triệu đồng/tháng")

            st.divider()
            st.caption("So sánh dài hạn (World Bank, theo năm):")
            _wb_category_tab("Lao động")

        with tab_money:
            _wb_category_tab("Tiền tệ")

        with tab_consumer:
            _wb_category_tab("Tiêu dùng")

        with tab_tax:
            _wb_category_tab("Thuế")

        with tab_rates:
            _wb_category_tab("Lãi suất")
