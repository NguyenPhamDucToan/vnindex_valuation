"""Streamlit dashboard — VNIndex Valuation Tool.

Run with:  streamlit run dashboard/app.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import select, func as sqlfunc

from models.database import get_session
from models.schema import Financial, Price, Company, Valuation, PinnedTicker
from valuation.inputs import compute_ttm, compute_fcff_ttm, prepare_dcf_inputs, build_quarter_history
from valuation.dcf import dcf_valuation, sensitivity_grid
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
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Page config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="VNIndex Valuation",
    page_icon="📈",
    layout="wide",
)

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


def compute_quality_score(roe_v, net_margin_v, profit_quality_v,
                           fcf_margin_v, current_ratio_v, debt_to_equity_v) -> float:
    """Composite quality score 0–100 from 6 metrics.

    Thresholds (maps raw value to 0-10 sub-score):
      ROE           0 % → 0,  25 %+ → 10
      Net margin    0 % → 0,  20 %+ → 10
      Profit quality 0 → 0,  1.5+  → 10
      FCF margin    0 % → 0,  15 %+ → 10
      Current ratio 0.5 → 0,  2.5+  → 10
      D/E           3.0+ → 0,  0     → 10
    """
    scores = []
    if roe_v is not None:
        scores.append(max(0.0, min(10.0, max(roe_v, 0.0) / 0.25 * 10.0)))
    if net_margin_v is not None:
        scores.append(max(0.0, min(10.0, max(net_margin_v, 0.0) / 0.20 * 10.0)))
    if profit_quality_v is not None:
        scores.append(max(0.0, min(10.0, max(profit_quality_v, 0.0) / 1.5 * 10.0)))
    if fcf_margin_v is not None:
        scores.append(max(0.0, min(10.0, max(fcf_margin_v, 0.0) / 0.15 * 10.0)))
    if current_ratio_v is not None:
        scores.append(max(0.0, min(10.0, (current_ratio_v - 0.5) / 2.0 * 10.0)))
    if debt_to_equity_v is not None:
        scores.append(max(0.0, min(10.0, (3.0 - min(debt_to_equity_v, 3.0)) / 3.0 * 10.0)))
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores) * 10.0, 1)


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

    # Auto-fetch if DB is sparse (< 200 rows in the last 2 years)
    cutoff = _date.today() - timedelta(days=730)
    recent = len(df_db[pd.to_datetime(df_db["date"]).dt.date >= cutoff]) if not df_db.empty else 0
    if recent < 200:
        try:
            start = incremental_start_date(ticker)
            df_fresh = fetch_prices(ticker, start, _date.today())
            if not df_fresh.empty:
                upsert_prices(ticker, df_fresh)
                df_db = _read_db()
        except Exception:
            pass

    return df_db


@st.cache_data(ttl=300)
def load_financials_q(ticker: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Q")
            .order_by(Financial.period.asc())
        ).scalars().all()
        cols = [c.key for c in Financial.__table__.columns]
        return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])


@st.cache_data(ttl=300)
def load_financials_y(ticker: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Y")
            .order_by(Financial.period.asc())
        ).scalars().all()
        cols = [c.key for c in Financial.__table__.columns]
        return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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
                "_avg_upside_raw": avg_upside if avg_upside is not None else -999,
            })

    return pd.DataFrame(records)


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
VIEWS = [
    "Company Analysis",
    "Valuation Screen",
    "Sector Analysis",
    "Undervalued Watchlist",
    "Portfolio Tracker",
]
view = st.sidebar.radio("View", VIEWS)


# ═══════════════════════════════════════════════════════════════
# VIEW 1 — COMPANY ANALYSIS
# ═══════════════════════════════════════════════════════════════
if view == "Company Analysis":

    _available = load_available_tickers()
    _default_idx = _available.index("VNM") if "VNM" in _available else 0
    ticker = st.sidebar.selectbox("Ticker", _available, index=_default_idx)

    prices_df   = load_prices(ticker)
    fin_q       = load_financials_q(ticker)
    ttm         = compute_ttm(ticker)
    dcf_result  = get_dcf(ticker)
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

    _last     = prices_df.iloc[-1]
    _prev_c   = float(prices_df["close"].iloc[-2]) * 1000 if len(prices_df) > 1 else current_price
    _high_d   = float(_last["high"]) * 1000
    _low_d    = float(_last["low"])  * 1000
    _chg      = current_price - _prev_c
    _chg_pct  = _chg / _prev_c * 100 if _prev_c else 0
    _chg_disp = f"+{_chg:,.0f}" if _chg >= 0 else f"{_chg:,.0f}"

    # Price band limit by exchange (HOSE ±7%, HNX ±10%, UPCOM ±15%)
    _exch_up = _co_exch.upper() if _co_exch else "HOSE"
    _band_pct = 0.10 if "HNX" in _exch_up else 0.15 if "UPCOM" in _exch_up or "UPC" in _exch_up else 0.07
    _ceil_p   = _prev_c * (1 + _band_pct)
    _floor_p  = _prev_c * (1 - _band_pct)

    if _prev_c and current_price >= _ceil_p * 0.9995:   # at ceiling (within 0.05% rounding)
        _cc  = "#a855f7"   # purple
        _cbg = "#4a1d96"
        _arrow = "▲"
    elif _chg > 0:
        _cc  = "#22c55e"   # green
        _cbg = "#166534"
        _arrow = "▲"
    elif _chg == 0:
        _cc  = "#eab308"   # yellow
        _cbg = "#713f12"
        _arrow = "—"
    elif _prev_c and current_price <= _floor_p * 1.0005:  # at floor
        _cc  = "#22d3ee"   # light blue / cyan
        _cbg = "#164e63"
        _arrow = "▼"
    else:
        _cc  = "#ef4444"   # red
        _cbg = "#7f1d1d"
        _arrow = "▼"

    _sh    = (ttm.get("shares_outstanding") or 0) if ttm else 0
    _eq    = (ttm.get("equity")             or 0) if ttm else 0
    _ni    = (ttm.get("net_income")         or 0) if ttm else 0
    _ebit_ = (ttm.get("ebit")               or 0) if ttm else 0
    _dep_  = (ttm.get("depreciation")       or 0) if ttm else 0
    _debt_ = (ttm.get("debt")               or 0) if ttm else 0
    _cash_ = (ttm.get("cash")               or 0) if ttm else 0

    _mcap     = round(current_price * _sh / 1e3) if _sh else None
    _bvps     = (_eq * 1e9 / (_sh * 1e6))        if _sh else None
    _eps_h    = (_ni * 1e9 / (_sh * 1e6))        if _sh else None
    _pe_h     = round(current_price / _eps_h, 1) if _eps_h and _eps_h > 0 else None
    _pb_h     = round(current_price / _bvps,   1) if _bvps  and _bvps  > 0 else None
    _ebitda_h = _ebit_ + _dep_
    _ev_h     = (_mcap or 0) + _debt_ - _cash_
    _evebitda = round(_ev_h / _ebitda_h, 1)      if _ebitda_h > 0 else None
    _avgvol   = prices_df["volume"].tail(15).mean() / 1000 if len(prices_df) >= 15 else None
    _rng_pct  = round((current_price - _low_d) / (_high_d - _low_d) * 100) if _high_d > _low_d else 50
    _rng_pct  = max(2, min(98, _rng_pct))  # keep dot inside bar

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

    _header = (
        f'<div style="padding:22px 0px;margin-bottom:18px;display:flex;gap:28px;align-items:center;">'

        # LEFT — ticker + name
        f'<div style="min-width:230px;padding-right:28px;">'
        f'<div style="font-size:33px;font-weight:800;color:#f9fafb;line-height:1.2;">'
        f'{ticker}'
        f'<span style="font-size:16px;background:#1e3a5f;color:#60a5fa;padding:3px 10px;'
        f'border-radius:6px;margin-left:9px;vertical-align:middle;">{_co_exch}</span>'
        f'</div>'
        f'<div style="font-size:18px;color:#9ca3af;margin-top:8px;">{_co_name}</div>'
        f'</div>'

        # CENTER — price + change + day range
        f'<div style="min-width:300px;padding-right:28px;">'
        f'<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-size:45px;font-weight:800;color:#f9fafb;">{current_price:,.0f}</span>'
        f'<span style="font-size:22px;color:{_cc};font-weight:600;">{_chg_disp}</span>'
        f'<span style="font-size:18px;background:{_cbg};color:{_cc};padding:3px 12px;'
        f'border-radius:8px;font-weight:600;">{_arrow}{abs(_chg_pct):.2f}%</span>'
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
        + _mc("Book Value (bn)", _hv(_eq))
        + _mc("P/E", _hf(_pe_h, 1, "x"), accent=True)
        + _mc("Avg Vol 15D (K)", _hv(_avgvol))
        + _mc("EPS (VND)", _hv(_eps_h))
        + _mc("P/B", _hf(_pb_h, 1, "x"), accent=True)
        + _mc("Shares (M)", _hv(_sh))
        + _mc("EV/EBITDA", _hf(_evebitda, 1, "x"))
        + _mc("Sector", _co_sect)
        + f'</div>'

        f'</div>'
    )
    st.markdown(_header, unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #2d3748;margin:0 0 8px 0;">', unsafe_allow_html=True)

    # ── Price chart ────────────────────────────────────────────
    col_chart, col_dcf = st.columns([5, 2])

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
            ("fcfe",   "#00bcd4", "Cash Flow to Equity",         "dash",  1),
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

    # ── Valuation panel ────────────────────────────────────────
    with col_dcf:
        st.subheader("Valuation Estimates")

        def _upside_delta(price_val):
            if not price_val or not current_price:
                return None, None
            u = (price_val - current_price) / current_price
            return f"{'+'if u>0 else ''}{u*100:.1f}% vs market", "normal"

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
                # RE: EV/EBITDA ×15, P/Revenue ×3.5 (lumpy project revenues)
                _ebitda_re = (ttm.get("ebit") or 0) + (ttm.get("depreciation") or 0)
                _nd_re = (ttm.get("debt") or 0) - (ttm.get("cash") or 0)
                if _ebitda_re > 0 and _sh_s > 0:
                    _ev_re = _ebitda_re * 15 - _nd_re
                    v["ev_ebitda"] = round(_ev_re * 1e9 / (_sh_s * 1e6)) if _ev_re > 0 else None
                v["ps"] = _ps_val(ttm.get("revenue"), 3.5)

            elif _co_sect == "Chứng khoán":
                # Securities: P/Revenue ×3, no EV/EBITDA
                v["ps"]        = _ps_val(ttm.get("revenue"), 3)
                v["ev_ebitda"] = None

            elif _co_sect == "Bảo hiểm":
                # Insurance: P/Revenue ×2, P/Book is primary method
                v["ps"]        = _ps_val(ttm.get("revenue"), 2)
                v["ev_ebitda"] = None

        # ── Build method label list (sector-aware) ─────────────────
        _IS_BANK   = (_co_sect == "Ngân hàng")
        _IS_RE     = (_co_sect == "Bất động sản")
        _IS_SEC    = (_co_sect == "Chứng khoán")
        _IS_INS    = (_co_sect == "Bảo hiểm")

        _ALL_METHODS = [
            ("dcf",       "DCF / FCFF",              "NOPAT-based DCF at WACC"),
            ("fcfe",      "Cash Flow to Equity",      "FCFE discounted at CoE"),
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
                "P/PPOP (×6)"     if _IS_BANK else "Earnings Power Value",
                "PPOP/share × 6×" if _IS_BANK else "NOPAT ÷ WACC, zero growth"),
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
        m1, m2 = st.columns(2)
        cols_cycle = [m1, m2]
        for i, (key, label, hint) in enumerate(_valid_methods):
            col = cols_cycle[i % 2]
            with col:
                d, dc = _upside_delta(v[key])
                st.metric(label, f"{v[key]:,.0f} ₫", delta=d, delta_color=dc, help=hint)

        valid_prices = [v[k] for k, *_ in _valid_methods if v.get(k) and v[k] > 0]
        if valid_prices:
            avg_val = sum(valid_prices) / len(valid_prices)
            d, dc = _upside_delta(avg_val)
            st.markdown("---")
            st.metric(
                f"Avg of {len(valid_prices)} Estimates",
                f"{avg_val:,.0f} ₫",
                delta=d, delta_color=dc,
                help="Simple average of all valid method estimates",
            )

        if dcf_result:
            inp = dcf_result["inputs"]
            st.markdown(f"""
| Parameter | Value |
|---|---|
| WACC | {dcf_result['wacc']*100:.2f}% |
| FCFF growth | {inp['fcff_growth_rate']*100:.1f}% ({inp['growth_source']}) |
| FCFF base (TTM) | {inp['fcff_base']:,.0f} bn |
| Net debt | {inp['net_debt_bn']:,.0f} bn |
| Enterprise value | {dcf_result['enterprise_value']:,.0f} bn |
| Equity value | {dcf_result['equity_value']:,.0f} bn |
""")

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
                       hovertemplate="%{y:,.0f} bn<extra></extra>"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=labels, y=yoy, name="YoY Growth %", mode="lines+markers",
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
            fig.update_yaxes(title_text="bn VND", secondary_y=False)
            fig.update_yaxes(title_text="YoY %", secondary_y=True, showgrid=False)
            return fig

        # Chart 1 — Doanh Thu
        fig_rev = _dual_bar(
            df_q["revenue"].tolist(), rev_yoy,
            bar_color="#5b9bd5", neg_color="#d62728",
            title="Revenue", bar_name="Revenue", yoy_color="#e07b39",
        )

        # Chart 2 — Lợi Nhuận Sau Thuế
        fig_ni = _dual_bar(
            df_q["net_income"].tolist(), ni_yoy,
            bar_color="#2ca02c", neg_color="#d62728",
            title="Net Profit After Tax", bar_name="Net Profit", yoy_color="#f5c518",
        )

        # Chart 3 — Biên Lợi Nhuận
        fig_mg = go.Figure()
        for series, name, color in [
            (gm_pct,  "Gross Margin",   "#f5c518"),
            (em_pct,  "EBITDA Margin",  "#e07b39"),
            (nm_pct,  "Net Margin",     "#2ca02c"),
        ]:
            if any(v is not None for v in series):
                fig_mg.add_trace(go.Scatter(
                    x=labels, y=series, name=name, mode="lines",
                    line=dict(color=color, width=2),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ))
        fig_mg.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
        fig_mg.update_layout(
            title="Profit Margins (%)", height=_CHART_H, margin=_CHART_M,
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
            fig_b1.add_trace(go.Bar(x=labels, y=ii_v, name="Interest Income",
                marker_color="#5b9bd5", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                secondary_y=False)
            fig_b1.add_trace(go.Bar(x=labels, y=ie_v, name="Interest Expense",
                marker_color="#ef4444", opacity=0.85,
                hovertemplate="%{y:,.0f} bn<extra></extra>"),
                secondary_y=False)
            fig_b1.add_trace(go.Scatter(x=labels, y=nim_v, name="NIM % (ann.)",
                mode="lines+markers", line=dict(color="#f5c518", width=2),
                marker=dict(size=5), hovertemplate="NIM %{y:.2f}%<extra></extra>"),
                secondary_y=True)
            fig_b1.add_trace(go.Scatter(x=labels, y=cof_v, name="Cost of Funds %",
                mode="lines+markers", line=dict(color="#fb923c", width=2, dash="dash"),
                marker=dict(size=5), hovertemplate="CoF %{y:.2f}%<extra></extra>"),
                secondary_y=True)
            fig_b1.update_layout(title="Interest Income & Expense", height=_CHART_H,
                margin=_CHART_M, barmode="relative", legend=_LEG_LAYOUT,
                hovermode="x unified", dragmode=False)
            fig_b1.update_yaxes(title_text="bn VND", secondary_y=False)
            fig_b1.update_yaxes(title_text="%", secondary_y=True, showgrid=False)

            # Chart B2 — Customer Deposits + YoY Growth
            dep_v   = df_q["payables"].tolist()
            dep_yoy = _yoy_full("payables")

            fig_b2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_b2.add_trace(go.Bar(x=labels, y=dep_v, name="Customer Deposits",
                marker_color="#60a5fa", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                secondary_y=False)
            fig_b2.add_trace(go.Scatter(x=labels, y=dep_yoy, name="YoY Growth %",
                mode="lines+markers", line=dict(color="#f5c518", width=2),
                marker=dict(size=5), hovertemplate="%{y:.1f}%<extra></extra>"),
                secondary_y=True)
            fig_b2.add_hline(y=0, line_dash="dot", line_color="gray",
                             opacity=0.4, secondary_y=True)
            fig_b2.update_layout(title="Customer Deposits", height=_CHART_H,
                margin=_CHART_M, legend=_LEG_LAYOUT,
                hovermode="x unified", dragmode=False)
            fig_b2.update_yaxes(title_text="bn VND", secondary_y=False)
            fig_b2.update_yaxes(title_text="YoY %", secondary_y=True, showgrid=False)

            # Chart B3 — Deposit Structure: Customer vs Interbank+SBV (% of total)
            cust_dep_v  = df_q["payables"].tolist()
            interbank_v = df_q["debt"].tolist()
            total_fund_v = [(c or 0) + (i or 0) for c, i in zip(cust_dep_v, interbank_v)]
            cust_pct_v      = [_pct(c, t) for c, t in zip(cust_dep_v, total_fund_v)]
            interbank_pct_v = [_pct(i, t) for i, t in zip(interbank_v, total_fund_v)]

            fig_b3 = go.Figure()
            fig_b3.add_trace(go.Bar(x=labels, y=cust_pct_v, name="Customer Deposits",
                marker_color="#60a5fa", hovertemplate="%{y:.1f}%<extra></extra>"))
            fig_b3.add_trace(go.Bar(x=labels, y=interbank_pct_v, name="Interbank & SBV",
                marker_color="#1e3a5f", hovertemplate="%{y:.1f}%<extra></extra>"))
            fig_b3.update_layout(title="Deposit Structure", height=_CHART_H,
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
                go.Bar(x=labels, y=ebit_vals, name="Operating Profit (EBIT)",
                       marker_color="#4472c4",
                       hovertemplate="%{y:,.0f} bn<extra></extra>"),
                secondary_y=False,
            )
            fig4.add_trace(
                go.Bar(x=labels, y=int_vals, name="Interest Expense (−)",
                       marker_color="#ffc000",
                       hovertemplate="%{y:,.0f} bn<extra></extra>"),
                secondary_y=False,
            )
            fig4.add_trace(
                go.Scatter(x=labels, y=ebit_yoy, name="EBIT YoY %",
                           mode="lines", line=dict(color="#c00000", width=2),
                           hovertemplate="%{y:.1f}%<extra></extra>"),
                secondary_y=True,
            )
            fig4.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=True)
            fig4.update_layout(
                title="Pre-tax Profit Structure", height=_CHART_H, margin=_CHART_M,
                barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified",
                dragmode=False,
            )
            fig4.update_yaxes(title_text="bn VND", secondary_y=False)
            fig4.update_yaxes(title_text="YoY %", secondary_y=True, showgrid=False)

            st.plotly_chart(fig4, width="stretch")

        # Chart 5 — SG&A Expenses (stacked: selling + G&A)
        with r2c2:
            # Stored as negatives → abs for display
            sell_vals = [abs(v) if v else 0 for v in df_q["selling_expense"].tolist()]
            ga_vals   = [abs(v) if v else 0 for v in df_q["ga_expense"].tolist()]

            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=labels, y=ga_vals, name="G&A Expenses",
                marker_color="#7030a0",
                hovertemplate="%{y:,.0f} bn<extra></extra>",
            ))
            fig5.add_trace(go.Bar(
                x=labels, y=sell_vals, name="Selling Expenses",
                marker_color="#b4a0e0",
                hovertemplate="%{y:,.0f} bn<extra></extra>",
            ))
            fig5.update_layout(
                title="SG&A Expenses", height=_CHART_H, margin=_CHART_M,
                barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified",
                yaxis_title="bn VND", dragmode=False,
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
                fig6.add_trace(go.Bar(x=labels, y=prov_v, name="Provisions",
                    marker_color="#ef4444", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig6.add_trace(go.Scatter(x=labels, y=cc_pct, name="Credit Cost % (ann.)",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="%{y:.2f}%<extra></extra>"),
                    secondary_y=True)
                fig6.update_layout(title="Credit Cost & Provisions", height=_CHART_H,
                    margin=_CHART_M, legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig6.update_yaxes(title_text="bn VND", secondary_y=False)
                fig6.update_yaxes(title_text="Credit Cost %", secondary_y=True, showgrid=False)
            else:
                capex_vals = df_q["capex"].tolist()
                dep_vals   = df_q["depreciation"].tolist()
                dep_ratio  = [(d / a * 100) if d and a and a > 0 else None
                              for d, a in zip(dep_vals, df_q["total_assets"].tolist())]
                fig6 = make_subplots(specs=[[{"secondary_y": True}]])
                fig6.add_trace(go.Bar(x=labels, y=capex_vals, name="CAPEX",
                    marker_color="#70ad47", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig6.add_trace(go.Bar(x=labels, y=dep_vals, name="Depreciation",
                    marker_color="#ffc000", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig6.add_trace(go.Scatter(x=labels, y=dep_ratio, name="Dep./Assets %",
                    mode="lines", fill="tozeroy", fillcolor="rgba(255,100,100,0.15)",
                    line=dict(color="rgba(255,100,100,0.6)", width=1.5),
                    hovertemplate="%{y:.2f}%<extra></extra>"), secondary_y=True)
                fig6.update_layout(title="CAPEX & Depreciation", height=_CHART_H,
                    margin=_CHART_M, barmode="group", legend=_LEG_LAYOUT,
                    hovermode="x unified", dragmode=False)
                fig6.update_yaxes(title_text="bn VND", secondary_y=False)
                fig6.update_yaxes(title_text="Dep./Assets %", secondary_y=True, showgrid=False)
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
                fig7b.add_trace(go.Bar(x=labels, y=nii_b, name="Net Interest Income",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig7b.add_trace(go.Bar(x=labels, y=non_ii, name="Non-Interest Income",
                    marker_color="#70ad47", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig7b.add_trace(go.Scatter(x=labels, y=nii_pct, name="NII %",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="%{y:.1f}%<extra></extra>"),
                    secondary_y=True)
                fig7b.update_layout(title="Income Mix", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig7b.update_yaxes(title_text="bn VND", secondary_y=False)
                fig7b.update_yaxes(title_text="NII %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig7b, width="stretch")

            with r3c2:
                opex_v = df_q["ga_expense"].tolist()
                ppop_v = df_q["ebit"].tolist()
                cir_b  = [_pct(o, rv) for o, rv in zip(opex_v, df_q["revenue"].tolist())]
                fig8b = make_subplots(specs=[[{"secondary_y": True}]])
                fig8b.add_trace(go.Bar(x=labels, y=opex_v, name="OPEX",
                    marker_color="#ef4444", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig8b.add_trace(go.Bar(x=labels, y=ppop_v, name="PPOP",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig8b.add_trace(go.Scatter(x=labels, y=cir_b, name="CIR %",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="CIR %{y:.1f}%<extra></extra>"),
                    secondary_y=True)
                fig8b.update_layout(title="OPEX & PPOP", height=_CHART_H, margin=_CHART_M,
                    barmode="group", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig8b.update_yaxes(title_text="bn VND", secondary_y=False)
                fig8b.update_yaxes(title_text="CIR %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig8b, width="stretch")

            with r3c3:
                roa_v  = [_pct((r.net_income or 0)*4, r.total_assets) for _, r in df_q.iterrows()]
                roe_v  = [_pct((r.net_income or 0)*4, r.equity)       for _, r in df_q.iterrows()]
                nim_v3 = [_pct((r.gross_profit or 0)*4, r.total_assets) for _, r in df_q.iterrows()]
                nm_v3  = [_pct(r.net_income, r.revenue)                  for _, r in df_q.iterrows()]
                fig9b  = go.Figure()
                for vals, name, color in [
                    (roa_v, "ROA % (ann.)", "#60a5fa"),
                    (roe_v, "ROE % (ann.)", "#f59e0b"),
                    (nim_v3,"NIM % (ann.)", "#22c55e"),
                    (nm_v3, "Net Margin %", "#c084fc"),
                ]:
                    if any(v is not None for v in vals):
                        fig9b.add_trace(go.Scatter(x=labels, y=vals, name=name,
                            mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                            hovertemplate="%{y:.2f}%<extra></extra>"))
                fig9b.update_layout(title="Profitability Ratios", height=_CHART_H,
                    margin=_CHART_M, yaxis_title="%", legend=_LEG_LAYOUT,
                    hovermode="x unified", dragmode=False)
                st.plotly_chart(fig9b, width="stretch")

        else:
            with r3c1:
                prov_st_rec = _ser(bal_raw, ["provision_for_doubtful_debts"],           periods_r3)
                prov_inv    = _ser(bal_raw, ["provision_for_decline_in_inventories"],    periods_r3)
                prov_lt_rec = _ser(bal_raw, ["provision_for_doubtful_lt_receivable"],    periods_r3)
                prov_lt_inv = _ser(bal_raw, ["provision_for_long_term_investments"],     periods_r3)
                fig7 = go.Figure()
                for vals, name, color in [
                    (prov_lt_rec, "LT Receivables Provision", "#f4a460"),
                    (prov_lt_inv, "LT Investment Provision",  "#ff8c00"),
                    (prov_inv,    "Inventory Provision",      "#ffd700"),
                    (prov_st_rec, "ST Receivables Provision", "#70ad47"),
                ]:
                    if any(v is not None for v in vals):
                        fig7.add_trace(go.Bar(x=labels, y=vals, name=name,
                            marker_color=color, hovertemplate="%{y:,.1f} bn<extra></extra>"))
                fig7.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig7.update_layout(title="Provisions", height=_CHART_H, margin=_CHART_M,
                    barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig7.update_yaxes(title_text="bn VND")
                st.plotly_chart(fig7, width="stretch")

            with r3c2:
                fin_income = _ser(inc_raw, ["financial_income"], periods_r3)
                import re as _re2
                _qpat = _re2.compile(r'^\d{4}-Q[1-4]$')
                _all_p = sorted([c for c in (inc_raw.columns if not inc_raw.empty else []) if _qpat.match(str(c))])
                _fi_full = [_rv(inc_raw, "financial_income", p) for p in _all_p]
                _fi_yoy_full = _yoy(pd.Series(_fi_full))
                _p2yoy = dict(zip(_all_p, _fi_yoy_full))
                fi_yoy = [_p2yoy.get(p) for p in periods_r3]
                fig8 = make_subplots(specs=[[{"secondary_y": True}]])
                fig8.add_trace(go.Bar(x=labels, y=fin_income, name="Financial Income",
                    marker_color="#0d6efd", hovertemplate="%{y:,.1f} bn<extra></extra>"),
                    secondary_y=False)
                if any(v is not None for v in fi_yoy):
                    fig8.add_trace(go.Scatter(x=labels, y=fi_yoy, name="YoY %",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:.1f}%<extra></extra>"), secondary_y=True)
                fig8.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
                fig8.update_layout(title="Financial Income", height=_CHART_H, margin=_CHART_M,
                    legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig8.update_yaxes(title_text="bn VND", secondary_y=False)
                fig8.update_yaxes(title_text="YoY %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig8, width="stretch")

            with r3c3:
                fin_exp_raw = _ser(inc_raw, ["financial_expenses"], periods_r3)
                int_exp_raw = _ser(inc_raw, ["interest_expenses"],  periods_r3)
                fin_exp_abs = [abs(v) if v is not None else None for v in fin_exp_raw]
                int_exp_abs = [abs(v) if v is not None else None for v in int_exp_raw]
                other_exp   = [round(fe - ie, 3) if (fe is not None and ie is not None) else fe
                               for fe, ie in zip(fin_exp_abs, int_exp_abs)]
                fig9 = make_subplots(specs=[[{"secondary_y": True}]])
                fig9.add_trace(go.Bar(x=labels, y=int_exp_abs, name="Interest Expense",
                    marker_color="#c00000", hovertemplate="%{y:,.1f} bn<extra></extra>"),
                    secondary_y=False)
                if any(v is not None and v > 0 for v in other_exp):
                    fig9.add_trace(go.Bar(x=labels, y=other_exp, name="Other Financial Costs",
                        marker_color="#4472c4", hovertemplate="%{y:,.1f} bn<extra></extra>"),
                        secondary_y=False)
                fig9.add_trace(go.Scatter(x=labels, y=fin_exp_abs, name="Total Financial Costs",
                    mode="lines", line=dict(color="#ffb3b3", width=2),
                    hovertemplate="%{y:,.1f} bn<extra></extra>"), secondary_y=True)
                fig9.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
                fig9.update_layout(title="Financial Costs", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig9.update_yaxes(title_text="bn VND", secondary_y=False)
                fig9.update_yaxes(title_text="bn VND (total)", secondary_y=True, showgrid=False)
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
                fig10b.add_trace(go.Bar(x=labels, y=cash_r4, name="Cash & SBV",
                    marker_color="#5bc0de", hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig10b.add_trace(go.Bar(x=labels, y=loans_r4, name="Loan Book",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig10b.add_trace(go.Bar(x=labels, y=other_r4, name="Securities & Other",
                    marker_color="#9467bd", hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig10b.update_layout(title="Asset Mix", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig10b.update_yaxes(title_text="bn VND")
                st.plotly_chart(fig10b, width="stretch")

            with r4c2:
                # Funding Structure: Deposits, Interbank, Equity, Other
                other_fund = [max(0, (ta or 0) - (d or 0) - (ib or 0) - (e or 0))
                              for ta, d, ib, e in zip(ta_r4, dep_r4, ib_r4, eq_r4)]
                fig11b = go.Figure()
                fig11b.add_trace(go.Bar(x=labels, y=dep_r4, name="Customer Deposits",
                    marker_color="#60a5fa", hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig11b.add_trace(go.Bar(x=labels, y=ib_r4, name="Interbank & SBV",
                    marker_color="#1e40af", hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig11b.add_trace(go.Bar(x=labels, y=eq_r4, name="Equity",
                    marker_color="#22c55e", hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig11b.add_trace(go.Bar(x=labels, y=other_fund, name="Other Liabilities",
                    marker_color="#6b7280", hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig11b.update_layout(title="Funding Structure", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig11b.update_yaxes(title_text="bn VND")
                st.plotly_chart(fig11b, width="stretch")

            with r4c3:
                # Liquidity: LDR, Loan/Asset, Cash/Deposit
                ldr_r4  = [_pct(l, d) for l, d in zip(loans_r4, dep_r4)]
                la_r4   = [_pct(l, ta) for l, ta in zip(loans_r4, ta_r4)]
                cd_r4   = [_pct(c, d)  for c, d  in zip(cash_r4, dep_r4)]
                fig12b = go.Figure()
                for vals, name, color in [
                    (ldr_r4, "LDR %",          "#f59e0b"),
                    (la_r4,  "Loan/Asset %",   "#60a5fa"),
                    (cd_r4,  "Cash/Deposit %", "#22c55e"),
                ]:
                    if any(v is not None for v in vals):
                        fig12b.add_trace(go.Scatter(x=labels, y=vals, name=name,
                            mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                            hovertemplate="%{y:.1f}%<extra></extra>"))
                fig12b.update_layout(title="Liquidity Ratios", height=_CHART_H, margin=_CHART_M,
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
                    (cash_v,     "Cash & Equivalents",   "#5bc0de"),
                    (recv_v,     "Receivables",          "#f0ad4e"),
                    (inv_v,      "Inventory",            "#5cb85c"),
                    (other_curr, "Other Current",        "#9b59b6"),
                    (non_curr,   "Non-current Assets",   "#e74c3c"),
                ]:
                    if any(v is not None and v > 0 for v in vals):
                        fig10.add_trace(go.Bar(x=labels, y=vals, name=name,
                            marker_color=color, hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig10.update_layout(title="Asset Structure", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig10.update_yaxes(title_text="bn VND")
                st.plotly_chart(fig10, width="stretch")

            with r4c2:
                fig11 = go.Figure()
                for vals, name, color in [
                    (equity_v,   "Equity",             "#2ca02c"),
                    (lt_borrow,  "LT Borrowings",      "#d62728"),
                    (st_borrow,  "ST Borrowings",      "#ff7f0e"),
                    (payable_v,  "Trade Payables",     "#1f77b4"),
                    (other_liab, "Other Liabilities",  "#9467bd"),
                ]:
                    if any(v is not None and v > 0 for v in vals):
                        fig11.add_trace(go.Bar(x=labels, y=vals, name=name,
                            marker_color=color, hovertemplate="%{y:,.0f} bn<extra></extra>"))
                fig11.update_layout(title="Capital Structure", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig11.update_yaxes(title_text="bn VND")
                st.plotly_chart(fig11, width="stretch")

            with r4c3:
                fig12 = make_subplots(specs=[[{"secondary_y": True}]])
                for vals, name, color in [
                    (cash_v, "Cash & Equivalents", "#5bc0de"),
                    (recv_v, "Receivables",        "#f0ad4e"),
                ]:
                    if any(v is not None for v in vals):
                        fig12.add_trace(go.Bar(x=labels, y=vals, name=name,
                            marker_color=color, hovertemplate="%{y:,.0f} bn<extra></extra>"),
                            secondary_y=False)
                if any(v is not None for v in cash_pct):
                    fig12.add_trace(go.Scatter(x=labels, y=cash_pct, name="Cash/Assets %",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:.1f}%<extra></extra>"), secondary_y=True)
                if any(v is not None for v in recv_pct):
                    fig12.add_trace(go.Scatter(x=labels, y=recv_pct, name="Recv/Assets %",
                        mode="lines", line=dict(color="#2c7bb6", width=2),
                        hovertemplate="%{y:.1f}%<extra></extra>"), secondary_y=True)
                fig12.update_layout(title="Liquid Assets", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig12.update_yaxes(title_text="bn VND", secondary_y=False)
                fig12.update_yaxes(title_text="%", secondary_y=True, showgrid=False)
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
                fig13b.add_trace(go.Bar(x=labels, y=loans_r5, name="Loan Book (net)",
                    marker_color="#5b9bd5", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig13b.add_trace(go.Scatter(x=labels, y=loans_yoy, name="YoY Growth %",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="%{y:.1f}%<extra></extra>"),
                    secondary_y=True)
                fig13b.add_hline(y=0, line_dash="dot", line_color="gray",
                                 opacity=0.4, secondary_y=True)
                fig13b.update_layout(title="Loan Book", height=_CHART_H, margin=_CHART_M,
                    legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig13b.update_yaxes(title_text="bn VND", secondary_y=False)
                fig13b.update_yaxes(title_text="YoY %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig13b, width="stretch")

            with r5c2:
                # Asset Quality: Provisions + Credit Cost %
                cc_r5  = [_pct((p or 0) * 4, l) for p, l in zip(prov_r5, loans_r5)]
                prv_r5 = [_pct(p, l) for p, l in zip(prov_r5, loans_r5)]
                fig14b = make_subplots(specs=[[{"secondary_y": True}]])
                fig14b.add_trace(go.Bar(x=labels, y=prov_r5, name="Provision (quarterly)",
                    marker_color="#ef4444", hovertemplate="%{y:,.0f} bn<extra></extra>"),
                    secondary_y=False)
                fig14b.add_trace(go.Scatter(x=labels, y=cc_r5, name="Credit Cost % (ann.)",
                    mode="lines+markers", line=dict(color="#f5c518", width=2),
                    marker=dict(size=5), hovertemplate="%{y:.2f}%<extra></extra>"),
                    secondary_y=True)
                fig14b.update_layout(title="Asset Quality", height=_CHART_H, margin=_CHART_M,
                    legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False)
                fig14b.update_yaxes(title_text="bn VND", secondary_y=False)
                fig14b.update_yaxes(title_text="Credit Cost %", secondary_y=True, showgrid=False)
                st.plotly_chart(fig14b, width="stretch")

            with r5c3:
                # Capital Ratios: Equity/Assets, Equity/Loans, ROE
                ea_r5  = [_pct(e, ta) for e, ta in zip(eq_r5, ta_r5)]
                el_r5  = [_pct(e, l)  for e, l  in zip(eq_r5, loans_r5)]
                roe_r5 = [_pct((r.net_income or 0) * 4, r.equity) for _, r in df_q.iterrows()]
                fig15b = go.Figure()
                for vals, name, color in [
                    (ea_r5,  "Equity/Assets %", "#60a5fa"),
                    (el_r5,  "Equity/Loans %",  "#f59e0b"),
                    (roe_r5, "ROE % (ann.)",    "#22c55e"),
                ]:
                    if any(v is not None for v in vals):
                        fig15b.add_trace(go.Scatter(x=labels, y=vals, name=name,
                            mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                            hovertemplate="%{y:.1f}%<extra></extra>"))
                fig15b.update_layout(title="Capital Ratios", height=_CHART_H, margin=_CHART_M,
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
                    (r_trade,      "Trade Receivables",      "#5bc0de"),
                    (r_other,      "Other ST Receivables",   "#f0ad4e"),
                    (r_lt,         "LT Receivables",         "#555555"),
                    (r_prov_st_neg,"ST Doubtful Provision",  "#d9534f"),
                    (r_prov_lt_neg,"LT Doubtful Provision",  "#e87c6e"),
                ]:
                    if any(v is not None and v != 0 for v in vals):
                        fig13.add_trace(go.Bar(
                            x=labels, y=vals, name=name, marker_color=color,
                            hovertemplate="%{y:,.0f} bn<extra></extra>",
                        ))
                if any(v != 0 for v in r_net):
                    fig13.add_trace(go.Scatter(
                        x=labels, y=r_net, name="Net Total",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:,.0f} bn<extra></extra>",
                    ))
                fig13.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig13.update_layout(
                    title="Receivables", height=_CHART_H, margin=_CHART_M,
                    barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified",
                    dragmode=False,
                )
                fig13.update_yaxes(title_text="bn VND")
                st.plotly_chart(fig13, width="stretch")

            # Chart 14 — Inventory Structure
            with r5c2:
                inv_gross = _ser(bal_raw, ["inventories"],                        periods_r3)
                inv_net   = _ser(bal_raw, ["inventories_net"],                    periods_r3)
                inv_prov  = _ser(bal_raw, ["provision_for_decline_in_inventories"],periods_r3)
                inv_prov_neg = [-(abs(v)) if v is not None and v != 0 else v for v in inv_prov]
                inv_pct = [
                    round(n / t * 100, 2) if n and t and t > 0 else None
                    for n, t in zip(inv_net, total_v)
                ]

                fig14 = make_subplots(specs=[[{"secondary_y": True}]])
                if any(v is not None and v > 0 for v in inv_gross):
                    fig14.add_trace(go.Bar(
                        x=labels, y=inv_gross, name="Gross Inventory",
                        marker_color="#f0ad4e",
                        hovertemplate="%{y:,.0f} bn<extra></extra>",
                    ), secondary_y=False)
                elif any(v is not None and v > 0 for v in inv_net):
                    fig14.add_trace(go.Bar(
                        x=labels, y=inv_net, name="Net Inventory",
                        marker_color="#f0ad4e",
                        hovertemplate="%{y:,.0f} bn<extra></extra>",
                    ), secondary_y=False)
                if any(v is not None and v != 0 for v in inv_prov_neg):
                    fig14.add_trace(go.Bar(
                        x=labels, y=inv_prov_neg, name="Inventory Provision",
                        marker_color="#d9534f",
                        hovertemplate="%{y:,.0f} bn<extra></extra>",
                    ), secondary_y=False)
                if any(v is not None for v in inv_pct):
                    fig14.add_trace(go.Scatter(
                        x=labels, y=inv_pct, name="Inventory/Assets %",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:.1f}%<extra></extra>",
                    ), secondary_y=True)
                fig14.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
                fig14.update_layout(
                    title="Inventory", height=_CHART_H, margin=_CHART_M,
                    barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified",
                    dragmode=False,
                )
                fig14.update_yaxes(title_text="bn VND", secondary_y=False)
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
                    (d_st,    "ST Borrowings",      "#ff7f0e"),
                    (d_lt,    "LT Borrowings",      "#1f77b4"),
                    (d_bonds, "Convertible Bonds",  "#e74c3c"),
                    (d_other, "Other LT Payables",  "#9467bd"),
                ]:
                    if any(v is not None and v > 0 for v in vals):
                        fig15.add_trace(go.Bar(
                            x=labels, y=vals, name=name, marker_color=color,
                            hovertemplate="%{y:,.0f} bn<extra></extra>",
                        ), secondary_y=False)
                if any(v is not None for v in de_ratio):
                    fig15.add_trace(go.Scatter(
                        x=labels, y=de_ratio, name="D/E Ratio",
                        mode="lines", line=dict(color="#c00000", width=2),
                        hovertemplate="%{y:.2f}x<extra></extra>",
                    ), secondary_y=True)
                fig15.update_layout(
                    title="Financial Leverage", height=_CHART_H, margin=_CHART_M,
                    barmode="stack", legend=_LEG_LAYOUT, hovermode="x unified",
                    dragmode=False,
                )
                fig15.update_yaxes(title_text="bn VND", secondary_y=False)
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
                (ocf_v, "Operating CF",  "#2ca02c"),
                (icf_v, "Investing CF",  "#1f77b4"),
                (fcf_v, "Financing CF",  "#ffc000"),
            ]:
                if any(v is not None for v in vals):
                    fig16.add_trace(go.Bar(
                        x=labels, y=vals, name=name, marker_color=color,
                        hovertemplate="%{y:,.0f} bn<extra></extra>",
                    ), secondary_y=False)
            if any(v is not None for v in cash_v):
                fig16.add_trace(go.Scatter(
                    x=labels, y=cash_v, name="Cash (EOP)",
                    mode="lines", line=dict(color="#d62728", width=2),
                    hovertemplate="%{y:,.0f} bn<extra></extra>",
                ), secondary_y=True)
            fig16.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
            fig16.update_layout(
                title="Cash Flow", height=_CHART_H, margin=_CHART_M,
                barmode="relative", legend=_LEG_LAYOUT, hovermode="x unified",
                dragmode=False,
            )
            fig16.update_yaxes(title_text="bn VND", secondary_y=False)
            fig16.update_yaxes(title_text="Cash bn", secondary_y=True, showgrid=False)
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
                    x=ylabels, y=div_vals, name="Cash Dividends",
                    marker_color="#5bc0de",
                    hovertemplate="%{y:,.0f} bn<extra></extra>",
                ), secondary_y=False)
            if any(v is not None for v in payout):
                fig17.add_trace(go.Scatter(
                    x=ylabels, y=payout, name="Payout Ratio %",
                    mode="lines", line=dict(color="#c00000", width=2),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ), secondary_y=True)
            fig17.update_layout(
                title="Dividends (Annual)", height=_CHART_H, margin=_CHART_M,
                legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False,
            )
            fig17.update_yaxes(title_text="bn VND", secondary_y=False)
            fig17.update_yaxes(title_text="Payout %", secondary_y=True, showgrid=False)
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
                title="Valuation (P/E & P/B)", height=_CHART_H, margin=_CHART_M,
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
                    x=act_yrs, y=rev_act, name="Revenue",
                    marker_color="#1f77b4",
                    hovertemplate="%{y:,.0f} bn<extra></extra>",
                ), secondary_y=False)
                fig19.add_trace(go.Bar(
                    x=act_yrs, y=ni_act, name="Net Income",
                    marker_color="#aec7e8",
                    hovertemplate="%{y:,.0f} bn<extra></extra>",
                ), secondary_y=False)
                fig19.add_trace(go.Scatter(
                    x=act_yrs, y=mgn_act, name="Net Margin %",
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
                        x=proj_yrs, y=proj_rev, name="Revenue (F)",
                        marker=dict(color="#1f77b4", opacity=0.7,
                                    pattern=dict(shape="/", size=6, solidity=0.4)),
                        showlegend=False,
                        hovertemplate=f"%{{x}}: %{{y:,.0f}} bn (CAGR {_cr_str}%)<extra></extra>",
                    ), secondary_y=False)
                    fig19.add_trace(go.Bar(
                        x=proj_yrs, y=proj_ni, name="Net Income (F)",
                        marker=dict(color="#aec7e8", opacity=0.7,
                                    pattern=dict(shape="/", size=6, solidity=0.4)),
                        showlegend=False,
                        hovertemplate=f"%{{x}}: %{{y:,.0f}} bn (CAGR {_cn_str}%)<extra></extra>",
                    ), secondary_y=False)
                    fig19.add_trace(go.Scatter(
                        x=proj_yrs, y=proj_mgn, name="Net Margin % (F)",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                        marker=dict(size=5),
                        showlegend=False,
                        hovertemplate="%{y:.1f}%<extra></extra>",
                    ), secondary_y=True)

            fig19.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4,
                            secondary_y=False)
            fig19.update_layout(
                title="Business Projection", height=_CHART_H, margin=_CHART_M,
                barmode="group", legend=_LEG_LAYOUT, hovermode="x unified",
                dragmode=False,
            )
            fig19.update_yaxes(title_text="bn VND", secondary_y=False)
            fig19.update_yaxes(title_text="Net Margin %", secondary_y=True, showgrid=False)
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
                    name="Market Price", mode="lines",
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
                        name="Avg Intrinsic Value", mode="lines",
                        line=dict(color="#c00000", width=2, dash="dash"),
                        hovertemplate="%{y:,.0f} VND<extra></extra>",
                    ))
            fig20.update_layout(
                title="Price vs Intrinsic Value", height=_CHART_H, margin=_CHART_M,
                legend=_LEG_LAYOUT, hovermode="x unified", dragmode=False,
            )
            fig20.update_yaxes(title_text="Price (VND)")
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
                    yaxis_title="Index (base=100)",
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
                        name=f"Market {current_price:,.0f}",
                        hovertemplate=f"Market: {current_price:,.0f} VND<extra></extra>",
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
                        name=f"Avg Est {_avg21:,.0f}",
                        hovertemplate=f"Avg Est: {_avg21:,.0f} VND ({_avg_pct})<extra></extra>",
                    ))
                fig21.update_layout(
                    title="Valuation Estimates vs Market Price",
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
        st.subheader("TTM Key Ratios")
        r1, r2, r3, r4 = st.columns(4)

        gm  = gross_margin(ttm.get("gross_profit"), ttm.get("revenue"))
        nm  = net_margin(ttm.get("net_income"),     ttm.get("revenue"))
        om  = operating_margin(ttm.get("ebit"),     ttm.get("revenue"))
        roe_val = roe(ttm.get("net_income"),         ttm.get("equity"))
        roa_val = roa(ttm.get("net_income"),         ttm.get("total_assets"))
        cr  = current_ratio(ttm.get("current_assets"), ttm.get("current_liabilities"))
        de  = debt_to_equity(ttm.get("debt"),        ttm.get("equity"))
        pq  = profit_quality(ttm.get("operating_cf"), ttm.get("net_income"))
        fcfm = fcf_margin(ttm.get("fcf"),             ttm.get("revenue"))

        with r1:
            st.markdown("**Profitability**")
            st.write(f"Gross margin: {fmt_pct(gm)}")
            st.write(f"Operating margin: {fmt_pct(om)}")
            st.write(f"Net margin: {fmt_pct(nm)}")
        with r2:
            st.markdown("**Returns**")
            st.write(f"ROE: {fmt_pct(roe_val)}")
            st.write(f"ROA: {fmt_pct(roa_val)}")
        with r3:
            st.markdown("**Cash Flow Quality**")
            st.write(f"Profit quality (OCF/NI): {fmt_pct(pq)}")
            st.write(f"FCF margin: {fmt_pct(fcfm)}")
        with r4:
            st.markdown("**Balance Sheet**")
            st.write(f"Current ratio: {cr:.2f}x" if cr else "Current ratio: —")
            st.write(f"Debt/Equity: {de:.2f}x" if de else "Debt/Equity: —")

    # ── Sensitivity table ──────────────────────────────────────
    if dcf_result:
        st.divider()
        st.subheader("DCF Sensitivity (WACC × FCFF Growth)")
        inp = dcf_result["inputs"]
        grid = sensitivity_grid(
            fcff_base      = inp["fcff_base"],
            net_debt_bn    = inp["net_debt_bn"],
            shares_millions= inp["shares_millions"],
            wacc_range     = [0.10, 0.13, 0.146, 0.16, 0.18],
            growth_range   = [0.05, 0.10, inp["fcff_growth_rate"], 0.20, 0.30],
        )
        grid_df = pd.DataFrame(grid)
        pivot = grid_df.pivot(index="wacc", columns="growth", values="price")
        pivot.index   = [f"{w*100:.1f}%" for w in pivot.index]
        pivot.columns = [f"{g*100:.0f}%" for g in pivot.columns]
        pivot = pivot.map(lambda x: f"{x:,.0f}" if x is not None else "—")
        st.dataframe(pivot, width="stretch")


# ═══════════════════════════════════════════════════════════════
# VIEW 2 — VALUATION SCREEN
# ═══════════════════════════════════════════════════════════════
elif view == "Valuation Screen":
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

        min_roe = st.sidebar.slider("Min ROE (%)", -50, 50, 0, step=5)
        max_de  = st.sidebar.slider("Max D/E (x)", 0.0, 10.0, 10.0, step=0.5)
        min_upside_pct = st.sidebar.slider("Min Avg upside (%)", -500, 200, -500, step=10)
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
        _all_tickers = sorted(screen_df["Ticker"].tolist())
        _sel_tickers = st.multiselect(
            "Ticker", _all_tickers, default=[],
            placeholder="Filter by ticker...",
            key="screen_ticker_ms",
            label_visibility="collapsed",
        )
        if _sel_tickers:
            # When specific tickers are selected, bypass other filters for those tickers
            filtered = screen_df[screen_df["Ticker"].isin(_sel_tickers)]

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
        signals = []
        for u, q in zip(filtered["_avg_upside_raw"], filtered["_qs_raw"]):
            if   u >=  0.20 and q >= 60: signals.append(_SIG_LABELS["Strong Buy"])
            elif u >=  0.10 and q >= 45: signals.append(_SIG_LABELS["Buy"])
            elif u >=  0.00:             signals.append(_SIG_LABELS["Watch"])
            elif u >= -0.10:             signals.append(_SIG_LABELS["Neutral"])
            elif u >= -0.30:             signals.append(_SIG_LABELS["Reduce"])
            elif u >= -0.50:             signals.append(_SIG_LABELS["Sell"])
            else:                        signals.append(_SIG_LABELS["Strong Sell"])
        display["Signal"] = signals
        _signal_rank = {v: 6 - i for i, v in enumerate(_SIG_LABELS.values())}
        display["_sig_rank"] = [_signal_rank.get(s, 3) for s in signals]

        # Quality as integer for color bar
        display["Quality"] = [int(round(q)) for q in filtered["_qs_raw"]]

        # Column order — Avg Estimate first, then DCF
        ordered = ["Signal", "Sector", "Price (VND)",
                   "Avg Estimate", "Avg Upside",
                   "DCF Estimate", "Upside",
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
                height=900,
                margin=dict(l=0, r=0, t=0, b=0),
                dragmode=False,
            )
            st.plotly_chart(fig_hm, width="stretch")

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
# VIEW 4 — SCREENING & WATCHLIST
# ═══════════════════════════════════════════════════════════════
elif view == "Undervalued Watchlist":
    st.title("Screening & Watchlist")

    screen_df = load_valuation_screen_data()
    pinned    = get_pinned_tickers()

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
    f_min_upside = st.sidebar.slider("Min DCF Upside (%)", -100, 200,
                     st.session_state.get("f_min_upside", 0), step=5)
    f_max_pe  = st.sidebar.slider("Max P/E (×)",   0, 100,
                     st.session_state.get("f_max_pe", 50))
    f_max_pb  = st.sidebar.slider("Max P/B (×)",   0.0, 10.0,
                     float(st.session_state.get("f_max_pb", 5.0)), step=0.1)
    f_min_roe = st.sidebar.slider("Min ROE (%)",  -30, 50,
                     st.session_state.get("f_min_roe", 0))
    f_min_nm  = st.sidebar.slider("Min Net Margin (%)", -50, 50,
                     st.session_state.get("f_min_nm", 0))
    f_min_qs  = st.sidebar.slider("Min Quality Score", 0, 100,
                     st.session_state.get("f_min_qs", 0), step=5)
    f_pinned_only = st.sidebar.checkbox("Saved watchlist only", value=False)

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
    res = res[res["_upside_raw"]   >= f_min_upside / 100]
    res = res[res["_roe_raw"]      >= f_min_roe / 100]
    res = res[res["_nm_raw"]       >= f_min_nm  / 100]
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

    # ── Results ─────────────────────────────────────────────────
    tab_screen, tab_saved = st.tabs([
        f"Screen Results ({len(res)})",
        f"Saved Watchlist ({len(pinned)})",
    ])

    with tab_screen:
        if res.empty:
            st.info("No tickers match the current filters. Adjust filters in the sidebar.")
        else:
            st.caption(f"{len(res)} tickers match · Click a row then use buttons below to save/remove")

            res_disp = res.copy()
            # Star column before ticker — ★ gold (saved), ✩ dim (not saved)
            res_disp["·"] = res_disp["Ticker"].apply(lambda t: "★" if t in pinned else "")

            # Signal column with colors
            _sc_sigs = []
            for u, q in zip(res_disp["_avg_upside_raw"], res_disp["_qs_raw"]):
                if   u >=  0.20 and q >= 60: _sc_sigs.append(_SIG_LABELS["Strong Buy"])
                elif u >=  0.10 and q >= 45: _sc_sigs.append(_SIG_LABELS["Buy"])
                elif u >=  0.00:             _sc_sigs.append(_SIG_LABELS["Watch"])
                elif u >= -0.10:             _sc_sigs.append(_SIG_LABELS["Neutral"])
                elif u >= -0.30:             _sc_sigs.append(_SIG_LABELS["Reduce"])
                elif u >= -0.50:             _sc_sigs.append(_SIG_LABELS["Sell"])
                else:                        _sc_sigs.append(_SIG_LABELS["Strong Sell"])
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

            # Star column before Ticker, then rest
            _final_cols = ["·", "Ticker"] + display_cols
            _final_cols = [c for c in _final_cols if c in res_disp.columns]
            _styled_sc = (
                res_disp[_final_cols].reset_index(drop=True)
                .style
                .map(_sc_sig_col, subset=["Signal"])
                .map(_sc_q_col,   subset=["Quality"] if "Quality" in display_cols else [])
            )
            st.dataframe(_styled_sc, width="stretch", hide_index=True,
                         column_config={"·": st.column_config.TextColumn("★", width="small")})

            # Save/remove buttons
            save_col, rem_col, _ = st.columns([2, 2, 6])
            _ticker_input = st.text_input("Ticker to save/remove", placeholder="e.g. VNM",
                                          key="wl_ticker_input").upper().strip()
            if save_col.button("★ Save to watchlist", use_container_width=True):
                if _ticker_input:
                    pin_ticker(_ticker_input)
                    st.success(f"Saved {_ticker_input}")
                    st.cache_data.clear()
                    st.rerun()
            if rem_col.button("✕ Remove from watchlist", use_container_width=True):
                if _ticker_input:
                    unpin_ticker(_ticker_input)
                    st.info(f"Removed {_ticker_input}")
                    st.cache_data.clear()
                    st.rerun()

    with tab_saved:
        if not pinned:
            st.info("No tickers saved yet. Use the Screen tab to find and save tickers.")
        else:
            saved_df = screen_df[screen_df["Ticker"].isin(pinned)].copy()

            # Compute signals for saved tickers
            _saved_sigs = []
            for u, q in zip(saved_df["_avg_upside_raw"], saved_df["_qs_raw"]):
                if   u >=  0.20 and q >= 60: _saved_sigs.append(_SIG_LABELS["Strong Buy"])
                elif u >=  0.10 and q >= 45: _saved_sigs.append(_SIG_LABELS["Buy"])
                elif u >=  0.00:             _saved_sigs.append(_SIG_LABELS["Watch"])
                elif u >= -0.10:             _saved_sigs.append(_SIG_LABELS["Neutral"])
                elif u >= -0.30:             _saved_sigs.append(_SIG_LABELS["Reduce"])
                elif u >= -0.50:             _saved_sigs.append(_SIG_LABELS["Sell"])
                else:                        _saved_sigs.append(_SIG_LABELS["Strong Sell"])
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
# VIEW 5 — PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════
elif view == "Portfolio Tracker":
    st.title("Portfolio Tracker")
    st.caption("Track your holdings - enter each ticker and share count, get live value + upside.")

    # ── Holdings input ─────────────────────────────────────────
    st.subheader("Your Holdings")
    holdings_text = st.text_area(
        "Enter holdings (one per line: TICKER SHARES)",
        value=st.session_state.get("holdings_text", "VNM 1000\nFPT 500\nVIC 200"),
        height=150,
        placeholder="VNM 1000\nFPT 500\nHPG 2000",
    )
    st.session_state["holdings_text"] = holdings_text

    # Parse holdings
    holdings: dict[str, float] = {}
    parse_errors = []
    for line in holdings_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            parse_errors.append(f"Invalid line: '{line}' - expected 'TICKER SHARES'")
            continue
        tkr, shares_str = parts
        try:
            shares = float(shares_str.replace(",", ""))
            if shares > 0:
                holdings[tkr.upper()] = shares
        except ValueError:
            parse_errors.append(f"Invalid shares for '{tkr}': '{shares_str}'")

    for err in parse_errors:
        st.warning(err)

    if not holdings:
        st.info("Enter your holdings above to see portfolio analysis.")
        st.stop()

    # ── Load data for portfolio tickers ────────────────────────
    price_map = load_latest_prices()
    screen_df = load_valuation_screen_data()

    # Build a lookup from screen data (raw values by ticker)
    val_lookup: dict[str, dict] = {}
    if not screen_df.empty:
        for _, row in screen_df.iterrows():
            val_lookup[row["Ticker"]] = row.to_dict()

    # ── Build portfolio table ──────────────────────────────────
    portfolio_rows = []
    for tkr, shares in holdings.items():
        cur_price = price_map.get(tkr)
        val_data  = val_lookup.get(tkr, {})

        dcf_est_raw = None
        if val_data.get("DCF Estimate", "-") != "-":
            try:
                dcf_est_raw = float(val_data["DCF Estimate"].replace(",", ""))
            except (ValueError, AttributeError):
                pass

        market_value = cur_price * shares if cur_price else None
        dcf_value    = dcf_est_raw * shares if dcf_est_raw else None
        upside       = val_data.get("_upside_raw")
        if upside == -999:
            upside = None

        portfolio_rows.append({
            "Ticker":        tkr,
            "Shares":        f"{shares:,.0f}",
            "Price (VND)":   f"{cur_price:,.0f}" if cur_price else "-",
            "Market Value":  f"{market_value:,.0f}" if market_value else "-",
            "DCF Estimate":  val_data.get("DCF Estimate", "-"),
            "DCF Value":     f"{dcf_value:,.0f}" if dcf_value else "-",
            "DCF Upside":    val_data.get("Upside", "-"),
            "Quality":       val_data.get("Quality", "-"),
            "ROE":           val_data.get("ROE", "-"),
            "Net Margin":    val_data.get("Net Margin", "-"),
            "_market_value": market_value or 0,
            "_dcf_value":    dcf_value or 0,
            "_upside":       upside,
            "_shares":       shares,
        })

    port_df = pd.DataFrame(portfolio_rows)

    # ── Summary KPIs ───────────────────────────────────────────
    total_market = port_df["_market_value"].sum()
    total_dcf    = port_df["_dcf_value"].sum()
    n_tickers    = len(port_df)

    # Weighted average upside (weighted by market value)
    has_upside = port_df.dropna(subset=["_upside"])
    if not has_upside.empty and total_market > 0:
        weights = has_upside["_market_value"] / total_market
        wavg_upside = (has_upside["_upside"] * weights).sum()
    else:
        wavg_upside = None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Market Value", f"{total_market/1e9:,.1f} bn VND" if total_market else "-")
    k2.metric("Total DCF Value",    f"{total_dcf/1e9:,.1f} bn VND" if total_dcf else "-")

    if wavg_upside is not None:
        portfolio_upside_delta = total_dcf - total_market if total_market and total_dcf else None
        k3.metric(
            "Weighted Avg Upside",
            f"{wavg_upside*100:+.1f}%",
            delta=f"DCF value {portfolio_upside_delta/1e9:+,.1f} bn VND" if portfolio_upside_delta else None,
        )
    else:
        k3.metric("Weighted Avg Upside", "-")

    k4.metric("Holdings", f"{n_tickers} tickers, {port_df['_shares'].sum():,.0f} shares total")

    st.divider()

    # ── Holdings table ─────────────────────────────────────────
    display_cols = ["Ticker", "Shares", "Price (VND)", "Market Value",
                    "DCF Estimate", "DCF Value", "DCF Upside", "Quality", "ROE", "Net Margin"]
    raw_cols = [c for c in port_df.columns if c.startswith("_")]
    display = port_df.drop(columns=raw_cols)
    st.dataframe(display.set_index("Ticker"), width="stretch")

    # ── Allocation chart ──────────────────────────────────────
    if total_market > 0:
        st.subheader("Portfolio Allocation")
        alloc_df = port_df[port_df["_market_value"] > 0].copy()
        alloc_df["pct"] = alloc_df["_market_value"] / total_market * 100

        fig_pie = px.pie(
            alloc_df,
            names="Ticker",
            values="_market_value",
            hover_data={"pct": ":.1f"},
            hole=0.4,
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} VND<br>%{percent}<extra></extra>",
        )
        fig_pie.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_pie, width="stretch")

    # ── CSV export ─────────────────────────────────────────────
    csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download Portfolio CSV",
        data=csv_bytes,
        file_name="portfolio.csv",
        mime="text/csv",
    )
