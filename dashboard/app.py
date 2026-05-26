"""Streamlit dashboard — VNIndex Valuation Tool.

Run with:  streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import select, func as sqlfunc

from models.database import get_session
from models.schema import Financial, Price, Company, Valuation
from valuation.inputs import compute_ttm, compute_fcff_ttm, prepare_dcf_inputs, build_quarter_history
from valuation.dcf import dcf_valuation, sensitivity_grid
from valuation.graham import graham_number, bvps_from_financials
from valuation.wacc import cost_of_equity, DEFAULT_BETA, DEFAULT_COD
from config import MARKET_PE
from valuation.ratios import (
    gross_margin, net_margin, operating_margin, roe, roa,
    current_ratio, debt_to_equity, profit_quality, fcf_margin,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VNIndex Valuation",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# Data helpers (cached so they don't re-query on every rerun)
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_prices(ticker: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(
            select(Price)
            .where(Price.ticker == ticker)
            .order_by(Price.date.asc())
        ).scalars().all()
        return pd.DataFrame([{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in rows])


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
    """Compute all 4 intrinsic price estimates for a ticker.

    Returns dict with keys: dcf, fcfe, graham, pe  (all in VND/share, or None).
    """
    ttm    = compute_ttm(ticker)
    dcf_r  = get_dcf(ticker)

    # 1. DCF / FCFF (existing)
    dcf_price = dcf_r["price_per_share"] if dcf_r else None

    if ttm is None:
        return {"dcf": dcf_price, "fcfe": None, "graham": None, "pe": None}

    shares = ttm.get("shares_outstanding") or 0
    equity = ttm.get("equity")
    net_income = ttm.get("net_income")

    # 2. FCFE price — operating CF already nets out interest in VAS, so:
    #    FCFE ≈ operating_cf − capex  (levered free cash flow to equity)
    #    Discount at cost of equity; no net-debt subtraction (it's already equity CF)
    fcfe_price = None
    fcfe_val = ttm.get("fcf")   # operating_cf - capex, stored in TTM
    if fcfe_val and fcfe_val > 0 and shares > 0:
        coe = cost_of_equity(DEFAULT_BETA)
        g   = (dcf_r["inputs"]["fcff_growth_rate"] if dcf_r else 0.12)
        res = dcf_valuation(
            fcff_base=fcfe_val, net_debt_bn=0,
            shares_millions=shares,
            fcff_growth_rate=g,
            wacc_override=coe,
        )
        fcfe_price = res.get("price_per_share")

    # 3. Graham Number — sqrt(22.5 × EPS × BVPS)
    graham_price = None
    if equity and shares > 0:
        bvps  = bvps_from_financials(equity, shares)
        # TTM EPS from net income (more accurate than summing quarterly EPS fields)
        eps   = (net_income * 1_000 / shares) if net_income and net_income > 0 else None
        graham_price = graham_number(eps, bvps) if eps else None

    # 4. P/E implied price — TTM EPS × HOSE market P/E
    pe_price = None
    if net_income and net_income > 0 and shares > 0:
        ttm_eps  = net_income * 1_000 / shares   # VND/share
        pe_price = ttm_eps * MARKET_PE

    return {"dcf": dcf_price, "fcfe": fcfe_price, "graham": graham_price, "pe": pe_price}


@st.cache_data(ttl=300)
def valuation_history(ticker: str) -> pd.DataFrame:
    """Compute intrinsic-value estimates at each quarterly TTM snapshot.

    For each quarter i (starting from index 3), builds a TTM from the
    4 quarters [i-3 … i] and runs all four valuation methods.
    Returns a DataFrame with columns: date, dcf, fcfe, graham, pe, avg.
    The date is the last day of the quarter (used to align with price chart).
    """
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

    # Use current growth-rate assumption consistently across history
    inputs = prepare_dcf_inputs(ticker)
    g = inputs["fcff_growth_rate"] if inputs else 0.12
    coe = cost_of_equity(DEFAULT_BETA)

    records = []
    for i in range(3, len(rows)):
        window = rows[i - 3: i + 1]

        # Build TTM (mirror of compute_ttm logic)
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

        # Quarter end date
        period = window[-1]["period"]           # e.g. "2025-Q3"
        yr, qn = int(period[:4]), int(period[6])
        mo = qn * 3
        dt = pd.Timestamp(f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}")

        shares = ttm.get("shares_outstanding") or 0
        equity = ttm.get("equity")
        net_income = ttm.get("net_income")

        # 1. DCF / FCFF
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

        # 2. FCFE
        fcfe_price = None
        fcfe_val = ttm.get("fcf")
        if fcfe_val and fcfe_val > 0 and shares > 0:
            res2 = dcf_valuation(fcff_base=fcfe_val, net_debt_bn=0,
                                 shares_millions=shares, fcff_growth_rate=g,
                                 wacc_override=coe)
            fcfe_price = res2.get("price_per_share")

        # 3. Graham
        graham_price = None
        if equity and shares > 0:
            bvps = bvps_from_financials(equity, shares)
            eps  = (net_income * 1_000 / shares) if net_income and net_income > 0 else None
            graham_price = graham_number(eps, bvps) if eps else None

        # 4. P/E
        pe_price = None
        if net_income and net_income > 0 and shares > 0:
            pe_price = (net_income * 1_000 / shares) * MARKET_PE

        valid = [p for p in [dcf_price, fcfe_price, graham_price, pe_price] if p and p > 0]
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
        from sqlalchemy import func as sqlfunc
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
    """Return pre-computed valuation rows joined with latest prices for all tickers.

    Reads from the valuations table (populated by collectors/compute_valuations.py).
    Falls back to empty DataFrame if no valuation rows exist.
    """
    with get_session() as s:
        # Latest calc_date per ticker subquery
        subq = (
            select(Valuation.ticker, sqlfunc.max(Valuation.calc_date).label("max_date"))
            .group_by(Valuation.ticker)
            .subquery()
        )
        # Join to get full row for latest date
        val_rows = s.execute(
            select(Valuation)
            .join(subq, (Valuation.ticker == subq.c.ticker) &
                         (Valuation.calc_date == subq.c.max_date))
            .order_by(Valuation.ticker)
        ).scalars().all()

        # Latest price per ticker
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
        price_map = {r[0]: r[1] * 1000 for r in price_rows if r[1]}  # thousands → VND

        records = []
        for v in val_rows:
            cur_price = price_map.get(v.ticker)
            upside = None
            if v.dcf_estimate and cur_price and cur_price > 0:
                upside = (v.dcf_estimate - cur_price) / cur_price
            records.append({
                "Ticker":          v.ticker,
                "Price (VND)":     f"{cur_price:,.0f}" if cur_price else "—",
                "DCF Estimate":    f"{v.dcf_estimate:,.0f}" if v.dcf_estimate else "—",
                "Upside":          f"{upside*100:+.1f}%" if upside is not None else "—",
                "Graham Number":   f"{v.graham_number:,.0f}" if v.graham_number else "—",
                "P/E":             f"{v.pe:.1f}x" if v.pe else "—",
                "P/B":             f"{v.pb:.1f}x" if v.pb else "—",
                "Net Margin":      fmt_pct(v.net_margin),
                "ROE":             fmt_pct(v.roe),
                "FCF Margin":      fmt_pct(v.fcf_margin),
                "D/E":             f"{v.debt_to_equity:.2f}x" if v.debt_to_equity else "—",
                "Current Ratio":   f"{v.current_ratio:.2f}x" if v.current_ratio else "—",
                "_upside_raw":     upside or -999,
            })

    return pd.DataFrame(records)


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


def fmt_bn(v) -> str:
    if v is None: return "—"
    return f"{v:,.0f} bn"


def fmt_pct(v) -> str:
    if v is None: return "—"
    return f"{v*100:.1f}%"


def fmt_vnd(v) -> str:
    if v is None: return "—"
    return f"{v:,.0f}"


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
VIEWS = ["Company Analysis", "Valuation Screen", "Undervalued Watchlist"]
view = st.sidebar.radio("View", VIEWS)

# ═══════════════════════════════════════════════════════════════
# VIEW 1 — COMPANY DRILLDOWN
# ═══════════════════════════════════════════════════════════════
if view == "Company Analysis":

    _available = load_available_tickers()
    _default_idx = _available.index("VNM") if "VNM" in _available else 0
    ticker = st.sidebar.selectbox("Ticker", _available, index=_default_idx)
    st.title(f"📊 {ticker} — Company Analysis")

    prices_df   = load_prices(ticker)
    fin_q       = load_financials_q(ticker)
    ttm         = compute_ttm(ticker)
    dcf_result  = get_dcf(ticker)
    valuations  = get_all_valuations(ticker)

    if prices_df.empty:
        st.warning("No price data in DB. Run: python collectors/prices.py")
        st.stop()

    current_price = float(prices_df["close"].iloc[-1]) * 1000  # VNM stored as thousands

    # ── KPI cards ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price", f"{current_price:,.0f} ₫")

    if ttm:
        c2.metric("TTM Revenue",    fmt_bn(ttm.get("revenue")))
        c3.metric("TTM Net Income", fmt_bn(ttm.get("net_income")))
        fcff_val = compute_fcff_ttm(ttm)
        c4.metric("TTM FCFF",       fmt_bn(fcff_val))
        c5.metric("Shares (M)",     f"{ttm.get('shares_outstanding', 0):,.0f}")

    st.divider()

    # ── Price chart ────────────────────────────────────────────
    col_chart, col_dcf = st.columns([3, 2])

    with col_chart:
        st.subheader("Price History (2 Years)")
        df1y = prices_df.tail(504).copy()
        df1y["close_vnd"] = df1y["close"] * 1000

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df1y["date"], y=df1y["close_vnd"],
            mode="lines", name="Price",
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
            hovertemplate="<b>Price</b>  %{y:,.0f}₫<extra></extra>",
            showlegend=False,
        ))
        vh = valuation_history(ticker)

        _val_series = [
            ("dcf",    "green",   "DCF Intrinsic Value",     "dash",  1),
            ("fcfe",   "#00bcd4", "Cash Flow to Equity",     "dash",  1),
            ("graham", "orange",  "Graham Number",           "dash",  1),
            ("pe",     "#9c27b0", f"P/E Implied (×{MARKET_PE})", "dash", 1),
            ("avg",    "red",     "Avg of Estimates",        "solid", 2),
        ]

        if not vh.empty:
            # Reindex quarterly snapshots to every trading day in the chart range,
            # then forward-fill so every pixel has a y value for hover.
            daily_idx = pd.to_datetime(df1y["date"])
            vh_daily = (
                vh.set_index(pd.to_datetime(vh["date"]))
                  .drop(columns=["date"])
                  .reindex(daily_idx)
                  .ffill()
                  .bfill()
            )

            for col, color, label, dash, width in _val_series:
                series = vh_daily[col].dropna()
                if series.empty:
                    continue
                fig_price.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    mode="lines",
                    line=dict(color=color, dash=dash, width=width),
                    name=label,
                    hovertemplate=f"<b>{label}</b>  %{{y:,.0f}}₫<extra></extra>",
                ))
        fig_price.update_layout(
            height=380, margin=dict(l=0, r=10, t=10, b=0),
            yaxis_title="VND", xaxis_title=None,
            hoverdistance=20,
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.15,
                xanchor="left", x=0,
                font=dict(size=11),
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ── Valuation panel ────────────────────────────────────────
    with col_dcf:
        st.subheader("Valuation Estimates")

        def _upside_delta(price_val):
            if not price_val or not current_price:
                return None, None
            u = (price_val - current_price) / current_price
            return f"{'+'if u>0 else ''}{u*100:.1f}% vs market", "normal"

        v = valuations
        m1, m2 = st.columns(2)

        with m1:
            if v["dcf"]:
                d, dc = _upside_delta(v["dcf"])
                st.metric("DCF Intrinsic Value", f"{v['dcf']:,.0f} ₫", delta=d, delta_color=dc)
            else:
                st.metric("DCF Intrinsic Value", "negative FCFF")

            if v["graham"]:
                d, dc = _upside_delta(v["graham"])
                st.metric("Graham Number", f"{v['graham']:,.0f} ₫", delta=d, delta_color=dc)
            else:
                st.metric("Graham Number", "—")

        with m2:
            if v["fcfe"]:
                d, dc = _upside_delta(v["fcfe"])
                st.metric("Cash Flow to Equity", f"{v['fcfe']:,.0f} ₫", delta=d, delta_color=dc)
            else:
                st.metric("Cash Flow to Equity", "negative FCFE")

            if v["pe"]:
                d, dc = _upside_delta(v["pe"])
                st.metric(f"P/E Implied (×{MARKET_PE})", f"{v['pe']:,.0f} ₫", delta=d, delta_color=dc)
            else:
                st.metric(f"P/E Implied (×{MARKET_PE})", "—")

        _panel_prices = [v[k] for k in ("dcf", "fcfe", "graham", "pe") if v.get(k) and v[k] > 0]
        if _panel_prices:
            avg_val = sum(_panel_prices) / len(_panel_prices)
            d, dc = _upside_delta(avg_val)
            st.markdown("---")
            st.metric("Avg of Estimates", f"{avg_val:,.0f} ₫", delta=d, delta_color=dc)

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

    # ── 8-quarter financial history ─────────────────────────────
    hist = build_quarter_history(ticker, n=8)

    if not hist.empty:
        st.subheader("Last 8 Quarters")

        def q_bar(col, title, solid_color, est_color):
            colors = [est_color if e else solid_color
                      for e in hist["is_estimated"]]
            labels = [p + " *" if e else p
                      for p, e in zip(hist["period"], hist["is_estimated"])]
            fig = go.Figure(go.Bar(x=labels, y=hist[col], marker_color=colors))
            fig.update_layout(title=title, height=280,
                              margin=dict(l=0, r=0, t=36, b=0),
                              xaxis_title=None, yaxis_title="bn VND")
            return fig

        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(q_bar("revenue",    "Revenue",    "#1f77b4", "#aec7e8"), use_container_width=True)
        with c2:
            st.plotly_chart(q_bar("net_income", "Net Income", "#2ca02c", "#98df8a"), use_container_width=True)
        with c3:
            # FCF: green/orange for positive, red for negative; lighter when estimated
            fcf_colors = []
            for v, est in zip(hist["fcf"].fillna(0), hist["is_estimated"]):
                if v < 0:
                    fcf_colors.append("#f4a8a8" if est else "#d62728")
                else:
                    fcf_colors.append("#fdd9b5" if est else "#ff7f0e")
            labels = [p + " *" if e else p
                      for p, e in zip(hist["period"], hist["is_estimated"])]
            fig_fcf = go.Figure(go.Bar(x=labels, y=hist["fcf"], marker_color=fcf_colors))
            fig_fcf.update_layout(title="FCF", height=280,
                                  margin=dict(l=0, r=0, t=36, b=0),
                                  xaxis_title=None, yaxis_title="bn VND")
            st.plotly_chart(fig_fcf, use_container_width=True)

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
        pivot = pivot.map(lambda v: f"{v:,.0f}" if v is not None else "—")
        st.dataframe(pivot, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# VIEW 2 — VALUATION SCREEN (comparison table)
# ═══════════════════════════════════════════════════════════════
elif view == "Valuation Screen":
    st.title("📋 Valuation Screen")

    screen_df = load_valuation_screen_data()

    if screen_df.empty:
        st.warning("No pre-computed valuation data. Run: `python -m collectors.compute_valuations`")
        st.info("If you haven't loaded tickers yet, run `python -m collectors.bulk_load` first.")
    else:
        n = len(screen_df)
        st.caption(f"{n} ticker{'s' if n != 1 else ''} with computed valuations. "
                   f"Sorted by DCF upside ↓")

        sort_col = st.selectbox(
            "Sort by",
            ["Upside (best first)", "Ticker (A–Z)", "ROE (best first)", "Net Margin (best first)"],
            label_visibility="collapsed",
        )

        display = screen_df.drop(columns=["_upside_raw"]).copy()
        if sort_col == "Ticker (A–Z)":
            display = display.sort_values("Ticker")
        elif sort_col == "ROE (best first)":
            display["_roe_sort"] = pd.to_numeric(
                display["ROE"].str.replace("%", "").str.replace("—", ""), errors="coerce"
            )
            display = display.sort_values("_roe_sort", ascending=False).drop(columns=["_roe_sort"])
        elif sort_col == "Net Margin (best first)":
            display["_nm_sort"] = pd.to_numeric(
                display["Net Margin"].str.replace("%", "").str.replace("—", ""), errors="coerce"
            )
            display = display.sort_values("_nm_sort", ascending=False).drop(columns=["_nm_sort"])
        else:
            display = display.sort_values(
                "Upside",
                key=lambda s: s.str.replace("+", "").str.replace("%", "").str.replace("—", "-999").astype(float),
                ascending=False,
            )

        st.dataframe(display.set_index("Ticker"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# VIEW 3 — UNDERVALUED WATCHLIST
# ═══════════════════════════════════════════════════════════════
elif view == "Undervalued Watchlist":
    st.title("🔍 Undervalued Watchlist")

    min_upside_pct = st.sidebar.slider("Min DCF upside (%)", 10, 100, 20, step=5)
    min_upside = min_upside_pct / 100

    watchlist = load_watchlist_data(min_upside=min_upside)

    st.caption(f"Screen: DCF upside > {min_upside_pct}%  ·  positive FCFF")

    if watchlist.empty:
        if load_valuation_screen_data().empty:
            st.warning("No pre-computed valuation data. Run: `python -m collectors.compute_valuations`")
        else:
            st.info(f"No tickers meet DCF upside > {min_upside_pct}% with positive FCFF.")
            st.caption("Try lowering the minimum upside threshold in the sidebar.")
    else:
        display = watchlist.drop(columns=["_upside_raw"])
        st.dataframe(display.set_index("Ticker"), use_container_width=True)
        st.success(f"Found {len(watchlist)} candidate{'s' if len(watchlist) != 1 else ''} "
                   f"with DCF upside > {min_upside_pct}% and positive FCFF.")
