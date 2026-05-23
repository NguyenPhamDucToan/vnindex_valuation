"""Streamlit dashboard — VNIndex Valuation Tool.

Run with:  streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import select

from models.database import get_session
from models.schema import Financial, Price, Company
from valuation.inputs import compute_ttm, compute_fcff_ttm, prepare_dcf_inputs
from valuation.dcf import dcf_valuation, sensitivity_grid
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

KNOWN_TICKERS = ["VNM", "FPT", "VIC"]

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
VIEWS = ["Company Drilldown", "Valuation Screen", "Undervalued Watchlist"]
view = st.sidebar.radio("View", VIEWS)

# ═══════════════════════════════════════════════════════════════
# VIEW 1 — COMPANY DRILLDOWN
# ═══════════════════════════════════════════════════════════════
if view == "Company Drilldown":

    ticker = st.sidebar.selectbox("Ticker", KNOWN_TICKERS, index=0)
    st.title(f"📊 {ticker} — Company Drilldown")

    prices_df   = load_prices(ticker)
    fin_q       = load_financials_q(ticker)
    ttm         = compute_ttm(ticker)
    dcf_result  = get_dcf(ticker)

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
        st.subheader("Price History (1 Year)")
        df1y = prices_df.tail(252).copy()
        df1y["close_vnd"] = df1y["close"] * 1000

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df1y["date"], y=df1y["close_vnd"],
            mode="lines", name="Close",
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
        ))
        if dcf_result and dcf_result.get("price_per_share"):
            intrinsic = dcf_result["price_per_share"]
            fig_price.add_hline(
                y=intrinsic, line_dash="dash", line_color="green",
                annotation_text=f"DCF {intrinsic:,.0f}₫",
                annotation_position="right",
            )
        fig_price.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="VND", xaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ── DCF result panel ───────────────────────────────────────
    with col_dcf:
        st.subheader("DCF Valuation")
        if dcf_result is None:
            st.warning("Cannot compute DCF — negative or missing FCFF.")
        else:
            intrinsic = dcf_result["price_per_share"]
            upside = (intrinsic - current_price) / current_price
            color = "green" if upside > 0 else "red"
            sign  = "+" if upside > 0 else ""

            st.metric(
                "Intrinsic Value (DCF)",
                f"{intrinsic:,.0f} ₫",
                delta=f"{sign}{upside*100:.1f}% vs market",
                delta_color="normal",
            )
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

    # ── Quarterly bar charts ────────────────────────────────────
    if not fin_q.empty:
        st.subheader("Quarterly Financials")
        c1, c2, c3 = st.columns(3)

        def bar_chart(df, col, title, color):
            fig = px.bar(df, x="period", y=col, title=title,
                         color_discrete_sequence=[color])
            fig.update_layout(height=260, margin=dict(l=0,r=0,t=30,b=0),
                              showlegend=False, xaxis_title=None)
            return fig

        with c1:
            st.plotly_chart(bar_chart(fin_q, "revenue",    "Revenue (bn VND)",    "#1f77b4"), use_container_width=True)
        with c2:
            st.plotly_chart(bar_chart(fin_q, "net_income", "Net Income (bn VND)", "#2ca02c"), use_container_width=True)
        with c3:
            fin_q["fcf_display"] = fin_q["fcf"]
            colors = ["#d62728" if v < 0 else "#ff7f0e" for v in fin_q["fcf"].fillna(0)]
            fig_fcf = go.Figure(go.Bar(
                x=fin_q["period"], y=fin_q["fcf"],
                marker_color=colors,
            ))
            fig_fcf.update_layout(title="FCF (bn VND)", height=260,
                                  margin=dict(l=0,r=0,t=30,b=0))
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
    st.caption("Comparing loaded tickers: VNM · FPT · VIC")

    rows = []
    for t in KNOWN_TICKERS:
        prices_df = load_prices(t)
        ttm       = compute_ttm(t)
        dcf_res   = get_dcf(t)
        if prices_df.empty or ttm is None:
            continue

        price = float(prices_df["close"].iloc[-1]) * 1000
        intrinsic = dcf_res["price_per_share"] if dcf_res else None
        upside = (intrinsic - price) / price if intrinsic else None

        nm  = net_margin(ttm.get("net_income"), ttm.get("revenue"))
        roe_val = roe(ttm.get("net_income"), ttm.get("equity"))
        de  = debt_to_equity(ttm.get("debt"), ttm.get("equity"))
        pq  = profit_quality(ttm.get("operating_cf"), ttm.get("net_income"))

        rows.append({
            "Ticker":         t,
            "Price (VND)":    f"{price:,.0f}",
            "TTM Revenue":    fmt_bn(ttm.get("revenue")),
            "TTM Net Income": fmt_bn(ttm.get("net_income")),
            "Net Margin":     fmt_pct(nm),
            "ROE":            fmt_pct(roe_val),
            "D/E":            f"{de:.2f}x" if de else "—",
            "Profit Quality": fmt_pct(pq),
            "DCF Price":      f"{intrinsic:,.0f}" if intrinsic else "neg. FCFF",
            "Upside":         f"{upside*100:+.1f}%" if upside else "—",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)
    else:
        st.warning("No data in DB. Run the collectors first.")


# ═══════════════════════════════════════════════════════════════
# VIEW 3 — UNDERVALUED WATCHLIST
# ═══════════════════════════════════════════════════════════════
elif view == "Undervalued Watchlist":
    st.title("🔍 Undervalued Watchlist")
    st.caption("Screen: DCF upside > 20%  ·  positive FCFF")

    candidates = []
    for t in KNOWN_TICKERS:
        prices_df = load_prices(t)
        ttm       = compute_ttm(t)
        dcf_res   = get_dcf(t)
        if prices_df.empty or ttm is None or dcf_res is None:
            continue

        price     = float(prices_df["close"].iloc[-1]) * 1000
        intrinsic = dcf_res["price_per_share"]
        upside    = (intrinsic - price) / price

        if upside > 0.20 and intrinsic > 0:
            candidates.append({
                "Ticker":      t,
                "Price":       f"{price:,.0f}",
                "DCF Price":   f"{intrinsic:,.0f}",
                "Upside":      f"{upside*100:+.1f}%",
                "WACC":        f"{dcf_res['wacc']*100:.2f}%",
                "FCFF Growth": f"{dcf_res['inputs']['fcff_growth_rate']*100:.1f}%",
                "TTM FCF":     fmt_bn(ttm.get("fcf")),
            })

    if candidates:
        st.dataframe(pd.DataFrame(candidates).set_index("Ticker"), use_container_width=True)
        st.success(f"Found {len(candidates)} candidate(s) with DCF upside > 20%.")
    else:
        st.info("No tickers in the watchlist meet screening criteria with current data.")
        st.caption("Load more tickers via collectors/ticker_list.py + collectors/financials.py to see more.")
