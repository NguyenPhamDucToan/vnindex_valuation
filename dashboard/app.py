"""Streamlit dashboard — 4 views.

Run with: streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="VNIndex Valuation",
    page_icon="📈",
    layout="wide",
)

VIEWS = [
    "Market Overview",
    "Valuation Screen",
    "Company Drilldown",
    "Undervalued Watchlist",
]

view = st.sidebar.radio("View", VIEWS)

# ---------------------------------------------------------------------------
# Market Overview
# ---------------------------------------------------------------------------
if view == "Market Overview":
    st.title("Market Overview")
    st.info("Sector treemap (size = market cap, colour = P/E vs sector median) — coming Phase 6.")

# ---------------------------------------------------------------------------
# Valuation Screen
# ---------------------------------------------------------------------------
elif view == "Valuation Screen":
    st.title("Valuation Screen")
    st.info("Sortable table of ~400 companies with P/E, P/B, EV/EBITDA, Graham & DCF upside — coming Phase 6.")

# ---------------------------------------------------------------------------
# Company Drilldown
# ---------------------------------------------------------------------------
elif view == "Company Drilldown":
    st.title("Company Drilldown")
    ticker = st.text_input("Ticker", value="VNM").upper()
    st.info(f"Price chart, 8Q financials, valuation history and peer table for {ticker} — coming Phase 6.")

# ---------------------------------------------------------------------------
# Undervalued Watchlist
# ---------------------------------------------------------------------------
elif view == "Undervalued Watchlist":
    st.title("Undervalued Watchlist")
    st.info("Auto-screened: DCF upside > 20%, P/B < 2, positive FCF — coming Phase 6.")
