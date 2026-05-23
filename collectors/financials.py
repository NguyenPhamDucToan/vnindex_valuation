"""Fetch income statement, balance sheet, and cash flow data per ticker."""
from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import select
from vnstock import Vnstock

from config import VNSTOCK_SOURCE
from models.database import get_session
from models.schema import Financial


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def fetch_financials(ticker: str, n_quarters: int = 8) -> pd.DataFrame:
    """Return a DataFrame with last n_quarters of consolidated financials.

    Columns: period, period_type, revenue, ebit, net_income, eps,
             total_assets, equity, debt, cash, operating_cf, capex, fcf,
             shares_outstanding
    All monetary amounts in VND billions.
    Always fetches consolidated statements (critical for FPT, VIC, VHM).
    """
    stock = Vnstock().stock(symbol=ticker, source=VNSTOCK_SOURCE)

    try:
        income = stock.finance.income_statement(period="quarter", lang="en")
        balance = stock.finance.balance_sheet(period="quarter", lang="en")
        cashflow = stock.finance.cash_flow(period="quarter", lang="en")
    except Exception as e:
        logger.warning(f"{ticker}: financial fetch error — {e}")
        return pd.DataFrame()

    if income is None or income.empty:
        return pd.DataFrame()

    # Normalise period index — vnstock returns a ticker-level MultiIndex or DatetimeIndex
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level=0) if ticker in df.index.get_level_values(0) else df.droplevel(0)
        df.index = df.index.astype(str)
        return df.iloc[:n_quarters]

    income = _normalize_index(income)
    balance = _normalize_index(balance)
    cashflow = _normalize_index(cashflow)

    rows = []
    for period_str in income.index:
        row: dict = {"period": period_str, "period_type": "Q"}

        def g(df: pd.DataFrame, *keys):
            for k in keys:
                if k in df.columns and period_str in df.index:
                    return _safe_float(df.loc[period_str, k])
            return None

        row["revenue"] = g(income, "revenue", "Net Revenue", "net_revenue")
        row["ebit"] = g(income, "ebit", "EBIT", "operating_profit")
        row["net_income"] = g(income, "net_income", "Net Income", "profit_after_tax")
        row["eps"] = g(income, "eps", "EPS", "basic_eps")
        row["total_assets"] = g(balance, "total_assets", "Total Assets")
        row["equity"] = g(balance, "equity", "owner_equity", "Total Equity")
        row["debt"] = g(balance, "debt", "short_term_borrowing", "long_term_debt")
        row["cash"] = g(balance, "cash", "cash_and_equivalents", "Cash and Equivalents")
        row["operating_cf"] = g(cashflow, "operating_cf", "Net Cash from Operating")
        row["capex"] = g(cashflow, "capex", "Purchase of Fixed Assets")
        ocf = row["operating_cf"] or 0.0
        cap = row["capex"] or 0.0
        row["fcf"] = ocf - abs(cap)
        row["shares_outstanding"] = g(balance, "shares_outstanding", "ordinary_shares")
        rows.append(row)

    return pd.DataFrame(rows)


def upsert_financials(ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    upserted = 0
    with get_session() as session:
        existing = {
            (r[0], r[1])
            for r in session.execute(
                select(Financial.period, Financial.period_type).where(Financial.ticker == ticker)
            )
        }
        for _, row in df.iterrows():
            key = (row["period"], row["period_type"])
            values = {k: row.get(k) for k in row.index if k not in ("period", "period_type")}
            if key in existing:
                session.execute(
                    Financial.__table__.update()
                    .where(
                        Financial.ticker == ticker,
                        Financial.period == row["period"],
                        Financial.period_type == row["period_type"],
                    )
                    .values(**values)
                )
            else:
                session.add(Financial(ticker=ticker, **row.to_dict()))
            upserted += 1
    return upserted


def run(tickers: list[str]) -> None:
    for ticker in tickers:
        try:
            df = fetch_financials(ticker)
            count = upsert_financials(ticker, df)
            logger.info(f"{ticker}: upserted {count} financial rows")
        except Exception as e:
            logger.error(f"{ticker}: financial upsert failed — {e}")


if __name__ == "__main__":
    run(["VNM", "FPT", "VIC"])
