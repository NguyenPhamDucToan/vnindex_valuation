"""Fetch income statement, balance sheet, and cash flow data per ticker.

All monetary values stored in VND billions.
Always uses consolidated statements (critical for FPT, VIC, VHM).
"""
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


def _get(df: pd.DataFrame, period_str: str, *keys) -> float | None:
    """Try multiple column name variants; return first match."""
    for k in keys:
        if k in df.columns and period_str in df.index:
            return _safe_float(df.loc[period_str, k])
    return None


def fetch_financials(ticker: str, n_quarters: int = 8) -> pd.DataFrame:
    """Return last n_quarters of consolidated financials as a DataFrame.

    Columns match all fields in the Financial ORM model.
    All monetary amounts in VND billions.
    """
    stock = Vnstock().stock(symbol=ticker, source=VNSTOCK_SOURCE)

    try:
        income = stock.finance.income_statement(period="quarter", lang="en")
        balance = stock.finance.balance_sheet(period="quarter", lang="en")
        cashflow = stock.finance.cash_flow(period="quarter", lang="en")
    except Exception as e:
        logger.warning(f"{ticker}: API error — {e}")
        return pd.DataFrame()

    if income is None or income.empty:
        return pd.DataFrame()

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(ticker, level=0)
            except KeyError:
                df = df.droplevel(0)
        df.index = df.index.astype(str)
        return df.iloc[:n_quarters]

    income = _normalize(income)
    balance = _normalize(balance)
    cashflow = _normalize(cashflow)

    rows = []
    for period_str in income.index:
        g_i = lambda *k: _get(income, period_str, *k)
        g_b = lambda *k: _get(balance, period_str, *k)
        g_c = lambda *k: _get(cashflow, period_str, *k)

        # --- Income statement ---
        revenue          = g_i("revenue", "Net Revenue", "net_revenue", "Revenue")
        cogs             = g_i("cost_of_goods_sold", "COGS", "cost_of_revenue", "Giá vốn hàng bán")
        gross_profit     = g_i("gross_profit", "Gross Profit") or (
                               (revenue or 0) - (cogs or 0) if revenue and cogs else None)
        selling_expense  = g_i("selling_expense", "Selling Expenses", "Chi phí bán hàng")
        ga_expense       = g_i("general_admin", "G&A", "admin_expense", "Chi phí QLDN")
        ebit             = g_i("ebit", "EBIT", "operating_profit", "Operating Income")
        depreciation     = g_i("depreciation", "Depreciation", "D&A", "Khấu hao")
        ebitda_val       = g_i("ebitda", "EBITDA") or (
                               (ebit or 0) + (depreciation or 0) if ebit is not None and depreciation else None)
        interest_expense = g_i("interest_expense", "Interest Expense", "Chi phí lãi vay")
        tax_expense      = g_i("income_tax", "Tax", "tax_expense", "Chi phí thuế TNDN")
        net_income       = g_i("net_income", "Net Income", "profit_after_tax", "PAT")
        eps              = g_i("eps", "EPS", "basic_eps")

        # --- Balance sheet ---
        total_assets        = g_b("total_assets", "Total Assets")
        current_assets      = g_b("current_assets", "Current Assets", "Short-term Assets")
        current_liabilities = g_b("current_liabilities", "Current Liabilities", "Short-term Liabilities")
        inventory           = g_b("inventory", "Inventories", "Hàng tồn kho")
        receivables         = g_b("receivables", "Accounts Receivable", "Trade Receivables")
        payables            = g_b("payables", "Accounts Payable", "Trade Payables")
        equity              = g_b("equity", "owner_equity", "Total Equity", "Shareholders Equity")
        debt                = g_b("debt", "total_debt", "Interest-bearing Debt")
        cash                = g_b("cash", "cash_and_equivalents", "Cash and Cash Equivalents")
        retained_earnings   = g_b("retained_earnings", "Undistributed Earnings", "Lợi nhuận chưa phân phối")
        shares_outstanding  = g_b("shares_outstanding", "Ordinary Shares", "shares")

        # --- Cash flow ---
        operating_cf  = g_c("operating_cf", "Net Cash from Operating", "CFO")
        capex_raw     = g_c("capex", "Purchase of Fixed Assets", "Capital Expenditure")
        capex         = abs(capex_raw) if capex_raw is not None else None
        investing_cf  = g_c("investing_cf", "Net Cash from Investing", "CFI")
        financing_cf  = g_c("financing_cf", "Net Cash from Financing", "CFF")

        ocf = operating_cf or 0.0
        cap = capex or 0.0
        fcf = ocf - cap

        rows.append({
            "period":            period_str,
            "period_type":       "Q",
            # income
            "revenue":           revenue,
            "cogs":              cogs,
            "gross_profit":      gross_profit,
            "selling_expense":   selling_expense,
            "ga_expense":        ga_expense,
            "ebit":              ebit,
            "ebitda":            ebitda_val,
            "depreciation":      depreciation,
            "interest_expense":  interest_expense,
            "tax_expense":       tax_expense,
            "net_income":        net_income,
            "eps":               eps,
            # balance sheet
            "total_assets":      total_assets,
            "current_assets":    current_assets,
            "current_liabilities": current_liabilities,
            "inventory":         inventory,
            "receivables":       receivables,
            "payables":          payables,
            "equity":            equity,
            "debt":              debt,
            "cash":              cash,
            "retained_earnings": retained_earnings,
            "shares_outstanding": shares_outstanding,
            # cash flow
            "operating_cf":      operating_cf,
            "capex":             capex,
            "fcf":               fcf if operating_cf is not None else None,
            "investing_cf":      investing_cf,
            "financing_cf":      financing_cf,
        })

    return pd.DataFrame(rows)


def upsert_financials(ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    meta_cols = {"period", "period_type"}
    value_cols = [c for c in df.columns if c not in meta_cols]
    upserted = 0

    with get_session() as session:
        existing = {
            (r[0], r[1])
            for r in session.execute(
                select(Financial.period, Financial.period_type)
                .where(Financial.ticker == ticker)
            )
        }
        for _, row in df.iterrows():
            key = (row["period"], row["period_type"])
            values = {c: row.get(c) for c in value_cols}
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
            logger.error(f"{ticker}: failed — {e}")


if __name__ == "__main__":
    run(["VNM", "FPT", "VIC"])
