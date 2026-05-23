"""Fetch income statement, balance sheet, and cash flow data per ticker.

All monetary values stored in VND billions.
Uses consolidated quarterly statements via vnstock v4.

vnstock v4 format:
  - Wide DataFrame: rows = line items, columns = periods ('2026-Q1', '2025-Q4', ...)
  - item_id column identifies each line item
  - Raw values in VND; divide by 1e9 to get VND billions
"""
from __future__ import annotations

import re
import pandas as pd
from loguru import logger
from sqlalchemy import select
from vnstock import Vnstock

from config import VNSTOCK_SOURCE
from models.database import get_session
from models.schema import Financial


_PERIOD_RE = re.compile(r'^\d{4}-Q[1-4]$')


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop ticker MultiIndex level; reset index so item_id stays as a column."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(ticker, level=0)
        except KeyError:
            df = df.droplevel(0)
    return df.reset_index(drop=True)


def _period_cols(df: pd.DataFrame) -> list[str]:
    """Return period columns sorted newest-first ('2026-Q1', '2025-Q4', ...)."""
    return sorted(
        [c for c in df.columns if _PERIOD_RE.match(str(c))],
        reverse=True,
    )


def _get(df: pd.DataFrame, item_id: str, period_col: str, scale: float = 1e9) -> float | None:
    """Extract value for item_id at period_col, divide by scale (default: 1e9 → VND billions)."""
    if df is None or df.empty or "item_id" not in df.columns or period_col not in df.columns:
        return None
    mask = df["item_id"] == item_id
    if not mask.any():
        return None
    try:
        raw = df.loc[mask, period_col].values[0]
        f = _safe_float(raw)
        return f / scale if f is not None else None
    except (IndexError, KeyError):
        return None


def fetch_financials(ticker: str, n_quarters: int = 8) -> pd.DataFrame:
    """Return last n_quarters of consolidated financials as a DataFrame.

    Columns match all fields in the Financial ORM model.
    All monetary amounts in VND billions; EPS in VND per share.
    """
    stock = Vnstock().stock(symbol=ticker, source=VNSTOCK_SOURCE)

    try:
        income   = stock.finance.income_statement(period="quarter", lang="en")
        balance  = stock.finance.balance_sheet(period="quarter", lang="en")
        cashflow = stock.finance.cash_flow(period="quarter", lang="en")
    except Exception as e:
        logger.warning(f"{ticker}: API error — {e}")
        return pd.DataFrame()

    if income is None or income.empty:
        return pd.DataFrame()

    income   = _normalize(income,   ticker)
    balance  = _normalize(balance,  ticker)
    cashflow = _normalize(cashflow, ticker)

    periods = _period_cols(income)[:n_quarters]
    if not periods:
        return pd.DataFrame()

    rows = []
    for p in periods:
        def gi(iid, s=1e9, _p=p): return _get(income,   iid, _p, s)
        def gb(iid, s=1e9, _p=p): return _get(balance,  iid, _p, s)
        def gc(iid, s=1e9, _p=p): return _get(cashflow, iid, _p, s)

        # --- Income statement ---
        revenue          = gi("net_sales")
        cogs_raw         = gi("cost_of_sales")
        cogs             = abs(cogs_raw) if cogs_raw is not None else None
        gross_profit     = gi("gross_profit")
        selling_expense  = gi("selling_expenses")
        ga_expense       = gi("general_and_admin_expenses")
        op_profit        = gi("operating_profit_loss")
        interest_raw     = gi("interest_expenses")
        interest_expense = abs(interest_raw) if interest_raw is not None else None
        # VAS line 30 (operating_profit_loss) already nets out interest; add it back for EBIT
        ebit             = (op_profit + interest_expense) if (
                               op_profit is not None and interest_expense is not None
                           ) else op_profit
        depreciation     = gc("depreciation_and_amortization")
        ebitda_val       = ((ebit or 0) + (depreciation or 0)) if (
                               ebit is not None and depreciation is not None
                           ) else None
        tax_raw          = gi("corporate_income_tax_expenses")
        tax_expense      = abs(tax_raw) if tax_raw is not None else None
        net_income       = gi("net_profit_loss_after_tax")
        eps              = gi("eps_basic_vnd", 1.0)  # already in VND/share, no scaling

        # --- Balance sheet ---
        total_assets        = gb("total_assets")
        current_assets      = gb("current_assets")
        current_liabilities = gb("current_liabilities")
        inventory           = gb("inventories_net")
        receivables         = gb("trade_accounts_receivable")
        payables            = gb("trade_accounts_payable")
        equity              = gb("owners_equity")
        cash                = gb("cash_and_cash_equivalents")
        retained_earnings   = gb("undistributed_earnings")
        st_debt             = gb("short_term_borrowings")
        lt_debt             = gb("long_term_borrowings")
        debt                = (st_debt or 0) + (lt_debt or 0) if (
                                  st_debt is not None or lt_debt is not None
                              ) else None
        # common_shares = total par value (10,000 VND/share)
        # after /1e9 → VND billions; billions / 10 → millions of shares
        common_shares_bn    = gb("common_shares")
        shares_outstanding  = common_shares_bn / 10 if common_shares_bn is not None else None

        # --- Cash flow ---
        operating_cf = gc("net_cash_inflows_outflows_from_operating_activities")
        capex_raw    = gc("purchases_of_fixed_assets_and_other_long_term_assets")
        capex        = abs(capex_raw) if capex_raw is not None else None
        investing_cf = gc("net_cash_inflows_outflows_from_investing_activities")
        financing_cf = gc("net_cash_inflows_outflows_from_financing_activities")
        fcf          = (operating_cf - capex) if (
                           operating_cf is not None and capex is not None
                       ) else None

        rows.append({
            "period":              p,
            "period_type":         "Q",
            "revenue":             revenue,
            "cogs":                cogs,
            "gross_profit":        gross_profit,
            "selling_expense":     selling_expense,
            "ga_expense":          ga_expense,
            "ebit":                ebit,
            "ebitda":              ebitda_val,
            "depreciation":        depreciation,
            "interest_expense":    interest_expense,
            "tax_expense":         tax_expense,
            "net_income":          net_income,
            "eps":                 eps,
            "total_assets":        total_assets,
            "current_assets":      current_assets,
            "current_liabilities": current_liabilities,
            "inventory":           inventory,
            "receivables":         receivables,
            "payables":            payables,
            "equity":              equity,
            "debt":                debt,
            "cash":                cash,
            "retained_earnings":   retained_earnings,
            "shares_outstanding":  shares_outstanding,
            "operating_cf":        operating_cf,
            "capex":               capex,
            "fcf":                 fcf,
            "investing_cf":        investing_cf,
            "financing_cf":        financing_cf,
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
