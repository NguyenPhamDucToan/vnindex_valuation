"""Fetch income statement, balance sheet, and cash flow data per ticker.

All monetary values stored in VND billions.
Uses VCI Finance._get_report(limit=50) to bypass the community 4-quarter cap.
The VCI API returns full history (33+ quarters); vnstock's public methods slice to 4.

vnstock v4 format:
  - Wide DataFrame: rows = line items, columns = periods ('2026-Q1', '2025-Q4', ...)
  - item_id column identifies each line item
  - Raw values in VND; divide by 1e9 to get VND billions

Banking stocks use completely different item_ids (detected by presence of
'net_interest_income' in income statement). Field mapping for banks:
  revenue        ← total_operating_income  (NII + fee + trading + other)
  gross_profit   ← net_interest_income     (core spread income)
  cogs           ← provision_for_credit_losses
  ga_expense     ← general_and_admin_expenses
  ebit           ← net_operating_profit_before_allowance_for_credit_loss (PPOP)
  interest_expense ← interest_and_similar_expenses
  receivables    ← loans_and_advances_to_customers_net  (loan book)
  payables       ← deposits_from_customers
  cash           ← cash_and_precious_metals
  shares_outstanding ← charter_capital / 10  (same 10,000 VND par formula)
  debt           ← interbank borrowings + SBV loans
"""
from __future__ import annotations

import re
import warnings
import pandas as pd
from loguru import logger
from sqlalchemy import select

from models.database import get_session
from models.schema import Financial


_QUARTER_RE = re.compile(r'^\d{4}-Q[1-4]$')
_ANNUAL_RE  = re.compile(r'^\d{4}$')

_API_LIMIT = 50


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _period_cols(df: pd.DataFrame, freq: str = "quarter") -> list[str]:
    pattern = _QUARTER_RE if freq == "quarter" else _ANNUAL_RE
    return sorted(
        [c for c in df.columns if pattern.match(str(c))],
        reverse=True,
    )


def _get(df: pd.DataFrame, item_id: str, period_col: str, scale: float = 1e9) -> float | None:
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


def _fetch_vci(ticker: str, freq: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from vnstock.explorer.vci.financial import Finance
        fin = Finance(ticker, period=freq, show_log=False)
        income   = fin._get_report("income_statement", period=freq, lang="en", show_log=False, limit=_API_LIMIT)
        balance  = fin._get_report("balance_sheet",    period=freq, lang="en", show_log=False, limit=_API_LIMIT)
        cashflow = fin._get_report("cash_flow",        period=freq, lang="en", show_log=False, limit=_API_LIMIT)
    return income, balance, cashflow


def _is_bank(income: pd.DataFrame) -> bool:
    """True if ticker is a bank/credit institution."""
    if income is None or income.empty or "item_id" not in income.columns:
        return False
    ids = set(income["item_id"].tolist())
    return "net_interest_income" in ids and "net_sales" not in ids


def _is_insurance(income: pd.DataFrame) -> bool:
    """True if ticker is an insurance company."""
    if income is None or income.empty or "item_id" not in income.columns:
        return False
    ids = set(income["item_id"].tolist())
    return "gross_written_premium" in ids or "net_sales_from_insurance_business" in ids


def _extract_insurance(income, balance, cashflow, p: str) -> dict:
    """Map insurance VAS item_ids to the standard Financial schema fields."""
    def gi(iid, s=1e9): return _get(income,   iid, p, s)
    def gb(iid, s=1e9): return _get(balance,  iid, p, s)
    def gc(iid, s=1e9): return _get(cashflow, iid, p, s)

    # Revenue = gross written premium (total insurance premium)
    revenue      = gi("gross_written_premium") or gi("net_sales_from_insurance_business")
    net_income   = gi("profit_after_tax") or gi("net_profit_attributable_to_shareholders_of_the_group")
    ebit         = gi("profit_before_tax")
    ga_expense_r = gi("general_and_administrative_expenses")
    ga_expense   = abs(ga_expense_r) if ga_expense_r is not None else None
    eps          = gi("eps_basic_vnd", 1.0)
    tax_raw      = gi("corporate_income_tax_for_the_year")
    tax_expense  = abs(tax_raw) if tax_raw is not None else None

    total_assets    = gb("total_assets")
    equity          = gb("owners_equity")
    cash            = gb("cash_and_cash_equivalents") or gb("cash_and_cash_equivalents_at_end_of_the_period")
    retained_earn   = gb("undistributed_earnings") or gb("retained_earnings")
    # Use common_shares or paid_in_capital (both = par value, same formula as regular cos)
    common_sh_bn    = gb("common_shares") or gb("paid_in_capital") or gb("charter_capital")
    shares_out      = common_sh_bn / 10 if common_sh_bn is not None else None

    operating_cf = gc("net_cash_from_operating_activities") or gc("net_cash_inflows_outflows_from_operating_activities")
    capex_raw    = gc("purchases_of_fixed_assets_and_other_long_term_assets")
    capex        = abs(capex_raw) if capex_raw is not None else None
    investing_cf = gc("net_cash_from_investing_activities") or gc("net_cash_inflows_outflows_from_investing_activities")
    fcf          = (operating_cf - capex) if (operating_cf is not None and capex is not None) else None

    return {
        "revenue": revenue, "cogs": None, "gross_profit": None,
        "selling_expense": None, "ga_expense": ga_expense,
        "ebit": ebit, "ebitda": None, "depreciation": None,
        "interest_expense": None, "tax_expense": tax_expense,
        "net_income": net_income, "eps": eps,
        "total_assets": total_assets, "current_assets": None,
        "current_liabilities": None, "inventory": None,
        "receivables": None, "payables": None, "equity": equity,
        "debt": None, "cash": cash, "retained_earnings": retained_earn,
        "shares_outstanding": shares_out,
        "operating_cf": operating_cf, "capex": capex, "fcf": fcf,
        "investing_cf": investing_cf, "financing_cf": None,
    }


def _extract_regular(income, balance, cashflow, p: str) -> dict:
    def gi(iid, s=1e9): return _get(income,   iid, p, s)
    def gb(iid, s=1e9): return _get(balance,  iid, p, s)
    def gc(iid, s=1e9): return _get(cashflow, iid, p, s)

    revenue          = gi("net_sales")
    cogs_raw         = gi("cost_of_sales")
    cogs             = abs(cogs_raw) if cogs_raw is not None else None
    gross_profit     = gi("gross_profit")
    selling_expense  = gi("selling_expenses") or gi("selling_cost")   # securities use selling_cost
    ga_expense       = gi("general_and_admin_expenses") or gi("general_and_administrative_expenses")
    op_profit        = gi("operating_profit_loss")
    interest_raw     = gi("interest_expenses")
    interest_expense = abs(interest_raw) if interest_raw is not None else None
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
    eps              = gi("eps_basic_vnd", 1.0)

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
    common_shares_bn    = gb("common_shares")
    shares_outstanding  = common_shares_bn / 10 if common_shares_bn is not None else None

    operating_cf = gc("net_cash_inflows_outflows_from_operating_activities")
    capex_raw    = gc("purchases_of_fixed_assets_and_other_long_term_assets")
    capex        = abs(capex_raw) if capex_raw is not None else None
    investing_cf = gc("net_cash_inflows_outflows_from_investing_activities")
    financing_cf = gc("net_cash_inflows_outflows_from_financing_activities")
    fcf          = (operating_cf - capex) if (
                       operating_cf is not None and capex is not None
                   ) else None

    return {
        "revenue": revenue, "cogs": cogs, "gross_profit": gross_profit,
        "selling_expense": selling_expense, "ga_expense": ga_expense,
        "ebit": ebit, "ebitda": ebitda_val, "depreciation": depreciation,
        "interest_expense": interest_expense, "tax_expense": tax_expense,
        "net_income": net_income, "eps": eps,
        "total_assets": total_assets, "current_assets": current_assets,
        "current_liabilities": current_liabilities, "inventory": inventory,
        "receivables": receivables, "payables": payables, "equity": equity,
        "debt": debt, "cash": cash, "retained_earnings": retained_earnings,
        "shares_outstanding": shares_outstanding,
        "operating_cf": operating_cf, "capex": capex, "fcf": fcf,
        "investing_cf": investing_cf, "financing_cf": financing_cf,
    }


def _extract_bank(income, balance, cashflow, p: str) -> dict:
    """Map bank VAS item_ids to the standard Financial schema fields."""
    def gi(iid, s=1e9): return _get(income,   iid, p, s)
    def gb(iid, s=1e9): return _get(balance,  iid, p, s)
    def gc(iid, s=1e9): return _get(cashflow, iid, p, s)

    # Income — map to nearest semantic equivalent
    revenue          = gi("total_operating_income")           # NII + fee + trading + other
    nii              = gi("net_interest_income")              # core spread income
    fee_raw          = gi("net_fee_and_commission_income")
    provision_raw    = gi("provision_for_credit_losses")
    cogs             = abs(provision_raw) if provision_raw is not None else None  # credit cost
    gross_profit     = nii                                    # NII ≈ gross profit
    ga_expense_raw   = gi("general_and_admin_expenses")
    ga_expense       = abs(ga_expense_raw) if ga_expense_raw is not None else None
    ppop             = gi("net_operating_profit_before_allowance_for_credit_loss")
    ebit             = ppop                                   # PPOP = EBIT analog
    interest_raw     = gi("interest_and_similar_expenses")
    interest_expense = abs(interest_raw) if interest_raw is not None else None
    tax_raw          = gi("business_income_tax_expenses")
    tax_expense      = abs(tax_raw) if tax_raw is not None else None
    net_income       = gi("net_profit_loss_after_tax")
    eps              = gi("eps_basic_vnd", 1.0)

    # Depreciation not separately reported; set None (banks have minimal D&A)
    depreciation     = None
    ebitda_val       = None

    # Balance sheet
    total_assets        = gb("total_assets")
    equity              = gb("owners_equity")
    cash                = gb("cash_and_precious_metals")
    retained_earnings   = gb("retained_earnings")
    loans_net           = gb("loans_and_advances_to_customers_net")  # loan book
    deposits            = gb("deposits_from_customers")
    interbank_borrow    = gb("deposits_and_loans_from_other_credit_institutions")
    sbv_loans           = gb("due_to_gov_and_loans_from_sbv")
    debt                = (interbank_borrow or 0) + (sbv_loans or 0) if (
                              interbank_borrow is not None or sbv_loans is not None
                          ) else None
    # charter_capital has same par-value formula as common_shares for regular cos
    charter_cap_bn      = gb("charter_capital")
    shares_outstanding  = charter_cap_bn / 10 if charter_cap_bn is not None else None

    # Banks have no inventory, current assets/liabilities not meaningful
    current_assets      = None
    current_liabilities = None
    inventory           = None
    selling_expense     = None

    # Cash flow
    operating_cf = gc("net_cash_from_operating_activities")
    capex_raw    = gc("purchases_of_fixed_assets_and_other_long_term_assets")
    capex        = abs(capex_raw) if capex_raw is not None else None
    investing_cf = gc("net_cash_from_investing_activities")
    financing_cf = None  # bank CF doesn't split financing separately
    fcf          = (operating_cf - capex) if (
                       operating_cf is not None and capex is not None
                   ) else None

    return {
        "revenue": revenue, "cogs": cogs, "gross_profit": gross_profit,
        "selling_expense": selling_expense, "ga_expense": ga_expense,
        "ebit": ebit, "ebitda": ebitda_val, "depreciation": depreciation,
        "interest_expense": interest_expense, "tax_expense": tax_expense,
        "net_income": net_income, "eps": eps,
        "total_assets": total_assets, "current_assets": current_assets,
        "current_liabilities": current_liabilities, "inventory": inventory,
        "receivables": loans_net, "payables": deposits, "equity": equity,
        "debt": debt, "cash": cash, "retained_earnings": retained_earnings,
        "shares_outstanding": shares_outstanding,
        "operating_cf": operating_cf, "capex": capex, "fcf": fcf,
        "investing_cf": investing_cf, "financing_cf": financing_cf,
    }


def fetch_financials(ticker: str, n_periods: int = 8, freq: str = "quarter") -> pd.DataFrame:
    """Return last n_periods of consolidated financials as a DataFrame.

    Automatically detects banking vs regular company and uses the correct item_id mapping.
    All monetary amounts in VND billions; EPS in VND per share.
    """
    try:
        income, balance, cashflow = _fetch_vci(ticker, freq)
    except Exception as e:
        logger.warning(f"{ticker}: API error ({freq}) — {e}")
        return pd.DataFrame()

    if income is None or income.empty:
        return pd.DataFrame()

    period_type = "Q" if freq == "quarter" else "Y"
    periods = _period_cols(income, freq)[:n_periods]
    if not periods:
        return pd.DataFrame()

    bank      = _is_bank(income)
    insurance = not bank and _is_insurance(income)
    if bank:
        logger.info(f"{ticker}: detected as bank — using banking item_id mapping")
    elif insurance:
        logger.info(f"{ticker}: detected as insurance — using insurance item_id mapping")

    rows = []
    for p in periods:
        if bank:
            fields = _extract_bank(income, balance, cashflow, p)
        elif insurance:
            fields = _extract_insurance(income, balance, cashflow, p)
        else:
            fields = _extract_regular(income, balance, cashflow, p)
        rows.append({"period": p, "period_type": period_type, **fields})

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
    """Fetch quarterly (20 periods) + annual (8 years) for each ticker."""
    for ticker in tickers:
        for freq, n in [("quarter", 20), ("year", 8)]:
            try:
                df = fetch_financials(ticker, n_periods=n, freq=freq)
                count = upsert_financials(ticker, df)
                logger.info(f"{ticker}/{freq}: upserted {count} rows")
            except Exception as e:
                logger.error(f"{ticker}/{freq}: failed — {e}")


if __name__ == "__main__":
    run(["VNM", "FPT", "VIC", "VCB", "TCB"])
