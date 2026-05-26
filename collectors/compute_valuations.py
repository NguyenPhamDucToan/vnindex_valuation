"""Compute and store Valuation rows for all tickers that have financial data.

For each active company with at least 4 quarterly rows:
  1. Compute TTM snapshot
  2. Run DCF (FCFF-based)
  3. Compute all financial ratios
  4. Look up current price
  5. Upsert into valuations table with calc_date = today

Usage:
    python -m collectors.compute_valuations
    python -m collectors.compute_valuations --tickers VNM FPT VIC
"""
from __future__ import annotations

import argparse
from datetime import date

from loguru import logger
from sqlalchemy import func, select

from config import MARKET_PE, TAX_RATE
from models.database import get_session
from models.schema import Company, Financial, Price, Valuation
from valuation.dcf import dcf_valuation
from valuation.graham import graham_number, bvps_from_financials
from valuation.inputs import compute_ttm, compute_fcff_ttm, annual_fcff_growth
from valuation.ratios import (
    gross_margin, operating_margin, ebitda_margin, net_margin,
    roe, roa, asset_turnover,
    profit_quality, ocf_to_revenue, fcf_margin, fcf_yield,
    capex_coverage, fcf_conversion, ocf_to_current_liabilities,
    cash_interest_coverage, cash_coverage,
    current_ratio, quick_ratio, debt_to_assets, debt_to_equity, financial_leverage,
    dso, dio, dpo, ccc,
    pe_ratio, pb_ratio, ev_ebitda,
)
from valuation.wacc import cost_of_equity, DEFAULT_BETA, DEFAULT_COD


def _latest_price(ticker: str) -> float | None:
    """Return the most recent closing price (in VND, not thousands)."""
    with get_session() as session:
        row = session.execute(
            select(Price.close, Price.date)
            .where(Price.ticker == ticker)
            .order_by(Price.date.desc())
            .limit(1)
        ).first()
    if row is None or row[0] is None:
        return None
    # Prices stored in thousands VND → convert to VND
    return row[0] * 1000


def _tickers_with_enough_data(min_quarters: int = 4) -> list[str]:
    """Return tickers that have at least min_quarters quarterly financial rows."""
    with get_session() as session:
        rows = session.execute(
            select(Financial.ticker, func.count(Financial.id).label("cnt"))
            .where(Financial.period_type == "Q")
            .group_by(Financial.ticker)
            .having(func.count(Financial.id) >= min_quarters)
            .order_by(Financial.ticker)
        ).all()
    return [r[0] for r in rows]


def compute_valuation_row(ticker: str, calc_date: date) -> dict | None:
    """Compute all valuation fields for ticker. Returns None if essential data missing."""
    ttm = compute_ttm(ticker)
    if ttm is None:
        return None

    # --- Current price ---
    price_vnd = _latest_price(ticker)

    # --- DCF (FCFF) ---
    fcff_base = compute_fcff_ttm(ttm)
    equity_bn = ttm.get("equity") or 1.0
    debt_bn = ttm.get("debt") or 0.0
    cash_bn = ttm.get("cash") or 0.0
    net_debt_bn = debt_bn - cash_bn
    shares_millions = ttm.get("shares_outstanding") or 0.0

    hist_growth = annual_fcff_growth(ticker)
    if hist_growth is not None:
        growth_rate = max(0.02, min(hist_growth, 0.35))
    else:
        growth_rate = 0.12

    dcf_result = {}
    dcf_price = None
    upside = None
    if fcff_base is not None and shares_millions > 0:
        dcf_result = dcf_valuation(
            fcff_base=fcff_base,
            net_debt_bn=net_debt_bn,
            shares_millions=shares_millions,
            fcff_growth_rate=growth_rate,
            beta=DEFAULT_BETA,
            cost_of_debt=DEFAULT_COD,
            debt_bn=debt_bn,
            equity_bn=equity_bn,
        )
        dcf_price = dcf_result.get("price_per_share")
        if dcf_price and price_vnd and price_vnd > 0:
            upside = (dcf_price - price_vnd) / price_vnd

    # --- Graham Number ---
    eps_vnd = None
    graham_price = None
    if ttm.get("net_income") and shares_millions and shares_millions > 0:
        eps_vnd = (ttm["net_income"] * 1_000) / shares_millions  # VND/share
    bvps_vnd = bvps_from_financials(equity_bn, shares_millions) if shares_millions > 0 else None
    if eps_vnd and bvps_vnd:
        graham_price = graham_number(eps_vnd, bvps_vnd)

    # --- P/E multiples ---
    pe_implied = (eps_vnd * MARKET_PE) if eps_vnd and eps_vnd > 0 else None
    pe_val = pe_ratio(price_vnd, eps_vnd) if (price_vnd and eps_vnd) else None
    pb_val = pb_ratio(price_vnd, equity_bn, shares_millions) if price_vnd else None

    # --- EV/EBITDA ---
    ebitda_bn = ttm.get("ebitda")
    ev_eb = None
    if price_vnd and shares_millions and shares_millions > 0 and ebitda_bn and ebitda_bn > 0:
        mktcap_bn = (price_vnd * shares_millions) / 1_000  # VND × millions / 1e9
        ev_eb = ev_ebitda(mktcap_bn, debt_bn, cash_bn, ebitda_bn)

    # --- Profitability ratios ---
    rev = ttm.get("revenue")
    ni = ttm.get("net_income")
    gp = ttm.get("gross_profit")
    ebit = ttm.get("ebit")
    ebitda_v = ttm.get("ebitda")
    ta = ttm.get("total_assets")
    eq = ttm.get("equity")

    gm = gross_margin(gp, rev)
    opm = operating_margin(ebit, rev)
    ebm = ebitda_margin(ebitda_v, rev)
    nm = net_margin(ni, rev)
    roe_v = roe(ni, eq)
    roa_v = roa(ni, ta)
    at = asset_turnover(rev, ta)

    # --- Cash flow quality ---
    ocf = ttm.get("operating_cf")
    capex = ttm.get("capex")
    fcf_v = ttm.get("fcf")
    ie = ttm.get("interest_expense")
    cl = ttm.get("current_liabilities")
    cogs_v = ttm.get("cogs")
    inv = ttm.get("inventory")
    rec = ttm.get("receivables")
    pay = ttm.get("payables")

    pq = profit_quality(ocf, ni)
    ocf_rev = ocf_to_revenue(ocf, rev)
    fm = fcf_margin(fcf_v, rev)
    fy = fcf_yield(fcf_v, ta)
    cc = capex_coverage(ocf, capex)
    fc = fcf_conversion(fcf_v, ocf)
    ocf_cl = ocf_to_current_liabilities(ocf, cl)
    cic = cash_interest_coverage(ocf, ie)
    cov = cash_coverage(ocf, debt_bn if debt_bn else None)

    # --- Balance sheet ---
    ca = ttm.get("current_assets")
    cr = current_ratio(ca, cl)
    qr = quick_ratio(ca, inv or 0, cl)
    dta = debt_to_assets(debt_bn, ta)
    dte = debt_to_equity(debt_bn, eq)
    fl = financial_leverage(ta, eq)

    # --- CCC ---
    dso_v = dso(rec, rev) if rec and rev else None
    dio_v = dio(inv, cogs_v) if inv and cogs_v else None
    dpo_v = dpo(pay, cogs_v) if pay and cogs_v else None
    ccc_v = ccc(dso_v, dio_v, dpo_v)

    return {
        "ticker": ticker,
        "calc_date": calc_date,
        # DCF
        "nopat": round(ebit * (1 - TAX_RATE), 2) if ebit else None,
        "fcff": round(fcff_base, 2) if fcff_base is not None else None,
        "wacc": dcf_result.get("wacc"),
        "ev": dcf_result.get("enterprise_value"),
        "equity_value": dcf_result.get("equity_value"),
        "dcf_estimate": dcf_price,
        "upside_pct": round(upside, 4) if upside is not None else None,
        # Multiples
        "pe": pe_val,
        "pb": pb_val,
        "ev_ebitda": ev_eb,
        "graham_number": round(graham_price, 0) if graham_price else None,
        # Profitability
        "gross_margin": gm,
        "operating_margin": opm,
        "ebitda_margin": ebm,
        "net_margin": nm,
        "roe": roe_v,
        "roa": roa_v,
        "asset_turnover": at,
        # CF quality
        "profit_quality": pq,
        "ocf_to_revenue": ocf_rev,
        "fcf_margin": fm,
        "fcf_yield": fy,
        "capex_coverage": cc,
        "fcf_conversion": fc,
        "ocf_to_current_liab": ocf_cl,
        "cash_interest_coverage": cic,
        "cash_coverage": cov,
        # Balance sheet
        "current_ratio": cr,
        "quick_ratio": qr,
        "debt_to_assets": dta,
        "debt_to_equity": dte,
        "financial_leverage": fl,
        # CCC
        "dso": dso_v,
        "dio": dio_v,
        "dpo": dpo_v,
        "ccc": ccc_v,
    }


def upsert_valuation(row: dict) -> None:
    """Insert or update a Valuation row."""
    ticker = row["ticker"]
    calc_date = row["calc_date"]
    with get_session() as session:
        existing = session.execute(
            select(Valuation)
            .where(Valuation.ticker == ticker, Valuation.calc_date == calc_date)
        ).scalar_one_or_none()

        if existing:
            for k, v in row.items():
                if k not in ("ticker", "calc_date") and hasattr(existing, k):
                    setattr(existing, k, v)
        else:
            session.add(Valuation(**{k: v for k, v in row.items() if hasattr(Valuation, k)}))


def run(tickers: list[str] | None = None) -> None:
    """Compute and store valuation rows for all eligible tickers (or a subset)."""
    today = date.today()

    if tickers is None:
        tickers = _tickers_with_enough_data(min_quarters=4)
        logger.info(f"Computing valuations for {len(tickers)} tickers with ≥4 quarterly rows")
    else:
        logger.info(f"Computing valuations for {len(tickers)} specified tickers")

    ok = skip = fail = 0
    for i, ticker in enumerate(tickers, 1):
        prefix = f"[{i}/{len(tickers)}] {ticker}"
        try:
            row = compute_valuation_row(ticker, today)
            if row is None:
                logger.warning(f"{prefix}: insufficient data, skipping")
                skip += 1
                continue
            upsert_valuation(row)
            dcf_est = row.get("dcf_estimate")
            upside = row.get("upside_pct")
            upside_str = f"{upside*100:+.1f}%" if upside is not None else "N/A"
            logger.info(f"{prefix}: DCF={dcf_est:,.0f}₫  upside={upside_str}" if dcf_est else f"{prefix}: stored (no DCF price)")
            ok += 1
        except Exception as e:
            logger.error(f"{prefix}: FAILED — {e}")
            fail += 1

    logger.info(f"Valuations done: {ok} ok, {skip} skipped, {fail} failed out of {len(tickers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute valuation rows for HOSE tickers")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Specific tickers to compute (default: all with ≥4Q data)")
    args = parser.parse_args()
    run(tickers=args.tickers)
