"""Day 18 — Historical signal backtest.

For every ticker with enough quarterly history, recompute the 7-level
Buy/Sell signal (avg intrinsic value upside + quality score) at each
quarterly TTM snapshot, then check the stock's actual return over the
following ~1 year.

This answers: "if you had followed the Strong Buy / Strong Sell signal
historically, how would it have performed?"

Output:
  - backtest_results.csv  — one row per (ticker, quarter snapshot)
  - backtest_summary.csv  — aggregated stats per signal bucket
  - printed summary table
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import calendar

import pandas as pd
from sqlalchemy import select, func

from models.database import get_session
from models.schema import Financial, Price
from config import MARKET_PE
from valuation.inputs import compute_fcff_ttm, prepare_dcf_inputs, _FLOW, _STOCK
from valuation.dcf import dcf_valuation
from valuation.graham import graham_number, bvps_from_financials
from valuation.wacc import cost_of_equity, DEFAULT_BETA, DEFAULT_COD
from valuation.multiples import (
    pb_implied, ev_ebitda_implied, epv_implied, ps_implied,
    residual_income_implied, pocf_implied,
)
from valuation.ratios import roe, net_margin, profit_quality, fcf_margin, current_ratio, debt_to_equity
from valuation.signals import compute_quality_score, classify_signal, SIGNAL_ORDER

FORWARD_DAYS = 365
PRICE_TOLERANCE_NOW = pd.Timedelta(days=10)
PRICE_TOLERANCE_FUTURE = pd.Timedelta(days=20)


def _tickers_with_history(min_quarters: int = 8) -> list[str]:
    with get_session() as s:
        rows = s.execute(
            select(Financial.ticker, func.count(Financial.id))
            .where(Financial.period_type == "Q")
            .group_by(Financial.ticker)
            .having(func.count(Financial.id) >= min_quarters)
            .order_by(Financial.ticker)
        ).all()
    return [r[0] for r in rows]


def _load_q_rows(ticker: str) -> list[dict]:
    cols = [c.key for c in Financial.__table__.columns]
    with get_session() as s:
        orm = s.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Q")
            .order_by(Financial.period.asc())
        ).scalars().all()
        return [{c: getattr(r, c) for c in cols} for r in orm]


def _load_price_df(ticker: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(
            select(Price.date, Price.close)
            .where(Price.ticker == ticker)
            .order_by(Price.date.asc())
        ).all()
    df = pd.DataFrame(rows, columns=["date", "close"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["price_vnd"] = df["close"] * 1000.0  # Price.close is in thousands VND
    return df[["date", "price_vnd"]]


def _avg_intrinsic_value(ttm: dict, fcff_growth: float) -> float | None:
    """Average of 10 intrinsic-value methods, in VND/share (mirrors dashboard valuation_history)."""
    shares = ttm.get("shares_outstanding") or 0
    if shares <= 0:
        return None
    equity = ttm.get("equity")
    net_income = ttm.get("net_income")
    debt_bn = ttm.get("debt") or 0.0
    cash_bn = ttm.get("cash") or 0.0
    net_debt_bn = debt_bn - cash_bn
    coe = cost_of_equity(DEFAULT_BETA)

    prices = []

    # 1. DCF (FCFF)
    fcff = compute_fcff_ttm(ttm)
    if fcff and fcff > 0:
        res = dcf_valuation(fcff_base=fcff, net_debt_bn=net_debt_bn,
                             shares_millions=shares, fcff_growth_rate=fcff_growth,
                             beta=DEFAULT_BETA, cost_of_debt=DEFAULT_COD,
                             debt_bn=debt_bn, equity_bn=equity or 1.0)
        if res.get("price_per_share"):
            prices.append(res["price_per_share"])

    # 2. FCFE
    fcfe_val = ttm.get("fcf")
    if fcfe_val and fcfe_val > 0:
        res2 = dcf_valuation(fcff_base=fcfe_val, net_debt_bn=0,
                              shares_millions=shares, fcff_growth_rate=fcff_growth,
                              wacc_override=coe)
        if res2.get("price_per_share"):
            prices.append(res2["price_per_share"])

    # 3. Graham Number
    if equity and net_income and net_income > 0:
        bvps = bvps_from_financials(equity, shares)
        eps = net_income * 1_000 / shares
        g = graham_number(eps, bvps)
        if g:
            prices.append(g)

    # 4. P/E implied
    if net_income and net_income > 0:
        prices.append((net_income * 1_000 / shares) * MARKET_PE)

    # 5. P/B implied
    pb_p = pb_implied(ttm.get("equity"), shares)
    if pb_p:
        prices.append(pb_p)

    # 6. EV/EBITDA implied
    ebitda_h = ttm.get("ebitda") or (
        ((ttm.get("ebit") or 0) + (ttm.get("depreciation") or 0)) if ttm.get("ebit") else None
    )
    ev_p = ev_ebitda_implied(ebitda_h, net_debt_bn, shares)
    if ev_p:
        prices.append(ev_p)

    # 7. EPV
    epv_p = epv_implied(ttm.get("ebit"), net_debt_bn, shares)
    if epv_p:
        prices.append(epv_p)

    # 8. P/Sales
    ps_p = ps_implied(ttm.get("revenue"), shares)
    if ps_p:
        prices.append(ps_p)

    # 9. Residual income
    ri_p = residual_income_implied(ttm.get("equity"), shares, net_income, g=fcff_growth)
    if ri_p:
        prices.append(ri_p)

    # 10. P/OCF
    pocf_p = pocf_implied(ttm.get("operating_cf"), shares)
    if pocf_p:
        prices.append(pocf_p)

    valid = [p for p in prices if p and p > 0]
    return (sum(valid) / len(valid)) if valid else None


def _build_ttm_window(rows: list[dict], i: int) -> dict:
    """Replicates valuation_history()'s rolling 4-quarter TTM window."""
    window = rows[i - 3: i + 1]
    ttm: dict = {}
    for col in _FLOW:
        vals = [r[col] for r in window if r.get(col) is not None]
        ttm[col] = sum(vals) if vals else None
    for col in _STOCK:
        ttm[col] = window[-1].get(col)

    ca0 = window[-1].get("current_assets"); cl0 = window[-1].get("current_liabilities")
    ca1 = window[0].get("current_assets");  cl1 = window[0].get("current_liabilities")
    nwc0 = (ca0 - cl0) if ca0 is not None and cl0 is not None else None
    nwc1 = (ca1 - cl1) if ca1 is not None and cl1 is not None else None
    ttm["delta_nwc"] = (nwc0 - nwc1) if nwc0 is not None and nwc1 is not None else 0.0
    return ttm


def _quarter_end_date(period: str) -> pd.Timestamp:
    yr, qn = int(period[:4]), int(period[6])
    mo = qn * 3
    return pd.Timestamp(f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}")


def _sector_of(ticker: str) -> str | None:
    """Sector for the quality score, which grades financials on their own bars."""
    from models.database import get_session
    from models.schema import Company
    with get_session() as session:
        row = session.execute(
            select(Company.sector).where(Company.ticker == ticker)).first()
    return row[0] if row else None


def backtest_ticker(ticker: str) -> list[dict]:
    rows = _load_q_rows(ticker)
    if len(rows) < 4:
        return []
    # Score the same way production does, or the backtest measures a different
    # rule than the one the app publishes.
    sector = _sector_of(ticker)

    price_df = _load_price_df(ticker)
    if price_df.empty:
        return []

    inputs = prepare_dcf_inputs(ticker)
    g = inputs["fcff_growth_rate"] if inputs else 0.08

    snapshots = []
    for i in range(3, len(rows)):
        ttm = _build_ttm_window(rows, i)
        period = rows[i]["period"]
        dt = _quarter_end_date(period)

        avg_val = _avg_intrinsic_value(ttm, g)
        if avg_val is None:
            continue

        qs = compute_quality_score(
            roe(ttm.get("net_income"), ttm.get("equity")),
            net_margin(ttm.get("net_income"), ttm.get("revenue")),
            profit_quality(ttm.get("operating_cf"), ttm.get("net_income")),
            fcf_margin(ttm.get("fcf"), ttm.get("revenue")),
            current_ratio(ttm.get("current_assets"), ttm.get("current_liabilities")),
            debt_to_equity(ttm.get("debt"), ttm.get("equity")),
            sector=sector,
        )

        snapshots.append({"period": period, "date": dt, "avg_value": avg_val, "quality": qs})

    if not snapshots:
        return []

    snap_df = pd.DataFrame(snapshots).sort_values("date")
    price_df = price_df.sort_values("date")

    # Price at snapshot date (nearest prior trading day)
    merged = pd.merge_asof(snap_df, price_df, on="date",
                            direction="backward", tolerance=PRICE_TOLERANCE_NOW)
    merged = merged.rename(columns={"price_vnd": "price_now"})

    # Price ~1 year later (nearest trading day)
    future = snap_df[["date"]].copy()
    future["target_date"] = future["date"] + pd.Timedelta(days=FORWARD_DAYS)
    future = future.sort_values("target_date")
    future_merged = pd.merge_asof(future, price_df.rename(columns={"date": "target_date"}),
                                   on="target_date", direction="nearest",
                                   tolerance=PRICE_TOLERANCE_FUTURE)
    future_merged = future_merged.rename(columns={"price_vnd": "price_future"})[["date", "price_future"]]

    merged = merged.merge(future_merged, on="date", how="left")
    merged = merged.dropna(subset=["price_now", "price_future"])
    if merged.empty:
        return []

    merged["upside"] = (merged["avg_value"] - merged["price_now"]) / merged["price_now"]
    merged["forward_return"] = (merged["price_future"] - merged["price_now"]) / merged["price_now"]
    merged["signal"] = [classify_signal(u, q) for u, q in zip(merged["upside"], merged["quality"])]
    merged["ticker"] = ticker

    return merged[["ticker", "period", "date", "price_now", "avg_value", "upside",
                    "quality", "signal", "price_future", "forward_return"]].to_dict("records")


def main():
    tickers = _tickers_with_history()
    print(f"Backtesting {len(tickers)} tickers with >=8 quarters of data...")

    all_records = []
    for n, t in enumerate(tickers, 1):
        try:
            recs = backtest_ticker(t)
            all_records.extend(recs)
        except Exception as e:
            print(f"  skip {t}: {e}")
        if n % 50 == 0:
            print(f"  ... {n}/{len(tickers)} tickers processed, {len(all_records)} snapshots so far")

    if not all_records:
        print("No backtest records produced — check data availability.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv("backtest_results.csv", index=False)
    print(f"\n{len(df)} (ticker, quarter) snapshots across {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    summary = (df.groupby("signal")["forward_return"]
                 .agg(count="count", mean_return="mean", median_return="median",
                      win_rate=lambda s: (s > 0).mean())
                 .reindex(SIGNAL_ORDER)
                 .dropna(how="all"))
    summary["mean_return"] = (summary["mean_return"] * 100).round(2)
    summary["median_return"] = (summary["median_return"] * 100).round(2)
    summary["win_rate"] = (summary["win_rate"] * 100).round(1)
    summary.to_csv("backtest_summary.csv")

    print("\n1-Year Forward Return by Signal (at time signal was issued):")
    print(summary.rename(columns={
        "count": "N", "mean_return": "Mean Fwd Return %",
        "median_return": "Median Fwd Return %", "win_rate": "Win Rate %",
    }).to_string())

    print("\nSaved: backtest_results.csv (per-snapshot), backtest_summary.csv (aggregated)")


if __name__ == "__main__":
    main()
