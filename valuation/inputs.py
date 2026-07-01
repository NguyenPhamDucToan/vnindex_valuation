"""Prepare DCF/FCFF inputs from stored Financial rows.

Two data sources available in the DB:
  period_type='Q' — latest 4 quarters (community vnstock limit)
  period_type='Y' — latest 4 annual years

Workflow:
  1. compute_ttm(ticker)        → aggregate 4Q rows into a TTM snapshot
  2. compute_fcff_ttm(ttm)      → proper FCFF = NOPAT + D&A - CapEx - ΔNWC
  3. annual_fcff_growth(ticker) → CAGR of annual FCF over available years
  4. prepare_dcf_inputs(ticker) → ready-to-pass dict for dcf_valuation()
"""
from __future__ import annotations

from sqlalchemy import select

from config import TAX_RATE
from models.database import get_session
from models.schema import Financial
from valuation.wacc import DEFAULT_COD

# Flow items: summed across the 4 quarters for TTM
_FLOW = [
    "revenue", "cogs", "gross_profit", "selling_expense", "ga_expense",
    "ebit", "ebitda", "depreciation", "interest_expense", "tax_expense",
    "net_income", "operating_cf", "capex", "fcf", "investing_cf", "financing_cf",
]

# Stock (balance sheet) items: taken from the most recent quarter only
_STOCK = [
    "total_assets", "current_assets", "current_liabilities",
    "inventory", "receivables", "payables",
    "equity", "debt", "cash", "retained_earnings", "shares_outstanding",
]


def compute_ttm(ticker: str) -> dict | None:
    """Aggregate the last 4 quarterly rows into a Trailing-Twelve-Month snapshot.

    Returns a plain dict with the same keys as Financial columns, or None
    if no quarterly rows exist.
    """
    _cols = [c.key for c in Financial.__table__.columns]

    with get_session() as session:
        orm_rows = session.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Q")
            .order_by(Financial.period.desc())
            .limit(4)
        ).scalars().all()
        # Convert to plain dicts inside the session to avoid DetachedInstanceError
        rows = [{c: getattr(r, c) for c in _cols} for r in orm_rows]

    if not rows:
        return None

    ttm: dict = {"ticker": ticker, "period": "TTM", "n_quarters": len(rows)}

    # Sum flow items over available quarters
    for col in _FLOW:
        vals = [r[col] for r in rows if r.get(col) is not None]
        ttm[col] = sum(vals) if vals else None

    # Take most recent value for balance-sheet items (rows[0] = most recent)
    for col in _STOCK:
        ttm[col] = rows[0].get(col)

    # NWC change: latest quarter vs oldest available (≈ 1-year delta for ΔNWC)
    def _nwc(r: dict) -> float | None:
        ca = r.get("current_assets")
        cl = r.get("current_liabilities")
        return (ca - cl) if (ca is not None and cl is not None) else None

    nwc_now  = _nwc(rows[0])
    nwc_prev = _nwc(rows[-1]) if len(rows) >= 2 else None
    ttm["delta_nwc"] = (nwc_now - nwc_prev) if (nwc_now is not None and nwc_prev is not None) else 0.0

    return ttm


def compute_fcff_ttm(ttm: dict, tax_rate: float = TAX_RATE) -> float | None:
    """FCFF from TTM components.

    Primary: NOPAT + D&A − CapEx − ΔNWC  (EBIT-based, less volatile than OCF)
    Fallback: OCF − CapEx  (when EBIT/D&A unavailable)
    """
    ebit      = ttm.get("ebit")
    da        = ttm.get("depreciation")
    capex     = ttm.get("capex")
    delta_nwc = ttm.get("delta_nwc", 0.0) or 0.0

    if ebit is not None and da is not None and capex is not None:
        return ebit * (1 - tax_rate) + da - capex - delta_nwc

    # Fallback: OCF-based
    ocf = ttm.get("operating_cf")
    if ocf is not None and capex is not None:
        return ocf - capex

    return ttm.get("fcf")


def annual_fcff_growth(ticker: str) -> float | None:
    """Estimate FCFF growth CAGR from annual rows (period_type='Y').

    Uses stored fcf (operating_cf − capex) from annual statements.
    Returns None if fewer than 2 years available or if base year FCF ≤ 0.
    """
    with get_session() as session:
        orm_rows = session.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Y")
            .order_by(Financial.period.desc())
            .limit(5)
        ).scalars().all()
        rows = [(r.period, r.fcf) for r in orm_rows]

    fcff_series = [(p, f) for p, f in rows if f is not None]
    if len(fcff_series) < 2:
        return None

    latest_val = fcff_series[0][1]
    oldest_val = fcff_series[-1][1]
    n_years    = len(fcff_series) - 1

    if oldest_val <= 0 or latest_val <= 0:
        return None

    return (latest_val / oldest_val) ** (1 / n_years) - 1


def prepare_dcf_inputs(ticker: str) -> dict | None:
    """Return a dict ready to pass directly to dcf_valuation().

    Keys match dcf_valuation() parameters:
        fcff_base, net_debt_bn, shares_millions, fcff_growth_rate,
        beta, cost_of_debt, debt_bn, equity_bn

    Also includes diagnostic fields: ttm_revenue, ttm_net_income,
    ttm_operating_cf, ttm_capex, annual_growth_used.

    Returns None when essential data is missing.
    """
    ttm = compute_ttm(ticker)
    if ttm is None:
        return None

    fcff_base = compute_fcff_ttm(ttm)
    if fcff_base is None:
        return None

    equity_bn      = ttm.get("equity") or 1.0
    debt_bn        = ttm.get("debt") or 0.0
    cash_bn        = ttm.get("cash") or 0.0
    net_debt_bn    = debt_bn - cash_bn
    shares_millions = ttm.get("shares_outstanding") or 0.0

    # Growth rate: prefer annual CAGR, cap at WACC-3% to ensure meaningful spread
    from valuation.wacc import wacc as calc_wacc, DEFAULT_COD, compute_beta
    beta = compute_beta(ticker)
    wacc_est = calc_wacc(beta, DEFAULT_COD, debt_bn, equity_bn)
    max_growth = max(0.02, wacc_est - 0.03)   # always keep 3% spread vs WACC

    hist_growth = annual_fcff_growth(ticker)
    if hist_growth is not None:
        fcff_growth_rate = max(0.02, min(hist_growth, max_growth))
        growth_source = "annual_cagr"
    else:
        fcff_growth_rate = min(0.08, max_growth)   # default 8%, capped at WACC-3%
        growth_source = "default"

    return {
        # --- DCF model inputs ---
        "fcff_base":       round(fcff_base, 2),
        "net_debt_bn":     round(net_debt_bn, 2),
        "shares_millions": round(shares_millions, 2),
        "fcff_growth_rate": round(fcff_growth_rate, 4),
        "beta":            round(beta, 3),
        "cost_of_debt":    DEFAULT_COD,
        "debt_bn":         round(debt_bn, 2),
        "equity_bn":       round(equity_bn, 2),
        # --- Diagnostics ---
        "ttm_revenue":     ttm.get("revenue"),
        "ttm_net_income":  ttm.get("net_income"),
        "ttm_operating_cf": ttm.get("operating_cf"),
        "ttm_capex":       ttm.get("capex"),
        "ttm_ebit":        ttm.get("ebit"),
        "ttm_depreciation": ttm.get("depreciation"),
        "ttm_delta_nwc":   ttm.get("delta_nwc"),
        "growth_source":   growth_source,
        "n_quarters_used": ttm.get("n_quarters"),
    }


# ---------------------------------------------------------------------------
# Quarter-history builder  (combines real Q rows + annual-derived estimates)
# ---------------------------------------------------------------------------

_CHART_COLS = ["revenue", "net_income", "ebit", "gross_profit",
               "operating_cf", "capex", "fcf", "depreciation"]


def build_quarter_history(ticker: str, n: int = 8) -> "pd.DataFrame":
    """Return up to n quarterly data points for charting.

    Priority (newest first):
      1. Real Q rows from DB.
      2. Derived quarters: if exactly 1 quarter is missing for a year that has
         annual data AND 3 known quarters, compute it as annual − sum(known).
      3. Estimated quarters: annual ÷ 4 for years with no quarterly DB rows,
         labeled is_estimated=True.

    Returns DataFrame sorted oldest → newest with columns:
        period, is_estimated, revenue, net_income, ebit,
        gross_profit, operating_cf, capex, fcf, depreciation
    """
    import pandas as pd

    _cols = [c.key for c in Financial.__table__.columns]

    with get_session() as session:
        q_orm = session.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Q")
            .order_by(Financial.period.desc())
        ).scalars().all()
        y_orm = session.execute(
            select(Financial)
            .where(Financial.ticker == ticker, Financial.period_type == "Y")
            .order_by(Financial.period.desc())
        ).scalars().all()
        q_rows = [{c: getattr(r, c) for c in _cols} for r in q_orm]
        y_rows = [{c: getattr(r, c) for c in _cols} for r in y_orm]

    if not q_rows and not y_rows:
        return pd.DataFrame()

    records = []

    # 1. Real quarterly rows
    known_by_year: dict[str, dict[int, dict]] = {}
    for row in q_rows:
        year = row["period"][:4]        # "2025" from "2025-Q3"
        qnum = int(row["period"][6])    # 3   from "2025-Q3"
        known_by_year.setdefault(year, {})[qnum] = row
        records.append({**{c: row.get(c) for c in _CHART_COLS},
                        "period": row["period"], "is_estimated": False})

    # 2 & 3. Fill from annual rows
    for y_row in y_rows:
        year = y_row["period"]          # "2025"
        known = known_by_year.get(year, {})
        missing = [q for q in [1, 2, 3, 4] if q not in known]

        if not missing:
            continue

        if len(missing) == 1:
            # Derive exactly: annual − sum of 3 known quarters
            qnum = missing[0]
            derived = {}
            for col in _CHART_COLS:
                annual_val = y_row.get(col) or 0
                known_sum  = sum((known[q].get(col) or 0) for q in known)
                derived[col] = annual_val - known_sum
            records.append({**derived, "period": f"{year}-Q{qnum}",
                             "is_estimated": False})

        else:
            # Estimate: annual ÷ 4 per quarter
            for qnum in missing:
                estimated = {col: (y_row.get(col) or 0) / 4 for col in _CHART_COLS}
                records.append({**estimated, "period": f"{year}-Q{qnum}",
                                 "is_estimated": True})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = (df.sort_values("period", ascending=False)
            .head(n)
            .sort_values("period", ascending=True)
            .reset_index(drop=True))
    return df
