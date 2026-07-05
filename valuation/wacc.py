"""WACC calculation via CAPM for Vietnamese (HOSE) companies."""
from __future__ import annotations

from config import DCF_DISCOUNT_RATE

# --- Vietnamese market parameters ---
RF = 0.05          # Risk-free rate: VN 10-year government bond
ERP = 0.08         # Equity risk premium for Vietnam (higher than developed markets)
DEFAULT_BETA = 1.0 # Market-average beta for HOSE fallback (updated from 1.2)
DEFAULT_COD = 0.08 # Average cost of debt on HOSE
TAX_RATE = 0.20    # Standard Vietnamese corporate income tax rate


def cost_of_equity(beta: float = DEFAULT_BETA, rf: float = RF, erp: float = ERP) -> float:
    """CAPM: CoE = Rf + β × ERP."""
    return rf + beta * erp


def wacc(
    beta: float = DEFAULT_BETA,
    cost_of_debt: float = DEFAULT_COD,
    debt_bn: float = 0.0,
    equity_bn: float = 1.0,
    tax_rate: float = TAX_RATE,
) -> float:
    """WACC = CoE × [E/(D+E)] + CoD × (1-T) × [D/(D+E)].

    debt_bn / equity_bn in VND billions (same unit, ratio is what matters).
    Falls back to DCF_DISCOUNT_RATE from config when equity is zero.
    """
    total = debt_bn + equity_bn
    if total <= 0:
        return DCF_DISCOUNT_RATE

    e_weight = equity_bn / total
    d_weight = debt_bn / total
    coe = cost_of_equity(beta)
    after_tax_cod = cost_of_debt * (1 - tax_rate)

    return coe * e_weight + after_tax_cod * d_weight


def default_wacc() -> float:
    """WACC for a typical HOSE company with no debt/equity breakdown available."""
    return wacc(
        beta=DEFAULT_BETA,
        cost_of_debt=DEFAULT_COD,
        debt_bn=1.0,
        equity_bn=1.0,
    )


def compute_beta(ticker: str, days: int = 252) -> float:
    """Compute beta from daily price history: β = Cov(R_stock, R_market) / Var(R_market).

    Uses 'days' most-recent trading days from the Price table (stock) and
    vnstock VNINDEX (market). Falls back to DEFAULT_BETA on any error or
    insufficient data (< 60 observations).
    """
    try:
        import numpy as np
        import pandas as pd
        import warnings
        from sqlalchemy import select as _sel
        from models.database import get_session as _gs
        from models.schema import Price as _P

        # ── Stock prices from DB ──────────────────────────────────
        with _gs() as _s:
            rows = _s.execute(
                _sel(_P.date, _P.close)
                .where(_P.ticker == ticker)
                .order_by(_P.date.desc())
                .limit(days + 10)
            ).all()

        if len(rows) < 60:
            return DEFAULT_BETA

        stock_df = (
            pd.DataFrame(rows, columns=["date", "close"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        start = pd.to_datetime(stock_df["date"].iloc[0]).strftime("%Y-%m-%d")
        end   = pd.to_datetime(stock_df["date"].iloc[-1]).strftime("%Y-%m-%d")

        # ── VNINDEX prices from vnstock ───────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from vnstock import Vnstock
            vni_raw = (
                Vnstock()
                .stock(symbol="VNINDEX", source="VCI")
                .quote.history(symbol="VNINDEX", start=start, end=end, interval="1D")
            )

        if vni_raw is None or len(vni_raw) < 60:
            return DEFAULT_BETA

        vni_df = (
            vni_raw[["time", "close"]]
            .rename(columns={"time": "date"})
            .assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)
        )
        stock_df = stock_df.assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)

        # ── Align on shared trading dates ─────────────────────────
        merged = stock_df.merge(vni_df, on="date", suffixes=("_s", "_m"))
        if len(merged) < 60:
            return DEFAULT_BETA

        r_s = merged["close_s"].pct_change().dropna().values
        r_m = merged["close_m"].pct_change().dropna().values
        n = min(len(r_s), len(r_m))
        r_s, r_m = r_s[-n:], r_m[-n:]

        cov_mat = np.cov(r_s, r_m)
        var_m   = float(np.var(r_m))
        if var_m == 0:
            return DEFAULT_BETA

        beta = float(cov_mat[0, 1] / var_m)
        return float(np.clip(beta, 0.2, 3.0))

    except (Exception, SystemExit):
        return DEFAULT_BETA
