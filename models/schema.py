from datetime import date, datetime
from sqlalchemy import (
    Column, String, Integer, Float, Date, DateTime, Boolean,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    ticker = Column(String(10), primary_key=True)
    name = Column(String(200), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    exchange = Column(String(10), default="HOSE")
    listed_date = Column(Date)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    prices = relationship("Price", back_populates="company", cascade="all, delete-orphan")
    financials = relationship("Financial", back_populates="company", cascade="all, delete-orphan")
    valuations = relationship("Valuation", back_populates="company", cascade="all, delete-orphan")


class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("companies.ticker"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adjusted_close = Column(Float)

    company = relationship("Company", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
        Index("ix_price_ticker_date", "ticker", "date"),
    )


class Financial(Base):
    """One row per ticker × period × period_type. All amounts in VND billions."""

    __tablename__ = "financials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("companies.ticker"), nullable=False)
    period = Column(String(10), nullable=False)     # e.g. "2024Q3" or "2024"
    period_type = Column(String(1), nullable=False) # "Q" or "Y"

    # --- Income statement ---
    revenue = Column(Float)          # Net revenue (doanh thu thuần, line 10)
    cogs = Column(Float)             # Cost of goods sold (giá vốn hàng bán, line 11)
    gross_profit = Column(Float)     # = revenue - cogs (line 20)
    selling_expense = Column(Float)  # Chi phí bán hàng (line 21)
    ga_expense = Column(Float)       # Chi phí quản lý doanh nghiệp (line 22)
    ebit = Column(Float)             # Operating profit before interest (PBT + interest)
    ebitda = Column(Float)           # ebit + depreciation + amortisation
    depreciation = Column(Float)     # Khấu hao TSCĐ — needed for EBITDA and FCFF
    interest_expense = Column(Float) # Chi phí lãi vay (sub-line of line 32)
    tax_expense = Column(Float)      # Chi phí thuế TNDN (line 52)
    net_income = Column(Float)       # PAT (line 60)
    eps = Column(Float)              # EPS = net_income / shares_outstanding

    # --- Balance sheet ---
    total_assets = Column(Float)
    current_assets = Column(Float)
    current_liabilities = Column(Float)
    inventory = Column(Float)        # Hàng tồn kho — for DIO
    receivables = Column(Float)      # Phải thu khách hàng — for DSO
    payables = Column(Float)         # Phải trả người bán — for DPO
    equity = Column(Float)           # Vốn chủ sở hữu
    debt = Column(Float)             # Total interest-bearing debt
    cash = Column(Float)             # Tiền và tương đương tiền
    retained_earnings = Column(Float) # Lợi nhuận chưa phân phối (code 421)

    # --- Cash flow ---
    operating_cf = Column(Float)     # Tiền thuần từ HĐKD (line 18)
    capex = Column(Float)            # |line 19| — always positive
    fcf = Column(Float)              # operating_cf - capex
    investing_cf = Column(Float)     # Tiền thuần từ HĐ đầu tư (line 26)
    financing_cf = Column(Float)     # Tiền thuần từ HĐ tài chính (line 32)

    shares_outstanding = Column(Float)  # millions of shares

    company = relationship("Company", back_populates="financials")

    __table_args__ = (
        UniqueConstraint("ticker", "period", "period_type", name="uq_fin_ticker_period"),
        Index("ix_fin_ticker_period", "ticker", "period", "period_type"),
    )


class Valuation(Base):
    """Computed valuation and ratio metrics for a ticker on a given calculation date."""

    __tablename__ = "valuations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("companies.ticker"), nullable=False)
    calc_date = Column(Date, nullable=False)

    # --- Multiples ---
    pe = Column(Float)               # Price / EPS (TTM)
    pb = Column(Float)               # Price / BVPS
    ev_ebitda = Column(Float)        # (Market cap + Debt - Cash) / EBITDA
    graham_number = Column(Float)    # sqrt(22.5 × EPS × BVPS)
    fcfe_estimate = Column(Float)    # FCFE (OCF−CapEx) discounted at cost of equity
    avg_intrinsic_value = Column(Float)  # simple avg of all valid method estimates

    # --- DCF (FCFF-based) ---
    nopat = Column(Float)            # EBIT × (1 - tax_rate) — VND billions TTM
    fcff = Column(Float)             # NOPAT + D&A - CapEx - ΔNWC — VND billions TTM
    wacc = Column(Float)             # CAPM-derived WACC
    ev = Column(Float)               # Enterprise value from DCF — VND billions
    equity_value = Column(Float)     # EV - Net Debt — VND billions
    dcf_estimate = Column(Float)     # Intrinsic price per share — VND
    upside_pct = Column(Float)       # (dcf_estimate - current_price) / current_price

    # --- Profitability margins (Income Statement dimension) ---
    gross_margin = Column(Float)     # Gross profit / Revenue
    operating_margin = Column(Float) # EBIT / Revenue
    ebitda_margin = Column(Float)    # EBITDA / Revenue
    net_margin = Column(Float)       # Net income / Revenue
    roe = Column(Float)              # Net income / Equity
    roa = Column(Float)              # Net income / Total assets
    asset_turnover = Column(Float)   # Revenue / Total assets

    # --- Cash flow quality (CF dimension) ---
    profit_quality = Column(Float)   # OCF / Net income  (≥1.0 = good)
    ocf_to_revenue = Column(Float)   # OCF / Revenue
    fcf_margin = Column(Float)       # FCF / Revenue
    fcf_yield = Column(Float)        # FCF / Total assets
    capex_coverage = Column(Float)   # OCF / CapEx  (≥1.5 = self-financing)
    fcf_conversion = Column(Float)   # FCF / OCF  (≥60% = good)
    ocf_to_current_liab = Column(Float)  # OCF / Current liabilities  (≥0.4)
    cash_interest_coverage = Column(Float)  # OCF / Interest expense  (≥5)
    cash_coverage = Column(Float)    # OCF / Total debt

    # --- Balance sheet ratios (BS dimension) ---
    current_ratio = Column(Float)    # Current assets / Current liabilities
    quick_ratio = Column(Float)      # (Current assets - Inventory) / Current liabilities
    debt_to_assets = Column(Float)   # Total debt / Total assets
    debt_to_equity = Column(Float)   # Total debt / Equity
    financial_leverage = Column(Float)  # Total assets / Equity

    # --- Working capital / CCC (Cash management dimension) ---
    dso = Column(Float)              # Receivables × 365 / Revenue  (days)
    dio = Column(Float)              # Inventory × 365 / COGS  (days)
    dpo = Column(Float)              # Payables × 365 / COGS  (days)
    ccc = Column(Float)              # DSO + DIO - DPO  (days)

    company = relationship("Company", back_populates="valuations")

    __table_args__ = (
        UniqueConstraint("ticker", "calc_date", name="uq_val_ticker_date"),
        Index("ix_val_ticker_date", "ticker", "calc_date"),
    )


class PinnedTicker(Base):
    """User-saved watchlist entries persisted in the local SQLite DB."""
    __tablename__ = "pinned_tickers"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(10), ForeignKey("companies.ticker"), nullable=False, unique=True)
    note        = Column(String(200), default="")
    added_date  = Column(Date, nullable=False)
