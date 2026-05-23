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
    period = Column(String(10), nullable=False)       # e.g. "2024Q3" or "2024"
    period_type = Column(String(1), nullable=False)   # "Q" or "Y"

    # Income statement
    revenue = Column(Float)
    ebit = Column(Float)
    net_income = Column(Float)
    eps = Column(Float)

    # Balance sheet
    total_assets = Column(Float)
    equity = Column(Float)
    debt = Column(Float)
    cash = Column(Float)

    # Cash flow
    operating_cf = Column(Float)
    capex = Column(Float)
    fcf = Column(Float)

    shares_outstanding = Column(Float)   # millions of shares

    company = relationship("Company", back_populates="financials")

    __table_args__ = (
        UniqueConstraint("ticker", "period", "period_type", name="uq_fin_ticker_period"),
        Index("ix_fin_ticker_period", "ticker", "period", "period_type"),
    )


class Valuation(Base):
    """Computed valuation metrics for a ticker on a given calculation date."""

    __tablename__ = "valuations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("companies.ticker"), nullable=False)
    calc_date = Column(Date, nullable=False)

    pe = Column(Float)
    pb = Column(Float)
    ev_ebitda = Column(Float)
    graham_number = Column(Float)
    dcf_estimate = Column(Float)     # VND per share
    upside_pct = Column(Float)       # (dcf_estimate - price) / price

    company = relationship("Company", back_populates="valuations")

    __table_args__ = (
        UniqueConstraint("ticker", "calc_date", name="uq_val_ticker_date"),
        Index("ix_val_ticker_date", "ticker", "calc_date"),
    )
