"""Restatement detection in collectors.financials, on an in-memory database (never a real one)."""
from contextlib import contextmanager

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import collectors.financials as fin
from models.schema import Base, Company, Financial, FinancialRevision


def test_find_revisions_thresholds():
    old = {"revenue": 1000.0, "net_income": 100.0, "equity": 50_000.0, "total_assets": None, "operating_cf": 10.0}
    new = {"revenue": 1000.4, "net_income": 92.0, "equity": 50_000.3, "total_assets": 900.0, "operating_cf": 10.5}
    # revenue: 0.4bn under the 1bn floor; equity: 0.3bn of 50,000; total_assets: backfill; ocf: 0.5bn
    assert fin.find_revisions(old, new) == [("net_income", 100.0, 92.0)]


def test_find_revisions_relative_bar():
    # 5bn on a 10,000bn revenue is 0.05% -- noise, not a restatement
    assert fin.find_revisions({"revenue": 10_000.0}, {"revenue": 10_005.0}) == []
    assert fin.find_revisions({"net_income": -20.0}, {"net_income": 15.0}) == [("net_income", -20.0, 15.0)]


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    @contextmanager
    def get_session():
        s = Session()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr(fin, "get_session", get_session)
    with get_session() as s:
        s.add(Company(ticker="AAA", name="AAA", exchange="HOSE"))
    return Session


def _df(ni_q2, ni_q3=None):
    rows = [{"period": "2026-Q2", "period_type": "Q", "revenue": 500.0, "net_income": ni_q2}]
    if ni_q3 is not None:
        rows.append({"period": "2026-Q3", "period_type": "Q", "revenue": 600.0, "net_income": ni_q3})
    return pd.DataFrame(rows)


def test_upsert_records_restatement(db):
    fin.upsert_financials("AAA", _df(40.0))
    fin.upsert_financials("AAA", _df(40.0, 55.0))       # same Q2, new Q3: nothing restated
    s = db()
    assert s.query(FinancialRevision).count() == 0
    fin.upsert_financials("AAA", _df(31.0, 55.0))       # Q2 profit cut after review
    revs = s.query(FinancialRevision).all()
    assert [(r.period, r.field, r.old_value, r.new_value) for r in revs] == [("2026-Q2", "net_income", 40.0, 31.0)]
    assert s.query(Financial).filter_by(ticker="AAA", period="2026-Q2").one().net_income == 31.0
