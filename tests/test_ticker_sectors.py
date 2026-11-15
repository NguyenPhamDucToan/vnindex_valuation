"""ICB -> house sector mapping in collectors.ticker_list (in-memory database, never a real one)."""
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import collectors.ticker_list as tl
from models.schema import Base, Company


def test_house_sector():
    assert tl.house_sector("ECO", "Hóa chất") == "SX Nhựa - Hóa chất"
    assert tl.house_sector("MZG", "Tài nguyên Cơ bản") == "SX Phụ trợ"
    assert tl.house_sector("GDA", "Tài nguyên Cơ bản") == "Vật liệu xây dựng"
    assert tl.house_sector("YEG", "Truyền thông") == "Truyền thông"
    assert tl.house_sector("XYZ", None) is None


def test_upsert_resectors_icb_names_only(monkeypatch):
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

    monkeypatch.setattr(tl, "get_session", get_session)
    with get_session() as s:
        s.add_all([Company(ticker="ECO", name="ECO", sector="Hóa chất"),          # raw ICB, stored earlier
                   Company(ticker="AAA", name="AAA", sector="SX Nhựa - Hóa chất"),  # house name: keep
                   Company(ticker="CLC", name="CLC", sector="Hàng cá nhân & Gia dụng")])
    df = pd.DataFrame([
        {"symbol": "ECO", "organ_name": "ECO", "exchange": "HOSE", "industry_name": "Hóa chất"},
        {"symbol": "AAA", "organ_name": "AAA", "exchange": "HOSE", "industry_name": "Hóa chất"},
        {"symbol": "CLC", "organ_name": "CLC", "exchange": "HOSE", "industry_name": "Hàng cá nhân & Gia dụng"},
        {"symbol": "DSH", "organ_name": "DSH", "exchange": "HOSE", "industry_name": "Xây dựng và Vật liệu"},
        {"symbol": "DVV", "organ_name": "DVV", "exchange": "HOSE", "industry_name": "Truyền thông"},
    ])
    tl.upsert_companies(df)
    s = Session()
    got = {c.ticker: c.sector for c in s.query(Company)}
    assert got == {"ECO": "SX Nhựa - Hóa chất", "AAA": "SX Nhựa - Hóa chất", "CLC": "SX Phụ trợ",
                   "DSH": "Xây dựng", "DVV": "Truyền thông"}
