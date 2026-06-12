from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from collectors.live_quote import is_market_hours_ict, fetch_live_quote, apply_live_overlay


# ── is_market_hours_ict ──────────────────────────────────────────

def test_market_hours_during_trading_session():
    # Wed 2026-06-10, 10:00 ICT (03:00 UTC) -> within 9:00-15:05 ICT
    now = datetime(2026, 6, 10, 3, 0, tzinfo=timezone.utc)
    assert is_market_hours_ict(now) is True


def test_market_hours_before_open():
    # Wed 2026-06-10, 08:00 ICT (01:00 UTC) -> before 9:00 ICT
    now = datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc)
    assert is_market_hours_ict(now) is False


def test_market_hours_after_close():
    # Wed 2026-06-10, 16:00 ICT (09:00 UTC) -> after 15:05 ICT
    now = datetime(2026, 6, 10, 9, 0, tzinfo=timezone.utc)
    assert is_market_hours_ict(now) is False


def test_market_hours_at_boundaries():
    # 09:00:00 ICT and 15:05:00 ICT are inclusive
    open_edge = datetime(2026, 6, 10, 2, 0, tzinfo=timezone.utc)   # 09:00 ICT
    close_edge = datetime(2026, 6, 10, 8, 5, tzinfo=timezone.utc)  # 15:05 ICT
    assert is_market_hours_ict(open_edge) is True
    assert is_market_hours_ict(close_edge) is True


@pytest.mark.parametrize("weekday_date", [
    datetime(2026, 6, 13, 5, 0, tzinfo=timezone.utc),  # Saturday 12:00 ICT
    datetime(2026, 6, 14, 5, 0, tzinfo=timezone.utc),  # Sunday 12:00 ICT
])
def test_market_hours_weekend(weekday_date):
    assert is_market_hours_ict(weekday_date) is False


def test_market_hours_naive_datetime_treated_as_utc():
    # Wed 2026-06-10, 03:00 (naive) -> treated as UTC -> 10:00 ICT -> open
    now = datetime(2026, 6, 10, 3, 0)
    assert is_market_hours_ict(now) is True


# ── fetch_live_quote ──────────────────────────────────────────────

def _make_price_board_df(overrides=None):
    row = {
        ("match", "match_price"):       59000.0,
        ("match", "open_price"):        59300.0,
        ("match", "highest"):           59800.0,
        ("match", "lowest"):            59000.0,
        ("match", "reference_price"):   59200.0,
        ("match", "accumulated_volume"): 3489300.0,
    }
    row.update(overrides or {})
    return pd.DataFrame([row])


def test_fetch_live_quote_success():
    df = _make_price_board_df()
    mock_stock = MagicMock()
    mock_stock.trading.price_board.return_value = df

    with patch("vnstock.Vnstock") as mock_vnstock:
        mock_vnstock.return_value.stock.return_value = mock_stock
        result = fetch_live_quote("VNM")

    assert result is not None
    assert result["price"] == 59000.0
    assert result["open"] == 59300.0
    assert result["high"] == 59800.0
    assert result["low"] == 59000.0
    assert result["ref"] == 59200.0
    assert result["volume"] == 3489300.0
    assert "as_of" in result


def test_fetch_live_quote_zero_price_returns_none():
    df = _make_price_board_df({("match", "match_price"): 0.0})
    mock_stock = MagicMock()
    mock_stock.trading.price_board.return_value = df

    with patch("vnstock.Vnstock") as mock_vnstock:
        mock_vnstock.return_value.stock.return_value = mock_stock
        result = fetch_live_quote("VNM")

    assert result is None


def test_fetch_live_quote_empty_df_returns_none():
    mock_stock = MagicMock()
    mock_stock.trading.price_board.return_value = pd.DataFrame()

    with patch("vnstock.Vnstock") as mock_vnstock:
        mock_vnstock.return_value.stock.return_value = mock_stock
        result = fetch_live_quote("VNM")

    assert result is None


def test_fetch_live_quote_none_df_returns_none():
    mock_stock = MagicMock()
    mock_stock.trading.price_board.return_value = None

    with patch("vnstock.Vnstock") as mock_vnstock:
        mock_vnstock.return_value.stock.return_value = mock_stock
        result = fetch_live_quote("VNM")

    assert result is None


def test_fetch_live_quote_exception_returns_none():
    with patch("vnstock.Vnstock") as mock_vnstock:
        mock_vnstock.return_value.stock.side_effect = RuntimeError("API down")
        result = fetch_live_quote("VNM")

    assert result is None


def test_fetch_live_quote_missing_optional_fields_fall_back_to_price():
    df = _make_price_board_df({
        ("match", "open_price"): None,
        ("match", "highest"): None,
        ("match", "lowest"): None,
        ("match", "reference_price"): None,
    })
    mock_stock = MagicMock()
    mock_stock.trading.price_board.return_value = df

    with patch("vnstock.Vnstock") as mock_vnstock:
        mock_vnstock.return_value.stock.return_value = mock_stock
        result = fetch_live_quote("VNM")

    assert result["open"] == result["price"]
    assert result["high"] == result["price"]
    assert result["low"] == result["price"]
    assert result["ref"] == result["price"]


# ── apply_live_overlay ──────────────────────────────────────────────

def test_apply_live_overlay_no_live_quote_keeps_eod_values():
    result = apply_live_overlay(
        current_price=59000.0, prev_close=59200.0, high=59800.0, low=58900.0,
        live_quote=None,
    )
    assert result == {
        "current_price": 59000.0,
        "prev_close": 59200.0,
        "high": 59800.0,
        "low": 58900.0,
        "as_of": None,
    }


def test_apply_live_overlay_with_live_quote_overrides_and_scales_to_vnd():
    live_quote = {
        "price": 59.0, "open": 59.3, "high": 59.8, "low": 59.0,
        "ref": 59.2, "volume": 3489300.0, "as_of": "10:30:00",
    }
    result = apply_live_overlay(
        current_price=58000.0, prev_close=57000.0, high=58500.0, low=57500.0,
        live_quote=live_quote,
    )
    # live values are in thousands VND -> scaled x1000 to raw VND
    assert result["current_price"] == 59000.0
    assert result["prev_close"] == 59200.0
    assert result["high"] == 59800.0
    assert result["low"] == 59000.0
    assert result["as_of"] == "10:30:00"


def test_apply_live_overlay_empty_dict_treated_as_no_quote():
    result = apply_live_overlay(
        current_price=59000.0, prev_close=59200.0, high=59800.0, low=58900.0,
        live_quote={},
    )
    assert result["current_price"] == 59000.0
    assert result["as_of"] is None
