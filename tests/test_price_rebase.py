"""collectors/prices.py rebase detection -- pure functions, no database."""
from datetime import date

import pandas as pd

from collectors.prices import needs_rebase, rebase_ratio


def _df(closes, start=date(2026, 9, 21)):
    d = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": [x.date() for x in d], "close": closes})


def test_unchanged_history_does_not_rebase():
    f = _df([51.4, 51.6, 51.9, 52.1])
    stored = dict(zip(f["date"], [51.40, 51.60, 51.87, 52.07]))   # stored rounded to 2dp
    r = rebase_ratio(stored, f)
    assert abs(r - 1) < 0.001 and not needs_rebase(r)


def test_bonus_applied_by_vci_since_last_run_rebases():
    # GMD: stored 77.x, VCI now prices the same days at 77.x / 1.5 after the bonus.
    stored_px = [76.8, 77.3, 77.0, 77.4, 77.8]
    f = _df([p / 1.5 for p in stored_px])
    r = rebase_ratio(dict(zip(f["date"], stored_px)), f)
    assert abs(r - 1.5) < 1e-9 and needs_rebase(r)


def test_cash_dividend_of_two_percent_rebases():
    stored_px = [50.0, 50.5, 49.8, 50.2]
    f = _df([p * 0.98 for p in stored_px])
    assert needs_rebase(rebase_ratio(dict(zip(f["date"], stored_px)), f))


def test_one_bad_row_does_not_trigger_and_short_overlap_is_none():
    stored_px = [50.0, 50.5, 49.8, 50.2, 50.1]
    fetched = list(stored_px)
    fetched[2] = 40.0                                  # one glitch
    f = _df(fetched)
    assert not needs_rebase(rebase_ratio(dict(zip(f["date"], stored_px)), f))
    g = _df([10.0, 10.0])
    assert rebase_ratio(dict(zip(g["date"], [10.0, 10.0])), g) is None
    assert not needs_rebase(None)
