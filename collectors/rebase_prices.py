"""One-off repair: reload price history that VCI has rescaled since it was stored.

The daily collector used to append new days only, so every entitlement left
the rows before it on the old scale (see collectors/prices.py). It now checks
a 14-day overlap each run, which catches new rescales but not the seams
already deep in the table. This compares each ticker's WHOLE stored history
with VCI's current one and reloads the tickers that differ.

    python -m collectors.rebase_prices                 # dry run: report only
    python -m collectors.rebase_prices --apply         # reload the stale tickers
    python -m collectors.rebase_prices --apply VCB FPT # just these

On 2026-11-07 the dry run found 215 of 403 tickers stale (46 by more than 15%,
e.g. GMD x1.50 before its 2026-10-07 bonus, FPT x1.115 before 2026-09-21).
"""
from __future__ import annotations

import sys
import time
from datetime import date

from loguru import logger

from collectors.prices import (REBASE_TOL, fetch_prices, stored_closes, stored_range,
                               upsert_prices)

_DELAY_S = 1.0


def stale_share(stored: dict, fetched) -> tuple[float, float]:
    """(share of overlapping days off by more than REBASE_TOL, worst ratio)."""
    r = [stored[d] / float(c) for d, c in zip(fetched["date"], fetched["close"])
         if d in stored and c and float(c) > 0]
    if not r:
        return 0.0, 1.0
    off = [x for x in r if abs(x - 1) > REBASE_TOL]
    worst = max(r, key=lambda x: abs(x - 1))
    return len(off) / len(r), worst


def main(argv: list[str]) -> None:
    apply = "--apply" in argv
    only = [a.upper() for a in argv if not a.startswith("--")]
    if only:
        tickers = only
    else:
        from collectors.bulk_load import _active_tickers
        tickers = [t for t in _active_tickers() if len(t) == 3]
    today = date.today()
    stale = fixed = failed = 0
    for i, t in enumerate(tickers, 1):
        first, _last = stored_range(t)
        if first is None:
            continue
        try:
            df = fetch_prices(t, first, today)
        except Exception as e:                          # noqa: BLE001 -- report and move on
            logger.error(f"[{i}/{len(tickers)}] {t}: fetch failed — {e}")
            failed += 1
            time.sleep(_DELAY_S * 3)
            continue
        share, worst = stale_share(stored_closes(t, first), df)
        if share > 0:
            stale += 1
            msg = f"[{i}/{len(tickers)}] {t}: {share:.0%} of days stale, worst x{worst:.4f}"
            if apply and not df.empty:
                n = upsert_prices(t, df)
                fixed += 1
                msg += f" — reloaded {n} rows"
            logger.info(msg)
        time.sleep(_DELAY_S)
    logger.info(f"{len(tickers)} checked: {stale} stale, {fixed} reloaded, {failed} failed"
                + ("" if apply else " (dry run — pass --apply to write)"))


if __name__ == "__main__":
    main(sys.argv[1:])
