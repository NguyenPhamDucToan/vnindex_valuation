"""Graham Number intrinsic value estimate."""
from __future__ import annotations

import math
from typing import Optional


def graham_number(eps: float, bvps: float) -> Optional[float]:
    """Classic Graham Number: sqrt(22.5 * EPS * BVPS).

    Both eps and bvps in VND per share.
    Returns None if either input is non-positive (loss-making or negative book).
    """
    if not eps or eps <= 0 or not bvps or bvps <= 0:
        return None
    return math.sqrt(22.5 * eps * bvps)


def bvps_from_financials(equity_bn: float, shares_millions: float) -> Optional[float]:
    """Convert equity (VND billions) and shares (millions) to book value per share (VND)."""
    if not shares_millions or shares_millions <= 0:
        return None
    return (equity_bn * 1_000) / shares_millions   # billions → millions VND per million shares = VND
