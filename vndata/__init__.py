"""Market data straight from the sources' own JSON APIs -- the vnstock replacement.

vnstock was a wrapper around these same endpoints (VCI for almost everything),
but version 5 metered them: 10-20 calls a minute on the guest tier, 57-second
penalties, a device limit that GitHub runners exhausted, and an API key plus
`login()` on every CI job. Called directly the endpoints need no key and no
throttle.

The classes keep vnstock's names, signatures and DataFrame columns, so a call
site changes only its import:

    from vndata import Quote, Company, Finance, Listing, Trading, Market

Sources: VCI (prices, statements, company data, listings, price board),
SSI iBoard (price fallback), VNDirect (USD/VND).

Every call raises `vndata.SourceError` (or a `requests` error) on failure, and
returns an empty DataFrame when the source simply has no rows.
"""
from ._http import SourceError
from .company import Company
from .finance import Finance
from .listing import Listing
from .market import Market
from .quote import Quote
from .trading import Trading

__all__ = ["Company", "Finance", "Listing", "Market", "Quote", "SourceError", "Trading"]
