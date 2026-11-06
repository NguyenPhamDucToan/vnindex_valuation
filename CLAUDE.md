# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A data pipeline + Streamlit dashboard for valuing HOSE (Ho Chi Minh Stock Exchange, Vietnam)
listed companies. It fetches financials/prices via the in-repo `vndata` package, computes DCF/multiples/ratio-based
intrinsic values, stores everything in SQLite (or Postgres/Supabase), and serves an interactive
dashboard for screening, watchlisting, and per-ticker deep dives.

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env              # fill in FIREANT_API_KEY, optionally DATABASE_URL

# Run the dashboard
streamlit run dashboard/app.py

# Tests
pytest                            # whole suite (pytest.ini: testpaths=tests)
pytest tests/test_dcf.py                     # one file
pytest tests/test_dcf.py::test_name -v       # one test

# Data pipeline (full HOSE universe — sequential, rate-limited, do not parallelize)
python run_pipeline.py                  # tickers -> financials -> prices -> valuations
python run_pipeline.py --from-prices    # skip financials, start from prices
python run_pipeline.py --valuations-only  # recompute valuations only, no API calls
python run_prices_then_valuations.py    # prices + valuations only (financials assumed current)
python run_macro.py                     # macro indicators (SBV, World Bank)
python -m collectors.bulk_load --tickers-only   # just refresh the ticker/company list
python -m collectors.compute_valuations --tickers VNM FPT VIC  # recompute specific tickers
python run_ticker_refresh.py            # weekly (financials_refresh.yml): ticker list, retire delisted, statements for new listings
python -m quality.audit [--dry-run]     # daily (refresh.yml): stale prices, missing quarters, unit errors -> data_quality_log

# Scheduled/background refresh (blocking process)
python daily_refresh.py --now     # run once immediately, then keep the scheduler running
                                   # Mon-Fri 18:30 ICT prices, Mon 03:00 ICT macro

# One-time DB migration (SQLite -> Supabase Postgres)
python migrate_to_supabase.py     # requires DATABASE_URL pointing at Postgres
```

There is no lint/format command configured in this repo.

## Architecture

### Data flow

```
VCI/SSI/VNDirect (vndata), FireAnt/NSO/SBV/World Bank APIs
        |
   collectors/*.py   (fetch + upsert into SQLite/Postgres via models/)
        |
   models/schema.py  (Company, Price, Financial, Valuation, PinnedTicker, MacroIndicator)
        |
   valuation/*.py    (pure functions: DCF, WACC, Graham, multiples, ratios, signals)
        |
   collectors/compute_valuations.py  (orchestrates valuation/* -> writes Valuation rows)
        |
   dashboard/app.py  (Streamlit UI reads from DB, renders screens/charts)
```

`config.py` is the single source of truth for tunables (VN market CAPM inputs, DB path/URL,
refresh schedule, screening thresholds) and is imported everywhere — check there first before
hardcoding a financial constant.

### Database layer

- `models/schema.py` defines all tables (SQLAlchemy 2.0 declarative). `Financial` is one row per
  `(ticker, period, period_type)` where `period_type` is `"Q"` (quarterly) or `"Y"` (annual); all
  monetary fields are stored in **VND billions**, prices in the `prices` table are in **thousands
  VND** (see the `* 1000` conversions at money boundaries, e.g.
  `collectors/compute_valuations.py::_latest_price`).
- `models/database.py` picks SQLite (local, WAL mode) vs Postgres (`DATABASE_URL` env var — e.g.
  Supabase) transparently based on `config.DB_URL`. Always use `get_session()` (context manager,
  auto commit/rollback) rather than instantiating sessions directly.
- Banking tickers (`sector == "Ngân hàng"`) use a different chart of accounts in vnstock's raw
  API; `collectors/financials.py` remaps bank-specific item_ids onto the same `Financial` schema
  columns (see the field-mapping table in its module docstring) so downstream valuation code
  doesn't need to know a company is a bank.

### Valuation layer (`valuation/`)

Pure, DB-agnostic functions — everything takes plain numbers/dicts in and returns numbers out,
which is why they're straightforward to unit test in isolation from `models`/`collectors`.

- `valuation/inputs.py` — turns stored quarterly `Financial` rows into a TTM (trailing-twelve-month)
  snapshot: flow items (revenue, net income, etc.) are **summed** over the last 4 quarters; stock
  items (balance sheet: assets, equity, debt) are taken from the **most recent quarter only**.
  This TTM dict is the shared input to everything downstream.
- `valuation/dcf.py` + `valuation/wacc.py` — FCFF-based DCF: `FCFF = NOPAT + D&A - CapEx - ΔNWC`,
  discounted at a CAPM-derived WACC, terminal value via Gordon growth.
- `valuation/graham.py`, `valuation/multiples.py` — Graham Number and 8 other implied-price
  methods (P/B, EV/EBITDA, EPV, P/S, residual income, P/OCF, FCFE...). `avg_intrinsic_value` in
  `compute_valuations.py` is the mean of whichever of these 10 methods produced a positive value
  for that ticker — methods silently drop out rather than raising when an input is missing.
- `valuation/ratios.py` — the 4 ratio dimensions (profitability, cash-flow quality, balance sheet,
  working-capital/CCC) that populate the rest of the `Valuation` row.
  All ratio functions return `None` on a zero/missing denominator instead of raising or dividing
  by zero.
  - `valuation/signals.py` — composite 0-100 quality score + 7-level Buy/Sell signal
  (`SIGNAL_ORDER = ["Strong Buy", ..., "Strong Sell"]`) combining upside % and quality score; used
  by both the dashboard's Watchlist/Screen views and `backtest_signals.py`.

### Market data (`vndata/`) and rate limiting

vnstock is gone. `vndata/` calls the sources' JSON APIs directly with `requests` — VCI for prices,
statements, company data, listings and the price board; SSI iBoard as the price fallback; VNDirect
for USD/VND — with no API key. Its classes (`Quote`, `Company`, `Finance`, `Listing`, `Trading`,
`Market`) keep vnstock's names, signatures and DataFrame columns, and `Finance._get_report` keeps
vnstock's `item_id` naming (`vndata/_names.py`, verified identical on all ~1,500 VCI titles), so the
item_id maps in `collectors/financials.py` and the web exporter are unchanged. Prices come back in
thousands of VND, indices in points. Tests: `tests/test_vndata.py` (offline, patched payloads).

The rate limit that shaped the collectors was vnstock's (Guest tier ~20 requests/minute, raised as
`SystemExit`); VCI itself served 30 statement calls in 4 s without complaint. The `SystemExit`
handling and inter-request `time.sleep()` calls in `collectors/` are still in place and harmless —
keep collection sequential until a CI run has shown VCI tolerates more from GitHub's runners.

### Dashboard (`dashboard/app.py`)

Single large Streamlit file (~7.5k lines). Structure to know:
- `load_*` functions are the DB read layer (most are `@st.cache_data`-wrapped queries feeding the
  UI — grep for `load_` to find the data a given screen uses).
- `vnstock_call()` / `vnstock_throttle()` (names kept from the vnstock era) wrap any live
  (non-cached) `vndata` call made directly from the dashboard (e.g. live quotes, intraday index) with the same retry/backoff discipline as
  the collectors.
- `dashboard/heatmap_component/` is a small custom Streamlit component (Plotly treemap with
  click-to-select-ticker) — `index.html` is the component's JS/HTML, wired up via
  `components.declare_component` in `__init__.py`.

### Tests

`tests/conftest.py` just adds the repo root to `sys.path` (no fixtures/DB setup) — tests import
`valuation/*` functions directly and exercise them with plain numbers, they don't hit the database.
