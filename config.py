import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# --- Timezone ---
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# --- VND scale: all financial figures stored in billions VND ---
VND_BILLION = 1_000_000_000
VND_MILLION = 1_000_000

# --- Settlement ---
# Vietnam uses T+2.5: trades execute 2.5 trading-days before settlement.
# When comparing "current price" to fundamental data, account for this lag.
SETTLEMENT_DAYS = 2.5

# --- Exchange ---
EXCHANGE = "HOSE"
MARKET_OPEN_HOUR = 9   # ICT (UTC+7)
MARKET_CLOSE_HOUR = 15  # ICT

# --- Data sources ---
VNSTOCK_SOURCE = "VCI"     # default; fallback to "TCBS"
FIREANT_API_BASE = "https://restv2.fireant.vn"
FIREANT_API_KEY = os.getenv("FIREANT_API_KEY", "")

# --- Database ---
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "vnindex.db")
# If DATABASE_URL is set (e.g. Supabase Postgres connection string), use it.
# Otherwise fall back to the local SQLite file.
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# --- Valuation / WACC parameters ---
# CAPM inputs for Vietnamese market
RF = 0.05                  # Risk-free rate: VN 10-year government bond
ERP = 0.08                 # Vietnam equity risk premium
DEFAULT_BETA = 1.2         # Market-average beta for HOSE
DEFAULT_COD = 0.08         # Average cost of debt on HOSE
TAX_RATE = 0.20            # Standard Vietnamese corporate income tax rate

# DCF model settings
DCF_DISCOUNT_RATE = RF + DEFAULT_BETA * ERP  # 14.6% — CAPM-derived WACC baseline
DCF_TERMINAL_GROWTH = 0.03   # VN long-run GDP growth
DCF_PROJECTION_YEARS = 5

# Market P/E reference for implied-price calculation (HOSE historical average)
MARKET_PE = 15

# --- Screening thresholds (Undervalued Watchlist) ---
SCREEN_MIN_DCF_UPSIDE = 0.20   # DCF upside > 20%
SCREEN_MAX_PB = 2.0            # P/B < 2
SCREEN_MIN_FCF = 0             # positive FCF only

# --- Refresh schedule (ICT) ---
PRICE_REFRESH_HOUR = 17    # daily at 5pm
FINANCIAL_REFRESH_DAY = 6  # Sunday (APScheduler: 0=Mon … 6=Sun)
