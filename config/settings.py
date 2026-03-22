"""
Central configuration for Polymarket trading bot.
Copy to settings_local.py and fill in credentials.
"""

# ─── Polymarket CLOB API ───────────────────────────────────────────────────────
POLYMARKET_HOST = "https://clob.polymarket.com"
GAMMA_HOST      = "https://gamma-api.polymarket.com"
CHAIN_ID        = 137          # Polygon mainnet

# Set these in environment or settings_local.py
PRIVATE_KEY     = ""           # EVM private key (0x...)
API_KEY         = ""
API_SECRET      = ""
API_PASSPHRASE  = ""

# ─── Trading Parameters ───────────────────────────────────────────────────────
MIN_NOTIONAL_USDC        = 10.0    # Min trade size in USDC
MAX_NOTIONAL_USDC        = 5000.0  # Max single trade size
MAX_PORTFOLIO_USDC       = 50_000  # Total capital at risk
FRACTIONAL_KELLY         = 0.25    # Use 25% of full Kelly
MAX_POSITION_FRACTION    = 0.10    # Max 10% of portfolio per market

# ─── Market Making ─────────────────────────────────────────────────────────────
MM_RISK_AVERSION         = 0.1     # γ in Avellaneda-Stoikov (higher = tighter quotes)
MM_INVENTORY_LIMIT       = 500     # Max token inventory per side before skewing
MM_MIN_SPREAD            = 0.02    # Never quote tighter than 2 cents
MM_MAX_SPREAD            = 0.15    # Never quote wider than 15 cents
MM_ORDER_SIZE_USDC       = 100     # Base order size per level
MM_QUOTE_LEVELS          = 3       # How many price levels to quote
MM_REFRESH_INTERVAL_S    = 10      # Requote every N seconds
MM_PULL_VPIN_THRESHOLD   = 0.6     # Pull quotes if VPIN > this

# ─── Arbitrage ─────────────────────────────────────────────────────────────────
ARB_MIN_PROFIT_CENTS     = 2.0     # Min profit in cents per contract after fees
ARB_VWAP_CHECK_DEPTH     = 500     # Tokens to simulate buying for VWAP check
ARB_MAX_SLIPPAGE         = 0.005   # Max acceptable slippage (0.5%)

# ─── VPIN ─────────────────────────────────────────────────────────────────────
VPIN_BUCKET_SIZE         = 50      # Trades per VPIN bucket (tune per market)
VPIN_WINDOW_BUCKETS      = 50      # Rolling window of buckets
VPIN_INFORMED_THRESHOLD  = 0.6

# ─── Hawkes Mean Reversion ────────────────────────────────────────────────────
HAWKES_REVERSION_LOOKBACK_H = 24   # Hours of history for Hawkes calibration
HAWKES_MIN_SHOCK_SIZE       = 0.05 # Min price move to consider a shock (5 cents)
HAWKES_TRADE_HORIZON_H      = 4    # Hours over which to expect mean reversion

# ─── Market Selection ─────────────────────────────────────────────────────────
TARGET_MARKET_CATEGORIES = [
    "politics", "crypto", "sports", "science", "economics",
]
MIN_MARKET_VOLUME_24H    = 1_000   # USDC
MAX_MARKET_RESOLUTION_DAYS = 90    # Don't trade markets resolving > 90d out
IGNORE_MARKETS_ENDING_IN  = 7      # Hours before resolution to stop trading

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = "logs/bot.log"
