# CLAUDE.md — polymarket-bot

Polymarket alpha trading bot. Python. Fully built and deployable.
**Always read this file before touching anything.**

## Architecture

```
bot.py                          # CLI entry point / orchestrator
config/settings.py              # All tunable params (thresholds, limits, intervals)
core/
  market_making.py              # Avellaneda-Stoikov MM (logit-space vol)
  vpin.py                       # VPIN adverse selection detector
  kelly.py                      # Kelly criterion + portfolio Kelly
  hawkes_reversion.py           # Hawkes process mean reversion
  arb_scanner.py                # 3-tier arb scanner
  bayesian_network.py           # Cross-market dependency inference
execution/executor.py           # Order executor (VWAP-checked, deduped)
strategies/
  market_maker_strategy.py      # A-S MM strategy wrapper
  crypto_arb.py                 # XRP/SOL UP/DOWN limit-order MM
  dispute_trader.py             # Ambiguous resolution scanner
data/
  polymarket_client.py          # API wrapper (Gamma + CLOB + data-api)
  websocket_stream.py           # Real-time WS + PriceHistoryFetcher
backtesting/
  backtest.py                   # Trade-replay backtest
  historical_backtest.py        # Price-history backtest (no auth needed)
```

## Running

```bash
python bot.py --mode report       # Intelligence scan (no creds)
python bot.py --mode crypto       # Crypto UP/DOWN MM (dry run)
python bot.py --mode dispute      # Dispute scanner (dry run)
python bot.py --mode backtest     # Historical backtest (no creds)
python bot.py --mode full --live  # Live trading (needs creds)
```

## Active Strategies

### 1. Crypto UP/DOWN Market Making (PRIMARY)
- Targets: XRP 5-min, SOL 5/15-min, BTC 1-hr UP/DOWN markets
- **NOT** BTC/ETH 5-min — dominated by gabagool22 at 1ms HFT latency
- Method: Post limit bids at mid-0.01 on both UP and DN sides simultaneously
- Edge: Both fill = total cost < $1.00 = locked risk-free profit
- Min edge: 2-4 cents per series
- File: `strategies/crypto_arb.py`

### 2. Dispute Trading (EPISODIC)
- Targets: Markets with ambiguous resolution language (invasion, suit, official, significant)
- Edge: 8-30c mispricing when resolution criteria are fuzzy
- NOT always-on — fires episodically on geopolitical/political events
- File: `strategies/dispute_trader.py`

### 3. Avellaneda-Stoikov MM
- Logit-space vol handles [0,1] bounded binary prices correctly
- VPIN-adjusted spread: widens on informed flow, pulls quotes when VPIN > 0.6
- Best for: markets with 5-15c spread + moderate volume
- File: `strategies/market_maker_strategy.py`

### 4. Hawkes Mean Reversion
- Detects 30-min price shocks ≥ 5c, fades them, targets 40% reversion in 4h
- Backtest: 86% win rate, $192 PnL on Netanyahu market over 2 days
- File: `core/hawkes_reversion.py`

### 5. Three-Tier Arb Scanner
- Tier 1: YES+NO < $1.00 on same market (rare)
- Tier 2: Complement set — buy all outcomes of multi-outcome event
- Tier 3: Parent-child dependency violations
- File: `core/arb_scanner.py`

## API — Critical Gotchas (DO NOT GET THESE WRONG)

- **Market discovery**: Use `gamma-api.polymarket.com` — NOT `/markets` on CLOB
- **Token IDs**: Parse from `clobTokenIds` JSON field in Gamma response (it's a JSON string, parse it)
- **Trade history**: `data-api.polymarket.com/trades?market=<conditionId>` (no auth needed)
- **Side param**: Use string `"BUY"` / `"SELL"` in OrderArgs — NOT the Side enum (it doesn't exist)
- **Sorting**: Never sort tuples containing `MarketInfo` objects — use only primitives

## Data Sources

| Source | URL | Use |
|--------|-----|-----|
| Gamma API | `https://gamma-api.polymarket.com` | Market metadata, discovery |
| CLOB API | `https://clob.polymarket.com` | Order books, trading |
| Data API | `https://data-api.polymarket.com` | Public trade history |
| WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | Real-time |
| Price History | `GET clob.polymarket.com/prices-history?market=<token_id>&interval=all&fidelity=1` | Historical prices |

## Backtest Results (March 2026)

- Historical MM: ~42.8% annualized return
- Netanyahu market mean reversion: 159 shocks, 86% win rate, $192 PnL
- XRP 5-min: 5-10c spreads (target)
- BTC 5-min: 1c spread (saturated, avoid)

## Conventions

- All new strategies go in `strategies/`, core math goes in `core/`
- `settings.py` is the single source of truth for all parameters — no hardcoded constants elsewhere
- Don't break existing dry-run modes — always keep `--live` flag gating real orders
- Run `python bot.py --mode report` first before any new strategy work to see current market state
- After changes: push to GitHub (https://github.com/wannabeaquant/Polymarket-Bot)

## Credentials (for live mode only)

Set as env vars:
```bash
export POLY_PRIVATE_KEY=0x...
export POLY_API_KEY=...
export POLY_API_SECRET=...
export POLY_API_PASSPHRASE=...
```
Never hardcode credentials. Never commit them.

## Git & Commits
- After each self-contained working change → suggest a commit with a pre-written message.
- Format: `<type>(<scope>): <short imperative description>`
  - feat(strategy): ... | fix(executor): ... | refactor(core): ... | test(backtest): ...
- One logical change per commit. Never batch unrelated changes.
- NEVER commit broken code, debug prints, or commented-out blocks.
- Push to GitHub at end of every session.

## Session Memory
- Read MEMORY.md at the start of every session.
- Never contradict a logged decision without flagging it first.
- On "session end" / "wrapping up": write summary to MEMORY.md.
  Include: Worked on / Completed / In progress / Decisions made / Next session priorities.

## Error Log
- Read ERRORS.md before suggesting any approach.
- When something takes 2+ attempts: log to ERRORS.md.
  Format: What didn't work / What worked / Note for next time.
