# Polymarket Bot — Session Memory

## Decisions (never re-litigate without flagging)
| Date | Decision | Why | What was rejected |
|------|----------|-----|-------------------|
| Mar 2026 | Target XRP/SOL UP/DOWN, not BTC/ETH 5-min | gabagool22 dominates BTC/ETH with 1ms HFT latency | BTC/ETH 5-min MM |
| Mar 2026 | Use Gamma API for discovery, not CLOB /markets | CLOB /markets doesn't return clobTokenIds correctly | CLOB /markets endpoint |
| Mar 2026 | Hawkes fade threshold: 5c shock, 4h reversion window | Backtest: 86% win rate on Netanyahu market | Smaller/larger thresholds |
| Mar 2026 | VPIN > 0.6 = pull quotes | Informed flow above this threshold destroys MM edge | Static spread widening |

## Backtest Baseline (March 2026)
- Historical MM: ~42.8% annualized
- Hawkes (Netanyahu): 159 shocks, 86% win, $192 PnL / 2 days
- Tier-1 arb: 0 opportunities found (market efficient)
- XRP 5-min spread: 5-10c ✅ | BTC 5-min: 1c ❌

## Session Logs
<!-- Claude writes here on "session end" -->

## Next Session Priorities
<!-- Updated each session -->
