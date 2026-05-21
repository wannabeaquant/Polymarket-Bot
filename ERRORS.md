# Polymarket Bot — Errors & Failures Log

<!-- Claude adds here when something takes 2+ attempts -->
| Date | What didn't work | What worked instead | Note for next time |
|------|-----------------|--------------------|--------------------|
| Mar 2026 | Side enum in OrderArgs | String "BUY"/"SELL" | SDK doesn't export Side enum — always use string literals |
| Mar 2026 | Sorting tuples with MarketInfo objects | Sort only on primitive fields (str, float) | MarketInfo is not comparable — never put it in sort key |
| Mar 2026 | CLOB /markets for discovery | gamma-api.polymarket.com/markets | CLOB /markets doesn't include clobTokenIds in response |
| Mar 2026 | Getting tokenId from Gamma response directly | Parse clobTokenIds field as JSON string first | clobTokenIds is a JSON-encoded string inside the JSON response |
