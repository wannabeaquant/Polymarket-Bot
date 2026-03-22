"""
Crypto UP/DOWN Market Strategy — Revised with Real Market Data

What we learned from live data:
  - BTC/ETH 5-min: 1-cent spread, gabagool22 dominates → DO NOT COMPETE HERE
  - XRP/SOL 5-min: 5-10 cent spreads, thin books → OUR OPPORTUNITY
  - The "arb" is really limit-order market making where BOTH sides fill < $1.00

Key mechanics (from actual order book data):
  XRP 5-min live book: UP bid=0.49 ask=0.54 | DN bid=0.42 ask=0.48
  - Sum of ASKS: 1.02 → no simultaneous-buy arb
  - Sum of MIDS: 0.965 → 3.5c edge if you can buy at mids
  - Strategy: post limit bids at mid, earn edge when both legs fill

The asymmetric edge:
  1. Post limit buy UP @ 0.49, DN @ 0.45 (both below fair value 0.50)
  2a. Both fill → combined cost 0.94, payout 1.00 → LOCK 6c risk-free
  2b. Only UP fills → you're long UP at 0.49 on a ~50/50 market → 1c edge
  2c. Only DN fills → you're long DN at 0.45 → 5c edge (you paid 10% below fair!)

  Either way, positive expected value. The key: only enter when price < 0.49.

Target markets (confirmed live, March 2026):
  - xrp-updown-5m: 5c spread, minimal bot competition
  - sol-updown-5m: 2-3c spread, moderate competition
  - sol-updown-15m: 2c spread
  - btc-updown-1h: occasionally 2-3c spread at market open
"""

import time
import logging
import json
import requests
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST  = "https://clob.polymarket.com"

# Market series to trade: (slug_prefix, interval_seconds, min_spread_cents_to_enter)
TARGET_SERIES = [
    ("xrp-updown-5m",   300,  4.0),   # Our primary target
    ("sol-updown-5m",   300,  2.5),
    ("sol-updown-15m",  900,  2.0),
    ("btc-updown-1h",   3600, 3.0),   # Only if spread is unusually wide
    ("eth-updown-15m",  900,  3.0),
]


@dataclass
class CryptoMarket:
    slug: str
    title: str
    up_token_id: str
    dn_token_id: str
    resolution_ts: int
    interval_sec: int
    up_bid: float
    up_ask: float
    dn_bid: float
    dn_ask: float

    @property
    def up_mid(self) -> float:
        return (self.up_bid + self.up_ask) / 2

    @property
    def dn_mid(self) -> float:
        return (self.dn_bid + self.dn_ask) / 2

    @property
    def sum_mids(self) -> float:
        return self.up_mid + self.dn_mid

    @property
    def mid_arb_profit_cents(self) -> float:
        """Potential profit if both sides fill at mid price."""
        return (1.0 - self.sum_mids) * 100

    @property
    def best_entry_profit_cents(self) -> float:
        """
        Profit if we post limit bids at slightly below mid.
        Conservative estimate: we fill at (mid - 0.01) on each side.
        """
        return self.mid_arb_profit_cents + 2.0  # 2c improvement from limit vs market

    @property
    def minutes_to_resolution(self) -> float:
        return (self.resolution_ts - time.time()) / 60

    @property
    def should_trade(self) -> bool:
        """Only trade if spread is wide AND we have time."""
        return (
            self.best_entry_profit_cents >= 2.0  # At least 2c expected edge
            and 2 < self.minutes_to_resolution < 50  # 2-50 min window
            and self.up_ask > 0 and self.dn_ask > 0
        )


class CryptoUPDOWNStrategy:
    """
    Markets thin crypto UP/DOWN markets with limit orders at slightly below mid.

    Unlike gabagool22 (who dominates BTC with 1-2ms latency):
    - We target XRP/SOL where the books are thin and bots are fewer
    - We use patience: post limit orders and wait for fills, don't chase
    - We size conservatively: $50-150 per position, many positions simultaneously
    """

    def __init__(
        self,
        client,
        executor,
        min_spread_to_trade: float = 2.0,   # cents
        position_size_usdc: float = 100,
        max_simultaneous: int = 8,
        dry_run: bool = True,
    ):
        self.client = client
        self.executor = executor
        self.min_spread = min_spread_to_trade / 100
        self.position_size = position_size_usdc
        self.max_simultaneous = max_simultaneous
        self.dry_run = dry_run
        self._session = requests.Session()
        self._open_up_positions: dict[str, float] = {}   # token_id → entry_price
        self._open_dn_positions: dict[str, float] = {}
        self._total_pnl = 0.0
        self._n_completed = 0

    def fetch_active_markets(self) -> list[CryptoMarket]:
        """Fetch all currently active crypto UP/DOWN markets from all series."""
        now = int(time.time())
        markets = []

        for series, interval, min_spread_c in TARGET_SERIES:
            # Check current and next 3 intervals
            current_ts = (now // interval) * interval
            for offset in range(4):
                ts = current_ts + offset * interval
                slug = f"{series}-{ts}"

                try:
                    r = self._session.get(
                        f"{GAMMA_HOST}/events",
                        params={"slug": slug},
                        timeout=5,
                    )
                    if r.status_code != 200 or not r.json():
                        continue

                    e = r.json()[0]
                    ms = e.get("markets", [])
                    if not ms or not ms[0].get("acceptingOrders"):
                        continue

                    m = ms[0]
                    clob_ids = m.get("clobTokenIds", [])
                    if isinstance(clob_ids, str):
                        clob_ids = json.loads(clob_ids)
                    if len(clob_ids) < 2:
                        continue

                    up_id, dn_id = clob_ids[0], clob_ids[1]

                    # Fetch live order books
                    up_r = self._session.get(
                        f"{CLOB_HOST}/book", params={"token_id": up_id}, timeout=5
                    )
                    dn_r = self._session.get(
                        f"{CLOB_HOST}/book", params={"token_id": dn_id}, timeout=5
                    )
                    if up_r.status_code != 200 or dn_r.status_code != 200:
                        continue

                    def parse_book(book_data, side):
                        levels = book_data.get(side, [])
                        valid = [
                            float(x["price"]) for x in levels
                            if 0.01 < float(x["price"]) < 0.99
                        ]
                        return valid[0] if valid else None

                    up_book = up_r.json()
                    dn_book = dn_r.json()

                    up_bids = sorted([float(b["price"]) for b in up_book.get("bids", [])
                                      if 0.01 < float(b["price"]) < 0.99], reverse=True)
                    up_asks = sorted([float(a["price"]) for a in up_book.get("asks", [])
                                      if 0.01 < float(a["price"]) < 0.99])
                    dn_bids = sorted([float(b["price"]) for b in dn_book.get("bids", [])
                                      if 0.01 < float(b["price"]) < 0.99], reverse=True)
                    dn_asks = sorted([float(a["price"]) for a in dn_book.get("asks", [])
                                      if 0.01 < float(a["price"]) < 0.99])

                    if not up_bids or not up_asks or not dn_bids or not dn_asks:
                        continue

                    market = CryptoMarket(
                        slug=slug,
                        title=e.get("title", "")[:50],
                        up_token_id=up_id,
                        dn_token_id=dn_id,
                        resolution_ts=ts + interval,
                        interval_sec=interval,
                        up_bid=up_bids[0],
                        up_ask=up_asks[0],
                        dn_bid=dn_bids[0],
                        dn_ask=dn_asks[0],
                    )

                    if market.best_entry_profit_cents >= min_spread_c:
                        markets.append(market)

                except Exception as ex:
                    log.debug(f"Error fetching {slug}: {ex}")
                    continue

                time.sleep(0.1)

        return markets

    def scan_and_log(self):
        """Scan all markets and log opportunities. Core intelligence loop."""
        markets = self.fetch_active_markets()

        log.info(f"\n{'='*60}")
        log.info(f"CRYPTO UP/DOWN SCAN | {len(markets)} tradeable markets")
        log.info(f"{'='*60}")

        for m in sorted(markets, key=lambda x: x.best_entry_profit_cents, reverse=True):
            status = "ENTRY" if m.should_trade else "watch"
            log.info(
                f"[{status}] {m.title}"
                f"\n  UP: {m.up_bid:.3f}/{m.up_ask:.3f} | "
                f"DN: {m.dn_bid:.3f}/{m.dn_ask:.3f}"
                f"\n  sum_mids={m.sum_mids:.4f} | "
                f"entry_profit={m.best_entry_profit_cents:.1f}c | "
                f"{m.minutes_to_resolution:.1f}min left"
            )

        return markets

    def run_once(self):
        """Single scan + execution cycle."""
        markets = self.scan_and_log()
        tradeable = [m for m in markets if m.should_trade]

        for m in tradeable[:3]:   # Max 3 new entries per cycle
            # Post limit bids at slightly below mid on both sides
            up_entry = round(m.up_mid - 0.01, 3)
            dn_entry = round(m.dn_mid - 0.01, 3)
            up_tokens = self.position_size / up_entry
            dn_tokens = self.position_size / dn_entry

            expected_profit = (1.0 - up_entry - dn_entry) * 100

            if self.dry_run:
                log.info(
                    f"\n[DRY RUN] ENTRY: {m.title}"
                    f"\n  BID UP @ {up_entry:.3f} ({up_tokens:.0f} tokens)"
                    f"\n  BID DN @ {dn_entry:.3f} ({dn_tokens:.0f} tokens)"
                    f"\n  If both fill: locked profit = {expected_profit:.1f}c = "
                    f"${expected_profit/100 * min(up_tokens, dn_tokens):.2f}"
                )
            else:
                # Place actual limit orders
                up_resp = self.client.place_limit_order(m.up_token_id, "BUY", up_entry, up_tokens)
                dn_resp = self.client.place_limit_order(m.dn_token_id, "BUY", dn_entry, dn_tokens)
                log.info(f"Orders placed: UP={up_resp}, DN={dn_resp}")

    def run_forever(self, interval: float = 30.0):
        log.info(f"Crypto UP/DOWN strategy started (dry_run={self.dry_run})")
        while True:
            try:
                self.run_once()
                time.sleep(interval)
            except KeyboardInterrupt:
                log.info("Stopped")
                break
            except Exception as e:
                log.error(f"Error: {e}", exc_info=True)
                time.sleep(5)
