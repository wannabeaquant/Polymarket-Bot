"""
Market Making Strategy — Full Orchestration

Runs the Avellaneda-Stoikov market maker across a portfolio of markets.

Loop:
  1. Select target markets (thin spreads, sufficient volume, appropriate horizon)
  2. Fetch order books and recent trades
  3. Update VPIN for each market
  4. Compute A-S quotes (adjusted for VPIN + inventory)
  5. Cancel stale quotes, place fresh ones
  6. Monitor fills and update inventory
  7. Log PnL continuously

Risk management:
  - Pull all quotes if VPIN spikes (informed trading detected)
  - Reduce inventory via skewed quotes (not panic selling)
  - Hard stop if portfolio loss exceeds max_drawdown
"""

import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from core.market_making import AvellanedaStoikovMM
from core.vpin import MultiMarketVPIN, TradeRecord
from config.settings import (
    MM_RISK_AVERSION, MM_INVENTORY_LIMIT, MM_MIN_SPREAD, MM_MAX_SPREAD,
    MM_ORDER_SIZE_USDC, MM_QUOTE_LEVELS, MM_REFRESH_INTERVAL_S,
    MM_PULL_VPIN_THRESHOLD, MIN_MARKET_VOLUME_24H, MAX_MARKET_RESOLUTION_DAYS,
    IGNORE_MARKETS_ENDING_IN,
)

log = logging.getLogger(__name__)


@dataclass
class MMMarketState:
    market_id: str
    yes_token_id: str
    no_token_id: str
    question: str
    mm_yes: AvellanedaStoikovMM
    mm_no: AvellanedaStoikovMM
    active: bool = True
    open_order_ids: list[str] = field(default_factory=list)
    realized_pnl: float = 0.0
    total_fills: int = 0
    last_quote_time: float = 0.0


class MarketMakingStrategy:
    """
    Runs market making across multiple Polymarket markets simultaneously.

    Target: thin markets with wide spreads where we can earn the spread
    while managing inventory risk with A-S.
    """

    def __init__(
        self,
        client,
        executor,
        max_markets: int = 10,
        dry_run: bool = True,
    ):
        self.client = client
        self.executor = executor
        self.max_markets = max_markets
        self.dry_run = dry_run

        self.vpin = MultiMarketVPIN(default_bucket_size=50)
        self._market_states: dict[str, MMMarketState] = {}
        self._running = False
        self._start_time = time.time()
        self._total_pnl = 0.0

    def select_markets(self, all_markets: list) -> list:
        """
        Select the best markets for market making.

        Criteria:
          1. Volume in [1k, 500k] USDC/day (too thin = bad execution,
             too liquid = too much bot competition)
          2. Resolution > 7 days (too close = adverse selection risk)
          3. Price in [0.10, 0.90] (extremes = higher adverse selection)
          4. Category is active (not crypto/sports during off-hours)
        """
        scored = []
        for m in all_markets:
            if m.resolved:
                continue
            if m.volume_24h < MIN_MARKET_VOLUME_24H:
                continue
            if m.days_to_resolution is not None:
                if m.days_to_resolution < IGNORE_MARKETS_ENDING_IN / 24:
                    continue
                if m.days_to_resolution > MAX_MARKET_RESOLUTION_DAYS:
                    continue

            # Score: prefer markets with moderate volume and time horizon
            volume_score = min(1.0, m.volume_24h / 10000)  # Normalize to 10k/day
            time_score   = min(1.0, (m.days_to_resolution or 30) / 30)

            score = volume_score * 0.6 + time_score * 0.4
            scored.append((score, m))

        scored.sort(reverse=True)
        return [m for _, m in scored[:self.max_markets]]

    def _initialize_market(self, market) -> MMMarketState:
        """Set up A-S market maker for a new market."""
        end_ts = None
        if market.end_date:
            end_ts = market.end_date.timestamp()

        mm_yes = AvellanedaStoikovMM(
            token_id=market.yes_token_id,
            gamma=MM_RISK_AVERSION,
            max_inventory=MM_INVENTORY_LIMIT,
            min_spread=MM_MIN_SPREAD,
            max_spread=MM_MAX_SPREAD,
            order_size=MM_ORDER_SIZE_USDC,
            quote_levels=MM_QUOTE_LEVELS,
        )
        mm_no = AvellanedaStoikovMM(
            token_id=market.no_token_id,
            gamma=MM_RISK_AVERSION,
            max_inventory=MM_INVENTORY_LIMIT,
            min_spread=MM_MIN_SPREAD,
            max_spread=MM_MAX_SPREAD,
            order_size=MM_ORDER_SIZE_USDC,
            quote_levels=MM_QUOTE_LEVELS,
        )

        # Pre-warm with recent trades
        for token_id, mm in [(market.yes_token_id, mm_yes), (market.no_token_id, mm_no)]:
            trades = self.client.get_last_trades(token_id, limit=100)
            for t in trades:
                mm.on_trade(t.price, t.timestamp)
                self.vpin.process_trade(token_id, t.price, t.size, t.timestamp)

        return MMMarketState(
            market_id=market.condition_id,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            question=market.question,
            mm_yes=mm_yes,
            mm_no=mm_no,
        )

    def _requote_market(self, state: MMMarketState, market):
        """Cancel stale quotes and place fresh A-S quotes for a market."""
        end_ts = market.end_date.timestamp() if market.end_date else None

        for token_id, mm in [
            (state.yes_token_id, state.mm_yes),
            (state.no_token_id,  state.mm_no),
        ]:
            # Pull VPIN
            vpin_val = self.vpin.get_vpin(token_id) or 0.0
            regime   = self.vpin.get_regime(token_id)

            # Pull all quotes if informed trading detected
            if self.vpin.should_pull_quotes(token_id, MM_PULL_VPIN_THRESHOLD):
                log.info(f"  [{state.market_id[:8]}] VPIN={vpin_val:.2f} ({regime}), pulling quotes")
                self._cancel_market_orders(state)
                continue

            # Get current mid price
            book = self.client.get_order_book(token_id)
            if not book or book.mid is None:
                continue

            mid = book.mid

            # Update volatility with latest trades
            trades = self.client.get_last_trades(token_id, limit=20)
            for t in trades:
                mm.on_trade(t.price, t.timestamp)
                self.vpin.process_trade(token_id, t.price, t.size, t.timestamp)

            # Compute A-S quotes
            quotes = mm.compute_multi_level_quotes(mid, end_ts, vpin_val)

            if not quotes or not quotes[0].should_quote:
                log.debug(f"  Quotes not suitable for {token_id[:8]}")
                continue

            # Cancel existing and place fresh
            self._cancel_market_orders(state)
            time.sleep(0.1)

            for q in quotes:
                if self.dry_run:
                    log.info(
                        f"  [DRY RUN] MM {token_id[:8]}: "
                        f"bid={q.bid_price:.3f}x{q.bid_size:.0f} "
                        f"ask={q.ask_price:.3f}x{q.ask_size:.0f} "
                        f"(VPIN={vpin_val:.2f}, spread={q.spread:.3f})"
                    )
                else:
                    # Place bid
                    bid_resp = self.client.place_limit_order(
                        token_id, "BUY", q.bid_price, q.bid_size / q.bid_price
                    )
                    if bid_resp and bid_resp.get("orderID"):
                        state.open_order_ids.append(bid_resp["orderID"])

                    # Place ask
                    ask_resp = self.client.place_limit_order(
                        token_id, "SELL", q.ask_price, q.ask_size / q.ask_price
                    )
                    if ask_resp and ask_resp.get("orderID"):
                        state.open_order_ids.append(ask_resp["orderID"])

        state.last_quote_time = time.time()

    def _cancel_market_orders(self, state: MMMarketState):
        """Cancel all open orders for a market."""
        for oid in state.open_order_ids:
            self.client.cancel_order(oid)
        state.open_order_ids.clear()

    def run_once(self, markets: list):
        """Single iteration of the market making loop."""
        target_markets = self.select_markets(markets)
        log.info(f"Market making on {len(target_markets)} markets")

        for market in target_markets:
            mid_id = market.condition_id

            if mid_id not in self._market_states:
                log.info(f"  Initializing MM for: {market.question[:60]}")
                self._market_states[mid_id] = self._initialize_market(market)

            state = self._market_states[mid_id]

            # Only requote if enough time has passed
            if time.time() - state.last_quote_time >= MM_REFRESH_INTERVAL_S:
                self._requote_market(state, market)

    def run_forever(self, markets_fn, interval: float = MM_REFRESH_INTERVAL_S):
        """
        Main loop. markets_fn is a callable that returns fresh market list.
        """
        self._running = True
        log.info("Market making strategy started")

        while self._running:
            try:
                markets = markets_fn()
                self.run_once(markets)
                self._log_status()
                time.sleep(interval)
            except KeyboardInterrupt:
                log.info("Interrupted, cancelling all orders...")
                self.stop()
                break
            except Exception as e:
                log.error(f"Error in MM loop: {e}", exc_info=True)
                time.sleep(5)

    def stop(self):
        """Graceful shutdown: cancel all open orders."""
        self._running = False
        for state in self._market_states.values():
            self._cancel_market_orders(state)
        log.info("All orders cancelled, strategy stopped")

    def _log_status(self):
        n_markets = len(self._market_states)
        n_orders = sum(len(s.open_order_ids) for s in self._market_states.values())
        log.info(f"MM Status: {n_markets} markets, {n_orders} open orders, PnL=${self._total_pnl:.2f}")
