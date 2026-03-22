"""
Real-time WebSocket streaming for Polymarket.

Two channels:
  - Market: public order book updates, price changes, fills (no auth)
  - User: your own orders and fills (requires API credentials)

RTDS (wss://ws-live-data.polymarket.com): ultra-low-latency feed.
Market (wss://ws-subscriptions-clob.polymarket.com/ws/market): standard.

This module handles reconnection, heartbeat, and dispatches events to
registered strategy callbacks.
"""

import json
import asyncio
import logging
import time
from typing import Callable, Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
USER_WS_URL   = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


@dataclass
class BookUpdate:
    token_id: str
    timestamp: float
    bids: list[tuple[float, float]]   # (price, size)
    asks: list[tuple[float, float]]
    event_type: str                    # "book", "best_bid_ask", "price_change"


@dataclass
class TradeEvent:
    market_id: str
    token_id: str
    price: float
    size: float
    side: str
    timestamp: float
    status: str   # MATCHED, MINED, CONFIRMED


class PolymarketWebSocketStream:
    """
    Subscribes to real-time order book and trade updates.

    Usage:
        stream = PolymarketWebSocketStream()
        stream.on_book_update(my_callback)
        stream.on_trade(my_trade_callback)
        await stream.subscribe(token_ids=["123...", "456..."])
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        reconnect_delay: float = 2.0,
        ping_interval: float = 10.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval

        self._book_callbacks: list[Callable] = []
        self._trade_callbacks: list[Callable] = []
        self._price_callbacks: list[Callable] = []
        self._running = False
        self._subscribed_tokens: list[str] = []

        # Latest state cache
        self._book_cache: dict[str, BookUpdate] = {}
        self._last_prices: dict[str, float] = {}

    def on_book_update(self, callback: Callable[[BookUpdate], None]):
        """Register callback for order book updates."""
        self._book_callbacks.append(callback)

    def on_trade(self, callback: Callable[[TradeEvent], None]):
        """Register callback for trade events."""
        self._trade_callbacks.append(callback)

    def on_price_change(self, callback: Callable[[str, float], None]):
        """Register callback for price changes: (token_id, new_price)."""
        self._price_callbacks.append(callback)

    def get_cached_book(self, token_id: str) -> Optional[BookUpdate]:
        return self._book_cache.get(token_id)

    def get_last_price(self, token_id: str) -> Optional[float]:
        return self._last_prices.get(token_id)

    async def subscribe(self, token_ids: list[str]):
        """Subscribe to market updates for given token IDs."""
        self._subscribed_tokens = token_ids
        self._running = True
        while self._running:
            try:
                await self._connect_market(token_ids)
            except Exception as e:
                log.error(f"WebSocket disconnected: {e}")
                if self._running:
                    log.info(f"Reconnecting in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)

    async def _connect_market(self, token_ids: list[str]):
        """Connect to the market channel and process messages."""
        import websockets
        log.info(f"Connecting to Polymarket market WS ({len(token_ids)} tokens)...")

        async with websockets.connect(
            MARKET_WS_URL,
            ping_interval=None,   # We handle pings manually
        ) as ws:
            # Subscribe
            await ws.send(json.dumps({
                "assets_ids": token_ids,
                "type": "market",
                "custom_feature_enabled": True,
            }))
            log.info(f"Subscribed to {len(token_ids)} tokens")

            # Start heartbeat task
            ping_task = asyncio.create_task(self._heartbeat(ws))

            try:
                async for raw_msg in ws:
                    if raw_msg == "PONG":
                        continue
                    try:
                        await self._handle_market_message(raw_msg)
                    except Exception as e:
                        log.error(f"Error handling message: {e}")
            finally:
                ping_task.cancel()

    async def _heartbeat(self, ws):
        """Send PING every ping_interval seconds."""
        while True:
            await asyncio.sleep(self.ping_interval)
            try:
                await ws.send("PING")
            except Exception:
                break

    async def _handle_market_message(self, raw: str):
        """Parse and dispatch a market channel message."""
        try:
            events = json.loads(raw)
            if not isinstance(events, list):
                events = [events]
        except json.JSONDecodeError:
            return

        for event in events:
            event_type = event.get("event_type", "")
            asset_id   = event.get("asset_id", "")
            ts = time.time()

            if event_type == "book":
                bids = [(float(b["price"]), float(b["size"]))
                        for b in event.get("bids", [])]
                asks = [(float(a["price"]), float(a["size"]))
                        for a in event.get("asks", [])]
                bids.sort(key=lambda x: x[0], reverse=True)
                asks.sort(key=lambda x: x[0])

                update = BookUpdate(
                    token_id=asset_id,
                    timestamp=ts,
                    bids=bids,
                    asks=asks,
                    event_type="book",
                )
                self._book_cache[asset_id] = update
                for cb in self._book_callbacks:
                    cb(update)

            elif event_type == "price_change":
                price = float(event.get("price", 0))
                self._last_prices[asset_id] = price
                for cb in self._price_callbacks:
                    cb(asset_id, price)

            elif event_type == "best_bid_ask":
                bid = float(event.get("bid_price", 0))
                ask = float(event.get("ask_price", 0))
                # Build minimal book update from BBA
                existing = self._book_cache.get(asset_id)
                if existing:
                    update = BookUpdate(
                        token_id=asset_id,
                        timestamp=ts,
                        bids=[(bid, existing.bids[0][1] if existing.bids else 0)],
                        asks=[(ask, existing.asks[0][1] if existing.asks else 0)],
                        event_type="best_bid_ask",
                    )
                    self._book_cache[asset_id] = update
                    for cb in self._book_callbacks:
                        cb(update)

            elif event_type == "last_trade_price":
                price = float(event.get("price", 0))
                self._last_prices[asset_id] = price
                trade = TradeEvent(
                    market_id=event.get("market_id", ""),
                    token_id=asset_id,
                    price=price,
                    size=float(event.get("size", 0)),
                    side=event.get("side", ""),
                    timestamp=ts,
                    status="CONFIRMED",
                )
                for cb in self._trade_callbacks:
                    cb(trade)

    def stop(self):
        self._running = False


class PriceHistoryFetcher:
    """
    Fetches OHLC price history for backtesting and signal calibration.

    Endpoint: GET https://clob.polymarket.com/prices-history
    """

    def __init__(self):
        import requests
        self._session = requests.Session()
        self._base = "https://clob.polymarket.com"

    def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",    # 1h, 6h, 1d, 1w, 1m, all
        fidelity: int = 1,       # minutes
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> list[tuple[float, float]]:
        """
        Returns list of (timestamp, price) tuples.
        """
        params: dict = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts

        try:
            r = self._session.get(
                f"{self._base}/prices-history",
                params=params,
                timeout=15,
            )
            r.raise_for_status()
            history = r.json().get("history", [])
            return [(float(h["t"]), float(h["p"])) for h in history]
        except Exception as e:
            log.error(f"get_price_history({token_id}) failed: {e}")
            return []

    def get_returns_for_hawkes(
        self,
        token_id: str,
        lookback_hours: int = 48,
    ) -> list[tuple[float, float]]:
        """
        Returns (timestamp, price) at 1-minute fidelity for Hawkes calibration.
        """
        end_ts = int(time.time())
        start_ts = end_ts - lookback_hours * 3600
        return self.get_price_history(
            token_id, interval="all", fidelity=1,
            start_ts=start_ts, end_ts=end_ts,
        )

    def detect_shocks_from_history(
        self,
        token_id: str,
        threshold_cents: float = 5.0,
        lookback_hours: int = 48,
    ) -> list[tuple[float, float, str]]:
        """
        Returns list of (timestamp, magnitude_cents, direction) for historical shocks.
        Used to calibrate the Hawkes process.
        """
        history = self.get_returns_for_hawkes(token_id, lookback_hours)
        if len(history) < 10:
            return []

        shocks = []
        window = 10  # 10-minute shock window
        for i in range(window, len(history)):
            ts_now, p_now = history[i]
            ts_prev, p_prev = history[i - window]
            move = (p_now - p_prev) * 100
            if abs(move) >= threshold_cents:
                direction = "UP" if move > 0 else "DOWN"
                shocks.append((ts_now, abs(move), direction))

        return shocks
