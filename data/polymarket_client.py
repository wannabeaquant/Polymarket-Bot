"""
Polymarket data client — wraps the CLOB API and Gamma API.
Provides clean interfaces for market data, order books, trades, and order execution.
"""

import time
import logging
import asyncio
import aiohttp
import requests
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds, OrderArgs, OrderType,
    MarketOrderArgs,
)

BUY  = "BUY"
SELL = "SELL"

log = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class OrderBookSnapshot:
    market_id: str
    token_id: str          # YES or NO token
    timestamp: float
    bids: list[tuple[float, float]]   # (price, size) sorted desc
    asks: list[tuple[float, float]]   # (price, size) sorted asc
    last_trade_price: Optional[float] = None

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def vwap_buy(self, usdc_amount: float) -> Optional[float]:
        """Simulate buying `usdc_amount` worth, return VWAP fill price."""
        remaining = usdc_amount
        total_tokens = 0.0
        for price, size in self.asks:
            available_usdc = price * size
            if available_usdc >= remaining:
                total_tokens += remaining / price
                remaining = 0
                break
            else:
                total_tokens += size
                remaining -= available_usdc
        if remaining > 0:
            return None  # Insufficient depth
        return usdc_amount / total_tokens

    def vwap_sell(self, usdc_amount: float) -> Optional[float]:
        """Simulate selling `usdc_amount` worth, return VWAP fill price."""
        remaining = usdc_amount
        total_usdc = 0.0
        for price, size in self.bids:
            available_usdc = price * size
            if available_usdc >= remaining:
                total_usdc += remaining
                remaining = 0
                break
            else:
                total_usdc += available_usdc
                remaining -= available_usdc
        if remaining > 0:
            return None
        return total_usdc / usdc_amount


@dataclass
class MarketInfo:
    condition_id: str
    question: str
    category: str
    end_date: Optional[datetime]
    volume_24h: float
    yes_token_id: str
    no_token_id: str
    active: bool
    resolved: bool
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def days_to_resolution(self) -> Optional[float]:
        if self.end_date is None:
            return None
        now = datetime.now(timezone.utc)
        end = self.end_date if self.end_date.tzinfo else self.end_date.replace(tzinfo=timezone.utc)
        return (end - now).total_seconds() / 86400


@dataclass
class Trade:
    market_id: str
    token_id: str
    side: str          # "BUY" or "SELL"
    price: float
    size: float
    timestamp: float


# ─── Client ───────────────────────────────────────────────────────────────────

class PolymarketClient:
    """
    Unified client for Polymarket CLOB + Gamma APIs.
    Use in read-only mode without credentials for scanning.
    Use with credentials for order execution.
    """

    def __init__(
        self,
        private_key: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        host: str = "https://clob.polymarket.com",
        gamma_host: str = "https://gamma-api.polymarket.com",
        chain_id: int = 137,
    ):
        self.host = host
        self.gamma_host = gamma_host
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-bot/1.0"})
        self._clob: Optional[ClobClient] = None

        if private_key:
            try:
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                )
                self._clob = ClobClient(
                    host=host,
                    chain_id=chain_id,
                    key=private_key,
                    creds=creds,
                )
                log.info("ClobClient initialized with credentials")
            except Exception as e:
                log.error(f"Failed to init ClobClient: {e}")

    # ── Market Data ──────────────────────────────────────────────────────────

    def get_markets(
        self,
        active: bool = True,
        category: Optional[str] = None,
        limit: int = 500,
    ) -> list[MarketInfo]:
        """
        Fetch markets from CLOB API (paginated) with token IDs.
        Falls back to Gamma API for volume/category metadata.
        """
        try:
            # Gamma API has active markets with volume + clobTokenIds
            params: dict = {
                "limit": min(limit, 500),
                "order": "volume24hr",
                "ascending": "false",
            }
            if active:
                params["active"] = "true"
                params["closed"] = "false"
            if category:
                params["category"] = category

            r = self._session.get(f"{self.gamma_host}/markets", params=params, timeout=15)
            r.raise_for_status()
            raw = r.json()

            markets = []
            for m in raw:
                # clobTokenIds is a JSON string: '["tokenA", "tokenB"]'
                clob_ids = m.get("clobTokenIds", [])
                if isinstance(clob_ids, str):
                    import json as _json
                    try:
                        clob_ids = _json.loads(clob_ids)
                    except Exception:
                        clob_ids = []

                yes_token = clob_ids[0] if len(clob_ids) > 0 else ""
                no_token  = clob_ids[1] if len(clob_ids) > 1 else ""

                if not yes_token:
                    continue  # Skip markets without tradeable tokens

                end_date = None
                raw_date = m.get("endDateIso") or m.get("endDate")
                if raw_date:
                    try:
                        end_date = datetime.fromisoformat(
                            str(raw_date).replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

                outcomes = m.get("outcomes", ["Yes", "No"])
                if isinstance(outcomes, str):
                    import json as _json
                    try:
                        outcomes = _json.loads(outcomes)
                    except Exception:
                        outcomes = ["Yes", "No"]

                markets.append(MarketInfo(
                    condition_id=m.get("conditionId", ""),
                    question=m.get("question", ""),
                    category=m.get("category", ""),
                    end_date=end_date,
                    volume_24h=float(m.get("volume24hr", 0) or 0),
                    yes_token_id=yes_token,
                    no_token_id=no_token,
                    active=m.get("active", True),
                    resolved=m.get("closed", False),
                    description=m.get("description", ""),
                    tags=m.get("tags", []),
                ))
            return markets
        except Exception as e:
            log.error(f"get_markets failed: {e}")
            return []

    def get_order_book(self, token_id: str) -> Optional[OrderBookSnapshot]:
        """Fetch current order book for a token."""
        try:
            if self._clob:
                book = self._clob.get_order_book(token_id)
                bids = [(float(b.price), float(b.size)) for b in (book.bids or [])]
                asks = [(float(a.price), float(a.size)) for a in (book.asks or [])]
            else:
                r = self._session.get(
                    f"{self.host}/book",
                    params={"token_id": token_id},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                bids = [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
                asks = [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]

            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])
            return OrderBookSnapshot(
                market_id="",
                token_id=token_id,
                timestamp=time.time(),
                bids=bids,
                asks=asks,
            )
        except Exception as e:
            log.error(f"get_order_book({token_id}) failed: {e}")
            return None

    def get_last_trades(self, token_id: str = "", market_id: str = "", limit: int = 200) -> list[Trade]:
        """Fetch recent trades via the public data API (no auth needed)."""
        try:
            params: dict = {"limit": limit}
            if market_id:
                params["market"] = market_id
            elif token_id:
                params["asset"] = token_id
            else:
                return []
            r = self._session.get(
                "https://data-api.polymarket.com/trades",
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            trades = []
            for t in r.json():
                trades.append(Trade(
                    market_id=t.get("conditionId", ""),
                    token_id=t.get("asset", token_id),
                    side=t.get("side", ""),
                    price=float(t.get("price", 0)),
                    size=float(t.get("size", 0)),
                    timestamp=float(t.get("timestamp", 0)),
                ))
            return trades
        except Exception as e:
            log.error(f"get_last_trades failed: {e}")
            return []

    def get_mid_price(self, token_id: str) -> Optional[float]:
        """Quick mid-price fetch."""
        try:
            r = self._session.get(
                f"{self.host}/midpoint",
                params={"token_id": token_id},
                timeout=5,
            )
            r.raise_for_status()
            return float(r.json().get("mid", 0))
        except Exception:
            book = self.get_order_book(token_id)
            return book.mid if book else None

    # ── Order Execution ───────────────────────────────────────────────────────

    def place_limit_order(
        self,
        token_id: str,
        side: str,       # "BUY" or "SELL"
        price: float,
        size: float,
    ) -> Optional[dict]:
        """Place a limit order. Requires credentials."""
        if not self._clob:
            raise RuntimeError("Credentials required for order placement")
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=round(price, 4),
                size=round(size, 2),
                side=side,  # "BUY" or "SELL"
            )
            resp = self._clob.create_order(order_args)
            log.info(f"Limit order placed: {side} {size} @ {price} | {resp}")
            return resp
        except Exception as e:
            log.error(f"place_limit_order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self._clob:
            return False
        try:
            self._clob.cancel(order_id)
            return True
        except Exception as e:
            log.error(f"cancel_order({order_id}) failed: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders across all markets."""
        if not self._clob:
            return False
        try:
            self._clob.cancel_all()
            return True
        except Exception as e:
            log.error(f"cancel_all_orders failed: {e}")
            return False

    def get_open_orders(self) -> list[dict]:
        """Fetch all open orders."""
        if not self._clob:
            return []
        try:
            return self._clob.get_orders() or []
        except Exception as e:
            log.error(f"get_open_orders failed: {e}")
            return []
