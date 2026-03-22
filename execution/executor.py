"""
Order Execution Engine

Handles the messy reality of CLOB trading:
  - Sequential order placement (not atomic)
  - Leg-by-leg execution with abort logic
  - Position tracking and deduplication
  - Rate limiting
  - Order confirmation and fill monitoring
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

log = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    PENDING   = "PENDING"
    PARTIAL   = "PARTIAL"
    COMPLETE  = "COMPLETE"
    FAILED    = "FAILED"
    ABORTED   = "ABORTED"


@dataclass
class ExecutionResult:
    trade_id: str
    status: ExecutionStatus
    legs_attempted: int
    legs_filled: int
    avg_fill_prices: dict[str, float]   # token_id → avg fill price
    total_cost_usdc: float
    pnl_estimate: float
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return self.status == ExecutionStatus.COMPLETE

    @property
    def fill_rate(self) -> float:
        return self.legs_filled / max(self.legs_attempted, 1)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 5.0):
        self.rate = calls_per_second
        self._tokens = calls_per_second
        self._last_refill = time.time()

    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def wait_and_acquire(self, timeout: float = 5.0):
        start = time.time()
        while not self.acquire():
            if time.time() - start > timeout:
                raise TimeoutError("Rate limit wait timeout")
            time.sleep(0.05)


class OrderExecutor:
    """
    Executes multi-leg trades on Polymarket's CLOB.

    Key design decisions:
      1. Check VWAP before committing to each leg
      2. Abort remaining legs if first leg fills at worse-than-expected price
      3. Never leave open orders dangling — always cancel on abort
      4. Deduplication: track recent trades to avoid double-execution
    """

    def __init__(
        self,
        client,
        dry_run: bool = True,            # ALWAYS start with dry_run=True
        max_slippage_pct: float = 0.005, # 0.5% max slippage per leg
        min_profit_to_abort: float = 0.01, # Abort arb if profit drops below this
    ):
        self.client = client
        self.dry_run = dry_run
        self.max_slippage = max_slippage_pct
        self.min_profit_to_abort = min_profit_to_abort
        self._rate_limiter = RateLimiter(calls_per_second=3.0)
        self._executed_trades: dict[str, float] = {}  # trade_id → timestamp
        self._open_order_ids: list[str] = []

    def execute_arb(self, opportunity, trade_id: str = "") -> ExecutionResult:
        """
        Execute an arbitrage opportunity leg by leg.
        Aborts gracefully if mid-execution conditions change.
        """
        import uuid
        trade_id = trade_id or str(uuid.uuid4())[:8]

        if not opportunity.legs:
            return ExecutionResult(
                trade_id=trade_id,
                status=ExecutionStatus.FAILED,
                legs_attempted=0, legs_filled=0,
                avg_fill_prices={}, total_cost_usdc=0, pnl_estimate=0,
                error_message="No legs in opportunity",
            )

        # Deduplication: don't execute the same opportunity twice in 60s
        opp_key = "_".join(sorted(leg["token_id"] for leg in opportunity.legs))
        if opp_key in self._executed_trades:
            if time.time() - self._executed_trades[opp_key] < 60:
                log.warning(f"Duplicate arb attempt blocked: {trade_id}")
                return ExecutionResult(
                    trade_id=trade_id,
                    status=ExecutionStatus.ABORTED,
                    legs_attempted=0, legs_filled=0,
                    avg_fill_prices={}, total_cost_usdc=0, pnl_estimate=0,
                    error_message="Duplicate execution blocked",
                )

        legs_filled = 0
        fill_prices = {}
        total_cost = 0.0
        placed_order_ids = []

        log.info(f"[{trade_id}] Executing arb: {opportunity.description}")

        for i, leg in enumerate(opportunity.legs):
            token_id = leg["token_id"]
            side     = leg["side"]
            target_price = leg["price"]
            size_usdc = leg["size_usdc"]

            # Re-check current VWAP before each leg
            current_book = self.client.get_order_book(token_id)
            if current_book is None:
                log.warning(f"[{trade_id}] Leg {i}: no order book, aborting")
                self._cancel_placed(placed_order_ids)
                return ExecutionResult(
                    trade_id=trade_id,
                    status=ExecutionStatus.ABORTED,
                    legs_attempted=i+1, legs_filled=legs_filled,
                    avg_fill_prices=fill_prices, total_cost_usdc=total_cost,
                    pnl_estimate=0,
                    error_message=f"No order book for leg {i}",
                )

            if side == "BUY":
                current_vwap = current_book.vwap_buy(size_usdc)
            else:
                current_vwap = current_book.vwap_sell(size_usdc)

            if current_vwap is None:
                log.warning(f"[{trade_id}] Leg {i}: insufficient depth, aborting")
                self._cancel_placed(placed_order_ids)
                return ExecutionResult(
                    trade_id=trade_id,
                    status=ExecutionStatus.ABORTED,
                    legs_attempted=i+1, legs_filled=legs_filled,
                    avg_fill_prices=fill_prices, total_cost_usdc=total_cost,
                    pnl_estimate=0,
                    error_message=f"Insufficient depth for leg {i}",
                )

            # Check slippage vs original target price
            slippage = abs(current_vwap - target_price) / max(target_price, 0.01)
            if slippage > self.max_slippage:
                log.warning(
                    f"[{trade_id}] Leg {i}: slippage {slippage:.2%} > max, aborting"
                )
                self._cancel_placed(placed_order_ids)
                return ExecutionResult(
                    trade_id=trade_id,
                    status=ExecutionStatus.ABORTED,
                    legs_attempted=i+1, legs_filled=legs_filled,
                    avg_fill_prices=fill_prices, total_cost_usdc=total_cost,
                    pnl_estimate=0,
                    error_message=f"Slippage {slippage:.2%} too high on leg {i}",
                )

            # Calculate token quantity from USDC amount
            token_size = size_usdc / current_vwap if current_vwap > 0 else 0

            if self.dry_run:
                log.info(
                    f"[DRY RUN] [{trade_id}] Leg {i}: {side} "
                    f"{token_size:.1f} {token_id[:8]}... @ {current_vwap:.3f}"
                )
                fill_prices[token_id] = current_vwap
                total_cost += size_usdc if side == "BUY" else -size_usdc
                legs_filled += 1
            else:
                self._rate_limiter.wait_and_acquire()
                resp = self.client.place_limit_order(
                    token_id=token_id,
                    side=side,
                    price=round(current_vwap * (1.002 if side == "BUY" else 0.998), 4),
                    size=round(token_size, 1),
                )
                if resp:
                    order_id = resp.get("orderID", "")
                    if order_id:
                        placed_order_ids.append(order_id)
                        self._open_order_ids.append(order_id)
                    fill_prices[token_id] = current_vwap
                    total_cost += size_usdc if side == "BUY" else -size_usdc
                    legs_filled += 1
                else:
                    log.error(f"[{trade_id}] Leg {i} failed to place order")
                    self._cancel_placed(placed_order_ids)
                    return ExecutionResult(
                        trade_id=trade_id,
                        status=ExecutionStatus.FAILED,
                        legs_attempted=i+1, legs_filled=legs_filled,
                        avg_fill_prices=fill_prices, total_cost_usdc=total_cost,
                        pnl_estimate=0,
                        error_message=f"Order placement failed on leg {i}",
                    )

            time.sleep(0.1)  # Small delay between legs

        # All legs executed
        self._executed_trades[opp_key] = time.time()

        pnl = opportunity.net_profit_cents / 100 * (size_usdc * len(opportunity.legs))

        log.info(
            f"[{trade_id}] Arb complete: {legs_filled}/{len(opportunity.legs)} legs, "
            f"est. PnL=${pnl:.2f}"
        )

        return ExecutionResult(
            trade_id=trade_id,
            status=ExecutionStatus.COMPLETE,
            legs_attempted=len(opportunity.legs),
            legs_filled=legs_filled,
            avg_fill_prices=fill_prices,
            total_cost_usdc=total_cost,
            pnl_estimate=pnl,
        )

    def execute_kelly_trade(
        self,
        token_id: str,
        side: str,
        size_usdc: float,
        kelly_result,
        trade_id: str = "",
    ) -> ExecutionResult:
        """Execute a Kelly-sized directional trade."""
        import uuid
        trade_id = trade_id or str(uuid.uuid4())[:8]

        book = self.client.get_order_book(token_id)
        if not book:
            return ExecutionResult(
                trade_id=trade_id, status=ExecutionStatus.FAILED,
                legs_attempted=1, legs_filled=0, avg_fill_prices={},
                total_cost_usdc=0, pnl_estimate=0,
                error_message="No order book",
            )

        if side == "BUY":
            vwap = book.vwap_buy(size_usdc)
        else:
            vwap = book.vwap_sell(size_usdc)

        if vwap is None:
            return ExecutionResult(
                trade_id=trade_id, status=ExecutionStatus.FAILED,
                legs_attempted=1, legs_filled=0, avg_fill_prices={},
                total_cost_usdc=0, pnl_estimate=0,
                error_message="Insufficient depth",
            )

        token_size = size_usdc / vwap

        if self.dry_run:
            log.info(
                f"[DRY RUN] [{trade_id}] Kelly {side} {token_size:.1f} @ {vwap:.3f} "
                f"(edge={kelly_result.edge*100:.1f}c)"
            )
            return ExecutionResult(
                trade_id=trade_id, status=ExecutionStatus.COMPLETE,
                legs_attempted=1, legs_filled=1,
                avg_fill_prices={token_id: vwap},
                total_cost_usdc=size_usdc,
                pnl_estimate=kelly_result.edge * token_size,
            )

        self._rate_limiter.wait_and_acquire()
        resp = self.client.place_limit_order(token_id, side, round(vwap, 4), round(token_size, 1))
        if resp:
            return ExecutionResult(
                trade_id=trade_id, status=ExecutionStatus.COMPLETE,
                legs_attempted=1, legs_filled=1,
                avg_fill_prices={token_id: vwap},
                total_cost_usdc=size_usdc,
                pnl_estimate=kelly_result.edge * token_size,
            )

        return ExecutionResult(
            trade_id=trade_id, status=ExecutionStatus.FAILED,
            legs_attempted=1, legs_filled=0, avg_fill_prices={},
            total_cost_usdc=0, pnl_estimate=0,
            error_message="Order placement failed",
        )

    def _cancel_placed(self, order_ids: list[str]):
        """Cancel all orders placed so far (abort cleanup)."""
        for oid in order_ids:
            try:
                self.client.cancel_order(oid)
                if oid in self._open_order_ids:
                    self._open_order_ids.remove(oid)
            except Exception as e:
                log.error(f"Failed to cancel order {oid}: {e}")

    def cancel_all_open(self):
        """Emergency: cancel everything."""
        self.client.cancel_all_orders()
        self._open_order_ids.clear()
