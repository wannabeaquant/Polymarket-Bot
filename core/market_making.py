"""
Avellaneda-Stoikov Optimal Market Making for Binary Prediction Markets

The A-S model was designed for continuous assets but works exceptionally well
on Polymarket's thin CLOBs. Key adaptation: price dynamics modeled in logit-space
since binary contracts are bounded [0,1] and variance explodes near extremes in
price-space but is well-behaved in log-odds space.

Theory:
  - Reservation price: r = s - q·γ·σ²·(T-t)
  - Optimal spread:    δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)

  Where:
    s = current mid price (in logit-space, converted back)
    q = current inventory (positive = long YES, negative = short YES)
    γ = risk aversion parameter
    σ² = price variance per unit time
    T-t = time remaining (normalized)
    k = order arrival intensity (orders per unit time per unit spread)
"""

import math
import logging
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ─── Quote Output ─────────────────────────────────────────────────────────────

@dataclass
class MarketMakerQuotes:
    token_id: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    mid: float
    reservation_price: float
    spread: float
    inventory_skew: float    # How much we skewed due to inventory
    confidence: float        # 0-1, how confident we are in this quote
    timestamp: float = field(default_factory=time.time)

    @property
    def should_quote(self) -> bool:
        """Don't quote if spread is too tight or confidence is low."""
        return self.spread >= 0.005 and self.confidence >= 0.3

    def __repr__(self):
        return (
            f"Quotes({self.bid_price:.3f}/{self.ask_price:.3f} "
            f"spread={self.spread:.3f} skew={self.inventory_skew:+.3f})"
        )


# ─── Logit-Space Volatility Estimator ─────────────────────────────────────────

class LogitVolatilityEstimator:
    """
    Estimates price volatility in logit-space.

    Raw price variance is useless near 0 and 1 (heteroskedastic).
    In logit-space: logit(p) = log(p / (1-p)), variance is homoskedastic.

    We estimate σ² in logit-space and convert quotes back to price-space.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._logit_prices: deque = deque(maxlen=window)
        self._timestamps: deque   = deque(maxlen=window)

    def update(self, price: float, timestamp: float):
        if 0.001 < price < 0.999:  # Guard against logit overflow
            self._logit_prices.append(math.log(price / (1 - price)))
            self._timestamps.append(timestamp)

    def sigma_squared(self) -> float:
        """Returns variance per second in logit-space."""
        if len(self._logit_prices) < 10:
            return 0.01  # Default: assume moderate volatility
        prices = np.array(self._logit_prices)
        times  = np.array(self._timestamps)
        if times[-1] == times[0]:
            return 0.01
        # Annualized variance in logit-space, then per-second
        returns = np.diff(prices)
        time_diffs = np.diff(times)
        time_diffs = np.where(time_diffs < 1e-6, 1e-6, time_diffs)
        var_per_sec = np.var(returns / np.sqrt(time_diffs))
        return max(var_per_sec, 1e-6)

    def sigma_squared_price_space(self, mid: float) -> float:
        """
        Convert logit-space variance to price-space variance at current mid.
        Using delta method: σ²_price ≈ [p(1-p)]² × σ²_logit
        """
        if not (0.001 < mid < 0.999):
            return 0.001
        logit_var = self.sigma_squared()
        jacobian = mid * (1 - mid)   # dp/d(logit(p))
        return jacobian ** 2 * logit_var


# ─── Order Arrival Intensity Estimator ───────────────────────────────────────

class OrderArrivalEstimator:
    """
    Estimates k (order arrival intensity) for the A-S model.
    k = λ × α where λ = order arrival rate, α = fill probability per spread unit.
    Empirically estimated from recent trade frequency.
    """

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self._trade_times: deque = deque()

    def record_trade(self, timestamp: float):
        self._trade_times.append(timestamp)
        # Prune old trades
        cutoff = time.time() - self.window_seconds
        while self._trade_times and self._trade_times[0] < cutoff:
            self._trade_times.popleft()

    def arrival_rate(self) -> float:
        """Trades per second in the observation window."""
        if not self._trade_times:
            return 0.1  # Default: assume 1 trade per 10 seconds
        return len(self._trade_times) / self.window_seconds

    def k(self, spread: float) -> float:
        """
        Order arrival intensity at the given spread.
        Approximation: k ≈ λ × e^(-α × spread)
        Where α ≈ 1.5 is empirically calibrated for thin markets.
        """
        lambda_ = self.arrival_rate()
        alpha = 1.5
        return lambda_ * math.exp(-alpha * spread)


# ─── Inventory Manager ────────────────────────────────────────────────────────

@dataclass
class InventoryState:
    token_id: str
    net_position: float  # Positive = long YES tokens, negative = short
    avg_entry_price: float
    max_position: float  # Configured limit

    @property
    def inventory_fraction(self) -> float:
        """Returns -1 to +1 normalized inventory."""
        if self.max_position == 0:
            return 0
        return self.net_position / self.max_position

    @property
    def is_at_limit(self) -> bool:
        return abs(self.net_position) >= self.max_position * 0.95

    def update(self, side: str, size: float, price: float):
        if side == "BUY":
            new_pos = self.net_position + size
            if self.net_position >= 0:
                # Adding to long position
                total_cost = self.net_position * self.avg_entry_price + size * price
                self.avg_entry_price = total_cost / new_pos if new_pos > 0 else price
            self.net_position = new_pos
        else:  # SELL
            self.net_position -= size


# ─── Avellaneda-Stoikov Market Maker ─────────────────────────────────────────

class AvellanedaStoikovMM:
    """
    Full Avellaneda-Stoikov market maker adapted for binary prediction markets.

    Key innovations over vanilla A-S:
    1. Logit-space volatility estimation (proper handling of [0,1] bounds)
    2. Time-remaining awareness (T-t affects optimal spread)
    3. VPIN-based quote pulling (from vpin.py integration)
    4. Inventory-aware skewing with hard limits
    5. Multi-level quoting (not just top of book)
    """

    def __init__(
        self,
        token_id: str,
        gamma: float = 0.1,         # Risk aversion
        max_inventory: float = 500,  # Max token inventory
        min_spread: float = 0.02,    # Min quote spread
        max_spread: float = 0.15,    # Max quote spread
        order_size: float = 100,     # Base order size (USDC)
        quote_levels: int = 3,       # Number of price levels to quote
    ):
        self.token_id = token_id
        self.gamma = gamma
        self.max_inventory = max_inventory
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.order_size = order_size
        self.quote_levels = quote_levels

        self.vol_estimator   = LogitVolatilityEstimator()
        self.arrival_estimator = OrderArrivalEstimator()
        self.inventory = InventoryState(
            token_id=token_id,
            net_position=0.0,
            avg_entry_price=0.5,
            max_position=max_inventory,
        )

        self._start_time = time.time()
        self._last_mid: Optional[float] = None

    def _time_remaining_normalized(self, market_end_timestamp: Optional[float]) -> float:
        """
        Returns T-t normalized to [0, 1].
        If no end date, assume 30-day horizon (standard).
        """
        if market_end_timestamp is None:
            total_horizon = 30 * 86400   # 30 days in seconds
            elapsed = time.time() - self._start_time
            return max(0.001, 1 - elapsed / total_horizon)
        remaining = market_end_timestamp - time.time()
        total = market_end_timestamp - self._start_time
        return max(0.001, remaining / max(total, 1))

    def compute_quotes(
        self,
        mid: float,
        market_end_timestamp: Optional[float] = None,
        vpin: float = 0.0,
    ) -> Optional[MarketMakerQuotes]:
        """
        Core A-S calculation. Returns optimal bid/ask quotes.

        Args:
            mid: Current mid price [0, 1]
            market_end_timestamp: Unix timestamp of market resolution
            vpin: Current VPIN value (0-1). High VPIN = pull/widen quotes.
        """
        if not (0.001 < mid < 0.999):
            log.debug(f"Mid {mid} out of bounds, skipping quote")
            return None

        # Update volatility estimator
        self.vol_estimator.update(mid, time.time())
        self._last_mid = mid

        # Get variance and time remaining
        sigma2 = self.vol_estimator.sigma_squared_price_space(mid)
        T_minus_t = self._time_remaining_normalized(market_end_timestamp)

        # Estimate order arrival intensity k at current spread
        current_spread = self.min_spread
        k = self.arrival_estimator.k(current_spread)
        k = max(k, 1e-6)

        # ── Reservation price (inventory-adjusted mid) ──────────────────────
        # r = s - q·γ·σ²·(T-t)
        q = self.inventory.inventory_fraction  # -1 to +1
        reservation_price = mid - q * self.gamma * sigma2 * T_minus_t
        reservation_price = np.clip(reservation_price, 0.001, 0.999)

        # ── Optimal half-spread ─────────────────────────────────────────────
        # δ* = γ·σ²·(T-t)/2 + (1/γ)·ln(1 + γ/k)
        delta_star = (
            0.5 * self.gamma * sigma2 * T_minus_t
            + (1 / self.gamma) * math.log(1 + self.gamma / k)
        )

        # ── VPIN adjustment: widen spread when informed trading detected ────
        # High VPIN means we're likely on the wrong side of informed flow.
        # Respond by widening spread to compensate for adverse selection.
        vpin_multiplier = 1.0 + 2.0 * max(0, vpin - 0.3)   # Starts widening at VPIN=0.3
        delta_adjusted = delta_star * vpin_multiplier

        # Enforce spread bounds
        delta_final = np.clip(delta_adjusted, self.min_spread / 2, self.max_spread / 2)

        # ── Compute final quotes ────────────────────────────────────────────
        bid = reservation_price - delta_final
        ask = reservation_price + delta_final

        # Ensure we don't quote outside [0, 1] and maintain minimum tick
        bid = max(0.01, round(bid, 3))
        ask = min(0.99, round(ask, 3))

        if ask - bid < self.min_spread:
            # Force minimum spread
            spread_adj = (self.min_spread - (ask - bid)) / 2
            bid -= spread_adj
            ask += spread_adj
            bid = max(0.01, round(bid, 3))
            ask = min(0.99, round(ask, 3))

        # ── Inventory-based size skewing ────────────────────────────────────
        # When long, reduce bid size and increase ask size to reduce inventory.
        # When short, increase bid size and reduce ask size.
        base_size = self.order_size
        inv_frac = self.inventory.inventory_fraction
        bid_size = base_size * (1 - 0.7 * inv_frac)   # Long → smaller bids
        ask_size = base_size * (1 + 0.7 * inv_frac)   # Long → larger asks

        bid_size = max(10.0, bid_size)
        ask_size = max(10.0, ask_size)

        # ── Confidence score ────────────────────────────────────────────────
        # Lower confidence when: near price extremes, high inventory, high VPIN
        confidence = 1.0
        confidence *= (1 - abs(mid - 0.5) * 0.8)   # Penalize extremes
        confidence *= (1 - abs(inv_frac) * 0.5)     # Penalize high inventory
        confidence *= (1 - vpin * 0.6)              # Penalize high VPIN
        confidence = max(0.0, min(1.0, confidence))

        return MarketMakerQuotes(
            token_id=self.token_id,
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            mid=mid,
            reservation_price=float(reservation_price),
            spread=ask - bid,
            inventory_skew=float(-q * delta_final),
            confidence=confidence,
        )

    def compute_multi_level_quotes(
        self,
        mid: float,
        market_end_timestamp: Optional[float] = None,
        vpin: float = 0.0,
    ) -> list[MarketMakerQuotes]:
        """
        Generate multiple quote levels for deeper order book presence.
        Each level is wider and larger than the last.
        """
        base = self.compute_quotes(mid, market_end_timestamp, vpin)
        if base is None:
            return []

        levels = [base]
        for i in range(1, self.quote_levels):
            level_multiplier = 1 + i * 0.5
            adj_bid = round(base.bid_price - i * 0.01, 3)
            adj_ask = round(base.ask_price + i * 0.01, 3)
            adj_bid = max(0.01, adj_bid)
            adj_ask = min(0.99, adj_ask)
            levels.append(MarketMakerQuotes(
                token_id=self.token_id,
                bid_price=adj_bid,
                ask_price=adj_ask,
                bid_size=base.bid_size * level_multiplier,
                ask_size=base.ask_size * level_multiplier,
                mid=mid,
                reservation_price=base.reservation_price,
                spread=adj_ask - adj_bid,
                inventory_skew=base.inventory_skew,
                confidence=base.confidence * (0.8 ** i),
            ))
        return levels

    def on_fill(self, side: str, size: float, price: float):
        """Update inventory when an order gets filled."""
        self.inventory.update(side, size, price)
        log.info(
            f"Fill: {side} {size:.1f} @ {price:.3f} | "
            f"Inventory: {self.inventory.net_position:.1f}"
        )

    def on_trade(self, price: float, timestamp: float):
        """Record a market trade for volatility and arrival estimation."""
        self.vol_estimator.update(price, timestamp)
        self.arrival_estimator.record_trade(timestamp)

    def pnl(self, current_mid: float) -> float:
        """Unrealized + realized PnL estimate."""
        inv = self.inventory
        if inv.net_position == 0:
            return 0.0
        return inv.net_position * (current_mid - inv.avg_entry_price)
