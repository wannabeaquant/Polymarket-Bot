"""
VPIN — Volume-Synchronized Probability of Informed Trading

Original paper: Easley, Lopez de Prado, O'Hara (2012)

Adapted for prediction markets:
  - Binary contract trades have a natural directional signal (closer to 1 = more bullish)
  - Tick rule for trade direction classification (no quote data needed for historical)
  - When VPIN is high, the market maker faces adverse selection risk → pull/widen quotes
  - When VPIN is low, market is noise-trader dominated → safe to quote aggressively

VPIN(τ) = Σ|V_buy_i - V_sell_i| / (n × V_bucket)

Where each bucket contains V_bucket total volume.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    price: float
    size: float
    timestamp: float
    classified_side: Optional[str] = None   # "BUY" or "SELL" after classification


@dataclass
class VPINBucket:
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0

    @property
    def imbalance(self) -> float:
        return abs(self.buy_volume - self.sell_volume)


class VPIN:
    """
    Compute real-time VPIN for a single Polymarket token.

    High VPIN (> 0.6) → informed traders active, adverse selection risk high.
    Low VPIN (< 0.3)  → noise traders dominant, safe to market-make.
    """

    def __init__(
        self,
        bucket_size: float = 50,      # Total volume per bucket (tokens)
        window_buckets: int = 50,     # Rolling window for VPIN computation
        min_buckets: int = 10,        # Minimum buckets before VPIN is reliable
    ):
        self.bucket_size = bucket_size
        self.window_buckets = window_buckets
        self.min_buckets = min_buckets

        self._current_bucket = VPINBucket()
        self._current_bucket_volume = 0.0
        self._completed_buckets: deque[VPINBucket] = deque(maxlen=window_buckets)
        self._last_price: Optional[float] = None
        self._total_trades_processed = 0

    def _classify_trade(self, trade: TradeRecord) -> str:
        """
        Classify trade as buyer- or seller-initiated using the tick rule.
        If price > last price → BUY (uptick).
        If price < last price → SELL (downtick).
        If no change → carry forward last classification.

        For prediction markets, we also use the bulk volume classification:
        fraction_buy = Φ(z) where z = (price - μ_price) / σ_price
        This is the Lee-Ready bulk classification adapted for binary contracts.
        """
        if self._last_price is None:
            self._last_price = trade.price
            return "BUY"

        if trade.price > self._last_price:
            return "BUY"
        elif trade.price < self._last_price:
            return "SELL"
        else:
            # No tick change — use bulk classification based on price level
            # Trades above 0.5 are more likely buys (bullish); below 0.5 are sells
            return "BUY" if trade.price >= 0.5 else "SELL"

    def process_trade(self, trade: TradeRecord):
        """
        Process a single trade. Updates VPIN buckets.
        Call this for each new trade in chronological order.
        """
        side = self._classify_trade(trade)
        trade.classified_side = side
        self._last_price = trade.price
        self._total_trades_processed += 1

        remaining_volume = trade.size

        while remaining_volume > 0:
            # How much more volume fits in current bucket?
            space_in_bucket = self.bucket_size - self._current_bucket_volume

            if remaining_volume <= space_in_bucket:
                # Trade fits entirely in current bucket
                if side == "BUY":
                    self._current_bucket.buy_volume  += remaining_volume
                else:
                    self._current_bucket.sell_volume += remaining_volume
                self._current_bucket.total_volume += remaining_volume
                self._current_bucket_volume += remaining_volume
                remaining_volume = 0
            else:
                # Fill current bucket and start a new one
                partial = space_in_bucket
                if side == "BUY":
                    self._current_bucket.buy_volume  += partial
                else:
                    self._current_bucket.sell_volume += partial
                self._current_bucket.total_volume += partial
                self._current_bucket_volume += partial
                remaining_volume -= partial

                # Complete this bucket
                self._completed_buckets.append(self._current_bucket)
                self._current_bucket = VPINBucket()
                self._current_bucket_volume = 0.0

    def process_trades(self, trades: list[TradeRecord]):
        """Process a list of trades sorted chronologically."""
        for trade in sorted(trades, key=lambda t: t.timestamp):
            self.process_trade(trade)

    @property
    def current_vpin(self) -> Optional[float]:
        """
        Current VPIN estimate.
        Returns None if insufficient data (< min_buckets completed).
        """
        n = len(self._completed_buckets)
        if n < self.min_buckets:
            return None

        total_imbalance = sum(b.imbalance for b in self._completed_buckets)
        total_volume = n * self.bucket_size  # Each bucket = bucket_size volume
        if total_volume == 0:
            return None

        return total_imbalance / total_volume

    @property
    def vpin_series(self) -> list[float]:
        """
        VPIN value after each completed bucket.
        Useful for calibration and plotting.
        """
        series = []
        buckets = list(self._completed_buckets)
        for i in range(self.min_buckets, len(buckets) + 1):
            window = buckets[max(0, i - self.window_buckets):i]
            n = len(window)
            total_imbalance = sum(b.imbalance for b in window)
            total_vol = n * self.bucket_size
            series.append(total_imbalance / total_vol if total_vol > 0 else 0)
        return series

    def regime(self) -> str:
        """
        Return current market regime based on VPIN.
        Used by market maker to decide whether to quote.
        """
        v = self.current_vpin
        if v is None:
            return "UNKNOWN"
        if v >= 0.7:
            return "HIGHLY_INFORMED"   # Pull quotes entirely
        if v >= 0.5:
            return "MODERATELY_INFORMED"  # Widen spreads significantly
        if v >= 0.3:
            return "MIXED"             # Normal widening
        return "NOISE_DOMINATED"       # Quote aggressively, tighten spreads

    def adverse_selection_multiplier(self) -> float:
        """
        Returns a multiplier (≥1.0) to apply to the spread.
        1.0 = no adjustment (noise market)
        3.0 = triple the spread (highly informed market)
        """
        v = self.current_vpin
        if v is None:
            return 1.5  # Conservative default
        # Linear interpolation: 0.0 → 1.0x, 1.0 → 3.0x
        return 1.0 + 2.0 * v


class MultiMarketVPIN:
    """
    Manages VPIN instances for multiple markets simultaneously.
    The market maker uses this to quickly check any token's regime.
    """

    def __init__(self, default_bucket_size: float = 50):
        self.default_bucket_size = default_bucket_size
        self._vpins: dict[str, VPIN] = {}

    def get_or_create(self, token_id: str) -> VPIN:
        if token_id not in self._vpins:
            self._vpins[token_id] = VPIN(bucket_size=self.default_bucket_size)
        return self._vpins[token_id]

    def process_trade(self, token_id: str, price: float, size: float, timestamp: float):
        vpin = self.get_or_create(token_id)
        vpin.process_trade(TradeRecord(price=price, size=size, timestamp=timestamp))

    def get_vpin(self, token_id: str) -> Optional[float]:
        if token_id not in self._vpins:
            return None
        return self._vpins[token_id].current_vpin

    def get_regime(self, token_id: str) -> str:
        if token_id not in self._vpins:
            return "UNKNOWN"
        return self._vpins[token_id].regime()

    def should_pull_quotes(self, token_id: str, threshold: float = 0.6) -> bool:
        """Returns True if we should pull all quotes for this token."""
        v = self.get_vpin(token_id)
        if v is None:
            return False
        return v >= threshold
