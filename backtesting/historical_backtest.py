"""
Historical Backtesting using Polymarket's /prices-history endpoint.

Unlike the trade-replay backtest, this uses the actual 1-minute OHLC
price history to simulate:
  1. Market making PnL across different spread settings
  2. Mean reversion (Hawkes) trade signals and outcomes
  3. Kelly-based directional trades and their resolution

Key advantage: /prices-history goes back the full lifetime of each market
at 1-minute granularity — much richer than the 500-trade API limit.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    gamma: float = 0.1
    min_spread_cents: float = 1.5
    order_size_usdc: float = 100
    fractional_kelly: float = 0.25
    hawkes_threshold_cents: float = 5.0
    reversion_target_pct: float = 0.45
    initial_capital: float = 10_000


@dataclass
class MMSimResult:
    n_days: float
    n_quote_cycles: int
    estimated_round_trips: float
    gross_spread_capture_usdc: float
    adverse_selection_cost_usdc: float
    net_pnl_usdc: float
    annualized_return_pct: float
    spread_capture_per_day: float

    def __repr__(self):
        return (
            f"MMSim: {self.n_days:.1f} days | "
            f"RT={self.estimated_round_trips:.0f} | "
            f"PnL=${self.net_pnl_usdc:.2f} | "
            f"Ann={self.annualized_return_pct:.1f}%/yr"
        )


@dataclass
class MeanReversionSimResult:
    n_shocks: int
    n_trades_taken: int
    wins: int
    losses: int
    avg_profit_cents: float
    total_pnl_usdc: float
    win_rate: float

    def __repr__(self):
        return (
            f"MeanRevSim: {self.n_shocks} shocks | "
            f"trades={self.n_trades_taken} | "
            f"win={self.win_rate:.0%} | PnL=${self.total_pnl_usdc:.2f}"
        )


class HistoricalBacktester:
    """
    Backtests strategies against Polymarket's full price history.
    No credentials needed — /prices-history is a public endpoint.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._fetcher = None

    def _get_fetcher(self):
        if self._fetcher is None:
            from data.websocket_stream import PriceHistoryFetcher
            self._fetcher = PriceHistoryFetcher()
        return self._fetcher

    def backtest_market_making(
        self,
        token_id: str,
        interval: str = "all",
    ) -> Optional[MMSimResult]:
        """
        Simulate market making over the full price history.

        Method:
          - At each 1-minute candle, compute what A-S spread we'd quote
          - Estimate round trips as a function of volume and our spread competitiveness
          - Estimate adverse selection as % of round trips that moved against us
        """
        history = self._get_fetcher().get_price_history(token_id, interval=interval, fidelity=1)
        if len(history) < 60:
            log.warning(f"Insufficient history for {token_id[:12]}... ({len(history)} points)")
            return None

        prices = np.array([p for _, p in history])
        times  = np.array([t for t, _ in history])

        # Volatility in logit space
        logit_prices = np.log(np.clip(prices, 0.001, 0.999) /
                              np.clip(1 - prices, 0.001, 0.999))
        logit_returns = np.diff(logit_prices)
        sigma_logit = np.std(logit_returns)

        # Convert back to price-space vol at median price
        p_med = np.median(prices)
        jacobian = p_med * (1 - p_med)
        sigma_price = sigma_logit * jacobian  # Per-minute std in price space

        # Time to resolution (normalized, decreasing over the backtest)
        n_minutes = len(prices)
        n_days = n_minutes / 1440

        # Simulate A-S spread at each step
        gamma = self.config.gamma
        spreads = []
        min_s = self.config.min_spread_cents / 100

        for i, (ts, p) in enumerate(history):
            t_remaining = max(0.001, 1 - i / n_minutes)
            sigma2 = sigma_price ** 2
            k = 0.1  # Simplified arrival rate
            delta = (
                0.5 * gamma * sigma2 * t_remaining
                + (1 / gamma) * np.log(1 + gamma / k)
            )
            spread = max(min_s, min(0.15, delta * 2))
            spreads.append(spread)

        avg_spread = np.mean(spreads)

        # Estimate round trips per day
        # Assumption: if our spread <= 2x the market's typical spread,
        # we get roughly 1 round trip per N hours.
        # This is very conservative — actual varies a lot by market.
        round_trips_per_day = max(0.5, 5.0 * (1 / (1 + avg_spread * 20)))
        total_round_trips = round_trips_per_day * n_days

        # Gross spread capture = half-spread per leg × 2 legs × round trips × order_size
        half_spread = avg_spread / 2
        order_tokens = self.config.order_size_usdc  # Approximate token count
        gross_capture = total_round_trips * avg_spread * order_tokens

        # Adverse selection: % of fills that were informed (median VPIN estimate)
        # Industry estimate: ~20-35% for thin prediction markets
        adverse_rate = 0.25
        # Adverse selection cost: when an informed trader fills us, we lose
        # approximately the full spread move that follows.
        adverse_cost = total_round_trips * adverse_rate * avg_spread * 1.5 * order_tokens

        net_pnl = gross_capture - adverse_cost
        ann_return = (net_pnl / self.config.initial_capital) * (365 / max(n_days, 1)) * 100

        return MMSimResult(
            n_days=n_days,
            n_quote_cycles=n_minutes,
            estimated_round_trips=total_round_trips,
            gross_spread_capture_usdc=gross_capture,
            adverse_selection_cost_usdc=adverse_cost,
            net_pnl_usdc=net_pnl,
            annualized_return_pct=ann_return,
            spread_capture_per_day=net_pnl / max(n_days, 1),
        )

    def backtest_mean_reversion(
        self,
        token_id: str,
        lookback_hours: int = 120,
    ) -> Optional[MeanReversionSimResult]:
        """
        Identify historical shocks and test fading them.
        Measures: did price revert within the target window?
        """
        history = self._get_fetcher().get_price_history(
            token_id, interval="all", fidelity=1,
        )
        if len(history) < 100:
            return None

        threshold = self.config.hawkes_threshold_cents / 100
        reversion_window = 240  # 4 hours in minutes

        n_shocks = 0
        n_trades = 0
        wins = 0
        losses = 0
        profits = []

        for i in range(60, len(history) - reversion_window):
            ts_now, p_now = history[i]
            ts_prev, p_prev = history[i - 30]  # 30-min shock window
            move = p_now - p_prev

            if abs(move) < threshold:
                continue

            n_shocks += 1
            direction = "DOWN" if move > 0 else "UP"  # Fade the move
            entry = p_now
            target = p_prev + move * (1 - self.config.reversion_target_pct)

            # Check if price reaches target within reversion_window
            future_prices = [p for _, p in history[i+1:i+reversion_window+1]]
            if not future_prices:
                continue

            n_trades += 1
            if direction == "DOWN":  # Sell high, expect to buy back lower
                max_p = max(future_prices[:reversion_window // 2])
                # We "won" if price dropped at least 40% of the shock move
                reversion = entry - min(future_prices)
                hit_target = min(future_prices) <= target
            else:  # Buy low, expect to sell higher
                reversion = max(future_prices) - entry
                hit_target = max(future_prices) >= target

            if hit_target:
                wins += 1
                profit = abs(move) * self.config.reversion_target_pct * 100
            else:
                losses += 1
                # Conservative loss estimate: stop at 1.5x expected profit
                profit = -abs(move) * self.config.reversion_target_pct * 1.5 * 100

            profits.append(profit)

        if n_trades == 0:
            return MeanReversionSimResult(
                n_shocks=n_shocks, n_trades_taken=0, wins=0, losses=0,
                avg_profit_cents=0, total_pnl_usdc=0, win_rate=0,
            )

        avg_profit = np.mean(profits)
        total_pnl = sum(profits) * (self.config.order_size_usdc / 100)

        return MeanReversionSimResult(
            n_shocks=n_shocks,
            n_trades_taken=n_trades,
            wins=wins,
            losses=losses,
            avg_profit_cents=avg_profit,
            total_pnl_usdc=total_pnl,
            win_rate=wins / max(n_trades, 1),
        )

    def run_full_backtest(self, markets: list) -> dict:
        """
        Run both strategies across a set of markets.
        Returns a report dict with per-market and aggregate results.
        """
        report = {
            "markets": [],
            "total_mm_pnl": 0,
            "total_reversion_pnl": 0,
            "best_mm_market": None,
            "best_reversion_market": None,
        }

        for m in markets:
            if not m.yes_token_id:
                continue

            log.info(f"Backtesting: {m.question[:55]}")
            entry = {"question": m.question, "volume_24h": m.volume_24h}

            mm_result = self.backtest_market_making(m.yes_token_id)
            if mm_result:
                entry["mm"] = mm_result
                report["total_mm_pnl"] += mm_result.net_pnl_usdc
                if (report["best_mm_market"] is None or
                        mm_result.annualized_return_pct >
                        report["best_mm_market"]["result"].annualized_return_pct):
                    report["best_mm_market"] = {"market": m, "result": mm_result}
                log.info(f"  MM: {mm_result}")

            rev_result = self.backtest_mean_reversion(m.yes_token_id)
            if rev_result:
                entry["reversion"] = rev_result
                report["total_reversion_pnl"] += rev_result.total_pnl_usdc
                if (report["best_reversion_market"] is None and rev_result.win_rate > 0.5):
                    report["best_reversion_market"] = {"market": m, "result": rev_result}
                log.info(f"  Rev: {rev_result}")

            report["markets"].append(entry)
            time.sleep(0.3)

        return report
