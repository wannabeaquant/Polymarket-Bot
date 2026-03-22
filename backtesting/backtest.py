"""
Backtesting Framework for Polymarket Strategies

Uses historical trade data from Polymarket's API to simulate strategy
performance. Key metrics: Sharpe, PnL, fill rate, adverse selection cost.

The framework simulates:
  - Market making: track spread captures and adverse selection losses
  - Arb: track how many opportunities existed and how many were executable
  - Kelly/directional: track edge realization vs. predicted
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from core.market_making import AvellanedaStoikovMM
from core.vpin import VPIN, TradeRecord
from core.kelly import kelly_binary

log = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    timestamp: float
    token_id: str
    side: str
    price: float
    size: float
    strategy: str
    pnl: float = 0.0
    is_adverse: bool = False  # True if filled against informed flow


@dataclass
class BacktestResults:
    strategy: str
    total_pnl: float
    n_trades: int
    n_adverse_fills: int
    adverse_selection_cost: float
    spread_capture: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float
    pnl_series: list[float] = field(default_factory=list)

    def __repr__(self):
        return (
            f"BacktestResults({self.strategy})\n"
            f"  PnL: ${self.total_pnl:.2f}\n"
            f"  Trades: {self.n_trades}\n"
            f"  Adverse fills: {self.n_adverse_fills} "
            f"(cost=${self.adverse_selection_cost:.2f})\n"
            f"  Spread capture: ${self.spread_capture:.2f}\n"
            f"  Sharpe: {self.sharpe_ratio:.2f}\n"
            f"  Max drawdown: {self.max_drawdown:.2%}\n"
            f"  Win rate: {self.win_rate:.1%}"
        )


class MarketMakingBacktest:
    """
    Backtests the A-S market making strategy against historical trade data.

    Simulation approach:
      - Replay historical trades chronologically
      - At each step, compute what our A-S quotes would have been
      - If a market trade crosses our quote, assume we get filled
      - Track PnL including adverse selection (VPIN-guided fills)
    """

    def __init__(self, gamma: float = 0.1, min_spread: float = 0.02):
        self.gamma = gamma
        self.min_spread = min_spread

    def run(
        self,
        token_id: str,
        trades: list,         # List of Trade objects from PolymarketClient
        initial_capital: float = 10_000,
    ) -> BacktestResults:
        """
        Replay historical trades and compute MM performance.
        """
        if not trades:
            log.warning("No trades provided for backtest")
            return self._empty_results("market_making")

        # Sort trades chronologically
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        mm = AvellanedaStoikovMM(
            token_id=token_id,
            gamma=self.gamma,
            min_spread=self.min_spread,
            order_size=200,
        )
        vpin = VPIN(bucket_size=50)

        backtest_trades = []
        pnl = 0.0
        pnl_series = []
        spread_capture = 0.0
        adverse_cost = 0.0
        n_adverse = 0

        for i, trade in enumerate(sorted_trades):
            # Update VPIN with this trade
            tr = TradeRecord(price=trade.price, size=trade.size, timestamp=trade.timestamp)
            vpin.process_trade(tr)

            # Get current VPIN
            vpin_val = vpin.current_vpin or 0.0

            # Compute what our quotes would have been
            quotes = mm.compute_quotes(trade.price, vpin=vpin_val)
            if quotes is None or not quotes.should_quote:
                continue

            # Did the market trade cross our bid or ask?
            filled_bid = (trade.side == "SELL" and trade.price <= quotes.ask_price and
                          trade.price >= quotes.bid_price)
            filled_ask = (trade.side == "BUY"  and trade.price >= quotes.bid_price and
                          trade.price <= quotes.ask_price)

            if filled_bid:
                # We bought (passive) — sold to us by the market
                fill_price = quotes.bid_price
                mm.on_fill("BUY", trade.size, fill_price)
                mm.on_trade(trade.price, trade.timestamp)

                # Check if this was an informed trade (adverse selection)
                is_adverse = vpin_val >= 0.5
                if is_adverse:
                    n_adverse += 1
                    adverse_cost += trade.size * abs(trade.price - fill_price)

                backtest_trades.append(BacktestTrade(
                    timestamp=trade.timestamp,
                    token_id=token_id,
                    side="BUY",
                    price=fill_price,
                    size=trade.size,
                    strategy="market_making",
                    is_adverse=is_adverse,
                ))

            elif filled_ask:
                fill_price = quotes.ask_price
                mm.on_fill("SELL", trade.size, fill_price)
                mm.on_trade(trade.price, trade.timestamp)

                is_adverse = vpin_val >= 0.5
                if is_adverse:
                    n_adverse += 1
                    adverse_cost += trade.size * abs(trade.price - fill_price)

                backtest_trades.append(BacktestTrade(
                    timestamp=trade.timestamp,
                    token_id=token_id,
                    side="SELL",
                    price=fill_price,
                    size=trade.size,
                    strategy="market_making",
                    is_adverse=is_adverse,
                ))

            # Compute running PnL (mark to market)
            current_pnl = mm.pnl(trade.price)
            pnl_series.append(current_pnl)

        # Final PnL accounting
        if backtest_trades:
            # Estimate spread capture: half-spread per fill
            n_fills = len(backtest_trades)
            avg_spread = 0.02  # Conservative default: 2 cent average spread
            spread_capture = n_fills * avg_spread * 0.5 * 100  # In USDC

        pnl_array = np.array(pnl_series)
        sharpe = (pnl_array.mean() / max(pnl_array.std(), 1e-6)) * np.sqrt(365) if len(pnl_array) > 1 else 0
        max_dd = self._max_drawdown(pnl_array)
        win_rate = (pnl_array > 0).mean() if len(pnl_array) > 0 else 0

        return BacktestResults(
            strategy="market_making",
            total_pnl=float(pnl_array[-1]) if len(pnl_array) > 0 else 0,
            n_trades=len(backtest_trades),
            n_adverse_fills=n_adverse,
            adverse_selection_cost=adverse_cost,
            spread_capture=spread_capture,
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            avg_trade_pnl=float(pnl_array.mean()) if len(pnl_array) > 0 else 0,
            pnl_series=pnl_series,
        )

    def _max_drawdown(self, pnl_series: np.ndarray) -> float:
        if len(pnl_series) < 2:
            return 0.0
        cumulative = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (np.abs(peak) + 1e-6)
        return float(drawdown.max())

    def _empty_results(self, strategy: str) -> BacktestResults:
        return BacktestResults(
            strategy=strategy, total_pnl=0, n_trades=0,
            n_adverse_fills=0, adverse_selection_cost=0,
            spread_capture=0, sharpe_ratio=0, max_drawdown=0,
            win_rate=0, avg_trade_pnl=0,
        )


class ArbBacktest:
    """
    Backtests the arbitrage scanner against historical price data.
    Measures: how many opportunities existed, how many were executable,
    and total extractable profit.
    """

    def run_single_market_arb(
        self,
        yes_prices: list[tuple[float, float]],  # (timestamp, price)
        no_prices:  list[tuple[float, float]],
        min_profit_cents: float = 2.0,
    ) -> dict:
        """
        Simulate single-market (YES+NO) arb scanning over time.
        yes_prices and no_prices are synchronized price time series.
        """
        opportunities = 0
        total_profit = 0.0
        opportunity_sizes = []

        paired = list(zip(yes_prices, no_prices))
        for (ts_yes, p_yes), (ts_no, p_no) in paired:
            total = p_yes + p_no
            profit_cents = (1.0 - total) * 100
            if profit_cents >= min_profit_cents:
                opportunities += 1
                total_profit += profit_cents
                opportunity_sizes.append(profit_cents)

        return {
            "n_opportunities": opportunities,
            "total_profit_cents": total_profit,
            "avg_opportunity_cents": np.mean(opportunity_sizes) if opportunity_sizes else 0,
            "max_opportunity_cents": max(opportunity_sizes) if opportunity_sizes else 0,
            "opportunity_rate": opportunities / max(len(paired), 1),
        }
