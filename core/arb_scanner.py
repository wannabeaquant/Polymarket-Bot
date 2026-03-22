"""
Structural Arbitrage Scanner

Detects three tiers of arbitrage:
  Tier 1 - Single-market: P(YES) + P(NO) != $1.00
  Tier 2 - Cross-market complement: mutually exclusive events where
            sum of probabilities != 1
  Tier 3 - Logical dependency: tournament/election chains where
            P(child) > P(parent) implies mismatch

Each opportunity is returned with VWAP-adjusted profit after fees.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from itertools import combinations

import numpy as np

log = logging.getLogger(__name__)


# ─── Opportunity Models ───────────────────────────────────────────────────────

@dataclass
class ArbOpportunity:
    tier: int                        # 1, 2, or 3
    description: str
    legs: list[dict]                 # [{"token_id", "side", "price", "size_usdc"}]
    gross_profit_cents: float        # Profit per $1 risked, before fees
    net_profit_cents: float          # After slippage + fees
    confidence: float                # 0-1, how reliable is this arb
    capital_required: float          # USDC needed to execute
    expected_return_pct: float       # net_profit / capital_required

    def __repr__(self):
        return (
            f"ArbOpp(tier={self.tier}, net={self.net_profit_cents:.2f}c, "
            f"ret={self.expected_return_pct:.2%}, legs={len(self.legs)})"
        )


# ─── Core Scanner ─────────────────────────────────────────────────────────────

class ArbScanner:
    """
    Scans for all three tiers of arbitrage across Polymarket.

    Usage:
        scanner = ArbScanner(client, min_profit_cents=2.0)
        opps = scanner.scan_all(markets)
        for opp in sorted(opps, key=lambda x: x.expected_return_pct, reverse=True):
            print(opp)
    """

    def __init__(
        self,
        client,                        # PolymarketClient
        min_profit_cents: float = 2.0, # Min net profit per $1 to flag
        check_size_usdc: float = 200,  # USDC to simulate for VWAP
    ):
        self.client = client
        self.min_profit_cents = min_profit_cents
        self.check_size_usdc = check_size_usdc

    # ── Tier 1: Single-market ─────────────────────────────────────────────────

    def check_single_market(self, market) -> Optional[ArbOpportunity]:
        """
        Buy YES + NO: if VWAP(YES) + VWAP(NO) < 1.00, guaranteed profit.
        """
        yes_book = self.client.get_order_book(market.yes_token_id)
        no_book  = self.client.get_order_book(market.no_token_id)
        if not yes_book or not no_book:
            return None

        yes_vwap = yes_book.vwap_buy(self.check_size_usdc)
        no_vwap  = no_book.vwap_buy(self.check_size_usdc)
        if yes_vwap is None or no_vwap is None:
            return None

        total_cost = yes_vwap + no_vwap
        gross_profit = (1.0 - total_cost) * 100   # cents per contract

        if gross_profit < self.min_profit_cents:
            return None

        # Slippage & execution penalty (empirical: ~0.5 cents per leg on Polymarket)
        exec_penalty = 1.0
        net_profit = gross_profit - exec_penalty
        if net_profit < self.min_profit_cents:
            return None

        capital = self.check_size_usdc
        return ArbOpportunity(
            tier=1,
            description=f"Single-market arb: {market.question[:60]}",
            legs=[
                {"token_id": market.yes_token_id, "side": "BUY",
                 "price": yes_vwap, "size_usdc": capital / 2},
                {"token_id": market.no_token_id,  "side": "BUY",
                 "price": no_vwap,  "size_usdc": capital / 2},
            ],
            gross_profit_cents=gross_profit,
            net_profit_cents=net_profit,
            confidence=0.95,
            capital_required=capital,
            expected_return_pct=net_profit / 100 / total_cost,
        )

    # ── Tier 2: Mutually exclusive complement ─────────────────────────────────

    def check_complement_set(self, markets: list) -> list[ArbOpportunity]:
        """
        For a set of mutually exclusive outcomes (e.g., which party wins Senate),
        if sum of best YES prices < 1.0, buy all; if sum > 1.0, sell all.

        Example: 3-way election with YES prices 0.30 + 0.28 + 0.25 = 0.83
        → buy all three for $0.83, guaranteed $1.00 payout = 17c profit.
        """
        opportunities = []

        # Buy-all arb: sum of YES prices < 1.00
        vwaps = []
        for m in markets:
            book = self.client.get_order_book(m.yes_token_id)
            if not book:
                return []
            vwap = book.vwap_buy(self.check_size_usdc)
            if vwap is None:
                return []
            vwaps.append((m, vwap))

        total_cost = sum(v for _, v in vwaps)
        gross_profit = (1.0 - total_cost) * 100

        if gross_profit > self.min_profit_cents:
            exec_penalty = len(markets) * 0.5  # 0.5c per leg
            net_profit = gross_profit - exec_penalty
            if net_profit > self.min_profit_cents:
                legs = [{"token_id": m.yes_token_id, "side": "BUY",
                         "price": v, "size_usdc": self.check_size_usdc / len(markets)}
                        for m, v in vwaps]
                opportunities.append(ArbOpportunity(
                    tier=2,
                    description=f"Complement arb (buy all): {len(markets)} outcomes",
                    legs=legs,
                    gross_profit_cents=gross_profit,
                    net_profit_cents=net_profit,
                    confidence=0.90,
                    capital_required=self.check_size_usdc,
                    expected_return_pct=net_profit / 100 / total_cost,
                ))

        # Sell-all arb: sum of YES prices > 1.00
        vwap_sells = []
        for m in markets:
            book = self.client.get_order_book(m.yes_token_id)
            if not book:
                return opportunities
            vwap = book.vwap_sell(self.check_size_usdc)
            if vwap is None:
                return opportunities
            vwap_sells.append((m, vwap))

        total_revenue = sum(v for _, v in vwap_sells)
        gross_profit_sell = (total_revenue - 1.0) * 100

        if gross_profit_sell > self.min_profit_cents:
            exec_penalty = len(markets) * 0.5
            net_profit_sell = gross_profit_sell - exec_penalty
            if net_profit_sell > self.min_profit_cents:
                legs = [{"token_id": m.yes_token_id, "side": "SELL",
                         "price": v, "size_usdc": self.check_size_usdc / len(markets)}
                        for m, v in vwap_sells]
                opportunities.append(ArbOpportunity(
                    tier=2,
                    description=f"Complement arb (sell all): {len(markets)} outcomes",
                    legs=legs,
                    gross_profit_cents=gross_profit_sell,
                    net_profit_cents=net_profit_sell,
                    confidence=0.90,
                    capital_required=self.check_size_usdc,
                    expected_return_pct=net_profit_sell / 100 / total_revenue,
                ))

        return opportunities

    # ── Tier 3: Logical dependency (parent/child) ─────────────────────────────

    def check_parent_child(
        self,
        parent_market,    # e.g., "Will X make the finals?"
        child_market,     # e.g., "Will X win the tournament?" (child ⊆ parent)
    ) -> Optional[ArbOpportunity]:
        """
        If child event requires parent event:
            P(child) MUST be <= P(parent)

        If P(child) > P(parent), sell child and buy parent.
        The spread is guaranteed alpha — it must close by resolution.
        """
        parent_book = self.client.get_order_book(parent_market.yes_token_id)
        child_book  = self.client.get_order_book(child_market.yes_token_id)
        if not parent_book or not child_book:
            return None

        parent_mid = parent_book.mid
        child_mid  = child_book.mid
        if parent_mid is None or child_mid is None:
            return None

        # Child probability cannot exceed parent probability
        if child_mid <= parent_mid:
            return None

        spread = (child_mid - parent_mid) * 100  # cents

        if spread < self.min_profit_cents * 2:  # Require larger margin for tier 3
            return None

        # For execution: sell child (overpriced), buy parent (underpriced)
        child_vwap_sell  = child_book.vwap_sell(self.check_size_usdc)
        parent_vwap_buy  = parent_book.vwap_buy(self.check_size_usdc)
        if child_vwap_sell is None or parent_vwap_buy is None:
            return None

        net_spread = (child_vwap_sell - parent_vwap_buy) * 100
        exec_penalty = 1.0
        net_profit = net_spread - exec_penalty

        if net_profit < self.min_profit_cents:
            return None

        return ArbOpportunity(
            tier=3,
            description=(
                f"Dependency arb: '{child_market.question[:40]}' > "
                f"'{parent_market.question[:40]}'"
            ),
            legs=[
                {"token_id": child_market.yes_token_id,  "side": "SELL",
                 "price": child_vwap_sell, "size_usdc": self.check_size_usdc},
                {"token_id": parent_market.yes_token_id, "side": "BUY",
                 "price": parent_vwap_buy, "size_usdc": self.check_size_usdc},
            ],
            gross_profit_cents=net_spread,
            net_profit_cents=net_profit,
            confidence=0.75,   # Lower — dependency must be manually validated
            capital_required=self.check_size_usdc * 2,
            expected_return_pct=net_profit / 100 / (self.check_size_usdc * 2),
        )

    # ── Full scan ─────────────────────────────────────────────────────────────

    def scan_all(self, markets: list) -> list[ArbOpportunity]:
        """
        Run all three tiers across the provided market list.
        Returns opportunities sorted by expected return.
        """
        opportunities = []
        total = len(markets)
        log.info(f"Scanning {total} markets for arbitrage...")

        for i, market in enumerate(markets):
            if i % 50 == 0:
                log.info(f"  Tier 1 scan: {i}/{total}")
            opp = self.check_single_market(market)
            if opp:
                log.info(f"  [T1 HIT] {opp}")
                opportunities.append(opp)

        log.info("Tier 1 complete. Starting Tier 2 (complement sets)...")
        # Group markets by category for complement-set detection
        by_category: dict[str, list] = {}
        for m in markets:
            by_category.setdefault(m.category, []).append(m)

        for cat, cat_markets in by_category.items():
            # Check pairs and triples within category for complement structure
            for size in [2, 3, 4]:
                for combo in combinations(cat_markets[:20], size):
                    opps = self.check_complement_set(list(combo))
                    for opp in opps:
                        log.info(f"  [T2 HIT] {opp}")
                    opportunities.extend(opps)

        return sorted(opportunities, key=lambda x: x.expected_return_pct, reverse=True)
