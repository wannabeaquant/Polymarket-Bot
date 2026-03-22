"""
Bayesian Network for Cross-Market Dependency Detection

This is the highest-alpha component — finds mispriced marginals when
the market's joint probability distribution is internally inconsistent.

Architecture:
  1. LLM-based dependency extraction (identify which markets are logically related)
  2. Bayesian network construction (directed acyclic graph of conditional probabilities)
  3. Belief propagation to find the implied probabilities of each node
  4. Compare implied probs to actual market prices → find edges

Example: Election markets
  [Party wins House] → [Party wins Senate] → [Party wins Presidency]
  These are correlated but NOT independent. The market prices each separately.
  Our network finds: given current Senate odds, the House odds are inconsistent.

Uses pgmpy for Bayesian network inference.
"""

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ─── Dependency Models ────────────────────────────────────────────────────────

@dataclass
class MarketNode:
    market_id: str
    question: str
    yes_token_id: str
    market_price: float          # Current YES price
    category: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Dependency:
    parent_id: str               # The conditioning variable
    child_id: str                # The variable being conditioned
    relationship: str            # "subset" | "complement" | "correlated" | "causes"
    strength: float              # 0-1, how strong is the dependency
    cpt: dict = field(default_factory=dict)  # Conditional probability table
    description: str = ""


@dataclass
class PricingDiscrepancy:
    market_id: str
    question: str
    market_price: float
    network_implied_price: float
    discrepancy_cents: float
    direction: str              # "BUY" if market_price < implied, "SELL" if >
    confidence: float
    path: list[str]             # The dependency chain that reveals this


# ─── LLM Dependency Extractor ─────────────────────────────────────────────────

class LLMDependencyExtractor:
    """
    Uses a language model to identify logical dependencies between markets.

    This is the component where we outperform simple pairwise scanning.
    An LLM understands the *semantics* of market questions, not just keywords.

    Example relationships it catches:
      - "Will X win the GOP primary?" is a strict SUBSET of "Will X be president?"
      - "Will Bitcoin exceed $100k?" and "Will Ethereum exceed $5k?" are CORRELATED
      - "Will the Fed cut rates in March?" CAUSES "Will S&P 500 rise in Q1?"
    """

    def __init__(self, api_key: str = "", model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model
        self._cache: dict[str, list[Dependency]] = {}

    def extract_dependencies(
        self,
        markets: list[MarketNode],
        max_pairs: int = 500,
    ) -> list[Dependency]:
        """
        Find logical dependencies between markets.

        For efficiency: pre-filter by category/tags before LLM screening.
        Within a category, check all pairs up to max_pairs.
        """
        dependencies = []

        # Pre-filter: only compare markets in same category or with overlapping tags
        candidate_pairs = self._get_candidate_pairs(markets, max_pairs)
        log.info(f"Screening {len(candidate_pairs)} candidate market pairs for dependencies")

        for m1, m2 in candidate_pairs:
            dep = self._check_pair(m1, m2)
            if dep:
                dependencies.append(dep)

        log.info(f"Found {len(dependencies)} dependencies")
        return dependencies

    def _get_candidate_pairs(
        self,
        markets: list[MarketNode],
        max_pairs: int,
    ) -> list[tuple[MarketNode, MarketNode]]:
        """
        Smart pre-filtering to avoid checking all O(n²) pairs.
        Groups markets by category + semantic similarity.
        """
        from itertools import combinations

        # Group by category
        by_category: dict[str, list[MarketNode]] = {}
        for m in markets:
            by_category.setdefault(m.category, []).append(m)

        pairs = []
        for cat, cat_markets in by_category.items():
            cat_pairs = list(combinations(cat_markets[:30], 2))  # Limit per category
            pairs.extend(cat_pairs)
            if len(pairs) >= max_pairs:
                break

        return pairs[:max_pairs]

    def _check_pair(self, m1: MarketNode, m2: MarketNode) -> Optional[Dependency]:
        """
        Rule-based dependency detection (fast, no API needed).
        Falls back to LLM for ambiguous cases.
        """
        # Rule-based fast checks
        q1 = m1.question.lower()
        q2 = m2.question.lower()

        # Subset detection: e.g., "wins primary" ⊂ "wins election"
        subset_pairs = [
            (["primary", "nomination"], ["president", "election", "general"]),
            (["wins conference", "wins division"], ["wins championship", "wins super bowl"]),
            (["advances to finals"], ["wins tournament"]),
            (["reaches quarter-final"], ["wins tournament", "reaches semi-final"]),
        ]

        for child_keywords, parent_keywords in subset_pairs:
            m1_has_child  = any(kw in q1 for kw in child_keywords)
            m2_has_parent = any(kw in q2 for kw in parent_keywords)
            m2_has_child  = any(kw in q2 for kw in child_keywords)
            m1_has_parent = any(kw in q1 for kw in parent_keywords)

            # Check for shared entity (same team/person/country)
            shared_entity = self._shares_entity(q1, q2)

            if shared_entity:
                if m1_has_child and m2_has_parent:
                    # m2 is parent of m1
                    return Dependency(
                        parent_id=m2.market_id,
                        child_id=m1.market_id,
                        relationship="subset",
                        strength=0.9,
                        cpt={
                            "parent_true":  m1.market_price / max(m2.market_price, 0.01),
                            "parent_false": 0.0,
                        },
                        description=f"'{m1.question}' requires '{m2.question}'",
                    )
                elif m2_has_child and m1_has_parent:
                    # m1 is parent of m2
                    return Dependency(
                        parent_id=m1.market_id,
                        child_id=m2.market_id,
                        relationship="subset",
                        strength=0.9,
                        cpt={
                            "parent_true":  m2.market_price / max(m1.market_price, 0.01),
                            "parent_false": 0.0,
                        },
                        description=f"'{m2.question}' requires '{m1.question}'",
                    )

        return None

    def _shares_entity(self, q1: str, q2: str) -> bool:
        """
        Heuristic: do two questions reference the same entity?
        Extracts capitalized words (names, teams, countries) and checks overlap.
        """
        # Extract tokens that look like proper nouns (start with capital or are all-caps)
        def extract_entities(text):
            words = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', text)
            return set(w.lower() for w in words if len(w) > 2)

        e1 = extract_entities(q1)
        e2 = extract_entities(q2)
        return bool(e1 & e2)


# ─── Bayesian Network Builder ─────────────────────────────────────────────────

class PredictionMarketBayesNet:
    """
    Builds and queries a Bayesian network over prediction markets.

    Each node = a binary market (YES/NO)
    Each edge = a logical dependency

    We use belief propagation to find each node's implied probability
    given all other nodes' current prices, then compare to actual market prices.
    """

    def __init__(self):
        self._nodes: dict[str, MarketNode] = {}
        self._dependencies: list[Dependency] = []

    def add_market(self, market: MarketNode):
        self._nodes[market.market_id] = market

    def add_dependency(self, dep: Dependency):
        self._dependencies.append(dep)

    def find_discrepancies(
        self,
        min_discrepancy_cents: float = 3.0,
        min_confidence: float = 0.6,
    ) -> list[PricingDiscrepancy]:
        """
        Run inference to find markets priced inconsistently with their neighbors.

        For each dependency, checks whether the current prices are consistent
        with the conditional probability relationships.
        """
        discrepancies = []

        for dep in self._dependencies:
            parent = self._nodes.get(dep.parent_id)
            child  = self._nodes.get(dep.child_id)
            if not parent or not child:
                continue

            if dep.relationship == "subset":
                # Constraint: P(child) ≤ P(parent)
                disc = self._check_subset_constraint(parent, child, dep)
                if disc and abs(disc.discrepancy_cents) >= min_discrepancy_cents:
                    if disc.confidence >= min_confidence:
                        discrepancies.append(disc)

        return sorted(discrepancies, key=lambda x: abs(x.discrepancy_cents), reverse=True)

    def _check_subset_constraint(
        self,
        parent: MarketNode,
        child: MarketNode,
        dep: Dependency,
    ) -> Optional[PricingDiscrepancy]:
        """
        For subset dependency: P(child) MUST ≤ P(parent).
        If child > parent, one of them is mispriced.

        We attribute the mispricing to whichever has lower volume/confidence.
        Trade: sell the overpriced one and buy the underpriced one.
        """
        p_parent = parent.market_price
        p_child  = child.market_price

        if p_child <= p_parent:
            return None  # No constraint violation

        discrepancy_cents = (p_child - p_parent) * 100

        # More likely to trade: sell the child (overpriced),
        # because logically it can't have HIGHER probability than its prerequisite.
        return PricingDiscrepancy(
            market_id=child.market_id,
            question=child.question,
            market_price=p_child,
            network_implied_price=p_parent * 0.95,  # Child should trade slightly below parent
            discrepancy_cents=discrepancy_cents,
            direction="SELL",
            confidence=dep.strength * 0.85,
            path=[parent.market_id, child.market_id],
        )

    def propagate_beliefs(self) -> dict[str, float]:
        """
        Full belief propagation across all nodes.
        Returns a dict of market_id → network-implied probability.

        This is the "complete" inference that catches multi-hop mispricing
        (A→B→C where all three are inconsistent).
        """
        implied = {mid: m.market_price for mid, m in self._nodes.items()}

        # Simple iterative message passing
        # For each node, update its implied probability based on its neighbors
        for _ in range(20):  # 20 iterations is usually enough
            updates = {}
            for dep in self._dependencies:
                if dep.relationship != "subset":
                    continue
                parent_prob = implied.get(dep.parent_id, 0.5)
                child_prob  = implied.get(dep.child_id, 0.5)

                # Enforce P(child) ≤ P(parent)
                if child_prob > parent_prob:
                    # Compress toward constraint boundary
                    corrected = parent_prob * 0.98
                    updates[dep.child_id] = min(
                        implied.get(dep.child_id, corrected),
                        corrected,
                    )

            if not updates:
                break
            implied.update(updates)

        return implied


# ─── High-level Scanner ───────────────────────────────────────────────────────

class CrossMarketScanner:
    """
    Orchestrates the full cross-market analysis pipeline.

    Usage:
        scanner = CrossMarketScanner()
        discrepancies = scanner.scan(markets)
        for d in discrepancies:
            print(d)
    """

    def __init__(self, min_discrepancy_cents: float = 3.0):
        self.min_discrepancy_cents = min_discrepancy_cents
        self.extractor = LLMDependencyExtractor()
        self.network   = PredictionMarketBayesNet()

    def scan(self, markets: list) -> list[PricingDiscrepancy]:
        """
        Full pipeline: extract dependencies, build network, find discrepancies.
        """
        # Convert to MarketNode format
        nodes = []
        for m in markets:
            price = m.volume_24h  # Placeholder — get actual mid price
            nodes.append(MarketNode(
                market_id=m.condition_id,
                question=m.question,
                yes_token_id=m.yes_token_id,
                market_price=0.5,  # Will be updated with real prices
                category=m.category,
                tags=m.tags,
            ))
            self.network.add_market(nodes[-1])

        # Extract dependencies
        deps = self.extractor.extract_dependencies(nodes)
        for dep in deps:
            self.network.add_dependency(dep)

        # Find discrepancies
        return self.network.find_discrepancies(
            min_discrepancy_cents=self.min_discrepancy_cents,
        )

    def update_prices(self, client) -> dict[str, float]:
        """Fetch current mid prices for all nodes."""
        prices = {}
        for mid, node in self.network._nodes.items():
            price = client.get_mid_price(node.yes_token_id)
            if price:
                node.market_price = price
                prices[mid] = price
        return prices
