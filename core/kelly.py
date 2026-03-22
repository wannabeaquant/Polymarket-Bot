"""
Kelly Criterion Position Sizing for Prediction Markets

Implements:
  1. Standard Kelly for binary outcomes
  2. Modified Kelly with execution uncertainty (RohOnChain's version)
  3. Fractional Kelly with variance targeting
  4. Kelly for multiple simultaneous correlated positions
  5. Probability estimation from multiple evidence sources

The key insight: Kelly maximizes long-run growth rate, but ONLY if your
probability estimate p is correct. The #1 killer of Kelly-based strategies
is overconfident probability estimates. We use conservative fractional Kelly
(0.25x) and explicit uncertainty adjustments.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)


# ─── Kelly Result ─────────────────────────────────────────────────────────────

@dataclass
class KellyResult:
    full_kelly_fraction: float      # Theoretical full Kelly
    recommended_fraction: float     # Adjusted Kelly (fractional + uncertainty)
    position_size_usdc: float       # Actual USDC to deploy
    edge: float                     # Expected value per $1 risked
    ev_per_dollar: float            # EV / cost
    confidence_adjusted: bool       # Whether we applied confidence penalty
    notes: str = ""

    @property
    def is_positive_ev(self) -> bool:
        return self.edge > 0

    def __repr__(self):
        return (
            f"Kelly(f={self.recommended_fraction:.3f}, "
            f"size=${self.position_size_usdc:.0f}, "
            f"EV={self.ev_per_dollar:.3f})"
        )


# ─── Core Kelly Functions ─────────────────────────────────────────────────────

def kelly_binary(
    p_true: float,
    market_price: float,
    bankroll: float,
    fractional: float = 0.25,
    min_edge_cents: float = 3.0,
    min_size_usdc: float = 10.0,
    max_fraction: float = 0.15,
) -> Optional[KellyResult]:
    """
    Standard Kelly for a binary YES/NO market.

    Args:
        p_true: Your probability estimate for YES resolving at $1
        market_price: Current YES token price (= market's implied probability)
        bankroll: Total capital in USDC
        fractional: Kelly fraction to use (0.25 = quarter Kelly)
        min_edge_cents: Don't trade if edge < this (cents per $1)
        min_size_usdc: Minimum trade size
        max_fraction: Cap fraction at this regardless of Kelly

    Returns:
        KellyResult if positive edge, None if no edge.
    """
    if not (0 < p_true < 1) or not (0.01 < market_price < 0.99):
        return None

    # b = net odds received on a $1 bet
    # If you buy YES at price p, you receive (1-p)/p net per dollar if it wins
    b = (1.0 - market_price) / market_price
    q = 1.0 - p_true

    # Kelly fraction: f* = (b*p - q) / b
    full_kelly = (b * p_true - q) / b

    # Edge = EV per $1 of contract
    edge = p_true * 1.0 - market_price  # = p_true - market_price

    if edge * 100 < min_edge_cents:
        return None

    if full_kelly <= 0:
        return None

    # Apply fractional Kelly and cap
    recommended = min(full_kelly * fractional, max_fraction)

    position_usdc = max(min_size_usdc, recommended * bankroll)
    position_usdc = min(position_usdc, max_fraction * bankroll)

    return KellyResult(
        full_kelly_fraction=full_kelly,
        recommended_fraction=recommended,
        position_size_usdc=position_usdc,
        edge=edge,
        ev_per_dollar=edge / market_price,
        confidence_adjusted=False,
        notes=f"b={b:.3f}, p={p_true:.3f}, market={market_price:.3f}",
    )


def kelly_with_execution_uncertainty(
    p_true: float,
    market_price: float,
    p_execution: float,    # Probability order actually fills at target price
    bankroll: float,
    fractional: float = 0.25,
    max_fraction: float = 0.15,
) -> Optional[KellyResult]:
    """
    Modified Kelly accounting for execution risk on CLOB markets.
    RohOnChain's formula: f = (b*p - q) / b × √p_execution

    This is conservative: if execution is only 80% certain,
    we size down by factor √0.8 ≈ 0.89.
    """
    base = kelly_binary(p_true, market_price, bankroll, fractional, max_fraction=max_fraction)
    if base is None:
        return None

    execution_adjustment = math.sqrt(max(0.1, p_execution))
    adjusted_fraction = base.recommended_fraction * execution_adjustment
    adjusted_size = min(adjusted_fraction * bankroll, max_fraction * bankroll)

    return KellyResult(
        full_kelly_fraction=base.full_kelly_fraction,
        recommended_fraction=adjusted_fraction,
        position_size_usdc=adjusted_size,
        edge=base.edge,
        ev_per_dollar=base.ev_per_dollar,
        confidence_adjusted=True,
        notes=f"{base.notes} | p_exec={p_execution:.2f} adj={execution_adjustment:.3f}",
    )


def kelly_portfolio(
    positions: list[dict],  # [{"p_true", "market_price", "correlation"}]
    bankroll: float,
    fractional: float = 0.25,
    max_total_fraction: float = 0.80,
) -> list[KellyResult]:
    """
    Multi-position Kelly with correlation adjustments.
    When positions are correlated (e.g., same election), Kelly overallocates.
    We use the Markowitz-Kelly extension to handle correlation.

    positions: list of dicts with keys:
        p_true: Your probability
        market_price: Current market price
        correlation: Correlation with other positions (0 = independent)
    """
    results = []
    # Compute individual Kelly fractions
    individual_kellys = []
    for pos in positions:
        k = kelly_binary(pos["p_true"], pos["market_price"], bankroll, fractional)
        individual_kellys.append(k)

    if not any(k for k in individual_kellys):
        return results

    # Build covariance matrix for Kelly
    n = len(positions)
    edges = np.array([
        k.edge if k else 0 for k in individual_kellys
    ])
    prices = np.array([p["market_price"] for p in positions])

    # Approximate variance: p*(1-p) for binary outcomes
    variances = prices * (1 - prices)

    # Build correlation matrix
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                corr = positions[i].get("correlation", 0) * positions[j].get("correlation", 0)
                corr_matrix[i, j] = corr

    cov_matrix = np.outer(np.sqrt(variances), np.sqrt(variances)) * corr_matrix

    # Portfolio Kelly: f* = Σ⁻¹ × μ / γ (where γ is risk aversion)
    try:
        gamma = 2.0  # Risk aversion for portfolio Kelly
        inv_cov = np.linalg.inv(cov_matrix + np.eye(n) * 1e-6)  # Regularize
        kelly_fracs = inv_cov @ edges / gamma
        kelly_fracs = np.clip(kelly_fracs, 0, 0.25)

        # Normalize if total exceeds max_total_fraction
        total = kelly_fracs.sum()
        if total > max_total_fraction:
            kelly_fracs *= max_total_fraction / total

        for i, (pos, k_orig) in enumerate(zip(positions, individual_kellys)):
            if k_orig is None:
                continue
            frac = float(kelly_fracs[i])
            results.append(KellyResult(
                full_kelly_fraction=k_orig.full_kelly_fraction,
                recommended_fraction=frac * fractional,
                position_size_usdc=frac * fractional * bankroll,
                edge=k_orig.edge,
                ev_per_dollar=k_orig.ev_per_dollar,
                confidence_adjusted=True,
                notes=f"Portfolio-Kelly, corr-adjusted | {k_orig.notes}",
            ))
    except np.linalg.LinAlgError:
        # Fallback to independent Kelly if covariance matrix is singular
        results = [k for k in individual_kellys if k is not None]

    return results


# ─── Probability Estimation ───────────────────────────────────────────────────

class ProbabilityEstimator:
    """
    Aggregates multiple probability signals into a calibrated estimate.

    Sources supported:
      - Base rate (historical resolution rate for similar markets)
      - Polling data (for political markets)
      - Market price (other correlated markets)
      - News sentiment (external signal)
      - Prediction aggregators (Metaculus, Manifold)

    Uses Bayesian updating with Beta distribution as conjugate prior.
    """

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        """
        Beta(2, 2) prior = mild center pull.
        Avoids extreme predictions when evidence is thin.
        """
        self.alpha = prior_alpha
        self.beta  = prior_beta
        self._sources: list[tuple[str, float, float]] = []  # (name, p, weight)

    def add_source(self, name: str, probability: float, weight: float = 1.0):
        """
        Add a probability signal.
        Weight = how many pseudo-observations this source is worth.
        """
        p = np.clip(probability, 0.01, 0.99)
        self._sources.append((name, p, weight))

        # Update Beta distribution: treat source as weight pseudo-observations
        self.alpha += weight * p
        self.beta  += weight * (1 - p)
        log.debug(f"Added source '{name}': p={p:.3f}, w={weight:.1f}")

    def add_base_rate(self, p: float, weight: float = 2.0):
        self.add_source("base_rate", p, weight)

    def add_market_price(self, p: float, weight: float = 3.0):
        """Market price gets higher weight — it aggregates many beliefs."""
        self.add_source("market_price", p, weight)

    def add_poll(self, p: float, weight: float = 5.0, sample_size: int = 1000):
        """Polls get weight proportional to sample size (log scale)."""
        adjusted_weight = weight * math.log10(max(sample_size, 100)) / 3
        self.add_source("poll", p, adjusted_weight)

    def add_news_sentiment(self, sentiment_score: float, weight: float = 1.5):
        """
        sentiment_score: -1 (very negative) to +1 (very positive)
        Convert to probability shift from current estimate.
        """
        current = self.mean
        # Sentiment shifts probability by up to ±15% from current
        p_shift = current + sentiment_score * 0.15
        p_shift = np.clip(p_shift, 0.01, 0.99)
        self.add_source("news_sentiment", p_shift, weight)

    @property
    def mean(self) -> float:
        """Maximum likelihood estimate (posterior mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% credible interval."""
        dist = stats.beta(self.alpha, self.beta)
        return dist.ppf(0.025), dist.ppf(0.975)

    @property
    def effective_observations(self) -> float:
        """Effective sample size of our belief."""
        return self.alpha + self.beta

    def edge_vs_market(self, market_price: float) -> float:
        """
        How many cents of edge do we have?
        Positive = we think YES is underpriced.
        """
        return (self.mean - market_price) * 100

    def uncertainty_discount(self) -> float:
        """
        How much should we discount Kelly bet due to uncertainty?
        High variance → more discount.
        Returns multiplier in [0, 1].
        """
        # Wide CI → low confidence → small discount
        ci_lo, ci_hi = self.confidence_interval
        ci_width = ci_hi - ci_lo
        # If CI width > 0.4, we're very uncertain → use 0.5x discount
        discount = max(0.3, 1.0 - ci_width * 1.5)
        return discount

    def kelly_size(
        self,
        market_price: float,
        bankroll: float,
        fractional: float = 0.25,
    ) -> Optional[KellyResult]:
        """Compute Kelly position size using this probability estimate."""
        k = kelly_binary(self.mean, market_price, bankroll, fractional)
        if k is None:
            return None
        # Apply uncertainty discount
        discount = self.uncertainty_discount()
        return KellyResult(
            full_kelly_fraction=k.full_kelly_fraction,
            recommended_fraction=k.recommended_fraction * discount,
            position_size_usdc=k.position_size_usdc * discount,
            edge=k.edge,
            ev_per_dollar=k.ev_per_dollar,
            confidence_adjusted=True,
            notes=f"{k.notes} | uncertainty_discount={discount:.2f}",
        )
