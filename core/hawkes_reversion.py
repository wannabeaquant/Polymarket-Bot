"""
Hawkes Process Mean Reversion for Prediction Markets

Prediction markets over-react to news events and mean-revert.
The Hawkes process models self-exciting event arrivals — large price
shocks cluster in time (news begets news, overreaction begets reversion).

We use this to:
  1. Detect "shock" events (large sudden price moves)
  2. Estimate the expected reversion magnitude and timing
  3. Generate faded trade signals to profit from the mean reversion

Model: λ(t) = μ + Σ α × e^(-β(t - tᵢ))
  μ = baseline event rate
  α = excitement factor (each event "excites" future events)
  β = decay rate (how fast excitement fades)

Key observation for prediction markets:
  - When a shock occurs, price moves X cents
  - Historical data shows N% of shocks revert 40-60% within T hours
  - We fade the shock with position sized by Kelly
"""

import time
import math
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)


# ─── Price Shock ──────────────────────────────────────────────────────────────

@dataclass
class PriceShock:
    token_id: str
    timestamp: float
    price_before: float
    price_after: float
    magnitude: float          # Absolute price move in cents
    direction: str            # "UP" or "DOWN"
    reversion_target: float   # Expected mean-reversion level

    @property
    def is_upshock(self) -> bool:
        return self.direction == "UP"

    def trade_signal(self) -> str:
        """Fade the shock: if up-shock, SELL; if down-shock, BUY."""
        return "SELL" if self.is_upshock else "BUY"


@dataclass
class ReversionSignal:
    shock: PriceShock
    expected_reversion_pct: float    # % of shock expected to revert
    reversion_price: float           # Price target
    confidence: float                # 0-1
    time_horizon_hours: float        # Expected hours until reversion
    kelly_edge: float                # Edge in cents per $1
    timestamp: float = field(default_factory=time.time)

    def is_live(self, max_age_hours: float = 6.0) -> bool:
        age_h = (time.time() - self.timestamp) / 3600
        return age_h < max_age_hours


# ─── Hawkes Process Parameter Estimation ─────────────────────────────────────

class HawkesProcess:
    """
    Univariate Hawkes process: λ(t) = μ + Σᵢ α × exp(-β(t - tᵢ))

    Parameters estimated via maximum likelihood from historical shock data.
    """

    def __init__(self, mu: float = 0.1, alpha: float = 0.5, beta: float = 1.0):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self._event_times: list[float] = []

    def intensity(self, t: float) -> float:
        """Current event arrival intensity at time t."""
        base = self.mu
        excitation = sum(
            self.alpha * math.exp(-self.beta * (t - ti))
            for ti in self._event_times
            if ti <= t
        )
        return base + excitation

    def add_event(self, t: float):
        """Record a new event (shock) at time t."""
        self._event_times.append(t)

    def log_likelihood(self, events: list[float], T: float) -> float:
        """
        Log-likelihood of observing `events` in [0, T].
        Used for parameter estimation.
        """
        if not events:
            return -np.inf

        mu, alpha, beta = self.mu, self.alpha, self.beta

        # Sum of log-intensities at event times
        log_sum = 0.0
        for i, ti in enumerate(events):
            lam = mu + sum(
                alpha * math.exp(-beta * (ti - tj))
                for tj in events[:i]
            )
            log_sum += math.log(max(lam, 1e-10))

        # Integral of intensity over [0, T]
        integral = mu * T
        for ti in events:
            integral += (alpha / beta) * (1 - math.exp(-beta * (T - ti)))

        return log_sum - integral

    def fit(self, shock_timestamps: list[float]) -> bool:
        """
        Fit Hawkes parameters via MLE on historical shock times.
        Returns True if fit converged.
        """
        if len(shock_timestamps) < 5:
            log.warning("Insufficient shocks for Hawkes calibration (need ≥5)")
            return False

        events = sorted(shock_timestamps)
        T = events[-1]

        def neg_ll(params):
            mu_t, alpha_t, beta_t = params
            if mu_t <= 0 or alpha_t <= 0 or beta_t <= 0:
                return 1e10
            if alpha_t >= beta_t:  # Stability condition
                return 1e10
            self.mu, self.alpha, self.beta = mu_t, alpha_t, beta_t
            return -self.log_likelihood(events, T)

        result = minimize(
            neg_ll,
            x0=[0.1, 0.3, 1.0],
            method="L-BFGS-B",
            bounds=[(1e-6, 10), (1e-6, 0.999), (1e-6, 10)],
        )

        if result.success:
            self.mu, self.alpha, self.beta = result.x
            log.info(f"Hawkes fit: μ={self.mu:.4f}, α={self.alpha:.4f}, β={self.beta:.4f}")
            return True
        else:
            log.warning(f"Hawkes MLE failed: {result.message}")
            return False

    def expected_events_in(self, dt_hours: float, t_now: float) -> float:
        """Expected number of events in the next dt_hours."""
        lam = self.intensity(t_now)
        dt = dt_hours * 3600
        # For a Hawkes process, expected count ≈ integral of intensity
        # Approximation using current intensity and decay
        return lam * dt * math.exp(-self.beta * dt / 2)


# ─── Mean Reversion Detector ─────────────────────────────────────────────────

class MeanReversionDetector:
    """
    Detects price shocks and generates fade trade signals.

    Pipeline:
      1. Maintain rolling price history
      2. Detect shocks (price moves > threshold in short time window)
      3. Model expected reversion using historical shock data
      4. Generate Kelly-sized trade signals to fade the shock
    """

    def __init__(
        self,
        shock_threshold_cents: float = 5.0,    # Min shock size to trade
        shock_window_minutes: float = 10.0,    # Time window to detect shock
        reversion_lookback_hours: float = 24,  # Historical data for calibration
        min_reversion_pct: float = 0.35,       # Min expected reversion to trade
        confidence_threshold: float = 0.55,    # Min confidence to generate signal
    ):
        self.shock_threshold = shock_threshold_cents / 100
        self.shock_window = shock_window_minutes * 60
        self.reversion_lookback = reversion_lookback_hours * 3600
        self.min_reversion_pct = min_reversion_pct
        self.confidence_threshold = confidence_threshold

        # Price history: (timestamp, price)
        self._price_history: deque = deque(maxlen=10000)
        # Historical shocks for Hawkes calibration
        self._historical_shocks: deque = deque(maxlen=500)
        # Historical reversion observations: (shock_size, reversion_pct)
        self._reversion_observations: deque = deque(maxlen=200)

        self.hawkes = HawkesProcess()
        self._active_signals: list[ReversionSignal] = []
        self._last_shock_price: Optional[float] = None

    def update_price(self, price: float, timestamp: Optional[float] = None):
        """Record a new price observation."""
        ts = timestamp or time.time()
        self._price_history.append((ts, price))
        self._check_for_shock(price, ts)

    def _get_price_n_minutes_ago(self, ts: float, minutes: float) -> Optional[float]:
        """Get price from approximately N minutes ago."""
        target_ts = ts - minutes * 60
        best = None
        best_diff = float("inf")
        for t, p in self._price_history:
            diff = abs(t - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = p
        return best

    def _check_for_shock(self, current_price: float, ts: float):
        """Detect if a shock has occurred."""
        price_before = self._get_price_n_minutes_ago(ts, self.shock_window / 60)
        if price_before is None:
            return

        move = current_price - price_before
        magnitude = abs(move)

        if magnitude < self.shock_threshold:
            return

        direction = "UP" if move > 0 else "DOWN"

        # Calculate expected reversion based on historical observations
        reversion_pct = self._estimate_reversion(magnitude)
        if reversion_pct < self.min_reversion_pct:
            return

        # Mean reversion target
        if direction == "UP":
            reversion_target = current_price - magnitude * reversion_pct
        else:
            reversion_target = current_price + magnitude * reversion_pct

        shock = PriceShock(
            token_id="",
            timestamp=ts,
            price_before=price_before,
            price_after=current_price,
            magnitude=magnitude * 100,  # Convert to cents
            direction=direction,
            reversion_target=reversion_target,
        )

        self._historical_shocks.append(ts)
        self.hawkes.add_event(ts)

        # Compute confidence based on:
        # 1. Number of historical observations (more = more confident)
        # 2. Hawkes process state (more excited = less confident, shocks cluster)
        # 3. Magnitude of shock (larger = more likely to revert, up to a point)
        n_obs = len(self._reversion_observations)
        obs_confidence = min(1.0, n_obs / 50)   # Saturates at 50+ observations
        size_confidence = min(1.0, magnitude * 200)  # Larger shock = higher conf
        hawkes_excite = min(0.5, self.hawkes.intensity(ts) / 10)
        excite_penalty = 1.0 - hawkes_excite  # Many shocks → less reliable signal

        confidence = obs_confidence * size_confidence * excite_penalty * 0.8

        if confidence < self.confidence_threshold:
            log.debug(f"Shock detected but confidence too low: {confidence:.2f}")
            return

        edge_cents = magnitude * 100 * reversion_pct  # Expected profit in cents

        signal = ReversionSignal(
            shock=shock,
            expected_reversion_pct=reversion_pct,
            reversion_price=reversion_target,
            confidence=confidence,
            time_horizon_hours=self._estimate_time_horizon(),
            kelly_edge=edge_cents,
        )
        self._active_signals.append(signal)
        log.info(f"Mean reversion signal: {direction} shock {magnitude*100:.1f}c, "
                 f"target={reversion_target:.3f}, conf={confidence:.2f}")

    def _estimate_reversion(self, shock_magnitude: float) -> float:
        """
        Estimate expected reversion % from historical shock data.
        Default: 45% reversion (conservative estimate from literature).
        """
        if len(self._reversion_observations) < 5:
            return 0.45  # Conservative default

        observations = np.array(self._reversion_observations)
        # Weight by similarity to current shock size
        magnitudes = observations[:, 0]
        reversions = observations[:, 1]
        weights = np.exp(-abs(magnitudes - shock_magnitude) * 10)
        if weights.sum() == 0:
            return 0.45
        return float(np.average(reversions, weights=weights))

    def _estimate_time_horizon(self) -> float:
        """Estimate hours until expected reversion completes."""
        # Default: 4 hours for prediction markets
        # Could be calibrated from historical reversion times
        return 4.0

    def record_reversion(self, shock_magnitude: float, actual_reversion_pct: float):
        """
        Called when we observe a shock resolving.
        Updates historical distribution for better future estimates.
        """
        self._reversion_observations.append((shock_magnitude, actual_reversion_pct))

    def get_active_signals(self) -> list[ReversionSignal]:
        """Return currently active (non-expired) reversion signals."""
        self._active_signals = [s for s in self._active_signals if s.is_live()]
        return self._active_signals

    def recalibrate_hawkes(self):
        """Re-fit Hawkes parameters from recent shock history."""
        recent = [t for t in self._historical_shocks
                  if t > time.time() - self.reversion_lookback]
        if len(recent) >= 5:
            self.hawkes.fit(recent)
