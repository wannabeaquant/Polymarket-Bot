"""
Dispute Trading Strategy — Exploiting Ambiguous Resolution Rules

Based on 0xRicker's framework: when Polymarket markets have fuzzy resolution
language, the market prices the "headline" interpretation while actual resolution
follows the literal rules. The gap between these is our edge.

How it works:
  1. Scan active markets for ambiguous resolution criteria (fuzzy language)
  2. For each ambiguous market, score the headline-implied probability vs.
     the rule-implied probability (stricter/looser reading of the resolution)
  3. Monitor UMA oracle: if a dispute is active, track uncommitted votes
  4. Enter positions when gap > min_edge and direction is clear

Key examples of mispricing patterns:
  - "Will X invade Y?" — market prices 20%, but resolution requires "large-scale
    conventional military incursion of significant territory" → lower probability
  - "Will X wear a suit?" — market prices 80%, but "business formal" definition
    is stricter than expected → lower probability
  - "Will X make official announcement?" — "official" is undefined → ambiguous

UMA Oracle mechanics:
  - Anyone can assert a resolution with $750 USDC bond
  - 24-hour dispute window (commit/reveal)
  - betmoardotfun tracks whale voters (>25K UMA tokens unrevealed)
  - If assertion fails: disputer wins 50% of asserter's bond
  - Vote outcome: YES resolves 1.0, NO resolves 0.0

Signal types:
  A. PRE-RESOLUTION: Market misprices before any dispute is filed
     - Buy what rules actually support, sell the headline narrative
  B. DISPUTE-ACTIVE: UMA oracle vote is in progress
     - Track large unrevealed voters as leading indicator
     - Enter in direction of likely vote outcome
  C. POST-DISPUTE: After a resolution is challenged but before final settlement
     - Typically 2-7 day window for appeal / committee override

Edge estimate: 70% accuracy claimed, average 15-30c mispricing when found.
"""

import time
import logging
import re
import json
import requests
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)

GAMMA_HOST  = "https://gamma-api.polymarket.com"
CLOB_HOST   = "https://clob.polymarket.com"
UMA_HOST    = "https://oracle.uma.xyz/api"         # UMA optimistic oracle API
BETTORSFUN  = "https://api.betmoardotfun.com"       # betmoardotfun whale tracker


# ─────────────────────────────────────────────────────────────────────────────
# Resolution ambiguity scoring
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_PATTERNS = {
    # Fuzzy threshold words — resolution criteria are inherently subjective
    "vague_threshold": [
        r"\bsignificant(ly)?\b", r"\bsubstantial(ly)?\b", r"\blarge.?scale\b",
        r"\bmajor\b", r"\bserious\b", r"\bnotable\b", r"\bconsiderable\b",
        r"\bwidespread\b", r"\bextensive\b", r"\bmeaningful\b",
    ],
    # Ambiguous event definitions
    "ambiguous_event": [
        r"\binvade?\b", r"\bannounce?\b", r"\bofficial(ly)?\b",
        r"\bformal(ly)?\b", r"\bconfirm\b", r"\brecognize?\b",
        r"\bbegin\b", r"\bstart\b", r"\blaunch\b", r"\bintroduce\b",
    ],
    # Dress / appearance criteria
    "appearance": [
        r"\bsuit\b", r"\btie\b", r"\bformal attire\b", r"\bbusiness\b",
        r"\bwear(ing)?\b", r"\bdress(ed)?\b",
    ],
    # Unclear timeframes
    "timeframe": [
        r"\bbefore\b.*?\bby\b", r"\bwithin\b", r"\bby end of\b",
        r"\bprior to\b", r"\bahead of\b",
    ],
    # Legal / political edge cases
    "legal_political": [
        r"\bconvict\b", r"\bindicted?\b", r"\bcharged?\b",
        r"\bimpeach\b", r"\bresign\b", r"\bremove\b",
        r"\bsanction\b", r"\bban\b",
    ],
}

# Words that REDUCE ambiguity (more concrete criteria)
CLARITY_PATTERNS = [
    r"\bprice above \$[\d,.]+\b",
    r"\bmore than \d+%\b",
    r"\bnew (high|low|record)\b",
    r"\bcloses? (above|below)\b",
    r"\bwins? (the )?(election|race|championship)\b",
    r"\bgets? \d+ (votes|seats)\b",
    r"\bapproved by .+congress\b",
    r"\bsigned into law\b",
]


@dataclass
class AmbiguityScore:
    market_id: str
    question: str
    description_snippet: str
    total_score: float           # 0.0 = crystal clear, 1.0 = maximally ambiguous
    matched_patterns: list[str]
    clarity_matches: list[str]
    category: str               # dominant ambiguity category

    def __repr__(self):
        return (
            f"Ambiguity({self.total_score:.2f}) | {self.question[:60]}\n"
            f"  Fuzzy: {self.matched_patterns[:3]}\n"
            f"  Clear: {self.clarity_matches[:3]}"
        )


def score_resolution_ambiguity(question: str, description: str) -> AmbiguityScore:
    """
    Score how ambiguous a market's resolution criteria are.
    Returns AmbiguityScore with 0.0 (clear) to 1.0 (maximally ambiguous).
    """
    text = (question + " " + description).lower()

    matched = []
    category_counts = {}

    for category, patterns in AMBIGUITY_PATTERNS.items():
        hits = []
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                hits.append(pat.strip(r"\b"))
        if hits:
            matched.extend(hits)
            category_counts[category] = len(hits)

    clarity = []
    for pat in CLARITY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            clarity.append(pat.strip(r"\b"))

    # Raw score: ambiguity hits - clarity hits, normalized
    raw = len(matched) - len(clarity) * 2
    score = max(0.0, min(1.0, raw / 6.0))

    dominant_cat = max(category_counts, key=category_counts.get) if category_counts else "none"

    return AmbiguityScore(
        market_id="",
        question=question,
        description_snippet=description[:200],
        total_score=score,
        matched_patterns=matched,
        clarity_matches=clarity,
        category=dominant_cat,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resolution direction estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DisputeSignal:
    market_id: str
    question: str
    current_price: float          # YES token price (market consensus)
    rule_implied_price: float     # Our estimate based on literal rule reading
    edge_cents: float             # (rule_implied - current) × 100, signed
    direction: str                # "BUY_YES", "SELL_YES" (buy NO)
    confidence: float             # 0.0-1.0
    ambiguity_score: float
    resolution_ts: int
    dispute_active: bool          # Is a UMA dispute already in progress?
    uma_vote_direction: str       # "YES", "NO", "UNKNOWN"
    rationale: str

    @property
    def minutes_to_resolution(self) -> float:
        return (self.resolution_ts - time.time()) / 60

    @property
    def is_tradeable(self) -> bool:
        return (
            abs(self.edge_cents) >= 8.0        # At least 8c edge
            and self.confidence >= 0.55
            and self.ambiguity_score >= 0.3    # Actually ambiguous
            and self.minutes_to_resolution > 30
        )


class ResolutionDirectionEstimator:
    """
    Given a market's question and description, estimates whether strict
    rule reading points YES or NO vs. the current market price.

    This is inherently heuristic — full NLP/LLM analysis would be ideal.
    We use pattern matching as a fast first pass.
    """

    # Patterns suggesting strict reading will resolve NO (market overprices YES)
    STRICT_NO_SIGNALS = [
        # Invasion requires more than skirmishes
        (r"\binvad(e|ing|ion)\b.*\b(territory|sovereign|border)\b", 0.3),
        # "Official" announcements — raw leaks/rumors don't count
        (r"\bofficial(ly)?\b.*\b(announc|statement|confirm)\b", 0.25),
        # Suits — must be full formal business attire
        (r"\bwear(ing)?\b.*\b(suit|formal)\b", 0.2),
        # "Significant" = usually a higher bar than people expect
        (r"\bsignificant\b.*\b(progress|advance|gain)\b", 0.25),
        # Formal declarations (war, emergency, etc.)
        (r"\bdeclar(e|ing|ation)\b.*\b(war|emergency|state)\b", 0.3),
    ]

    # Patterns suggesting strict reading will resolve YES (market underprices YES)
    STRICT_YES_SIGNALS = [
        # Technical defaults — can happen without dramatic news
        (r"\bdefault\b.*\b(debt|bond|payment)\b", 0.2),
        # Price thresholds that are near current levels
        (r"\b(close|end|settle)\b.*\b(above|below|over|under)\b.*\$[\d,]+", 0.15),
        # Election wins (binary, clear outcome)
        (r"\bwin\b.*\b(election|primary|race|seat)\b", 0.1),
    ]

    def estimate(
        self,
        question: str,
        description: str,
        current_yes_price: float,
    ) -> tuple[float, float, str]:
        """
        Returns (rule_implied_price, confidence, rationale).
        """
        text = (question + " " + description).lower()

        bias = 0.0  # Negative = toward NO, positive = toward YES
        max_conf = 0.0
        rationale_parts = []

        for pattern, weight in self.STRICT_NO_SIGNALS:
            if re.search(pattern, text, re.IGNORECASE):
                bias -= weight
                max_conf = max(max_conf, weight)
                rationale_parts.append(f"Strict-NO: {pattern[:40]}")

        for pattern, weight in self.STRICT_YES_SIGNALS:
            if re.search(pattern, text, re.IGNORECASE):
                bias += weight
                max_conf = max(max_conf, weight)
                rationale_parts.append(f"Strict-YES: {pattern[:40]}")

        # Rule-implied price = current price shifted by bias
        # Clamp to [0.05, 0.95] — extreme edges not tradeable
        rule_price = max(0.05, min(0.95, current_yes_price + bias))
        confidence = min(0.85, max_conf * 2.5)  # Scale 0.3 weight → 0.75 conf
        rationale = "; ".join(rationale_parts) if rationale_parts else "No strong signal"

        return rule_price, confidence, rationale


# ─────────────────────────────────────────────────────────────────────────────
# UMA Oracle tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UMADisputeStatus:
    market_id: str
    assertion_id: str
    asserter_direction: str    # What the asserter claims: "YES" or "NO"
    assertion_ts: int
    expiry_ts: int             # When voting closes
    total_yes_votes: float     # UMA tokens committed YES
    total_no_votes: float
    large_unrevealed: int      # Whale voters (>25K UMA) who haven't revealed
    estimated_outcome: str     # "YES", "NO", "UNKNOWN"

    @property
    def hours_to_expiry(self) -> float:
        return (self.expiry_ts - time.time()) / 3600

    @property
    def vote_ratio(self) -> float:
        total = self.total_yes_votes + self.total_no_votes
        return self.total_yes_votes / total if total > 0 else 0.5


class UMAOracleTracker:
    """
    Monitors UMA Optimistic Oracle for active disputes on Polymarket markets.
    """

    def __init__(self):
        self._session = requests.Session()
        self._cache: dict[str, UMADisputeStatus] = {}
        self._last_fetch = 0

    def get_active_disputes(self) -> list[UMADisputeStatus]:
        """
        Fetch active UMA disputes related to Polymarket.
        Uses UMA's public oracle API.
        """
        if time.time() - self._last_fetch < 60:
            return list(self._cache.values())

        try:
            # UMA oracle v2 active assertions
            r = self._session.get(
                f"{UMA_HOST}/assertions",
                params={
                    "chainId": 137,          # Polygon
                    "status": "active",
                    "requester": "0x0000000000000000000000000000000000000000",  # Polymarket's oracle
                },
                timeout=10,
            )
            if r.status_code != 200:
                log.debug(f"UMA API returned {r.status_code}")
                return []

            disputes = []
            for item in r.json().get("data", []):
                status = UMADisputeStatus(
                    market_id=item.get("ancillaryData", "")[:64],
                    assertion_id=item.get("assertionId", ""),
                    asserter_direction="YES" if item.get("assertion") else "NO",
                    assertion_ts=item.get("timestamp", 0),
                    expiry_ts=item.get("expirationTime", 0),
                    total_yes_votes=float(item.get("yesVotes", 0)),
                    total_no_votes=float(item.get("noVotes", 0)),
                    large_unrevealed=item.get("largeUnrevealedVoters", 0),
                    estimated_outcome=self._estimate_outcome(item),
                )
                disputes.append(status)
                self._cache[status.market_id] = status

            self._last_fetch = time.time()
            return disputes

        except Exception as e:
            log.debug(f"UMA oracle fetch failed: {e}")
            return []

    def _estimate_outcome(self, item: dict) -> str:
        yes_v = float(item.get("yesVotes", 0))
        no_v  = float(item.get("noVotes", 0))
        if yes_v + no_v < 1000:
            return "UNKNOWN"
        return "YES" if yes_v > no_v * 1.5 else ("NO" if no_v > yes_v * 1.5 else "UNKNOWN")


# ─────────────────────────────────────────────────────────────────────────────
# Main Dispute Trading Strategy
# ─────────────────────────────────────────────────────────────────────────────

class DisputeTrader:
    """
    Scans Polymarket for mispriced markets due to ambiguous resolution criteria.

    Three signal types:
      A. PRE_RESOLUTION: Market price diverges from strict rule reading
      B. DISPUTE_ACTIVE: UMA dispute is live, track vote direction
      C. NEAR_EXPIRY:    Market near resolution with unresolved ambiguity

    Entry: Limit order toward rule-implied price.
    Exit:  At resolution, or if market corrects >50% of mispricing.
    Size:  Kelly-scaled, 2-10% of bankroll per trade (ambiguity = uncertainty).
    """

    def __init__(
        self,
        client,
        executor,
        min_edge_cents: float = 8.0,
        min_ambiguity_score: float = 0.3,
        min_confidence: float = 0.55,
        position_size_usdc: float = 200,
        max_positions: int = 5,
        dry_run: bool = True,
    ):
        self.client = client
        self.executor = executor
        self.min_edge = min_edge_cents / 100
        self.min_ambiguity = min_ambiguity_score
        self.min_confidence = min_confidence
        self.position_size = position_size_usdc
        self.max_positions = max_positions
        self.dry_run = dry_run

        self._session = requests.Session()
        self._estimator = ResolutionDirectionEstimator()
        self._uma = UMAOracleTracker()
        self._open_positions: dict[str, DisputeSignal] = {}
        self._scanned_ids: set[str] = set()

    def fetch_candidate_markets(self) -> list[dict]:
        """
        Fetch active markets that might have ambiguous resolution.
        Focus on: politics, geopolitics, crypto narratives, celebrity.
        These categories have the most fuzzy language.
        """
        candidates = []

        for tag in ["politics", "geopolitics", "crypto", "world", "sports", "entertainment"]:
            try:
                r = self._session.get(
                    f"{GAMMA_HOST}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "tag": tag,
                        "order": "volume24hr",
                        "ascending": "false",
                        "limit": 50,
                    },
                    timeout=8,
                )
                if r.status_code == 200:
                    candidates.extend(r.json() if isinstance(r.json(), list) else r.json().get("markets", []))
            except Exception as e:
                log.debug(f"fetch_candidate_markets({tag}): {e}")

            time.sleep(0.2)

        # Deduplicate
        seen = set()
        deduped = []
        for m in candidates:
            mid = m.get("id") or m.get("conditionId", "")
            if mid and mid not in seen:
                seen.add(mid)
                deduped.append(m)

        return deduped

    def _get_yes_price(self, market: dict) -> Optional[float]:
        """Get current YES token mid price from CLOB."""
        clob_ids = market.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                return None
        if not clob_ids:
            return None

        yes_id = clob_ids[0]
        try:
            r = self._session.get(
                f"{CLOB_HOST}/book",
                params={"token_id": yes_id},
                timeout=5,
            )
            if r.status_code != 200:
                return None
            book = r.json()
            bids = sorted([float(b["price"]) for b in book.get("bids", [])
                           if 0.01 < float(b["price"]) < 0.99], reverse=True)
            asks = sorted([float(a["price"]) for a in book.get("asks", [])
                           if 0.01 < float(a["price"]) < 0.99])
            if bids and asks:
                return (bids[0] + asks[0]) / 2
        except Exception:
            pass
        return None

    def scan_markets(self) -> list[DisputeSignal]:
        """
        Main scan loop. Returns list of actionable dispute signals.
        """
        candidates = self.fetch_candidate_markets()
        active_disputes = self._uma.get_active_disputes()
        dispute_map = {d.market_id: d for d in active_disputes}

        signals = []
        log.info(f"Scanning {len(candidates)} markets for resolution ambiguity...")

        for market in candidates:
            question = market.get("question", "") or market.get("title", "")
            description = market.get("description", "") or market.get("resolutionSource", "")
            market_id = market.get("conditionId") or market.get("id", "")

            if not question or not market_id:
                continue

            # Score ambiguity
            ambi = score_resolution_ambiguity(question, description)
            if ambi.total_score < self.min_ambiguity:
                continue

            # Get current price
            yes_price = self._get_yes_price(market)
            if yes_price is None or yes_price < 0.03 or yes_price > 0.97:
                continue  # Near-resolved, skip

            # Estimate rule-implied price
            rule_price, confidence, rationale = self._estimator.estimate(
                question, description, yes_price
            )

            edge = rule_price - yes_price
            if abs(edge) < self.min_edge:
                continue
            if confidence < self.min_confidence:
                continue

            direction = "BUY_YES" if edge > 0 else "SELL_YES"

            # Check for active UMA dispute
            dispute = dispute_map.get(market_id)
            dispute_active = dispute is not None
            uma_dir = dispute.estimated_outcome if dispute else "UNKNOWN"

            # If dispute vote direction contradicts our signal, reduce confidence
            if dispute_active and uma_dir != "UNKNOWN":
                signal_yes = direction == "BUY_YES"
                dispute_yes = uma_dir == "YES"
                if signal_yes != dispute_yes:
                    confidence *= 0.6  # Penalty for contradiction
                    rationale += f" [WARNING: UMA vote says {uma_dir}]"
                else:
                    confidence = min(0.9, confidence * 1.3)  # Bonus for alignment
                    rationale += f" [UMA vote confirms {uma_dir}]"

            res_ts = market.get("endDate") or market.get("resolutionTime", 0)
            if isinstance(res_ts, str):
                try:
                    from datetime import datetime
                    res_ts = int(datetime.fromisoformat(res_ts.replace("Z", "+00:00")).timestamp())
                except Exception:
                    res_ts = int(time.time()) + 86400

            signal = DisputeSignal(
                market_id=market_id,
                question=question,
                current_price=yes_price,
                rule_implied_price=rule_price,
                edge_cents=edge * 100,
                direction=direction,
                confidence=confidence,
                ambiguity_score=ambi.total_score,
                resolution_ts=int(res_ts),
                dispute_active=dispute_active,
                uma_vote_direction=uma_dir,
                rationale=rationale,
            )

            if signal.is_tradeable:
                signals.append(signal)

            time.sleep(0.1)

        return sorted(signals, key=lambda s: abs(s.edge_cents) * s.confidence, reverse=True)

    def run_once(self):
        """Single scan + execution cycle."""
        signals = self.scan_markets()

        log.info(f"\n{'='*65}")
        log.info(f"DISPUTE SCAN | {len(signals)} tradeable signals found")
        log.info(f"{'='*65}")

        for s in signals:
            dispute_tag = " [DISPUTE ACTIVE]" if s.dispute_active else ""
            uma_tag = f" UMA→{s.uma_vote_direction}" if s.dispute_active else ""
            log.info(
                f"[{s.direction}] {s.question[:60]}{dispute_tag}\n"
                f"  Price: {s.current_price:.3f} | Rule-implied: {s.rule_implied_price:.3f} "
                f"| Edge: {s.edge_cents:+.1f}c | Conf: {s.confidence:.0%}{uma_tag}\n"
                f"  Ambiguity: {s.ambiguity_score:.2f} | {s.minutes_to_resolution:.0f}min left\n"
                f"  Rationale: {s.rationale}"
            )

        if len(self._open_positions) >= self.max_positions:
            log.info(f"Max positions ({self.max_positions}) reached, skipping new entries")
            return

        slots = self.max_positions - len(self._open_positions)
        for signal in signals[:slots]:
            if signal.market_id in self._open_positions:
                continue

            clob_ids = []
            # Determine token ID based on direction
            try:
                market_data = self._session.get(
                    f"{GAMMA_HOST}/markets/{signal.market_id}", timeout=5
                ).json()
                raw = market_data.get("clobTokenIds", "[]")
                clob_ids = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                pass

            if not clob_ids:
                continue

            # YES = clob_ids[0], NO = clob_ids[1]
            if signal.direction == "BUY_YES":
                token_id = clob_ids[0]
                entry_price = round(signal.current_price + 0.005, 3)  # Slightly aggressive
                side = "BUY"
            else:  # SELL_YES = BUY_NO
                token_id = clob_ids[1]
                entry_price = round(1.0 - signal.current_price + 0.005, 3)
                side = "BUY"

            tokens = self.position_size / entry_price
            expected_payout = tokens * 1.0  # Binary market pays $1 per token
            expected_profit = expected_payout - self.position_size

            if self.dry_run:
                log.info(
                    f"\n[DRY RUN] DISPUTE ENTRY: {signal.question[:55]}\n"
                    f"  {side} {tokens:.0f} tokens @ {entry_price:.3f} "
                    f"(token: {token_id[:12]}...)\n"
                    f"  Expected profit if correct: ${expected_profit:.2f} "
                    f"({expected_profit/self.position_size:.0%})\n"
                    f"  Edge: {signal.edge_cents:+.1f}c | Confidence: {signal.confidence:.0%}"
                )
            else:
                resp = self.client.place_limit_order(token_id, side, entry_price, tokens)
                log.info(f"Dispute order placed: {resp}")
                self._open_positions[signal.market_id] = signal

    def run_forever(self, interval: float = 120.0):
        """
        Scan every 2 minutes. Dispute signals are slow-moving — no need for
        sub-second latency here. The alpha comes from research, not speed.
        """
        log.info(f"Dispute trader started (dry_run={self.dry_run})")
        while True:
            try:
                self.run_once()
                time.sleep(interval)
            except KeyboardInterrupt:
                log.info("Stopped")
                break
            except Exception as e:
                log.error(f"Error: {e}", exc_info=True)
                time.sleep(10)
