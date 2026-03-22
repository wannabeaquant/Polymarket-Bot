"""
Polymarket Alpha Bot — Main Orchestrator

Runs all strategies in parallel:
  1. Arb Scanner (all 3 tiers) — continuous sweep
  2. Market Maker — quote management
  3. Mean Reversion — fade shocks
  4. Cross-market Bayesian — dependency mispricing
  5. Crypto UP/DOWN — limit-order market making on XRP/SOL 5-15min
  6. Dispute Trader — ambiguous resolution language mispricing

Usage:
  python bot.py --mode scanner        # Scan only, no execution
  python bot.py --mode mm             # Market making only
  python bot.py --mode arb            # Arb only
  python bot.py --mode crypto         # XRP/SOL UP/DOWN market making
  python bot.py --mode dispute        # Dispute/ambiguity scanner
  python bot.py --mode full           # Everything (requires credentials)
  python bot.py --mode backtest       # Backtest on historical data
"""

import argparse
import logging
import time
import sys
import os
from pathlib import Path

# ─── Setup logging ────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log"),
    ],
)
log = logging.getLogger("bot")


# ─── Lazy imports (after logging setup) ──────────────────────────────────────
from data.polymarket_client import PolymarketClient
from core.arb_scanner import ArbScanner
from core.market_making import AvellanedaStoikovMM
from core.vpin import MultiMarketVPIN
from core.kelly import kelly_binary, ProbabilityEstimator
from core.hawkes_reversion import MeanReversionDetector
from core.bayesian_network import CrossMarketScanner
from execution.executor import OrderExecutor
from strategies.market_maker_strategy import MarketMakingStrategy
from strategies.crypto_arb import CryptoUPDOWNStrategy
from strategies.dispute_trader import DisputeTrader
from backtesting.backtest import MarketMakingBacktest, ArbBacktest
from config import settings


# ─── Scanner Mode (no credentials needed) ─────────────────────────────────────

def run_scanner(client: PolymarketClient):
    """
    Continuously scan for all opportunity types and log them.
    No trades executed — pure intelligence gathering.
    """
    arb_scanner = ArbScanner(client, min_profit_cents=settings.ARB_MIN_PROFIT_CENTS)
    cross_market = CrossMarketScanner(min_discrepancy_cents=3.0)

    log.info("=" * 60)
    log.info("POLYMARKET ALPHA SCANNER")
    log.info("=" * 60)

    iteration = 0
    while True:
        iteration += 1
        log.info(f"\n{'─'*40} Scan #{iteration} {'─'*40}")

        # 1. Fetch active markets
        log.info("Fetching active markets...")
        markets = client.get_markets(active=True, limit=500)
        log.info(f"Found {len(markets)} active markets")

        # Filter: only markets with meaningful volume
        tradeable = [
            m for m in markets
            if m.volume_24h >= settings.MIN_MARKET_VOLUME_24H
            and not m.resolved
            and (m.days_to_resolution is None or m.days_to_resolution > 1)
        ]
        log.info(f"Tradeable: {len(tradeable)} markets after filters")

        # 2. Arb scan
        log.info("\n[ARB SCAN] Running...")
        opps = arb_scanner.scan_all(tradeable[:100])  # Top 100 by volume for speed

        if opps:
            log.info(f"\n{'*'*40}")
            log.info(f"ARB OPPORTUNITIES FOUND: {len(opps)}")
            for opp in opps[:10]:  # Show top 10
                log.info(f"  {opp}")
                for leg in opp.legs:
                    log.info(f"    Leg: {leg['side']} {leg['token_id'][:12]}... "
                             f"@ {leg['price']:.3f} (${leg['size_usdc']:.0f})")
            log.info(f"{'*'*40}")
        else:
            log.info("[ARB SCAN] No opportunities above threshold")

        # 3. Cross-market dependency scan
        log.info("\n[CROSS-MARKET SCAN] Running...")
        cross_market.network._nodes.clear()
        cross_market.network._dependencies.clear()
        discrepancies = cross_market.scan(tradeable[:200])

        if discrepancies:
            log.info(f"\n{'*'*40}")
            log.info(f"CROSS-MARKET DISCREPANCIES: {len(discrepancies)}")
            for d in discrepancies[:5]:
                log.info(f"  {d.direction} '{d.question[:50]}': "
                         f"market={d.market_price:.3f}, implied={d.network_implied_price:.3f}, "
                         f"delta={d.discrepancy_cents:.1f}c (conf={d.confidence:.2f})")
            log.info(f"{'*'*40}")
        else:
            log.info("[CROSS-MARKET] No significant discrepancies")

        log.info(f"\nScan complete. Waiting 60s before next scan...")
        time.sleep(60)


# ─── Market Making Mode ────────────────────────────────────────────────────────

def run_market_maker(client: PolymarketClient, dry_run: bool = True):
    """Run the A-S market maker. Set dry_run=False to actually trade."""
    executor = OrderExecutor(client, dry_run=dry_run)
    mm_strategy = MarketMakingStrategy(
        client=client,
        executor=executor,
        max_markets=5,
        dry_run=dry_run,
    )

    log.info(f"Starting market maker (dry_run={dry_run})")

    def get_markets():
        return client.get_markets(active=True, limit=200)

    mm_strategy.run_forever(get_markets)


# ─── Arb-Only Mode ────────────────────────────────────────────────────────────

def run_arb_bot(client: PolymarketClient, dry_run: bool = True):
    """Run arb scanner with execution."""
    executor = OrderExecutor(client, dry_run=dry_run)
    arb_scanner = ArbScanner(client, min_profit_cents=settings.ARB_MIN_PROFIT_CENTS)

    log.info(f"Starting arb bot (dry_run={dry_run})")

    while True:
        markets = client.get_markets(active=True, limit=300)
        tradeable = [m for m in markets
                     if m.volume_24h >= settings.MIN_MARKET_VOLUME_24H and not m.resolved]

        opps = arb_scanner.scan_all(tradeable)

        for opp in opps:
            log.info(f"Executing: {opp}")
            result = executor.execute_arb(opp)
            log.info(f"Result: {result}")

        time.sleep(30)


# ─── Backtest Mode ────────────────────────────────────────────────────────────

def run_backtest(client: PolymarketClient):
    """
    Run backtests on historical data for a sample of markets.
    """
    log.info("Running backtests...")

    markets = client.get_markets(active=True, limit=50)
    if not markets:
        log.error("No markets returned, check API connectivity")
        return

    results = []
    mm_backtest = MarketMakingBacktest(gamma=0.1, min_spread=0.02)

    for market in markets[:5]:   # Test on first 5 markets
        log.info(f"Backtesting: {market.question[:60]}")

        # Fetch historical trades
        yes_trades = client.get_last_trades(market.yes_token_id, limit=500)
        if len(yes_trades) < 10:
            log.info(f"  Insufficient trade history, skipping")
            continue

        result = mm_backtest.run(
            token_id=market.yes_token_id,
            trades=yes_trades,
        )
        results.append((market.question, result))
        log.info(f"\n{result}\n")

    # Summary
    if results:
        log.info("\n" + "="*60)
        log.info("BACKTEST SUMMARY")
        log.info("="*60)
        total_pnl = sum(r.total_pnl for _, r in results)
        avg_sharpe = sum(r.sharpe_ratio for _, r in results) / len(results)
        log.info(f"Markets tested: {len(results)}")
        log.info(f"Total simulated PnL: ${total_pnl:.2f}")
        log.info(f"Average Sharpe: {avg_sharpe:.2f}")


# ─── Live Market Intelligence Report ──────────────────────────────────────────

def run_intelligence_report(client: PolymarketClient):
    """
    One-shot intelligence report: snapshot of current market conditions,
    opportunities, and recommended strategy allocation.
    """
    log.info("Generating market intelligence report...")

    markets = client.get_markets(active=True, limit=500)
    tradeable = [m for m in markets
                 if m.volume_24h >= settings.MIN_MARKET_VOLUME_24H and not m.resolved]

    log.info(f"\n{'='*60}")
    log.info("POLYMARKET INTELLIGENCE REPORT")
    log.info(f"{'='*60}")
    log.info(f"Total active markets: {len(markets)}")
    log.info(f"Tradeable (>$1k volume): {len(tradeable)}")

    # Volume distribution
    volumes = [m.volume_24h for m in tradeable]
    if volumes:
        log.info(f"Volume range: ${min(volumes):.0f} - ${max(volumes):.0f}")
        log.info(f"Median volume: ${sorted(volumes)[len(volumes)//2]:.0f}")

    # Category breakdown
    from collections import Counter
    cats = Counter(m.category for m in tradeable)
    log.info(f"\nBy category:")
    for cat, count in cats.most_common(10):
        log.info(f"  {cat or 'unknown'}: {count}")

    # Time to resolution
    soon = [m for m in tradeable if m.days_to_resolution and m.days_to_resolution < 7]
    log.info(f"\nResolving within 7 days: {len(soon)} markets")

    # MM opportunity: markets with wide spreads
    log.info("\nChecking spreads on sample markets...")
    wide_spread_markets = []
    for m in tradeable[:30]:
        book = client.get_order_book(m.yes_token_id)
        if book and book.spread and book.spread > 0.05:
            wide_spread_markets.append((m, book.spread))

    if wide_spread_markets:
        log.info(f"Wide-spread opportunities (>{5}c spread):")
        for m, spread in sorted(wide_spread_markets, key=lambda x: x[1], reverse=True)[:10]:
            log.info(f"  {m.question[:60]}: spread={spread:.3f} ({spread*100:.1f}c)")

    log.info(f"\n{'='*60}")


# ─── Crypto UP/DOWN Mode ──────────────────────────────────────────────────────

def run_crypto_arb(client: PolymarketClient, dry_run: bool = True):
    """Run XRP/SOL UP/DOWN limit-order market making."""
    executor = OrderExecutor(client, dry_run=dry_run)
    strategy = CryptoUPDOWNStrategy(
        client=client,
        executor=executor,
        min_spread_to_trade=2.0,
        position_size_usdc=100,
        dry_run=dry_run,
    )
    strategy.run_forever(interval=30.0)


# ─── Dispute Trader Mode ───────────────────────────────────────────────────────

def run_dispute_trader(client: PolymarketClient, dry_run: bool = True):
    """Scan for ambiguous resolution markets and trade the mispricing."""
    executor = OrderExecutor(client, dry_run=dry_run)
    strategy = DisputeTrader(
        client=client,
        executor=executor,
        min_edge_cents=8.0,
        min_ambiguity_score=0.3,
        min_confidence=0.55,
        position_size_usdc=200,
        max_positions=5,
        dry_run=dry_run,
    )
    strategy.run_forever(interval=120.0)


# ─── Full Mode (all strategies) ───────────────────────────────────────────────

def run_full(client: PolymarketClient, dry_run: bool = True):
    """
    Run all strategies concurrently using threads.
    Each strategy runs in its own thread with its own scan interval.
    """
    import threading

    executor = OrderExecutor(client, dry_run=dry_run)

    strategies = [
        ("scanner",  lambda: run_scanner(client)),
        ("crypto",   lambda: run_crypto_arb(client, dry_run)),
        ("dispute",  lambda: run_dispute_trader(client, dry_run)),
        ("mm",       lambda: run_market_maker(client, dry_run)),
    ]

    threads = []
    for name, fn in strategies:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        threads.append(t)
        log.info(f"Started {name} thread")

    try:
        while True:
            alive = [t.name for t in threads if t.is_alive()]
            dead = [t.name for t in threads if not t.is_alive()]
            if dead:
                log.error(f"Threads died: {dead}")
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Shutting down all strategies...")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polymarket Alpha Bot")
    parser.add_argument(
        "--mode",
        choices=["scanner", "mm", "arb", "crypto", "dispute", "backtest", "report", "full"],
        default="report",
        help="Operating mode",
    )
    parser.add_argument("--live", action="store_true", help="Execute real trades (default: dry run)")
    parser.add_argument("--private-key", default=os.environ.get("POLY_PRIVATE_KEY", ""))
    parser.add_argument("--api-key",     default=os.environ.get("POLY_API_KEY", ""))
    parser.add_argument("--api-secret",  default=os.environ.get("POLY_API_SECRET", ""))
    parser.add_argument("--api-pass",    default=os.environ.get("POLY_API_PASSPHRASE", ""))
    args = parser.parse_args()

    # Initialize client
    client = PolymarketClient(
        private_key=args.private_key,
        api_key=args.api_key,
        api_secret=args.api_secret,
        api_passphrase=args.api_pass,
    )

    dry_run = not args.live
    if args.live:
        log.warning("⚠️  LIVE TRADING MODE — real funds at risk")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            log.info("Aborted")
            return

    mode_map = {
        "scanner":  lambda: run_scanner(client),
        "mm":       lambda: run_market_maker(client, dry_run),
        "arb":      lambda: run_arb_bot(client, dry_run),
        "crypto":   lambda: run_crypto_arb(client, dry_run),
        "dispute":  lambda: run_dispute_trader(client, dry_run),
        "backtest": lambda: run_backtest(client),
        "report":   lambda: run_intelligence_report(client),
        "full":     lambda: run_full(client, dry_run),
    }

    mode_map[args.mode]()


if __name__ == "__main__":
    main()
