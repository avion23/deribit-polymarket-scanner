#!/usr/bin/env python3
"""Deribit options-implied probability scanner.

Compares Polymarket crypto binary markets against Deribit options-implied
probabilities to find mispricings.

Usage:
    python scripts/deribit_scanner.py
    python scripts/deribit_scanner.py --loop --interval 600
    python scripts/deribit_scanner.py --min-edge 5 --no-llm
"""
import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

sys.path.insert(0, "src")

from polymarket_analyzer.api.deribit_api import DeribitAPI, DeribitTicker, SpotPrice
from polymarket_analyzer.api.polymarket_api import PolymarketAPI
from polymarket_analyzer.analysis.bs_digital import VolPoint, VolSurface
from polymarket_analyzer.analysis.deribit_edge_calculator import BarrierScanResult, DeribitEdgeCalculator, DeribitScanResult, RangeScanResult
from polymarket_analyzer.analysis.market_matcher import CryptoMarketMatcher

logger = logging.getLogger(__name__)


class DeribitScanner:
    def __init__(
        self,
        min_edge_pct: float = 3.0,
        min_liquidity: float = 10_000.0,
        use_llm_fallback: bool = True,
        risk_free_rate: float = 0.05,
    ):
        self.pm_api = PolymarketAPI()
        self.deribit_api = DeribitAPI()
        self.matcher = CryptoMarketMatcher(use_llm_fallback=use_llm_fallback)
        self.edge_calc = DeribitEdgeCalculator(risk_free_rate=risk_free_rate)
        self.min_edge_pct = min_edge_pct
        self.min_liquidity = min_liquidity

    async def scan(self) -> tuple[list[DeribitScanResult], list[RangeScanResult], list[BarrierScanResult], int, int, int, int]:
        print("Fetching Polymarket crypto markets via events API...")
        all_markets = await self.pm_api.get_crypto_events()
        print(f"  Got {len(all_markets)} crypto markets")

        print("Matching crypto digital option markets...")
        matches = await self.matcher.match_batch(all_markets)
        range_matches = self.matcher.match_range_batch(all_markets)
        # Barrier matches with no spot yet — direction will be fixed after Deribit fetch
        barrier_matches_prelim = self.matcher.match_barrier_batch(all_markets)
        print(
            f"  Matched {len(matches)} digital + {len(range_matches)} range + "
            f"{len(barrier_matches_prelim)} barrier markets"
        )

        market_by_id = {m.id: m for m in all_markets}

        needed_currencies = sorted(
            set(m.asset for m in matches)
            | set(m.asset for m in range_matches)
            | set(m.asset for m in barrier_matches_prelim)
        )
        if not needed_currencies:
            return [], [], [], len(all_markets), 0, 0, 0

        print(f"Fetching Deribit data for: {', '.join(needed_currencies)}")
        vol_surfaces: dict[str, VolSurface] = {}
        spots: dict[str, float] = {}
        for currency in needed_currencies:
            spot = await self.deribit_api.get_spot_price(currency)
            summaries = await self.deribit_api.get_book_summaries(currency)
            surface = self._build_vol_surface(currency, spot, summaries)
            if surface is not None:
                vol_surfaces[currency] = surface
                spots[currency] = spot.price
                print(f"  {currency}: spot=${spot.price:,.0f}, {len(summaries)} option quotes, {len(surface._expiries)} expiries")

        # Re-match barriers now that we have spot prices so direction is correct
        barrier_matches = self.matcher.match_barrier_batch(all_markets, spots=spots)

        print("Computing edges...")
        digital_results: list[DeribitScanResult] = []
        if matches:
            matched_pairs = [(match, market_by_id[match.market_id]) for match in matches]
            digital_results = self.edge_calc.compute_batch(matched_pairs, vol_surfaces)

        range_results: list[RangeScanResult] = []
        if range_matches:
            range_pairs = [(match, market_by_id[match.market_id]) for match in range_matches]
            range_results = self.edge_calc.compute_range_batch(range_pairs, vol_surfaces)

        barrier_results: list[BarrierScanResult] = []
        if barrier_matches:
            barrier_pairs = [(match, market_by_id[match.market_id]) for match in barrier_matches]
            barrier_results = self.edge_calc.compute_barrier_batch(barrier_pairs, vol_surfaces)

        filtered_digital = [
            r for r in digital_results
            if abs(r.edge_percent) >= self.min_edge_pct
            and r.liquidity >= self.min_liquidity
        ]
        filtered_range = [
            r for r in range_results
            if abs(r.edge_percent) >= self.min_edge_pct
            and r.liquidity >= self.min_liquidity
        ]
        filtered_barrier = [
            r for r in barrier_results
            if abs(r.edge_percent) >= self.min_edge_pct
            and r.liquidity >= self.min_liquidity
        ]
        return filtered_digital, filtered_range, filtered_barrier, len(all_markets), len(matches), len(range_matches), len(barrier_matches)

    def _build_vol_surface(
        self, currency: str, spot: SpotPrice, summaries: list[DeribitTicker]
    ) -> VolSurface | None:
        now = datetime.now(UTC)

        # key: (strike, expiry_str) -> list of (iv, open_interest, underlying_price)
        by_strike_expiry: dict[tuple[float, str], list[tuple[float, float, float]]] = {}
        exp_dates: dict[str, datetime] = {}

        for ticker in summaries:
            if ticker.mark_iv <= 0 or ticker.mark_iv > 5.0:
                continue
            if ticker.open_interest <= 0:
                continue

            parts = ticker.instrument_name.split("-")
            if len(parts) != 4:
                continue

            option_type = parts[3]
            if option_type not in ("C", "P"):
                continue

            try:
                strike = float(parts[2])
            except ValueError:
                continue

            expiry_str = parts[1]
            if expiry_str not in exp_dates:
                try:
                    exp_dates[expiry_str] = datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=UTC)
                except ValueError:
                    continue

            T = (exp_dates[expiry_str] - now).total_seconds() / (365.25 * 86400)
            if T <= 0:
                continue

            key = (strike, expiry_str)
            by_strike_expiry.setdefault(key, []).append(
                (ticker.mark_iv, ticker.open_interest, ticker.underlying_price)
            )

        if not by_strike_expiry:
            return None

        # Collect per-expiry forwards: average underlying_price across all strikes
        expiry_forwards: dict[str, list[float]] = {}
        for (strike, expiry_str), entries in by_strike_expiry.items():
            for _, _, fwd in entries:
                if fwd > 0:
                    expiry_forwards.setdefault(expiry_str, []).append(fwd)

        points = []
        forwards: dict[float, float] = {}
        for (strike, expiry_str), entries in by_strike_expiry.items():
            exp_date = exp_dates[expiry_str]
            T = (exp_date - now).total_seconds() / (365.25 * 86400)

            if len(entries) == 1:
                iv = entries[0][0]
            else:
                # prefer higher open interest; if tied, average
                best_oi = max(e[1] for e in entries)
                top = [e for e in entries if e[1] == best_oi]
                iv = sum(e[0] for e in top) / len(top)

            points.append(VolPoint(strike=strike, expiry=exp_date, time_to_expiry=T, iv=iv))

            if T not in forwards and expiry_str in expiry_forwards:
                fwd_vals = expiry_forwards[expiry_str]
                forwards[T] = sum(fwd_vals) / len(fwd_vals)

        return VolSurface(
            currency=currency,
            spot=spot.price,
            points=points,
            timestamp=now,
            forwards=forwards,
        )

    def print_results(
        self,
        results: list[DeribitScanResult],
        range_results: list[RangeScanResult],
        barrier_results: list[BarrierScanResult],
        total_markets: int,
        matched: int,
        range_matched: int,
        barrier_matched: int,
    ):
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        print()
        print(f"DERIBIT OPTIONS-IMPLIED SCANNER — {now}")
        print("=" * 100)
        print(
            f"Markets scanned: {total_markets} | Digital matched: {matched} | Range matched: {range_matched} | "
            f"Barrier matched: {barrier_matched} | "
            f"Actionable digital: {len(results)} | Actionable range: {len(range_results)} | "
            f"Actionable barrier: {len(barrier_results)}"
        )
        print()

        if results:
            print("--- DIGITAL OPTIONS ---")
            header = (
                f"{'EDGE':>7} | {'ACTION':>8} | {'COST':>5} | {'MO%':>5} | {'APY':>6} | {'DAYS':>4} | "
                f"{'PM YES':>6} | {'DERIBIT':>7} | {'ASSET':>5} | {'STRIKE':>10} | "
                f"{'EXPIRY':>10} | {'IV':>6} | {'LIQ':>8}"
            )
            print(header)
            print("-" * len(header))
            for r in results:
                edge_str = f"{r.edge_percent:+.1f}%"
                strike_str = f"${r.strike:,.0f}"
                iv_str = f"{r.interpolated_iv * 100:.1f}%"
                liq_str = f"${r.liquidity / 1000:.0f}K" if r.liquidity >= 1000 else f"${r.liquidity:.0f}"
                mo_str = f"{r.monthly_return * 100:.1f}%"
                apy_str = f"{r.annualized_yield * 100:.0f}%"
                cost = r.no_price if r.action == "BUY_NO" else r.polymarket_yes_price
                extrap = "~" if r.extrapolated else " "
                print(
                    f"{edge_str:>7} | {r.action:>8} | {cost:>5.2f} | {mo_str:>5} | {apy_str:>6} | {r.days_to_expiry:>4} | "
                    f"{r.polymarket_yes_price:>6.3f} | {r.implied_probability:>7.3f} | {r.asset:>5} | {strike_str:>10} | "
                    f"{r.resolution_date.strftime('%Y-%m-%d'):>10} | {iv_str:>5}{extrap} | {liq_str:>8}"
                )
            print()
        else:
            print("No actionable digital option edges found.")
            print()

        if range_results:
            print("--- RANGE MARKETS ---")
            range_header = (
                f"{'EDGE':>7} | {'ACTION':>8} | {'COST':>5} | {'MO%':>5} | {'APY':>6} | {'DAYS':>4} | "
                f"{'PM YES':>6} | {'DERIBIT':>7} | {'ASSET':>5} | {'RANGE':>22} | "
                f"{'EXPIRY':>10} | {'LIQ':>8}"
            )
            print(range_header)
            print("-" * len(range_header))
            for r in range_results:
                edge_str = f"{r.edge_percent:+.1f}%"
                range_str = f"${r.lower_strike:,.0f}-${r.upper_strike:,.0f}"
                liq_str = f"${r.liquidity / 1000:.0f}K" if r.liquidity >= 1000 else f"${r.liquidity:.0f}"
                mo_str = f"{r.monthly_return * 100:.1f}%"
                apy_str = f"{r.annualized_yield * 100:.0f}%"
                cost = r.no_price if r.action == "BUY_NO" else r.polymarket_yes_price
                extrap = "~" if r.extrapolated else " "
                print(
                    f"{edge_str:>7} | {r.action:>8} | {cost:>5.2f} | {mo_str:>5} | {apy_str:>6} | {r.days_to_expiry:>4} | "
                    f"{r.polymarket_yes_price:>6.3f} | {r.implied_probability:>7.3f}{extrap}| {r.asset:>5} | {range_str:>22} | "
                    f"{r.resolution_date.strftime('%Y-%m-%d'):>10} | {liq_str:>8}"
                )
            print()
        else:
            print("No actionable range market edges found.")
            print()

        if barrier_results:
            print("--- BARRIER / ONE-TOUCH ---")
            barrier_header = (
                f"{'EDGE':>7} | {'ACTION':>8} | {'COST':>5} | {'MO%':>5} | {'APY':>6} | {'DAYS':>4} | "
                f"{'PM YES':>6} | {'TOUCH P':>7} | {'ASSET':>5} | {'DIR':>4} | {'BARRIER':>10} | "
                f"{'EXPIRY':>10} | {'IV':>6} | {'LIQ':>8}"
            )
            print(barrier_header)
            print("-" * len(barrier_header))
            for r in barrier_results:
                edge_str = f"{r.edge_percent:+.1f}%"
                barrier_str = f"${r.barrier:,.0f}"
                iv_str = f"{r.interpolated_iv * 100:.1f}%"
                liq_str = f"${r.liquidity / 1000:.0f}K" if r.liquidity >= 1000 else f"${r.liquidity:.0f}"
                mo_str = f"{r.monthly_return * 100:.1f}%"
                apy_str = f"{r.annualized_yield * 100:.0f}%"
                cost = r.no_price if r.action == "BUY_NO" else r.polymarket_yes_price
                extrap = "~" if r.extrapolated else " "
                print(
                    f"{edge_str:>7} | {r.action:>8} | {cost:>5.2f} | {mo_str:>5} | {apy_str:>6} | {r.days_to_expiry:>4} | "
                    f"{r.polymarket_yes_price:>6.3f} | {r.implied_probability:>7.3f} | {r.asset:>5} | {r.direction:>4} | {barrier_str:>10} | "
                    f"{r.resolution_date.strftime('%Y-%m-%d'):>10} | {iv_str:>5}{extrap} | {liq_str:>8}"
                )
            print()
        else:
            print("No actionable barrier option edges found.")
            print()

        print("~ = extrapolated IV (strike outside Deribit grid, lower confidence)")
        print("NOTE: N(d2) uses risk-neutral probabilities. Vol risk premium means implied")
        print("probabilities systematically understate upside / overstate downside.")
        print("BARRIER: one-touch prices use Reiner-Rubinstein continuous-monitoring formula.")
        print("Treat edges < 5% with skepticism.")
        print()

    async def close(self):
        await self.deribit_api.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Deribit options-implied probability scanner")
    parser.add_argument("--min-edge", type=float, default=3.0, help="Minimum edge %% to display (default: 3.0)")
    parser.add_argument("--min-liquidity", type=float, default=10_000, help="Minimum PM liquidity (default: 10000)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback for market matching")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between scans in loop mode (default: 600)")
    parser.add_argument("--risk-free-rate", type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


async def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    noisy_loggers = ["httpx", "httpcore", "hpack", "h2", "h11", "asyncio"]
    if not args.verbose:
        noisy_loggers.extend([
            "polymarket_analyzer.api.polymarket_api",
            "polymarket_analyzer.api.deribit_api",
        ])
    for noisy in noisy_loggers:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    scanner = DeribitScanner(
        min_edge_pct=args.min_edge,
        min_liquidity=args.min_liquidity,
        use_llm_fallback=not args.no_llm,
        risk_free_rate=args.risk_free_rate,
    )

    try:
        if args.loop:
            while True:
                try:
                    results, range_results, barrier_results, total, matched, range_matched, barrier_matched = await scanner.scan()
                    scanner.print_results(results, range_results, barrier_results, total, matched, range_matched, barrier_matched)
                except Exception:
                    logger.exception("Scan cycle failed")
                await asyncio.sleep(args.interval)
        else:
            results, range_results, barrier_results, total, matched, range_matched, barrier_matched = await scanner.scan()
            scanner.print_results(results, range_results, barrier_results, total, matched, range_matched, barrier_matched)
    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(main())
