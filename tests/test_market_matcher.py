import pytest
from datetime import UTC, datetime

from polymarket_analyzer.analysis import market_matcher
from polymarket_analyzer.analysis.market_matcher import (
    CryptoMarketMatch,
    CryptoMarketMatcher,
    _barrier_regex_match,
    _regex_match,
)
from polymarket_analyzer.models import Market


def market(title: str) -> Market:
    return Market(
        id="m1",
        title=title,
        description="",
        end_date="2026-12-31",
        liquidity=10_000,
        volume=0,
        price=0.5,
    )


@pytest.mark.parametrize("word", ["hit", "hits", "reach", "reaches", "touch", "touches"])
def test_one_touch_markets_match_barrier_not_digital(word):
    m = market(f"Will BTC {word} $90,000 by December 31, 2026?")

    assert _regex_match(m) is None
    barrier = _barrier_regex_match(m, spot=80_000)

    assert barrier is not None
    assert barrier.asset == "BTC"
    assert barrier.barrier == 90_000
    assert barrier.direction == "up"


def test_plain_above_market_matches_digital_not_barrier():
    m = market("Will BTC be above $90,000 by December 31, 2026?")

    digital = _regex_match(m)

    assert digital is not None
    assert digital.asset == "BTC"
    assert digital.strike == 90_000
    assert digital.direction == "above"
    assert _barrier_regex_match(m, spot=80_000) is None


@pytest.mark.asyncio
async def test_barrier_market_skips_llm_fallback(monkeypatch):
    async def fail_if_called(market):
        raise AssertionError("LLM fallback should not parse barrier markets as digital")

    monkeypatch.setattr(market_matcher, "_llm_match", fail_if_called)

    matcher = CryptoMarketMatcher(use_llm_fallback=True)

    assert await matcher.match(market("Will BTC touch $90,000 by December 31, 2026?")) is None


@pytest.mark.asyncio
async def test_non_barrier_can_use_llm_fallback(monkeypatch):
    expected = CryptoMarketMatch(
        market_id="m1",
        question="Can BTC close above $90,000?",
        asset="BTC",
        strike=90_000,
        direction="above",
        resolution_date=datetime(2026, 12, 31, tzinfo=UTC),
        confidence=0.75,
        match_method="llm",
    )

    async def fake_llm(market):
        return expected

    monkeypatch.setattr(market_matcher, "_llm_match", fake_llm)
    monkeypatch.setattr(market_matcher, "_regex_match", lambda market: None)

    matcher = CryptoMarketMatcher(use_llm_fallback=True)

    assert await matcher.match(market("Can BTC close above $90,000?")) is expected
