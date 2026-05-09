import logging
import re
from dataclasses import dataclass
from datetime import datetime

from dateutil import parser as dateutil_parser

from ..models import Market

logger = logging.getLogger(__name__)

CRYPTO_KEYWORDS = {"bitcoin", "btc", "ethereum", "ether", "eth"}

ASSET_MAP = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
}

ABOVE_WORDS = {
    "above", "exceed", "exceeds", "reach", "reaches", "surpass", "surpasses",
    "over", "higher than", "greater than", "hit", "hits", "top",
    "rise", "rise to", "rise above", "climb", "climb to", "climb above",
    "rally", "rally to", "rally above", "break above", "surge", "surge to", "surge above",
}
BELOW_WORDS = {
    "below", "under", "lower than", "less than", "beneath",
    "fall", "fall to", "fall below",
    "drop", "drop to", "drop below",
    "dip", "dip to", "dip below",
    "crash", "crash to", "crash below",
    "sink", "sink to", "sink below",
    "plunge", "plunge to", "plunge below",
    "decline", "decline to", "decline below",
}

_ASSET_RE = r"(?P<asset>bitcoin|btc|ethereum|ether|eth)"
_DIRECTION_RE = (
    r"(?P<direction>"
    r"above|below|exceed|exceeds?|reach|reaches?|surpass|surpasses?|over|under|hit|hits?|top"
    r"|higher\s+than|lower\s+than|greater\s+than|less\s+than"
    r"|rise(?:\s+(?:to|above))?|climb(?:\s+(?:to|above))?|rally(?:\s+(?:to|above))?"
    r"|break\s+above|surge(?:\s+(?:to|above))?"
    r"|fall(?:\s+(?:to|below))?|drop(?:\s+(?:to|below))?|dip(?:\s+(?:to|below))?"
    r"|crash(?:\s+(?:to|below))?|sink(?:\s+(?:to|below))?|plunge(?:\s+(?:to|below))?"
    r"|decline(?:\s+(?:to|below))?"
    r")"
)
_PRICE_RE = r"\$?\s*(?P<price>[\d,]+(?:\.[\d]+)?)\s*(?P<suffix>[kKmM])?"
_DATE_RE = r"(?:by|on|before|after|at\s+the\s+end\s+of|end\s+of)?\s*(?P<date>.+?)$"

PATTERNS = [
    re.compile(
        rf"(?:will|if|can)\s+{_ASSET_RE}(?:'s?\s+price)?\s+(?:be\s+)?{_DIRECTION_RE}\s+{_PRICE_RE}\s+{_DATE_RE}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:will|if|can)\s+{_ASSET_RE}(?:'s?\s+price)?\s+{_DIRECTION_RE}\s+{_PRICE_RE}\s*\??$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_ASSET_RE}\s+(?:price\s+)?{_DIRECTION_RE}\s+{_PRICE_RE}\s+{_DATE_RE}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:will|if)\s+the\s+price\s+of\s+{_ASSET_RE}\s+(?:be\s+)?{_DIRECTION_RE}\s+{_PRICE_RE}\s+{_DATE_RE}",
        re.IGNORECASE,
    ),
]

BARRIER_WORDS = {"at any point", "at any time", "ever", "any moment", "touches", "once"}
MULTI_CONDITION_WORDS = {" or ", " and ", " first", " versus ", " vs "}

# One-touch barrier patterns: "Will BTC reach/hit $X in May" or "by December 31, 2026"
# These resolve YES as soon as price touches the barrier at any point before expiry.
_BARRIER_ASSET_RE = r"(?P<asset>bitcoin|btc|ethereum|ether|eth)"
_BARRIER_TOUCH_RE = r"(?P<direction>reach|reaches|hit|hits)"
_BARRIER_PRICE_RE = r"\$?\s*(?P<price>[\d,]+(?:\.[\d]+)?)\s*(?P<suffix>[kKmM])?"

# "Will Bitcoin reach $150,000 in May?" / "Will Bitcoin hit $150k by December 31, 2026?"
# Also handles "May 4-10" style date ranges (e.g. weekly candle markets)
_BARRIER_PATTERN = re.compile(
    rf"(?:will|if|can)\s+{_BARRIER_ASSET_RE}(?:'s?\s+price)?\s+{_BARRIER_TOUCH_RE}\s+{_BARRIER_PRICE_RE}"
    rf"(?:\s+(?:in|by|before|on)\s+(?P<date>.+?))?$",
    re.IGNORECASE,
)

# "Will the price of Bitcoin be between $80,000 and $82,000 on May 10?"
_RANGE_PATTERN = re.compile(
    rf"(?:will\s+(?:the\s+price\s+of\s+)?)?{_ASSET_RE}(?:'s?\s+price)?\s+(?:be\s+)?between\s+"
    rf"\$?\s*(?P<lower>[\d,]+(?:\.[\d]+)?)\s*(?P<lower_suffix>[kKmM])?"
    rf"\s+and\s+"
    rf"\$?\s*(?P<upper>[\d,]+(?:\.[\d]+)?)\s*(?P<upper_suffix>[kKmM])?"
    rf"\s+{_DATE_RE}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CryptoMarketMatch:
    market_id: str
    question: str
    asset: str
    strike: float
    direction: str
    resolution_date: datetime
    confidence: float
    match_method: str


@dataclass(frozen=True)
class RangeMarketMatch:
    market_id: str
    question: str
    asset: str
    lower_strike: float
    upper_strike: float
    resolution_date: datetime
    confidence: float
    match_method: str


@dataclass(frozen=True)
class BarrierMarketMatch:
    """A one-touch/barrier market that resolves YES if price reaches the barrier
    at any point before expiry (path-dependent, not European)."""
    market_id: str
    question: str
    asset: str
    barrier: float
    direction: str  # "up" (barrier > spot) or "down" (barrier < spot)
    resolution_date: datetime
    confidence: float
    match_method: str


def _has_crypto_keywords(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in CRYPTO_KEYWORDS)


def _is_barrier_option(title: str) -> bool:
    title_lower = title.lower()
    return any(w in title_lower for w in BARRIER_WORDS)


def _parse_price(price_str: str, suffix: str | None) -> float | None:
    try:
        val = float(price_str.replace(",", ""))
    except ValueError:
        return None
    if suffix:
        s = suffix.lower()
        if s == "k":
            val *= 1_000
        elif s == "m":
            val *= 1_000_000
    if val <= 0:
        return None
    return val


def _parse_direction(direction_str: str) -> str | None:
    d = direction_str.lower().strip()
    for w in BELOW_WORDS:
        if w in d:
            return "below"
    for w in ABOVE_WORDS:
        if w in d:
            return "above"
    return None


def _parse_asset(asset_str: str) -> str | None:
    return ASSET_MAP.get(asset_str.lower().strip())


_END_OF_RE = re.compile(r"(?:end\s+of|year[- ]end)\s+(\d{4})", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})$", re.IGNORECASE)
_MONTH_ONLY_RE = re.compile(r"^(?:in\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"Q([1-4])\s+(\d{4})", re.IGNORECASE)

_QUARTER_END = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}
_MONTH_LAST_DAY = {
    "january": "01-31", "february": "02-28", "march": "03-31", "april": "04-30",
    "may": "05-31", "june": "06-30", "july": "07-31", "august": "08-31",
    "september": "09-30", "october": "10-31", "november": "11-30", "december": "12-31",
}


def _parse_date(date_str: str, market_end_date: str | None) -> datetime | None:
    from datetime import timezone

    date_str = date_str.strip().rstrip("?.,!;")
    if not date_str and market_end_date:
        date_str = market_end_date

    if not date_str:
        return None

    m = _END_OF_RE.search(date_str)
    if m:
        return datetime(int(m.group(1)), 12, 31, tzinfo=timezone.utc)

    m = _QUARTER_RE.search(date_str)
    if m:
        md = _QUARTER_END[m.group(1)]
        return dateutil_parser.parse(f"{m.group(2)}-{md}").replace(tzinfo=timezone.utc)

    m = _MONTH_YEAR_RE.match(date_str)
    if m:
        md = _MONTH_LAST_DAY[m.group(1).lower()]
        return dateutil_parser.parse(f"{m.group(2)}-{md}").replace(tzinfo=timezone.utc)

    m = _MONTH_ONLY_RE.match(date_str)
    if m:
        month_name = m.group(1).lower()
        md = _MONTH_LAST_DAY[month_name]
        month_num = int(md.split("-")[0])
        now = datetime.now(timezone.utc)
        year = now.year if month_num >= now.month else now.year + 1
        return dateutil_parser.parse(f"{year}-{md}").replace(tzinfo=timezone.utc)

    try:
        dt = dateutil_parser.parse(date_str, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, OverflowError):
        pass

    if market_end_date:
        try:
            dt = dateutil_parser.parse(market_end_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, OverflowError):
            pass

    return None


def _is_multi_condition(title: str) -> bool:
    title_lower = title.lower()
    return any(w in title_lower for w in MULTI_CONDITION_WORDS)


def _regex_match(market: Market) -> CryptoMarketMatch | None:
    title = market.title
    if _is_barrier_option(title):
        return None
    if _is_multi_condition(title):
        return None

    for pattern in PATTERNS:
        m = pattern.search(title)
        if not m:
            continue

        asset = _parse_asset(m.group("asset"))
        if not asset:
            continue

        price = _parse_price(m.group("price"), m.groupdict().get("suffix"))
        if not price:
            continue

        direction = _parse_direction(m.group("direction"))
        if not direction:
            continue

        date_str = m.groupdict().get("date", "")
        resolution_date = _parse_date(date_str, market.end_date)
        if not resolution_date:
            continue

        return CryptoMarketMatch(
            market_id=market.id,
            question=title,
            asset=asset,
            strike=price,
            direction=direction,
            resolution_date=resolution_date,
            confidence=0.9,
            match_method="regex",
        )

    return None


try:
    import dspy

    class CryptoMarketParser(dspy.Signature):
        """Parse a Polymarket question about cryptocurrency price into structured fields.
        Only parse markets that are clearly binary YES/NO about BTC or ETH price
        being above or below a specific USD threshold on a specific date.
        Return is_crypto_digital=False for anything ambiguous."""

        market_title: str = dspy.InputField()
        market_description: str = dspy.InputField(default="")

        is_crypto_digital: bool = dspy.OutputField(
            desc="True only if binary YES/NO about BTC or ETH price vs a USD threshold on a date"
        )
        asset: str = dspy.OutputField(desc="BTC or ETH, empty if not crypto digital")
        strike_usd: float = dspy.OutputField(desc="Strike price in USD, 0 if not parseable")
        direction: str = dspy.OutputField(desc="above or below, empty if ambiguous")
        resolution_date_iso: str = dspy.OutputField(desc="ISO date YYYY-MM-DD, empty if not parseable")
        confidence: float = dspy.OutputField(desc="0.0 to 1.0 confidence in parsing accuracy")

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False


async def _llm_match(market: Market) -> CryptoMarketMatch | None:
    if not _HAS_DSPY:
        return None

    try:
        predictor = dspy.Predict(CryptoMarketParser)
        result = predictor(
            market_title=market.title,
            market_description=market.description or "",
        )

        if not result.is_crypto_digital:
            return None
        if not result.asset or result.asset not in ("BTC", "ETH"):
            return None
        if not result.direction or result.direction not in ("above", "below"):
            return None
        if not result.strike_usd or result.strike_usd <= 0:
            return None
        if not result.resolution_date_iso:
            return None

        try:
            dt = dateutil_parser.parse(result.resolution_date_iso)
            if dt.tzinfo is None:
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, OverflowError):
            return None

        confidence = min(float(result.confidence), 0.8)

        return CryptoMarketMatch(
            market_id=market.id,
            question=market.title,
            asset=result.asset,
            strike=float(result.strike_usd),
            direction=result.direction,
            resolution_date=dt,
            confidence=confidence,
            match_method="llm",
        )
    except Exception:
        logger.warning("LLM market parsing failed for %s", market.id, exc_info=True)
        return None


def _barrier_regex_match(market: Market, spot: float | None = None) -> BarrierMarketMatch | None:
    """Match one-touch barrier markets like 'Will Bitcoin reach $150K in May?'

    The direction (up/down) is determined by comparing the barrier to the
    current spot price when provided.  Without a spot price, we default to
    "up" since the vast majority of observed barrier markets have upside targets.

    Excludes double-barrier / race markets (" or ... first").
    """
    title = market.title
    if not _has_crypto_keywords(title):
        return None
    # Exclude double-barrier race markets
    if _is_multi_condition(title):
        return None

    m = _BARRIER_PATTERN.search(title)
    if not m:
        return None

    asset = _parse_asset(m.group("asset"))
    if not asset:
        return None

    price = _parse_price(m.group("price"), m.groupdict().get("suffix"))
    if not price:
        return None

    date_str = m.groupdict().get("date") or ""
    resolution_date = _parse_date(date_str, market.end_date)
    if not resolution_date:
        return None

    direction = "up" if (spot is None or price > spot) else "down"

    return BarrierMarketMatch(
        market_id=market.id,
        question=title,
        asset=asset,
        barrier=price,
        direction=direction,
        resolution_date=resolution_date,
        confidence=0.9,
        match_method="regex",
    )


def _range_regex_match(market: Market) -> RangeMarketMatch | None:
    title = market.title
    if not _has_crypto_keywords(title):
        return None

    m = _RANGE_PATTERN.search(title)
    if not m:
        return None

    asset = _parse_asset(m.group("asset"))
    if not asset:
        return None

    lower = _parse_price(m.group("lower"), m.group("lower_suffix"))
    upper = _parse_price(m.group("upper"), m.group("upper_suffix"))
    if not lower or not upper or lower >= upper:
        return None

    date_str = m.groupdict().get("date", "")
    resolution_date = _parse_date(date_str, market.end_date)
    if not resolution_date:
        return None

    return RangeMarketMatch(
        market_id=market.id,
        question=title,
        asset=asset,
        lower_strike=lower,
        upper_strike=upper,
        resolution_date=resolution_date,
        confidence=0.95,
        match_method="regex",
    )


class CryptoMarketMatcher:
    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback

    async def match(self, market: Market) -> CryptoMarketMatch | None:
        if not _has_crypto_keywords(market.title):
            return None

        result = _regex_match(market)
        if result is not None:
            return result

        if self.use_llm_fallback:
            result = await _llm_match(market)
            if result is not None and result.confidence >= 0.7:
                return result

        return None

    async def match_batch(self, markets: list[Market]) -> list[CryptoMarketMatch]:
        results = []
        for market in markets:
            match = await self.match(market)
            if match is not None:
                results.append(match)
        return results

    def match_range(self, market: Market) -> RangeMarketMatch | None:
        return _range_regex_match(market)

    def match_range_batch(self, markets: list[Market]) -> list[RangeMarketMatch]:
        return [m for market in markets if (m := _range_regex_match(market)) is not None]

    def match_barrier(self, market: Market, spot: float | None = None) -> BarrierMarketMatch | None:
        return _barrier_regex_match(market, spot=spot)

    def match_barrier_batch(
        self, markets: list[Market], spots: dict[str, float] | None = None
    ) -> list[BarrierMarketMatch]:
        """Match all barrier markets.  spots maps asset ('BTC'/'ETH') -> current spot price."""
        results = []
        for market in markets:
            spot = None
            if spots:
                # We don't know the asset yet; pass None and let the regex handle it
                # Then fix direction after we know the asset
                match = _barrier_regex_match(market, spot=None)
                if match is not None:
                    spot = spots.get(match.asset)
                    if spot is not None:
                        direction = "up" if match.barrier > spot else "down"
                        if direction != match.direction:
                            match = BarrierMarketMatch(
                                market_id=match.market_id,
                                question=match.question,
                                asset=match.asset,
                                barrier=match.barrier,
                                direction=direction,
                                resolution_date=match.resolution_date,
                                confidence=match.confidence,
                                match_method=match.match_method,
                            )
                    results.append(match)
            else:
                match = _barrier_regex_match(market, spot=None)
                if match is not None:
                    results.append(match)
        return results
