import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from ..exceptions import APIError
from .polymarket_api import CircuitBreaker, TokenBucket

logger = logging.getLogger(__name__)

BASE_URL = "https://www.deribit.com/api/v2/public"


@dataclass(frozen=True)
class DeribitInstrument:
    instrument_name: str
    currency: str
    strike: float
    expiration_timestamp: int
    expiration_date: datetime
    option_type: str
    is_active: bool


@dataclass(frozen=True)
class DeribitTicker:
    instrument_name: str
    mark_iv: float
    mark_price: float
    underlying_price: float
    open_interest: float
    best_bid_price: float | None = None
    best_ask_price: float | None = None


@dataclass(frozen=True)
class SpotPrice:
    currency: str
    price: float
    timestamp: int


class DeribitAPI:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self.rate_limiter = TokenBucket(rate=10.0, capacity=20)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_period=120.0)

    async def _request(self, method: str, params: dict | None = None) -> dict:
        await self.circuit_breaker.acquire()
        await self.rate_limiter.acquire()

        url = f"{BASE_URL}/{method}"
        try:
            resp = await self.client.get(url, params=params or {})
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise APIError(f"Deribit API error: {data['error']}")
            self.circuit_breaker.record_success()
            return data.get("result", data)
        except httpx.HTTPStatusError as e:
            self.circuit_breaker.record_failure()
            raise APIError(f"Deribit HTTP {e.response.status_code}", status_code=e.response.status_code) from e
        except httpx.RequestError as e:
            self.circuit_breaker.record_failure()
            raise APIError(f"Deribit request failed: {e}") from e

    async def get_spot_price(self, currency: str) -> SpotPrice:
        index_name = f"{currency.lower()}_usd"
        result = await self._request("get_index_price", {"index_name": index_name})
        return SpotPrice(
            currency=currency.upper(),
            price=result["index_price"],
            timestamp=int(datetime.now(UTC).timestamp() * 1000),
        )

    async def get_instruments(self, currency: str) -> list[DeribitInstrument]:
        result = await self._request("get_instruments", {
            "currency": currency.upper(),
            "kind": "option",
            "expired": "false",
        })
        instruments = []
        for item in result:
            exp_ts = item["expiration_timestamp"]
            instruments.append(DeribitInstrument(
                instrument_name=item["instrument_name"],
                currency=currency.upper(),
                strike=float(item["strike"]),
                expiration_timestamp=exp_ts,
                expiration_date=datetime.fromtimestamp(exp_ts / 1000, tz=UTC),
                option_type=item["option_type"],
                is_active=item.get("is_active", True),
            ))
        return instruments

    async def get_book_summaries(self, currency: str) -> list[DeribitTicker]:
        result = await self._request("get_book_summary_by_currency", {
            "currency": currency.upper(),
            "kind": "option",
        })
        tickers = []
        for item in result:
            mark_iv = item.get("mark_iv")
            if mark_iv is None or mark_iv <= 0:
                continue
            tickers.append(DeribitTicker(
                instrument_name=item["instrument_name"],
                mark_iv=mark_iv / 100.0,
                mark_price=item.get("mark_price", 0.0),
                underlying_price=item.get("underlying_price", 0.0),
                open_interest=item.get("open_interest", 0.0),
                best_bid_price=item.get("bid_price"),
                best_ask_price=item.get("ask_price"),
            ))
        return tickers

    async def close(self):
        await self.client.aclose()
