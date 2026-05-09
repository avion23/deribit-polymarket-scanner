"""Polymarket API client using httpx for async operations.

This module provides async access to Polymarket data through:
1. Gamma API (for market metadata)
2. CLOB API (for order book and historical prices)
"""

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, Dict, Optional

import httpx

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import TradeParams
except ModuleNotFoundError:
    ClobClient = None
    TradeParams = dict[str, Any]

from ..exceptions import APIError, AuthenticationError, GammaAPIError
from ..models import Market
from ..settings import settings
from ..utils import safe_float

logger = logging.getLogger(__name__)


def _redact_address(address: str | None) -> str:
    """Redact an Ethereum address for safe logging.

    Args:
        address: Ethereum address string

    Returns:
        Redacted address showing only first 6 and last 4 characters
    """
    if not address or len(address) < 10:
        return "<redacted>"
    return f"{address[:6]}...{address[-4:]}"


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    Opens after failure_threshold failures, remains open for cooldown_period.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_period: float = 300.0):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            cooldown_period: Seconds to keep circuit open (default 5 minutes)
        """
        self.failure_threshold = failure_threshold
        self.cooldown_period = cooldown_period
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Check if circuit is open, raise exception if so."""
        async with self._lock:
            now = time.time()

            # Check if circuit should reset after cooldown
            if (
                self.failure_count >= self.failure_threshold
                and now - self.last_failure_time > self.cooldown_period
            ):
                logger.info("Circuit breaker cooldown expired, resetting")
                self.failure_count = 0
                self.last_failure_time = 0.0

            # If circuit is open, raise exception
            if self.failure_count >= self.failure_threshold:
                remaining_time = self.cooldown_period - (now - self.last_failure_time)
                raise APIError(
                    f"Circuit breaker is open. "
                    f"{self.failure_count} failures occurred. "
                    f"Retry in {remaining_time:.0f} seconds."
                )

    def record_success(self) -> None:
        """Record a successful request, reset failure count."""
        if self.failure_count > 0:
            logger.debug(
                f"Circuit breaker: Success after {self.failure_count} failures, resetting"
            )
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_failure(self) -> None:
        """Record a failed request, increment failure count."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning(
            f"Circuit breaker: Failure recorded ({self.failure_count}/{self.failure_threshold})"
        )

        if self.failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit breaker: OPEN due to {self.failure_count} failures. "
                f"Cooldown: {self.cooldown_period}s"
            )


# Contract addresses (Polygon Mainnet)
USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# ERC20 ABI (simplified)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
]

# ERC1155 ABI (for position tokens)
ERC1155_ABI = [
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }
]

PUBLIC_RPC_URL = "https://polygon-bor.publicnode.com"


class TokenBucket:
    """Token bucket rate limiter.

    Allows bursts up to capacity, then refills at rate tokens per second.
    Thread-safe for async operations.
    """

    def __init__(self, rate: float = 1.0, capacity: int = 2):
        """Initialize token bucket.

        Args:
            rate: Tokens added per second (default 1.0 = 1 RPS)
            capacity: Maximum bucket size (default 2, allows small bursts)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens from bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1.0)

        Raises:
            ValueError: If tokens requested exceeds capacity
        """
        if tokens > self.capacity:
            raise ValueError(
                f"Requested {tokens} tokens exceeds capacity {self.capacity}"
            )

        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            # If not enough tokens, wait for refill
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for tokens")
                await asyncio.sleep(wait_time)

                # Update after waiting
                self.tokens = 0.0
                self.last_update = time.time()
            else:
                self.tokens -= tokens

    def get_available_tokens(self) -> float:
        """Get current available tokens without acquiring.

        Returns:
            Current token count (for monitoring)
        """
        now = time.time()
        elapsed = now - self.last_update
        return min(self.capacity, self.tokens + elapsed * self.rate)


class PolymarketAPI:
    """Unified async API client for Polymarket data."""

    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    def __init__(self):
        proxy_url = self._get_proxy_url()
        client_kwargs = {
            "headers": {"User-Agent": "polymarket-analyzer/1.0"},
            "timeout": 30.0,
            "limits": httpx.Limits(max_connections=100, max_keepalive_connections=20),
        }
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
            logger.info(f"Using HTTP proxy: {proxy_url.split('@')[-1]}")

        self.client = httpx.AsyncClient(**client_kwargs)

        # Initialize rate limiter (1 RPS with burst capacity of 2)
        self.rate_limiter = TokenBucket(rate=1.0, capacity=2)
        logger.info("Rate limiter initialized: 1 RPS with burst capacity 2")

        # Initialize circuit breaker (3 failures triggers 5min pause)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3, cooldown_period=300.0
        )
        logger.info("Circuit breaker initialized: 3 failures -> 5min pause")

        # Initialize CLOB client and trading
        self.clob = None
        self.web3 = None
        self.signer_address = None  # For blockchain queries (actual wallet)
        self.address = None  # For trading operations (may differ for POLY_PROXY)
        self.trading_enabled = False
        self._init_trading_clients()

    def _get_proxy_url(self) -> str | None:
        """Get proxy URL from environment variables."""

        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        if proxy:
            return proxy
        return None

    def _init_trading_clients(self):
        """Initialize CLOB client and Web3 for trading operations."""
        try:
            clob_host = settings.polymarket_host or "https://clob.polymarket.com"

            if settings.dry_run:
                self.clob = ClobClient(clob_host) if ClobClient else None
                self.trading_enabled = False
                logger.info("Dry run enabled; initialized read-only CLOB client")
                return

            from eth_account import Account
            from web3 import Web3

            if settings.polymarket_private_key:
                # Validate and normalize private key
                pk = settings.polymarket_private_key
                if not pk.startswith("0x"):
                    pk = "0x" + pk

                if len(pk) != 66:
                    raise AuthenticationError(
                        f"Invalid private key format: expected 66 characters (0x + 64 hex), got {len(pk)}"
                    )

                try:
                    account = Account.from_key(pk)
                except Exception as e:
                    raise AuthenticationError(f"Invalid private key: {e}") from e

                self.signer_address = account.address

                # Determine signature type and funder
                if settings.polymarket_proxy_address:
                    signature_type = 1  # POLY_PROXY for email/social login
                    funder_address = settings.polymarket_proxy_address
                    logger.info(
                        f"Using POLY_PROXY mode (type=1) with signer={_redact_address(self.signer_address)}, funder={_redact_address(funder_address)}"
                    )
                else:
                    signature_type = 0  # EOA for direct MetaMask
                    funder_address = self.signer_address
                    logger.info(
                        f"Using EOA mode (type=0) with signer={_redact_address(self.signer_address)}"
                    )

                # Create temporary client to derive API credentials
                temp_client = ClobClient(
                    host=clob_host,
                    key=settings.polymarket_private_key,
                    chain_id=settings.polymarket_chain_id,
                )

                # Derive API credentials
                try:
                    api_creds = temp_client.create_or_derive_api_creds()
                    logger.info(
                        f"Derived API credentials for address: {_redact_address(self.signer_address)}"
                    )
                except Exception as e:
                    logger.error(f"Failed to derive API credentials: {e}")
                    raise AuthenticationError(
                        f"API credential derivation failed: {e}"
                    ) from e

                # Initialize authenticated client with funder
                self.clob = ClobClient(
                    host=clob_host,
                    key=settings.polymarket_private_key,
                    chain_id=settings.polymarket_chain_id,
                    creds=api_creds,
                    signature_type=signature_type,
                    funder=funder_address,
                )

                self.trading_enabled = True
                self.address = (
                    funder_address  # Use funder address for balance/trade queries
                )

                # Initialize web3 for balance checks
                alchemy_key = os.getenv("ALCHEMY_API_KEY", "")
                if alchemy_key:
                    rpc_url = f"https://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}"
                else:
                    rpc_url = "https://polygon-bor.publicnode.com"
                self.web3 = Web3(Web3.HTTPProvider(rpc_url))
                logger.info(
                    f"Wallet signer: {_redact_address(self.signer_address)}, Funder: {_redact_address(funder_address)}"
                )

            elif settings.polymarket_api_key and settings.polymarket_api_secret:
                # API credentials provided but no private key - can't sign orders
                logger.warning(
                    "API credentials provided but no private key - trading disabled"
                )
                self.clob = ClobClient(clob_host)
                self.trading_enabled = False

            else:
                # Read-only client
                self.clob = ClobClient(clob_host)
                self.trading_enabled = False
                logger.info("Initialized read-only CLOB client (no credentials)")

        except ImportError as e:
            logger.warning(f"py-clob-client or web3 not installed: {e}")
            self.trading_enabled = False
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CLOB client: {e}")
            raise APIError(f"CLOB initialization failed: {e}") from e

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _parse_price(self, data: dict) -> float:
        """Parse YES price from market data."""
        try:
            prices = data.get("outcomePrices")
            if isinstance(prices, str):
                prices = json.loads(prices)
            if isinstance(prices, list) and len(prices) > 0:
                return float(prices[0])
            return 0.5
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logger.exception(f"Failed to parse price: {e}")
            raise

    def _parse_market_odds(self, data: dict) -> dict[str, float]:
        """Parse market odds from data."""
        try:
            outcomes = data.get("outcomes")
            prices = data.get("outcomePrices")

            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(prices, str):
                prices = json.loads(prices)

            if outcomes and prices and len(outcomes) == len(prices):
                return {
                    str(outcome): safe_float(price)
                    for outcome, price in zip(outcomes, prices, strict=True)
                }

            price = self._parse_price(data)
            return {"yes": price, "no": 1.0 - price}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.exception(f"Failed to parse market odds: {e}")
            raise

    def _extract_resolved_outcome(self, data: dict) -> str | None:
        """Extract resolved outcome from market data."""
        try:
            # Check explicit outcome fields
            explicit = (
                data.get("resolvedOutcome") or data.get("winner") or data.get("result")
            )
            if explicit is not None:
                return str(explicit).lower()

            # Infer from outcome prices
            outcomes = data.get("outcomes")
            prices = data.get("outcomePrices")

            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(prices, str):
                prices = json.loads(prices)

            if outcomes and prices and len(outcomes) == len(prices):
                prices = [safe_float(p) for p in prices]
                if prices:
                    max_price = max(prices)
                    if max_price > 0.5:
                        winner_idx = prices.index(max_price)
                        return str(outcomes[winner_idx]).lower()

            return None
        except (json.JSONDecodeError, IndexError, TypeError, ValueError) as e:
            logger.exception(f"Failed to extract resolved outcome: {e}")
            raise

    def _parse_token_ids(self, data: dict) -> list[str] | None:
        """Parse CLOB token IDs from market data.

        The Gamma API returns clobTokenIds as a JSON-encoded string.
        Returns list of token IDs where index 0 is YES and index 1 is NO.
        """
        try:
            token_ids_str = data.get("clobTokenIds")
            if not token_ids_str:
                return None

            if isinstance(token_ids_str, list):
                return [str(t) for t in token_ids_str]

            if isinstance(token_ids_str, str):
                parsed = json.loads(token_ids_str)
                if isinstance(parsed, list):
                    return [str(t) for t in parsed]

            return None
        except (json.JSONDecodeError, TypeError) as e:
            logger.exception(f"Failed to parse token IDs: {e}")
            raise

    async def _request(self, url: str, params: dict) -> httpx.Response:
        """Make a single async request with rate limiting and circuit breaker."""
        # Check circuit breaker
        await self.circuit_breaker.acquire()

        # Apply rate limiting
        await self.rate_limiter.acquire()

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            self.circuit_breaker.record_success()
            return response
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

    async def get_markets(
        self,
        limit: int = 20,
        active: bool = True,
        future_only: bool = True,
        min_liquidity: float | None = None,
        min_volume: float | None = None,
    ) -> list[Market]:
        """Fetch markets from Gamma API with offset-based pagination.

        Args:
            limit: Maximum number of markets to fetch (default 20, max 10000 total)
            active: Only fetch active markets
            future_only: Only fetch markets ending in the future
            min_liquidity: Minimum liquidity filter (server-side)
            min_volume: Minimum volume filter (server-side)

        Returns:
            List of Market objects
        """
        all_markets: list[Market] = []
        offset = 0
        page_size = min(500, limit)  # API returns max 500 per request
        seen_ids = set()

        while len(all_markets) < limit:
            params: dict[str, Any] = {
                "limit": min(page_size, limit - len(all_markets)),
                "offset": offset,
                "active": active,
                "closed": False,
                "archived": False,
            }
            if future_only:
                params["endDateAfter"] = datetime.now(UTC).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            if min_liquidity is not None:
                params["liquidity_num_min"] = min_liquidity
            if min_volume is not None:
                params["volume_num_min"] = min_volume

            try:
                response = await self._request(f"{self.GAMMA_URL}/markets", params)
                data = response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Gamma API HTTP error: {e.response.status_code} - {e.response.text}"
                )
                raise GammaAPIError(
                    f"Failed to fetch markets: HTTP {e.response.status_code}",
                    status_code=e.response.status_code,
                    response_body=e.response.text,
                ) from e
            except httpx.RequestError as e:
                logger.error(f"Gamma API request error: {e}")
                raise GammaAPIError(
                    f"Failed to fetch markets: Request error - {e}",
                ) from e
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gamma API response: {e}")
                raise GammaAPIError(
                    f"Failed to parse markets response: {e}",
                ) from e

            # Handle both list and dict responses
            if isinstance(data, list):
                market_list = data
            else:
                market_list = data.get("data", [])

            # Parse markets from this page (avoid duplicates)
            new_markets = 0
            for m in market_list:
                try:
                    market_id = str(m.get("conditionId", m.get("id", "")))
                    if not market_id or market_id in seen_ids:
                        continue

                    seen_ids.add(market_id)

                    # Parse token IDs for CLOB API
                    token_ids = self._parse_token_ids(m)

                    all_markets.append(
                        Market(
                            id=market_id,
                            title=m.get("question", "").strip(),
                            description=m.get("description", "").strip(),
                            url=f"https://polymarket.com/event/{m.get('slug', '')}",
                            end_date=m.get("endDate"),
                            volume=safe_float(m.get("volume", m.get("volumeNum", 0))),
                            liquidity=safe_float(
                                m.get("liquidity", m.get("liquidityNum", 0))
                            ),
                            price=self._parse_price(m),
                            market_odds=self._parse_market_odds(m),
                            token_ids=token_ids,
                        )
                    )
                    new_markets += 1
                except (KeyError, ValueError, TypeError) as e:
                    logger.exception(
                        f"Failed to parse market {m.get('conditionId', m.get('id', 'unknown'))}: {e}"
                    )
                    raise

            # Check if we should continue pagination
            if new_markets == 0 or len(all_markets) >= limit:
                break

            offset += page_size

        return all_markets

    async def get_crypto_events(self) -> list[Market]:
        """Fetch all active crypto markets via the events endpoint, paginating until exhausted."""
        crypto_keywords = {"bitcoin", "btc", "ethereum", "ether", "eth"}
        now_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        all_markets: list[Market] = []
        seen_ids: set[str] = set()
        offset = 0
        page_size = 500
        total_events = 0
        crypto_events = 0

        while True:
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
                "active": True,
                "closed": False,
                "archived": False,
            }
            try:
                response = await self._request(f"{self.GAMMA_URL}/events", params)
                data = response.json()
            except httpx.HTTPStatusError as e:
                raise GammaAPIError(
                    f"Failed to fetch events: HTTP {e.response.status_code}",
                    status_code=e.response.status_code,
                    response_body=e.response.text,
                ) from e
            except httpx.RequestError as e:
                raise GammaAPIError(f"Failed to fetch events: Request error - {e}") from e
            except json.JSONDecodeError as e:
                raise GammaAPIError(f"Failed to parse events response: {e}") from e

            event_list = data if isinstance(data, list) else data.get("data", [])
            if not event_list:
                break

            total_events += len(event_list)
            for event in event_list:
                title = (event.get("title") or "").lower()
                if not any(kw in title for kw in crypto_keywords):
                    continue
                crypto_events += 1
                for m in event.get("markets") or []:
                    end_date = m.get("endDate")
                    if end_date and end_date < now_str:
                        continue
                    market_id = str(m.get("conditionId", m.get("id", "")))
                    if not market_id or market_id in seen_ids:
                        continue
                    seen_ids.add(market_id)
                    try:
                        token_ids = self._parse_token_ids(m)
                        all_markets.append(
                            Market(
                                id=market_id,
                                title=m.get("question", "").strip(),
                                description=m.get("description", "").strip(),
                                url=f"https://polymarket.com/event/{m.get('slug', '')}",
                                end_date=end_date,
                                volume=safe_float(m.get("volume", m.get("volumeNum", 0))),
                                liquidity=safe_float(m.get("liquidity", m.get("liquidityNum", 0))),
                                price=self._parse_price(m),
                                market_odds=self._parse_market_odds(m),
                                token_ids=token_ids,
                            )
                        )
                    except (KeyError, ValueError, TypeError) as e:
                        logger.exception(f"Failed to parse market {market_id}: {e}")
                        raise

            if len(event_list) < page_size:
                break
            offset += page_size

        logger.info(
            f"Events: fetched={total_events}, crypto_events={crypto_events}, markets_extracted={len(all_markets)}"
        )
        return all_markets

    async def get_market(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single market by ID from Gamma API.

        Args:
            market_id: The market ID (conditionId) to fetch

        Returns:
            Market data as dictionary, or None if not found
        """
        try:
            response = await self._request(
                f"{self.GAMMA_URL}/markets", params={"id": market_id}
            )
            data = response.json()

            # Handle both list and dict responses
            if isinstance(data, list):
                if len(data) > 0:
                    return data[0]
                else:
                    return None
            else:
                # Single market returned as dict
                return data

        except Exception as e:
            logger.exception(f"Failed to fetch market {market_id}: {e}")
            return None

    async def get_resolved_markets(
        self, limit: int = 50, resolved_after: str = "2025-01-01T00:00:00Z"
    ) -> list[Market]:
        """Fetch resolved markets for backtesting with pagination.

        Args:
            limit: Maximum number of markets to fetch
            resolved_after: Only fetch markets ending after this date

        Returns:
            List of resolved Market objects
        """
        all_markets = []
        offset = 0
        page_size = min(500, limit)  # API returns max 500 per request
        seen_ids = set()

        while len(all_markets) < limit:
            params: dict[str, Any] = {
                "limit": min(page_size, limit - len(all_markets)),
                "offset": offset,
                "active": False,
                "closed": True,
                "archived": False,
            }
            if resolved_after:
                params["endDateAfter"] = resolved_after

            response = await self._request(f"{self.GAMMA_URL}/markets", params)
            data = response.json()

            market_list = data if isinstance(data, list) else data.get("data", [])

            if not market_list:
                break  # No more markets

            for m in market_list:
                try:
                    market_id = str(m.get("conditionId", m.get("id", "")))
                    if not market_id or market_id in seen_ids:
                        continue
                    seen_ids.add(market_id)

                    # Parse token IDs for CLOB API
                    token_ids = self._parse_token_ids(m)

                    all_markets.append(
                        Market(
                            id=market_id,
                            title=m.get("question", "").strip(),
                            description=m.get("description", "").strip(),
                            url=f"https://polymarket.com/event/{m.get('slug', '')}",
                            end_date=m.get("endDate"),
                            volume=safe_float(m.get("volume", m.get("volumeNum", 0))),
                            liquidity=safe_float(
                                m.get("liquidity", m.get("liquidityNum", 0))
                            ),
                            price=self._parse_price(m),
                            resolved_outcome=self._extract_resolved_outcome(m),
                            market_odds=self._parse_market_odds(m),
                            token_ids=token_ids,
                        )
                    )
                except (KeyError, ValueError, TypeError) as e:
                    logger.exception(
                        f"Failed to parse resolved market {m.get('conditionId', m.get('id', 'unknown'))}: {e}"
                    )
                    raise

            # Check if we got fewer than requested - end of data
            if len(market_list) < page_size:
                break

            offset += page_size

        return all_markets

    async def get_diverse_markets(self, count: int = 20) -> list[Market]:
        """Fetch diverse set of markets."""
        all_markets = await self.get_markets(limit=count * 2, future_only=True)
        seen: set[str] = set()
        unique: list[Market] = []

        for m in all_markets:
            if m.id not in seen:
                seen.add(m.id)
                unique.append(m)
                if len(unique) >= count:
                    break

        return unique

    async def get_price_history(
        self, market_id: str, interval: str = "1d", fidelity: int | None = None
    ) -> list[dict]:
        """Fetch historical price data for a market.

        Args:
            market_id: The condition_id of the market
            interval: Time interval (1m, 1w, 1d, 6h, 1h, max)
            fidelity: Resolution in minutes (alternative to interval)

        Returns:
            List of {t: timestamp, p: price} dicts
        """
        url = f"{self.CLOB_URL}/prices-history"
        params: dict = {"market": market_id}

        if fidelity:
            params["fidelity"] = fidelity
        else:
            params["interval"] = interval

        response = await self._request(url, params)
        return response.json().get("history", [])

    async def get_market_trades(
        self,
        market: Market | str,
        before: int | None = None,
        after: int | None = None,
    ) -> list[dict]:
        """Fetch trades for a market.

        Args:
            market: Market object or token_id string (YES token ID for CLOB API)
            before: Unix timestamp for end time
            after: Unix timestamp for start time

        Returns:
            List of trade records
        """
        # Extract token_id from Market object or use provided string
        if isinstance(market, Market):
            if not market.token_ids or len(market.token_ids) == 0:
                print(f"Warning: No token_ids available for market {market.id}")
                return []
            token_id = market.token_ids[0]  # YES token
        else:
            # Assume it's already a token_id string
            token_id = market

        # Note: py_clob_client doesn't support async, so we run in thread pool
        params_kwargs: dict = {"market": token_id}
        if before is not None:
            params_kwargs["before"] = before
        if after is not None:
            params_kwargs["after"] = after
        params = TradeParams(**params_kwargs)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.clob.get_trades(params=params)
        )
        return result.get("data", []) if isinstance(result, dict) else []

    async def get_order_book_spread(self, token_id: str) -> dict:
        """Fetch order book and calculate spread for a token.

        Args:
            token_id: The CLOB token ID (YES token for a market)

        Returns:
            Dictionary with spread info: {
                'best_bid': float,
                'best_ask': float,
                'spread': float,
                'spread_pct': float,
                'mid_price': float
            }
        """
        try:
            # Use CLOB API to get order book
            url = f"{self.CLOB_URL}/book"
            params = {"token_id": token_id}

            response = await self._request(url, params)
            data = response.json()

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids or not asks:
                return {
                    "best_bid": 0.0,
                    "best_ask": 1.0,
                    "spread": 1.0,
                    "spread_pct": 100.0,
                    "mid_price": 0.5,
                }

            # Get best bid (highest price someone will pay)
            best_bid = max(float(b["price"]) for b in bids) if bids else 0.0
            # Get best ask (lowest price someone will sell for)
            best_ask = min(float(a["price"]) for a in asks) if asks else 1.0

            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (spread / mid_price * 100) if mid_price > 0 else 100.0

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_pct": spread_pct,
                "mid_price": mid_price,
            }

        except Exception as e:
            logger.exception(
                f"Failed to get order book spread for {_redact_address(token_id)}: {e}"
            )
            raise

    async def get_current_prices(self, markets: list[Market]) -> dict[str, float]:
        """Get current prices for multiple markets in parallel.

        Args:
            markets: List of Market objects with token_ids

        Returns:
            Dict mapping market_id to YES price
        """
        # Note: py_clob_client doesn't support async, so we run in thread pool
        prices = {}
        loop = asyncio.get_event_loop()

        async def fetch_price(market: Market) -> tuple[str, float] | None:
            """Fetch price for a single market."""
            if not market.token_ids or len(market.token_ids) == 0:
                print(f"Warning: No token_ids for market {market.id}")
                return None

            token_id = market.token_ids[0]  # YES token

            try:
                midpoint = await loop.run_in_executor(
                    None, lambda: self.clob.get_midpoint(token_id)
                )
                if midpoint:
                    return (market.id, float(midpoint))
            except Exception as e:
                logger.exception(f"Failed to get price for market {market.id}: {e}")
                raise

        # Run all fetches in parallel using asyncio.gather
        tasks = [fetch_price(m) for m in markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        for result in results:
            if isinstance(result, Exception):
                continue
            if result is not None:
                market_id, price = result
                prices[market_id] = price

        return prices

    async def get_order_book(self, market: Market | str) -> dict:
        """Fetch order book for a market to get real bid/ask spreads.

        Uses direct HTTP request to CLOB API with controlled retry logic.
        Avoids py_clob_client's aggressive retry behavior on 404 errors.

        Args:
            market: Market object or token_id string (YES token ID for CLOB API)

        Returns:
            Dict with 'bids' and 'asks' lists, or empty dict on error
            Each bid/ask is {'price': float, 'size': float}
        """
        # Extract token_id from Market object or use provided string
        if isinstance(market, Market):
            if not market.token_ids or len(market.token_ids) == 0:
                logger.debug(f"No token_ids available for market {market.id}")
                return {"bids": [], "asks": []}
            token_id = market.token_ids[0]  # YES token
            market_id = market.id
        else:
            # Assume it's already a token_id string
            token_id = market
            market_id = market

        # Use direct HTTP request with controlled retry logic
        url = f"{self.CLOB_URL}/book"
        params = {"token_id": token_id}

        max_retries = 2
        base_delay = 0.5

        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                response = await self.client.get(url, params=params, timeout=5.0)

                # Handle 404 - market/orderbook doesn't exist (don't retry)
                if response.status_code == 404:
                    logger.debug(f"Order book not found for {market_id}: 404")
                    return {"bids": [], "asks": []}

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.debug(f"Rate limited, waiting {delay}s before retry")
                        await asyncio.sleep(delay)
                        continue
                    return {"bids": [], "asks": []}

                response.raise_for_status()
                data = response.json()

                # Parse bids and asks from response
                bids = []
                raw_bids = data.get("bids", [])
                if isinstance(raw_bids, list):
                    for bid in raw_bids:
                        if isinstance(bid, list) and len(bid) >= 2:
                            # Format: [price, size]
                            bids.append(
                                {
                                    "price": float(bid[0]),
                                    "size": float(bid[1]),
                                }
                            )
                        elif isinstance(bid, dict):
                            bids.append(
                                {
                                    "price": float(bid.get("price", 0)),
                                    "size": float(bid.get("size", 0)),
                                }
                            )

                asks = []
                raw_asks = data.get("asks", [])
                if isinstance(raw_asks, list):
                    for ask in raw_asks:
                        if isinstance(ask, list) and len(ask) >= 2:
                            asks.append(
                                {
                                    "price": float(ask[0]),
                                    "size": float(ask[1]),
                                }
                            )
                        elif isinstance(ask, dict):
                            asks.append(
                                {
                                    "price": float(ask.get("price", 0)),
                                    "size": float(ask.get("size", 0)),
                                }
                            )

                return {"bids": bids, "asks": asks}

            except TimeoutError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.debug(f"Timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.debug(f"Timeout getting order book for {market_id}")
                    return {"bids": [], "asks": []}

            except Exception as e:
                error_str = str(e)
                # Don't retry on 404 or "No orderbook exists"
                if "404" in error_str or "No orderbook exists" in error_str:
                    return {"bids": [], "asks": []}

                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.debug(f"Error getting order book, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.debug(f"Failed to get order book for {market_id}: {e}")
                    return {"bids": [], "asks": []}

        return {"bids": [], "asks": []}

    async def get_best_bid_ask(self, market: Market | str) -> dict:
        """Get best bid and ask prices for a market.

        Args:
            market: Market object or token_id string (YES token ID for CLOB API)

        Returns:
            Dict with 'best_bid', 'best_ask', 'spread', 'mid_price'
            Returns zeros if order book unavailable
        """
        book = await self.get_order_book(market)

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            return {
                "best_bid": 0.0,
                "best_ask": 0.0,
                "spread": 0.0,
                "mid_price": 0.0,
                "available": False,
            }

        # Best bid is highest price someone will pay for YES
        best_bid = max(b["price"] for b in bids) if bids else 0.0

        # Best ask is lowest price someone will sell YES for
        best_ask = min(a["price"] for a in asks) if asks else 1.0

        spread = best_ask - best_bid if best_ask > best_bid else 0.0
        mid_price = (best_bid + best_ask) / 2

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": (spread / mid_price * 100) if mid_price > 0 else 0.0,
            "mid_price": mid_price,
            "available": True,
        }

    async def get_order_book_depth(
        self, market: Market | str, max_depth: int = 10
    ) -> dict:
        """Get order book depth for slippage calculation.

        For NO positions, we need YES bid depth (since NO ask = 1 - YES bid).
        For YES positions, we need YES ask depth.

        Args:
            market: Market object or token_id string (YES token ID for CLOB API)
            max_depth: Number of price levels to fetch

        Returns:
            Dict with:
                - 'yes_bids': List of (price, size) for YES bids, sorted by price desc
                - 'yes_asks': List of (price, size) for YES asks, sorted by price asc
                - 'total_yes_bid_liquidity': Sum of YES bid sizes
                - 'total_yes_ask_liquidity': Sum of YES ask sizes
                - 'available': True if order book data available
        """
        book = await self.get_order_book(market)

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            return {
                "yes_bids": [],
                "yes_asks": [],
                "total_yes_bid_liquidity": 0.0,
                "total_yes_ask_liquidity": 0.0,
                "available": False,
            }

        # Sort bids by price descending (best bid first)
        yes_bids = sorted(
            [(b["price"], b["size"]) for b in bids if b["size"] > 0],
            key=lambda x: x[0],
            reverse=True,
        )[:max_depth]

        # Sort asks by price ascending (best ask first)
        yes_asks = sorted(
            [(a["price"], a["size"]) for a in asks if a["size"] > 0],
            key=lambda x: x[0],
        )[:max_depth]

        total_yes_bid_liquidity = sum(size for _, size in yes_bids)
        total_yes_ask_liquidity = sum(size for _, size in yes_asks)

        return {
            "yes_bids": yes_bids,
            "yes_asks": yes_asks,
            "total_yes_bid_liquidity": total_yes_bid_liquidity,
            "total_yes_ask_liquidity": total_yes_ask_liquidity,
            "available": True,
        }

    # ===== AUTHENTICATED TRADING METHODS =====

    async def create_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
    ) -> dict:
        """Place a limit order on the CLOB.

        Args:
            token_id: CLOB token ID (YES or NO token)
            price: Limit price (0-1)
            size: Order size in dollars
            side: "BUY" or "SELL"

        Returns:
            Order response with order_id

        Raises:
            APIError: If trading not enabled or order fails
        """
        if not (0.01 <= price <= 0.99):
            raise APIError(f"Invalid price: {price}. Must be between 0.01 and 0.99")
        if size <= 0:
            raise APIError(f"Invalid size: {size}. Must be > 0")
        if side.upper() not in ("BUY", "SELL"):
            raise APIError(f"Invalid side: {side}. Must be BUY or SELL")

        if settings.dry_run:
            logger.info(
                f"Dry run order: token={_redact_address(token_id)}, side={side.upper()}, "
                f"price={price:.3f}, size=${size:.2f}"
            )
            return {
                "dry_run": True,
                "orderID": "dry-run",
                "token_id": token_id,
                "price": price,
                "size": size,
                "side": side.upper(),
            }

        if not self.trading_enabled or not self.clob:
            raise APIError("Trading not enabled - no credentials configured")

        try:
            from py_clob_client.clob_types import OrderArgs

            loop = asyncio.get_event_loop()

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.upper(),
            )

            response = await loop.run_in_executor(
                None, lambda: self.clob.create_and_post_order(order_args)
            )

            order_id = response.get("orderID", "N/A")
            logger.info(
                f"Order placed: token={_redact_address(token_id)}, side={side}, "
                f"price={price:.3f}, size=${size:.2f}, order_id={order_id[:16] if order_id != 'N/A' else 'N/A'}..."
            )
            return response

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise APIError(f"Order creation failed: {e}") from e

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation succeeded

        Raises:
            APIError: If trading not enabled or cancellation fails
        """
        if settings.dry_run:
            logger.info(f"Dry run cancel order: {order_id[:16]}...")
            return True

        if not self.trading_enabled or not self.clob:
            raise APIError("Trading not enabled - no credentials configured")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.clob.cancel(order_id)
            )
            success = result.get("canceled") == [order_id]
            if success:
                logger.debug(f"Cancelled order: {order_id[:16]}...")
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id[:16]}...: {e}")
            return False

    async def get_open_orders(self, market_id: str | None = None) -> list[dict]:
        """Get open orders.

        Args:
            market_id: Optional filter by market ID

        Returns:
            List of open orders

        Raises:
            APIError: If trading not enabled
        """
        if not self.trading_enabled or not self.clob:
            raise APIError("Trading not enabled - no credentials configured")

        try:
            loop = asyncio.get_event_loop()
            orders = await loop.run_in_executor(None, lambda: self.clob.get_orders())
            return orders if isinstance(orders, list) else orders.get("data", [])
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise APIError(f"Failed to get orders: {e}") from e

    async def get_order_status(self, order_id: str) -> dict | None:
        """Get status of a specific order.

        Args:
            order_id: Order ID to check

        Returns:
            Order details including fill status, or None if not found

        Raises:
            APIError: If trading not enabled or query fails
        """
        if not self.trading_enabled or not self.clob:
            raise APIError("Trading not enabled - no credentials configured")

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.clob.get_order(order_id)
            )
            return response if isinstance(response, dict) else None

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id[:16]}...: {e}")
            raise APIError(f"Failed to get order status: {e}") from e

    # ===== WALLET INSPECTION METHODS =====

    async def get_balance(self) -> float:
        """Get USDC balance from blockchain.

        Returns:
            USDC balance

        Raises:
            AuthenticationError: If trading not enabled or Web3 not initialized
            APIError: If balance check fails
        """
        if settings.dry_run:
            return 1000.0

        if not self.trading_enabled or not self.signer_address or not self.web3:
            raise AuthenticationError("Trading not enabled - no credentials configured")

        try:
            contract = self.web3.eth.contract(address=USDC_CONTRACT, abi=ERC20_ABI)
            loop = asyncio.get_event_loop()

            balance_raw = await loop.run_in_executor(
                None, lambda: contract.functions.balanceOf(self.signer_address).call()
            )
            decimals = await loop.run_in_executor(
                None, lambda: contract.functions.decimals().call()
            )

            balance = float(balance_raw) / (10**decimals)
            logger.info(
                f"USDC balance for {_redact_address(self.signer_address)}: ${balance:.2f}"
            )
            return balance

        except Exception as e:
            logger.error(f"Failed to fetch USDC balance: {e}")
            raise APIError(f"Balance check failed: {e}") from e

    async def get_matic_balance(self) -> float:
        """Get MATIC balance for gas fees.

        Returns:
            MATIC balance

        Raises:
            AuthenticationError: If web3 not initialized
            APIError: If balance check fails
        """
        if settings.dry_run:
            return 10.0

        if not self.web3 or not self.signer_address:
            raise AuthenticationError(
                "Web3 not initialized - cannot check MATIC balance"
            )

        try:
            loop = asyncio.get_event_loop()
            balance_wei = await loop.run_in_executor(
                None, lambda: self.web3.eth.get_balance(self.signer_address)
            )
            balance_matic = float(self.web3.from_wei(balance_wei, "ether"))
            logger.debug(f"MATIC balance: {balance_matic:.4f}")
            return balance_matic
        except Exception as e:
            logger.error(f"Failed to fetch MATIC balance: {e}")
            raise APIError(f"MATIC balance check failed: {e}") from e

    async def get_usdc_allowance(self, spender: str | None = None) -> float | dict:
        """Get USDC allowance for CTF exchange contracts.

        Args:
            spender: Specific spender address, or check all if None

        Returns:
            Allowance amount (single float or dict of spender -> amount)

        Raises:
            AuthenticationError: If web3 not initialized
        """
        from ..settings import settings

        if settings.dry_run:
            return 1_000_000.0

        if not self.web3 or not self.address:
            return 0.0 if spender else {}

        erc20_allowance_abi = [
            {
                "constant": True,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"},
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function",
            },
        ]

        try:
            contract = self.web3.eth.contract(
                address=USDC_CONTRACT, abi=erc20_allowance_abi
            )
            loop = asyncio.get_event_loop()

            decimals = await loop.run_in_executor(
                None, lambda: contract.functions.decimals().call()
            )

            spenders = (
                [spender]
                if spender
                else [CTF_EXCHANGE, NEG_RISK_CTF_EXCHANGE, NEG_RISK_ADAPTER]
            )
            allowances = {}

            for sp in spenders:
                allowance_raw = await loop.run_in_executor(
                    None,
                    lambda s=sp: contract.functions.allowance(self.address, s).call(),
                )
                allowances[sp] = float(allowance_raw) / (10**decimals)

            return allowances if not spender else allowances[spender]
        except Exception as e:
            logger.error(f"Failed to fetch USDC allowance: {e}")
            return 0.0 if spender else {}

    async def verify_position_on_chain(self, token_id: str) -> float:
        """Verify position balance directly from blockchain (source of truth).

        This bypasses all API caching and reads the ERC1155 balance directly
        from the Conditional Token Framework contract.

        Args:
            token_id: The ERC1155 token ID (hex string)

        Returns:
            Balance in token units (6 decimals = USDC equivalent)

        Raises:
            AuthenticationError: If web3 not initialized
            APIError: If verification fails
        """
        if not self.web3 or not self.address:
            raise AuthenticationError("Web3 not initialized")

        erc1155_abi = [
            {
                "inputs": [
                    {"name": "account", "type": "address"},
                    {"name": "id", "type": "uint256"},
                ],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function",
            }
        ]

        try:
            contract = self.web3.eth.contract(address=CTF_CONTRACT, abi=erc1155_abi)
            loop = asyncio.get_event_loop()

            # token_id can be hex string or int
            if isinstance(token_id, str) and token_id.startswith("0x"):
                token_id_int = int(token_id, 16)
            else:
                token_id_int = int(token_id)

            balance_raw = await loop.run_in_executor(
                None,
                lambda: contract.functions.balanceOf(self.address, token_id_int).call(),
            )

            # Polymarket uses 6 decimals for outcome tokens
            balance = float(balance_raw) / (10**6)
            logger.debug(f"On-chain balance for {token_id[:16]}...: {balance:.6f}")
            return balance

        except Exception as e:
            logger.error(f"Failed to verify position on-chain: {e}")
            raise APIError(f"On-chain verification failed: {e}") from e

    async def get_positions(self, use_cache_buster: bool = True) -> list[dict]:
        """Fetch positions from Polymarket data API.

        WARNING: data-api.polymarket.com is an indexed cache that can lag
        behind real-time data by 30+ seconds. For critical trading decisions,
        verify with verify_position_on_chain().

        Args:
            use_cache_buster: If True, adds headers to prevent CDN caching

        Returns:
            List of position dicts

        Raises:
            AuthenticationError: If no address configured
        """
        if not self.address:
            raise AuthenticationError("No address configured - cannot fetch positions")

        import time

        headers = {}
        if use_cache_buster:
            headers = {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            }
            timestamp = int(time.time() * 1000)
            url = f"https://data-api.polymarket.com/positions?user={self.address}&_t={timestamp}"
        else:
            url = f"https://data-api.polymarket.com/positions?user={self.address}"

        await self.circuit_breaker.acquire()
        await self.rate_limiter.acquire()
        response = await self.client.get(url, headers=headers)
        response.raise_for_status()
        self.circuit_breaker.record_success()
        return response.json()

    # ===== POSITION MANAGEMENT =====

    async def cancel_all_orders(self) -> dict:
        """Cancel all open orders.

        Returns:
            Cancellation result dict

        Raises:
            APIError: If trading not enabled or cancellation fails
        """
        from ..settings import settings

        if settings.dry_run:
            return {"canceled": [], "not_canceled": {}, "dry_run": True}

        if not self.clob:
            raise APIError("CLOB client not initialized")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.clob.cancel_all())
            canceled = result.get("canceled", [])
            logger.info(f"Cancelled {len(canceled)} orders")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            raise APIError(f"Cancel all orders failed: {e}") from e
