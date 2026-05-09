from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    provider: str
    dspy_name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Keys (auto-loaded from env by BaseSettings)
    google_api_key: str | None = None
    openrouter_api_key: str | None = None

    # Polymarket Credentials (auto-loaded from env)
    polymarket_private_key: str | None = None
    polymarket_api_key: str | None = None
    polymarket_api_secret: str | None = None
    polymarket_api_passphrase: str | None = None
    polymarket_proxy_address: str | None = None

    # Polymarket Settings
    polymarket_chain_id: int = 137
    polymarket_host: str = "https://clob.polymarket.com"
    dry_run: bool = True

    # Builder API (for gasless transactions)
    builder_api_key: str | None = None
    builder_secret: str | None = None
    builder_passphrase: str | None = None
    relayer_url: str = "https://relayer-v2.polymarket.com"
    alchemy_api_key: str | None = None

    # LLM Settings
    llm_model: str = "gemini/gemini-2.5-flash"

    models: dict[str, ModelConfig] = {
        "gemini-pro": ModelConfig(
            provider="google",
            dspy_name="gemini/gemini-2.5-pro",
            kwargs={
                "temperature": 0.0,
                "max_tokens": 8192,
                "num_retries": 5,
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ],
            },
        ),
        "gemini-flash": ModelConfig(
            provider="google",
            dspy_name="gemini/gemini-2.5-flash",
            kwargs={
                "temperature": 0.3,
                "max_tokens": 16384,
                "num_retries": 5,
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ],
            },
        ),
        "openrouter-free": ModelConfig(
            provider="openrouter",
            dspy_name="openrouter/openrouter/free",
            kwargs={"temperature": 0.7, "max_tokens": 4096, "num_retries": 5},
        ),
        "pony-alpha": ModelConfig(
            provider="openrouter",
            dspy_name="openrouter/openrouter/pony-alpha",
            kwargs={"temperature": 0.7, "max_tokens": 4096, "num_retries": 5},
        ),
        "gemini-flash-lite": ModelConfig(
            provider="google",
            dspy_name="gemini/gemini-1.5-flash-lite",
            kwargs={
                "temperature": 0.1,
                "max_tokens": 1024,
                "num_retries": 3,
            },
        ),
    }

    ANALYST_MODELS: list[str] = ["pony-alpha"]
    JUDGE_MODEL: str = "gemini-pro"

    MODEL_SHORT_NAMES: dict[str, str] = {
        "gemini-flash": "flash",
        "gemini-flash-lite": "flash-lite",
        "kimi-free": "kimi",
        "openrouter-free": "router",
        "pony-alpha": "pony",
    }

    @property
    def optimizer_judge_model(self) -> str:
        return self.JUDGE_MODEL

    @property
    def analysis_models(self) -> list[str]:
        return self.ANALYST_MODELS

    log_level: str = "INFO"
    cache_dir: str = ".dspy_cache"
    optimizer_path: str = "models/optimized_assessor.pkl"

    MIN_EDGE_PERCENT: float = 0.30
    MIN_YES_PRICE: float = 0.15
    DEFAULT_POSITION_SIZE: float = 1000.0
    DEFAULT_SPREAD: float = 0.06
    DEFAULT_FEE_RATE: float = 0.002
    MIN_PROFIT_AFTER_COSTS_PCT: float = 1.0

    MIN_APY: float = 0.08
    MAX_DAYS_TO_RESOLUTION: int = 365
    MIN_PROFIT_PER_DAY: float = 0.10

    MIN_PROFIT_MARGIN_PCT: float = 0.30

    MAX_SPREAD: float = 0.30
    MAX_SPREAD_HIGH_LIQUIDITY: float = 0.50
    HIGH_LIQUIDITY_THRESHOLD: float = 50000

    MIN_VOLUME_EFFICIENT: float = 50000
    MIN_VOLUME_ALPHA: float = 5000
    MIN_VOLUME_EXPLORATORY: float = 500

    MIN_LIQUIDITY: float = 100000
    MIN_LIQUIDITY_ALPHA: float = 10000

    MAX_POSITION_PCT_OF_BANKROLL: float = 0.05
    MAX_PORTFOLIO_EXPOSURE_PCT: float = 0.20
    MAX_DRAWDOWN_PCT: float = 0.20


settings = Settings()
