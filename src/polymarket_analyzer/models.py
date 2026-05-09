import logging
from dataclasses import dataclass
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

T = TypeVar("T")

logger = logging.getLogger(__name__)

_dict_adapter = TypeAdapter(dict)
_list_str_adapter = TypeAdapter(list[str])


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _safe_dict(value: dict | str | None, field_name: str = "unknown") -> dict | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return _dict_adapter.validate_json(value)
        except (ValidationError, Exception):
            return None
    return None


def _safe_token_ids(
    value: list | str | None, field_name: str = "clobTokenIds"
) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(t) for t in value]
    if isinstance(value, str):
        try:
            parsed = _list_str_adapter.validate_json(value)
            return [str(t) for t in parsed]
        except ValidationError as e:
            logger.exception(f"Failed to parse token IDs for field '{field_name}': {e}")
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error parsing token IDs for field '{field_name}': {e}"
            )
            raise
    logger.debug(f"Unexpected type for token_ids field '{field_name}': {type(value)}")
    return None


@dataclass(frozen=True)
class Market:
    id: str
    title: str
    description: str
    url: str | None = None
    end_date: str | None = None
    volume: float | None = None
    liquidity: float | None = None
    price: float | None = None
    resolved_outcome: str | None = None
    market_odds: dict[str, float] | None = None
    extra_fields: dict | None = None
    no_bias_score: float | None = None
    yes_overpricing_pct: float | None = None
    procedural_friction_score: float | None = None
    token_ids: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict):
        core_keys = {
            "id",
            "title",
            "description",
            "url",
            "endDate",
            "end_date",
            "volume",
            "liquidity",
            "price",
            "resolved_outcome",
            "market_odds",
            "no_bias_score",
            "yes_overpricing_pct",
            "procedural_friction_score",
            "token_ids",
            "clobTokenIds",
        }
        extra = {
            k: v
            for k, v in data.items()
            if k not in core_keys and v is not None and v != ""
        }

        token_ids = _safe_token_ids(
            data.get("clobTokenIds"), "clobTokenIds"
        ) or data.get("token_ids")

        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")).strip(),
            description=str(data.get("description", "")).strip(),
            url=data.get("url"),
            end_date=data.get("endDate") or data.get("end_date"),
            volume=_safe_float(data.get("volume")),
            liquidity=_safe_float(data.get("liquidity")),
            price=_safe_float(data.get("price")),
            resolved_outcome=data.get("resolved_outcome"),
            market_odds=_safe_dict(data.get("market_odds"), "market_odds"),
            extra_fields=extra if extra else None,
            no_bias_score=_safe_float(data.get("no_bias_score")),
            yes_overpricing_pct=_safe_float(data.get("yes_overpricing_pct")),
            procedural_friction_score=_safe_float(
                data.get("procedural_friction_score")
            ),
            token_ids=token_ids,
        )

    def to_dict(self):
        result = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "endDate": self.end_date,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "price": self.price,
            "resolved_outcome": self.resolved_outcome,
            "market_odds": self.market_odds,
        }
        if self.extra_fields:
            result.update(self.extra_fields)
        if self.no_bias_score is not None:
            result["no_bias_score"] = self.no_bias_score
        if self.yes_overpricing_pct is not None:
            result["yes_overpricing_pct"] = self.yes_overpricing_pct
        if self.procedural_friction_score is not None:
            result["procedural_friction_score"] = self.procedural_friction_score
        if self.token_ids is not None:
            result["token_ids"] = self.token_ids
        return result
