from dataclasses import dataclass
from datetime import UTC, datetime

from ..models import Market
from .bs_digital import VolSurface, digital_call_price, digital_put_price, one_touch_down, one_touch_up, range_binary_price
from .market_matcher import BarrierMarketMatch, CryptoMarketMatch, RangeMarketMatch


@dataclass(frozen=True)
class DeribitScanResult:
    market_id: str
    question: str
    polymarket_yes_price: float
    polymarket_url: str | None

    asset: str
    strike: float
    direction: str
    resolution_date: datetime
    match_confidence: float
    match_method: str

    spot_price: float
    interpolated_iv: float
    time_to_expiry_years: float

    implied_probability: float
    risk_free_rate: float

    edge: float
    edge_percent: float
    edge_direction: str
    action: str

    no_price: float
    days_to_expiry: int
    monthly_return: float
    annualized_yield: float
    extrapolated: bool

    vol_risk_premium_note: str
    moneyness: float
    liquidity: float
    volume: float


@dataclass(frozen=True)
class RangeScanResult:
    market_id: str
    question: str
    polymarket_yes_price: float
    polymarket_url: str | None

    asset: str
    lower_strike: float
    upper_strike: float
    resolution_date: datetime
    match_confidence: float
    match_method: str

    spot_price: float
    iv_lower: float
    iv_upper: float
    time_to_expiry_years: float

    implied_probability: float
    risk_free_rate: float

    edge: float
    edge_percent: float
    edge_direction: str
    action: str

    no_price: float
    days_to_expiry: int
    monthly_return: float
    annualized_yield: float
    extrapolated: bool
    liquidity: float
    volume: float


@dataclass(frozen=True)
class BarrierScanResult:
    """Edge result for a one-touch/barrier market (path-dependent)."""
    market_id: str
    question: str
    polymarket_yes_price: float
    polymarket_url: str | None

    asset: str
    barrier: float
    direction: str  # "up" or "down"
    resolution_date: datetime
    match_confidence: float
    match_method: str

    spot_price: float
    interpolated_iv: float
    time_to_expiry_years: float

    implied_probability: float
    risk_free_rate: float

    edge: float
    edge_percent: float
    edge_direction: str
    action: str

    no_price: float
    days_to_expiry: int
    monthly_return: float
    annualized_yield: float
    extrapolated: bool
    moneyness: float
    liquidity: float
    volume: float


class DeribitEdgeCalculator:
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def compute(
        self,
        match: CryptoMarketMatch,
        market: Market,
        vol_surface: VolSurface,
    ) -> DeribitScanResult | None:
        now = datetime.now(UTC)
        T = (match.resolution_date - now).total_seconds() / (365.25 * 86400)
        if T <= 0:
            return None

        iv = vol_surface.interpolate_vol(match.strike, T)
        if iv is None:
            return None

        forward = vol_surface.get_forward(T)
        if match.direction == "above":
            implied_prob = digital_call_price(forward, match.strike, T, iv)
        else:
            implied_prob = digital_put_price(forward, match.strike, T, iv)

        pm_yes = market.price if market.price is not None else 0.5
        edge = pm_yes - implied_prob

        if edge > 0.001:
            edge_dir = "PM_OVERPRICED"
            action = "BUY_NO"
        elif edge < -0.001:
            edge_dir = "PM_UNDERPRICED"
            action = "BUY_YES"
        else:
            edge_dir = "FAIR"
            action = "HOLD"

        no_price = 1.0 - pm_yes
        days_to_expiry = max(1, int(T * 365.25))
        roi = abs(edge) / no_price if no_price > 0 and edge > 0 else abs(edge) / pm_yes if pm_yes > 0 and edge < 0 else 0.0
        months = days_to_expiry / 30.44
        monthly_return = roi / months if months > 0 else 0.0
        annualized_yield = roi / T if T > 0 else 0.0
        extrapolated = vol_surface.is_extrapolated(match.strike, T)

        moneyness = vol_surface.spot / match.strike

        if match.direction == "above":
            vrp_note = (
                "N(d2) understates real-world upside probability due to vol risk premium. "
                "True edge may be smaller if PM is overpriced."
            )
        else:
            vrp_note = (
                "N(d2) overstates real-world downside probability due to vol risk premium. "
                "True edge may be smaller if PM is underpriced."
            )

        return DeribitScanResult(
            market_id=match.market_id,
            question=match.question,
            polymarket_yes_price=pm_yes,
            polymarket_url=market.url,
            asset=match.asset,
            strike=match.strike,
            direction=match.direction,
            resolution_date=match.resolution_date,
            match_confidence=match.confidence,
            match_method=match.match_method,
            spot_price=vol_surface.spot,
            interpolated_iv=iv,
            time_to_expiry_years=T,
            implied_probability=implied_prob,
            risk_free_rate=self.risk_free_rate,
            edge=edge,
            edge_percent=edge * 100,
            edge_direction=edge_dir,
            action=action,
            no_price=no_price,
            days_to_expiry=days_to_expiry,
            monthly_return=monthly_return,
            annualized_yield=annualized_yield,
            extrapolated=extrapolated,
            vol_risk_premium_note=vrp_note,
            moneyness=moneyness,
            liquidity=market.liquidity or 0.0,
            volume=market.volume or 0.0,
        )

    def compute_barrier(
        self,
        match: BarrierMarketMatch,
        market: Market,
        vol_surface: VolSurface,
    ) -> BarrierScanResult | None:
        """Price a one-touch barrier option using the Reiner-Rubinstein formula."""
        now = datetime.now(UTC)
        T = (match.resolution_date - now).total_seconds() / (365.25 * 86400)
        if T <= 0:
            return None

        iv = vol_surface.interpolate_vol(match.barrier, T)
        if iv is None:
            return None

        forward = vol_surface.get_forward(T)
        if match.direction == "up":
            implied_prob = one_touch_up(forward, match.barrier, T, iv, self.risk_free_rate)
        else:
            implied_prob = one_touch_down(forward, match.barrier, T, iv, self.risk_free_rate)

        pm_yes = market.price if market.price is not None else 0.5
        edge = pm_yes - implied_prob

        if edge > 0.001:
            edge_dir = "PM_OVERPRICED"
            action = "BUY_NO"
        elif edge < -0.001:
            edge_dir = "PM_UNDERPRICED"
            action = "BUY_YES"
        else:
            edge_dir = "FAIR"
            action = "HOLD"

        no_price = 1.0 - pm_yes
        days_to_expiry = max(1, int(T * 365.25))
        roi = abs(edge) / no_price if no_price > 0 and edge > 0 else abs(edge) / pm_yes if pm_yes > 0 and edge < 0 else 0.0
        months = days_to_expiry / 30.44
        monthly_return = roi / months if months > 0 else 0.0
        annualized_yield = roi / T if T > 0 else 0.0
        extrapolated = vol_surface.is_extrapolated(match.barrier, T)
        moneyness = vol_surface.spot / match.barrier

        return BarrierScanResult(
            market_id=match.market_id,
            question=match.question,
            polymarket_yes_price=pm_yes,
            polymarket_url=market.url,
            asset=match.asset,
            barrier=match.barrier,
            direction=match.direction,
            resolution_date=match.resolution_date,
            match_confidence=match.confidence,
            match_method=match.match_method,
            spot_price=vol_surface.spot,
            interpolated_iv=iv,
            time_to_expiry_years=T,
            implied_probability=implied_prob,
            risk_free_rate=self.risk_free_rate,
            edge=edge,
            edge_percent=edge * 100,
            edge_direction=edge_dir,
            action=action,
            no_price=no_price,
            days_to_expiry=days_to_expiry,
            monthly_return=monthly_return,
            annualized_yield=annualized_yield,
            extrapolated=extrapolated,
            moneyness=moneyness,
            liquidity=market.liquidity or 0.0,
            volume=market.volume or 0.0,
        )

    def compute_barrier_batch(
        self,
        matches: list[tuple[BarrierMarketMatch, Market]],
        vol_surfaces: dict[str, VolSurface],
    ) -> list[BarrierScanResult]:
        results = []
        for match, market in matches:
            surface = vol_surfaces.get(match.asset)
            if surface is None:
                continue
            result = self.compute_barrier(match, market, surface)
            if result is not None:
                results.append(result)
        return sorted(results, key=lambda r: abs(r.edge), reverse=True)

    def compute_range(
        self,
        match: RangeMarketMatch,
        market: Market,
        vol_surface: VolSurface,
    ) -> RangeScanResult | None:
        now = datetime.now(UTC)
        T = (match.resolution_date - now).total_seconds() / (365.25 * 86400)
        if T <= 0:
            return None

        iv_lower = vol_surface.interpolate_vol(match.lower_strike, T)
        iv_upper = vol_surface.interpolate_vol(match.upper_strike, T)
        if iv_lower is None or iv_upper is None:
            return None

        forward = vol_surface.get_forward(T)
        implied_prob = range_binary_price(forward, match.lower_strike, match.upper_strike, T, iv_lower, iv_upper)

        pm_yes = market.price if market.price is not None else 0.5
        edge = pm_yes - implied_prob

        if edge > 0.001:
            edge_dir = "PM_OVERPRICED"
            action = "BUY_NO"
        elif edge < -0.001:
            edge_dir = "PM_UNDERPRICED"
            action = "BUY_YES"
        else:
            edge_dir = "FAIR"
            action = "HOLD"

        no_price = 1.0 - pm_yes
        days_to_expiry = max(1, int(T * 365.25))
        roi = abs(edge) / no_price if no_price > 0 and edge > 0 else abs(edge) / pm_yes if pm_yes > 0 and edge < 0 else 0.0
        months = days_to_expiry / 30.44
        monthly_return = roi / months if months > 0 else 0.0
        annualized_yield = roi / T if T > 0 else 0.0
        extrapolated = vol_surface.is_extrapolated(match.lower_strike, T) or vol_surface.is_extrapolated(match.upper_strike, T)

        return RangeScanResult(
            market_id=match.market_id,
            question=match.question,
            polymarket_yes_price=pm_yes,
            polymarket_url=market.url,
            asset=match.asset,
            lower_strike=match.lower_strike,
            upper_strike=match.upper_strike,
            resolution_date=match.resolution_date,
            match_confidence=match.confidence,
            match_method=match.match_method,
            spot_price=vol_surface.spot,
            iv_lower=iv_lower,
            iv_upper=iv_upper,
            time_to_expiry_years=T,
            implied_probability=implied_prob,
            risk_free_rate=self.risk_free_rate,
            edge=edge,
            edge_percent=edge * 100,
            edge_direction=edge_dir,
            action=action,
            no_price=no_price,
            days_to_expiry=days_to_expiry,
            monthly_return=monthly_return,
            annualized_yield=annualized_yield,
            extrapolated=extrapolated,
            liquidity=market.liquidity or 0.0,
            volume=market.volume or 0.0,
        )

    def compute_range_batch(
        self,
        matches: list[tuple[RangeMarketMatch, Market]],
        vol_surfaces: dict[str, VolSurface],
    ) -> list[RangeScanResult]:
        results = []
        for match, market in matches:
            surface = vol_surfaces.get(match.asset)
            if surface is None:
                continue
            result = self.compute_range(match, market, surface)
            if result is not None:
                results.append(result)
        return sorted(results, key=lambda r: abs(r.edge), reverse=True)

    def compute_batch(
        self,
        matches: list[tuple[CryptoMarketMatch, Market]],
        vol_surfaces: dict[str, VolSurface],
    ) -> list[DeribitScanResult]:
        results = []
        for match, market in matches:
            surface = vol_surfaces.get(match.asset)
            if surface is None:
                continue
            result = self.compute(match, market, surface)
            if result is not None:
                results.append(result)
        return sorted(results, key=lambda r: abs(r.edge), reverse=True)
