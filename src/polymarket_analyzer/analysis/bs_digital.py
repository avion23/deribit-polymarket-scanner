import math
from dataclasses import dataclass
from datetime import datetime


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def digital_call_price(
    forward: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
) -> float:
    """N(d₂) — risk-neutral probability that S_T > K."""
    if time_to_expiry <= 0:
        return 1.0 if forward > strike else 0.0
    if vol <= 0:
        return 1.0 if forward > strike else 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d2 = (math.log(forward / strike) - 0.5 * vol * vol * time_to_expiry) / (vol * sqrt_t)
    return normal_cdf(d2)


def digital_put_price(
    forward: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
) -> float:
    """1 - N(d₂) — risk-neutral probability that S_T < K."""
    return 1.0 - digital_call_price(forward, strike, time_to_expiry, vol)


def range_binary_price(
    forward: float,
    lower_strike: float,
    upper_strike: float,
    time_to_expiry: float,
    vol_lower: float,
    vol_upper: float,
) -> float:
    """N(d₂ at lower) - N(d₂ at upper) — risk-neutral probability that lower < S_T < upper."""
    return digital_call_price(forward, lower_strike, time_to_expiry, vol_lower) - digital_call_price(
        forward, upper_strike, time_to_expiry, vol_upper
    )


@dataclass(frozen=True)
class VolPoint:
    strike: float
    expiry: datetime
    time_to_expiry: float
    iv: float


class VolSurface:
    """Volatility surface for a single underlying, built from Deribit data."""

    def __init__(
        self,
        currency: str,
        spot: float,
        points: list[VolPoint],
        timestamp: datetime,
        forwards: dict[float, float] | None = None,
    ):
        self.currency = currency
        self.spot = spot
        self.timestamp = timestamp
        self._forwards: dict[float, float] = forwards or {}

        self._expiries: list[float] = sorted(set(p.time_to_expiry for p in points))
        self._by_expiry: dict[float, list[VolPoint]] = {}
        for p in points:
            self._by_expiry.setdefault(p.time_to_expiry, []).append(p)
        for t in self._by_expiry:
            self._by_expiry[t].sort(key=lambda p: p.strike)

    def get_forward(self, time_to_expiry: float) -> float:
        if not self._forwards:
            return self.spot
        sorted_T = sorted(self._forwards)
        if time_to_expiry <= sorted_T[0]:
            return self._forwards[sorted_T[0]]
        if time_to_expiry >= sorted_T[-1]:
            return self._forwards[sorted_T[-1]]
        for i in range(len(sorted_T) - 1):
            t1, t2 = sorted_T[i], sorted_T[i + 1]
            if t1 <= time_to_expiry <= t2:
                w = (time_to_expiry - t1) / (t2 - t1)
                return self._forwards[t1] * (1.0 - w) + self._forwards[t2] * w
        return self.spot

    def interpolate_vol(self, strike: float, time_to_expiry: float) -> float | None:
        if not self._expiries:
            return None

        t1_idx = _find_lower_bracket(self._expiries, time_to_expiry)
        if t1_idx is None:
            return None

        t1 = self._expiries[t1_idx]

        if t1_idx + 1 >= len(self._expiries):
            return self._interp_strike(self._by_expiry[t1], strike)

        t2 = self._expiries[t1_idx + 1]

        if abs(t1 - time_to_expiry) < 1e-9:
            return self._interp_strike(self._by_expiry[t1], strike)

        iv1 = self._interp_strike(self._by_expiry[t1], strike)
        iv2 = self._interp_strike(self._by_expiry[t2], strike)
        if iv1 is None or iv2 is None:
            return iv1 if iv2 is None else iv2

        sqrt_t = math.sqrt(time_to_expiry)
        sqrt_t1 = math.sqrt(t1)
        sqrt_t2 = math.sqrt(t2)
        denom = sqrt_t2 - sqrt_t1
        if abs(denom) < 1e-12:
            return iv1

        w = (sqrt_t - sqrt_t1) / denom
        return iv1 * (1.0 - w) + iv2 * w

    def is_extrapolated(self, strike: float, time_to_expiry: float) -> bool:
        if not self._expiries:
            return True

        t1_idx = _find_lower_bracket(self._expiries, time_to_expiry)
        if t1_idx is None:
            return True

        for t_idx in (t1_idx, min(t1_idx + 1, len(self._expiries) - 1)):
            points = self._by_expiry[self._expiries[t_idx]]
            if len(points) >= 2 and (strike < points[0].strike or strike > points[-1].strike):
                return True

        return False

    def _interp_strike(self, points: list[VolPoint], strike: float) -> float | None:
        if not points:
            return None
        if len(points) == 1:
            return points[0].iv

        if strike <= points[0].strike:
            return points[0].iv
        if strike >= points[-1].strike:
            return points[-1].iv

        for i in range(len(points) - 1):
            if points[i].strike <= strike <= points[i + 1].strike:
                k1, k2 = points[i].strike, points[i + 1].strike
                denom = k2 - k1
                if abs(denom) < 1e-12:
                    return points[i].iv
                w = (strike - k1) / denom
                return points[i].iv * (1.0 - w) + points[i + 1].iv * w

        return None


def one_touch_up(
    forward: float,
    barrier: float,
    time_to_expiry: float,
    vol: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Probability of price touching barrier from below at any point before expiry.

    GBM first-passage formula (continuous monitoring).  Matches the standard
    closed-form result: N(d1) + e^(2*mu*b/sigma^2)*N(d2) where b = ln(B/F)
    and mu = r - sigma^2/2 is the risk-neutral drift of ln(S).

    Args:
        forward: current forward price for the relevant expiry
        barrier: the barrier level (must be > forward for an up-touch)
        time_to_expiry: years to expiry
        vol: annualised implied volatility (e.g. 0.80 for 80%)
        risk_free_rate: continuously-compounded risk-free rate (used for drift)

    Returns:
        Risk-neutral probability of touching barrier before expiry, in [0, 1].
    """
    if time_to_expiry <= 0:
        return 1.0 if forward >= barrier else 0.0
    if vol <= 0:
        return 1.0 if forward >= barrier else 0.0
    if forward >= barrier:
        return 1.0

    sqrt_t = math.sqrt(time_to_expiry)
    var = vol * vol
    # drift of ln(S) under risk-neutral measure
    mu = risk_free_rate - 0.5 * var

    # First-passage probability formula for GBM hitting barrier B from below.
    # b = ln(B/S) > 0 (log-distance to barrier).
    # P = N((mu*T - b) / (sigma*sqrt(T))) + e^(2*mu*b/sigma^2) * N((-mu*T - b) / (sigma*sqrt(T)))
    b = math.log(barrier / forward)  # positive
    d1 = (mu * time_to_expiry - b) / (vol * sqrt_t)
    d2 = (-mu * time_to_expiry - b) / (vol * sqrt_t)
    reflection = math.exp(2.0 * mu * b / var)

    p = normal_cdf(d1) + reflection * normal_cdf(d2)
    return max(0.0, min(1.0, p))


def one_touch_down(
    forward: float,
    barrier: float,
    time_to_expiry: float,
    vol: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Probability of price touching barrier from above at any point before expiry.

    GBM first-passage formula (continuous monitoring) for the down-touch case.
    Symmetric to one_touch_up with reflected log-distance and inverted drift sign.

    Args:
        forward: current forward price for the relevant expiry
        barrier: the barrier level (must be < forward for a down-touch)
        time_to_expiry: years to expiry
        vol: annualised implied volatility
        risk_free_rate: continuously-compounded risk-free rate

    Returns:
        Risk-neutral probability of touching barrier before expiry, in [0, 1].
    """
    if time_to_expiry <= 0:
        return 1.0 if forward <= barrier else 0.0
    if vol <= 0:
        return 1.0 if forward <= barrier else 0.0
    if forward <= barrier:
        return 1.0

    sqrt_t = math.sqrt(time_to_expiry)
    var = vol * vol
    # drift of ln(S) under risk-neutral measure
    mu = risk_free_rate - 0.5 * var

    # First-passage probability formula for GBM hitting barrier B from above.
    # b = ln(S/B) > 0 (log-distance to barrier).
    # P = N((-mu*T - b) / (sigma*sqrt(T))) + e^(-2*mu*b/sigma^2) * N((mu*T - b) / (sigma*sqrt(T)))
    b = math.log(forward / barrier)  # positive
    d1 = (-mu * time_to_expiry - b) / (vol * sqrt_t)
    d2 = (mu * time_to_expiry - b) / (vol * sqrt_t)
    reflection = math.exp(-2.0 * mu * b / var)

    p = normal_cdf(d1) + reflection * normal_cdf(d2)
    return max(0.0, min(1.0, p))


def _find_lower_bracket(sorted_vals: list[float], target: float) -> int | None:
    if not sorted_vals:
        return None
    if target < sorted_vals[0]:
        return 0
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i] <= target <= sorted_vals[i + 1]:
            return i
    if target >= sorted_vals[-1]:
        return len(sorted_vals) - 1
    return None
