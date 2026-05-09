class PolymarketError(Exception):
    pass


class APIError(PolymarketError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GammaAPIError(APIError):
    pass


class ClobAPIError(APIError):
    pass


class DataValidationError(PolymarketError):
    def __init__(self, message: str, field: str | None = None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value


class MarketDataError(DataValidationError):
    pass


class TradingError(PolymarketError):
    pass


class AuthenticationError(PolymarketError):
    pass


class PositionError(TradingError):
    pass


class RiskLimitError(TradingError):
    def __init__(
        self, message: str, limit_type: str, current_value: float, limit_value: float
    ):
        super().__init__(message)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class ConfigurationError(PolymarketError):
    pass


class DatabaseError(PolymarketError):
    pass
