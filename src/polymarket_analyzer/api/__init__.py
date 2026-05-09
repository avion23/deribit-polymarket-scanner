try:
    from .polymarket_api import PolymarketAPI
except ModuleNotFoundError:
    PolymarketAPI = None

try:
    from .deribit_api import DeribitAPI
except ModuleNotFoundError:
    DeribitAPI = None

__all__ = ["PolymarketAPI", "DeribitAPI"]
