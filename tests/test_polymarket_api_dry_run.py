import pytest

from polymarket_analyzer.api import polymarket_api
from polymarket_analyzer.api.polymarket_api import PolymarketAPI
from polymarket_analyzer.settings import settings


class FakeClob:
    def create_and_post_order(self, order_args):
        raise AssertionError("dry-run order touched CLOB")

    def cancel(self, order_id):
        raise AssertionError("dry-run cancel touched CLOB")


@pytest.fixture(autouse=True)
def restore_dry_run():
    original = settings.dry_run
    settings.dry_run = True
    yield
    settings.dry_run = original


@pytest.mark.asyncio
async def test_create_limit_order_dry_run_does_not_post_order():
    api = object.__new__(PolymarketAPI)
    api.trading_enabled = True
    api.clob = FakeClob()

    result = await api.create_limit_order("0x1234567890abcdef", 0.42, 10, "buy")

    assert result == {
        "dry_run": True,
        "orderID": "dry-run",
        "token_id": "0x1234567890abcdef",
        "price": 0.42,
        "size": 10,
        "side": "BUY",
    }


@pytest.mark.asyncio
async def test_cancel_order_dry_run_does_not_cancel_order():
    api = object.__new__(PolymarketAPI)
    api.trading_enabled = True
    api.clob = FakeClob()

    assert await api.cancel_order("order-123") is True


@pytest.mark.asyncio
async def test_create_limit_order_validates_inputs_before_dry_run():
    api = object.__new__(PolymarketAPI)
    api.trading_enabled = True
    api.clob = FakeClob()

    with pytest.raises(Exception, match="Invalid price"):
        await api.create_limit_order("0x1234567890abcdef", 1.5, 10, "BUY")


def test_init_dry_run_does_not_authenticate_clob(monkeypatch):
    class ReadOnlyClob:
        def __init__(self, host, **kwargs):
            assert kwargs == {}
            self.host = host

    monkeypatch.setattr(polymarket_api, "ClobClient", ReadOnlyClob)
    monkeypatch.setattr(settings, "polymarket_private_key", "0x" + "1" * 64)

    api = PolymarketAPI()

    assert api.trading_enabled is False
    assert isinstance(api.clob, ReadOnlyClob)
