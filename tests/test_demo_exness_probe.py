"""Límites de la prueba manual: jamás enviar a una cuenta real o distinta."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import demo_exness_probe as probe


@pytest.fixture
def terminal(monkeypatch):
    account = SimpleNamespace(login=probe.LOGIN, server=probe.SERVER, currency="USD", trade_mode=0,
                              trade_allowed=True, trade_expert=True)
    terminal = SimpleNamespace(connected=True, trade_allowed=True, tradeapi_disabled=False)
    monkeypatch.setattr(probe.mt5, "account_info", lambda: account)
    monkeypatch.setattr(probe.mt5, "terminal_info", lambda: terminal)
    send = Mock()
    monkeypatch.setattr(probe.mt5, "order_send", send)
    return account, terminal, send


@pytest.mark.parametrize("field,value", [("trade_mode", 2), ("login", 123), ("server", "Exness-Real"),
                                         ("currency", "USC")])
def test_wrong_account_never_sends(terminal, field, value):
    account, _, send = terminal
    setattr(account, field, value)
    with pytest.raises(RuntimeError, match="Cuenta inesperada"):
        probe.send({"symbol": probe.SYMBOL, "magic": probe.MAGIC})
    send.assert_not_called()


def test_disabled_terminal_never_sends(terminal):
    _, term, send = terminal
    term.trade_allowed = False
    with pytest.raises(RuntimeError, match="no habilitado"):
        probe.send({"symbol": probe.SYMBOL, "magic": probe.MAGIC})
    send.assert_not_called()


def test_account_switch_during_check_never_sends(terminal, monkeypatch):
    account, _, send = terminal

    def check(_):
        account.trade_mode = 2
        return SimpleNamespace(retcode=0)

    monkeypatch.setattr(probe.mt5, "order_check", check)
    with pytest.raises(RuntimeError, match="Cuenta inesperada"):
        probe.send({"symbol": probe.SYMBOL, "magic": probe.MAGIC})
    send.assert_not_called()


def test_invalid_quote_blocks(terminal, monkeypatch):
    monkeypatch.setattr(probe.time, "time", lambda: 100)
    monkeypatch.setattr(probe.mt5, "symbol_info_tick", lambda _: SimpleNamespace(time=50, bid=1.1, ask=1.2))
    with pytest.raises(RuntimeError, match="desactualizada"):
        probe.fresh_tick()


def test_foreign_order_is_rejected(terminal):
    with pytest.raises(RuntimeError, match="ajena"):
        probe.send({"symbol": "GBPUSD", "magic": probe.MAGIC})
    terminal[2].assert_not_called()
