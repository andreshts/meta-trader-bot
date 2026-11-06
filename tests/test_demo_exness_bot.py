"""Límites del lanzador: solo la demo autorizada, duración acotada y hora del modelo."""
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import config
import demo_exness_bot as lanzador
from features import build_features


@pytest.fixture
def cuenta(monkeypatch):
    datos = SimpleNamespace(login=lanzador.LOGIN, server=lanzador.SERVER, trade_mode=0)
    monkeypatch.setattr(lanzador.mt5, "account_info", lambda: datos)
    return datos


def test_acepta_la_cuenta_autorizada(cuenta):
    assert lanzador.verificar_cuenta() == lanzador.LOGIN


@pytest.mark.parametrize("campo,valor", [("trade_mode", 2), ("trade_mode", 1), ("login", 123), ("server", "Exness-Real")])
def test_cuenta_distinta_detiene_el_bot(cuenta, campo, valor):
    setattr(cuenta, campo, valor)
    with pytest.raises(SystemExit, match="Cuenta inesperada"):
        lanzador.verificar_cuenta()


def test_el_limite_de_tiempo_termina_sin_ejecutar_el_ciclo(monkeypatch):
    ciclo = Mock()
    monkeypatch.setattr(lanzador, "_ciclo", ciclo)
    with pytest.raises(SystemExit):
        lanzador.ciclo_con_limite(fin=0)({}, {})
    ciclo.assert_not_called()


def test_antes_del_limite_ejecuta_el_ciclo(monkeypatch):
    ciclo = Mock()
    monkeypatch.setattr(lanzador, "_ciclo", ciclo)
    lanzador.ciclo_con_limite(fin=float("inf"))({}, {})
    ciclo.assert_called_once()


def test_el_modelo_se_valida_contra_su_simbolo_y_se_restaura(monkeypatch):
    visto = []
    monkeypatch.setattr(config, "SYMBOL", lanzador.SYMBOL)
    monkeypatch.setattr(lanzador, "_cargar_modelo", lambda: visto.append(config.SYMBOL) or {})
    lanzador.cargar_modelo()
    assert visto == [lanzador.SIMBOLO_MODELO]
    assert config.SYMBOL == lanzador.SYMBOL


def test_la_hora_del_modelo_va_desplazada_y_el_resto_no_cambia():
    n = 60
    velas = pd.DataFrame({"time": [1793994300 + 900 * i for i in range(n)], "open": 1.1, "high": 1.2, "low": 1.0,
                          "close": [1.1 + i * 1e-4 for i in range(n)], "tick_volume": 100, "spread": 8})
    desplazadas = lanzador.features_en_hora_del_modelo(velas, 1e-5)
    esperadas = build_features(velas.assign(time=velas["time"] + 7200), 1e-5)
    pd.testing.assert_series_equal(desplazadas["hora_sin"], esperadas["hora_sin"])
    assert not desplazadas["hora_sin"].equals(build_features(velas, 1e-5)["hora_sin"])
    pd.testing.assert_series_equal(desplazadas["atr"], build_features(velas, 1e-5)["atr"])
    assert velas["time"].iloc[0] == 1793994300  # no modifica las velas originales
